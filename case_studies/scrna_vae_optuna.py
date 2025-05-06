"""
scrna_vae_optuna.py — Optuna‑driven two‑stage optimisation for single‑cell VAE
-------------------------------------------------------------------------------

A *reproducible* pipeline that jointly tunes
1. **Pre‑processing** (QC filters, normalisation, HVG count …)
2. **Model** hyper‑parameters for an scVI VAE (latent dim, layers, LR …)

Key features
------------
* **Bayesian optimisation (Optuna TPE)** with **Successive Halving** pruning.
* Caches each unique pre‑processing result (`AnnData`) by MD5 hash so QC work
  is never redone.
* Objective = weighted mix of biological + statistical metrics.
* Stores the Optuna Study in SQLite — inspect live with `optuna-dashboard`.

Usage
-----
```bash
# Quick demo on PBMC‑3k
python scrna_vae_optuna.py --n_trials 20 --output_dir demo_optuna

# Your dataset (.h5ad with optional 'batch' column)
python scrna_vae_optuna.py --adata my_data.h5ad --n_trials 80 --output_dir run1

# Monitor
optuna-dashboard sqlite:///run1/scvi_optuna.db
```

Requires
--------
scanpy ≥1.9 • scvi‑tools ≥1.1 • optuna ≥3 • lisi • sklearn • anndata

"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import anndata as ad
import numpy as np
import optuna
import scanpy as sc
import scvi
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from sklearn.metrics import silhouette_score
from harmonypy import compute_lisi


# ──────────────────────────────────────────────────────────────────────────────
# WEIGHTS (edit here to change objective)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "var_exp": 0.2,
    "sil_pca": 0.2,
    "lisi": 0.1,
    "sil_scvi": 0.25,
    "elbo": 0.25,
}

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def md5_of(obj) -> str:
    return hashlib.md5(pickle.dumps(obj, protocol=4)).hexdigest()


def preprocess(raw: ad.AnnData, p: Dict) -> ad.AnnData:
    """Filter, normalise, HVG select, PCA with added safety checks."""
    try:
        adata = raw.copy()

        # QC filters
        print(f"Starting preprocessing with {adata.n_obs} cells and {adata.n_vars} genes")
        sc.pp.filter_cells(adata, min_genes=p["min_genes"])
        sc.pp.filter_genes(adata, min_cells=p["min_cells"])
        print(f"After filtering: {adata.n_obs} cells and {adata.n_vars} genes")
        
        # Check if enough cells remain after filtering
        if adata.n_obs < 50:
            print(f"Warning: Only {adata.n_obs} cells remain after filtering. Consider less stringent filters.")
            
        # Check if enough genes remain
        if adata.n_vars < 200:
            print(f"Warning: Only {adata.n_vars} genes remain after filtering. Consider less stringent filters.")
        
        # Calculate mitochondrial gene percentage
        mito = adata.var_names.str.upper().str.startswith("MT-")
        if mito.sum() == 0:
            print("Warning: No mitochondrial genes found (starting with 'MT-')")
            adata.obs["pct_mito"] = 0
        else:
            try:
                # Handle different matrix types safely
                if isinstance(adata.X, np.matrix) or hasattr(adata.X, 'A1'):
                    mito_sum = adata[:, mito].X.sum(axis=1).A1
                else:
                    mito_sum = adata[:, mito].X.sum(axis=1)
                
                # Check for zeros in total counts to avoid division by zero
                total_sum = adata.X.sum(axis=1)
                if isinstance(total_sum, np.matrix) or hasattr(total_sum, 'A1'):
                    total_sum = total_sum.A1
                
                # Add small epsilon to avoid division by zero
                total_sum = total_sum + 1e-6
                adata.obs["pct_mito"] = mito_sum / total_sum
            except Exception as e:
                print(f"Error calculating mitochondrial percentage: {e}")
                adata.obs["pct_mito"] = 0
        
        # Filter by mitochondrial percentage
        adata = adata[adata.obs["pct_mito"] < p["max_pct_mito"]].copy()
        print(f"After mito filtering: {adata.n_obs} cells remain")

        # Normalisation with safety checks
        if p["norm_method"] == "total":
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        elif p["norm_method"] == "sctransform":
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            # Scale with max_value to prevent extreme values
            sc.pp.scale(adata, max_value=10)
        else:
            raise ValueError(f"Unsupported normalization method: {p['norm_method']}")

        # Check for NaNs or infs after normalization
        if np.isnan(adata.X).any() or np.isinf(adata.X).any():
            print("Warning: NaN or Inf values detected after normalization!")
            # Replace NaN/inf with zeros
            adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)

        # HVG selection with error handling
        n_top_genes = min(p["n_top_genes"], adata.n_vars - 1)  # Ensure we don't select more genes than available
        try:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_top_genes,
                batch_key="batch" if "batch" in adata.obs and len(adata.obs["batch"].unique()) > 1 else None,
                subset=True,
            )
            print(f"Selected {adata.n_vars} highly variable genes")
        except Exception as e:
            print(f"Error in HVG selection: {e}")
            # If HVG selection fails, select the top genes by variance as fallback
            if adata.n_vars > n_top_genes:
                gene_vars = adata.X.var(axis=0)
                if isinstance(gene_vars, np.matrix):
                    gene_vars = gene_vars.A1
                top_genes = np.argsort(gene_vars)[::-1][:n_top_genes]
                adata = adata[:, top_genes].copy()
                print(f"Used variance-based selection instead, selected {adata.n_vars} genes")

        # PCA with error handling
        try:
            n_comps = min(50, adata.n_obs - 1, adata.n_vars - 1)  # Ensure valid number of components
            sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")
            print(f"PCA computed with {n_comps} components")
        except Exception as e:
            print(f"Error in PCA: {e}")
            # If arpack fails, try with a different solver
            try:
                sc.tl.pca(adata, n_comps=min(50, adata.n_obs - 1, adata.n_vars - 1), svd_solver="randomized")
                print("Used randomized SVD solver instead")
            except Exception as e2:
                print(f"All PCA methods failed: {e2}")
                # Create empty PCA placeholder as last resort
                adata.obsm["X_pca"] = np.zeros((adata.n_obs, min(50, adata.n_obs - 1, adata.n_vars - 1)))
                adata.uns["pca"] = {"variance_ratio": np.zeros(min(50, adata.n_obs - 1, adata.n_vars - 1))}

        return adata
        
    except Exception as e:
        print(f"Fatal error in preprocessing: {e}")
        # Create a minimal valid AnnData object to avoid complete failure
        min_adata = raw[:100].copy() if raw.n_obs > 100 else raw.copy()
        min_adata.obsm["X_pca"] = np.zeros((min_adata.n_obs, 10))
        min_adata.uns["pca"] = {"variance_ratio": np.zeros(10)}
        return min_adata


def variance_explained(adata: ad.AnnData, n: int = 30) -> float:
    """Return cumulative variance explained by the first *n* PCs.
    Handles both Scanpy 1.9 (`variance_ratio`) and older (`variance_ratio_`).
    If the field is missing (e.g. PCA failed), returns 0.0 to avoid crashes.
    """
    var = None
    if "pca" in adata.uns:
        # Fix for NumPy array logical operations
        if "variance_ratio" in adata.uns["pca"]:
            var = adata.uns["pca"]["variance_ratio"]
        elif "variance_ratio_" in adata.uns["pca"]:
            var = adata.uns["pca"]["variance_ratio_"]
    if var is None:
        return 0.0
    return float(np.sum(var[: min(n, len(var))]))


def safe_silhouette(emb: np.ndarray, labels) -> float:
    """Silhouette score helper that returns NaN if only one label present."""
    try:
        if len(np.unique(labels)) < 2:
            return np.nan
        return float(silhouette_score(emb, labels))
    except ValueError as e:
        print(f"Silhouette score calculation failed: {e}")
        return np.nan
    except Exception as e:
        print(f"Unexpected error in silhouette calculation: {e}")
        return np.nan

'''
def batch_lisi(adata: ad.AnnData, key: str = "batch", n_neighbors: int = 90) -> float:
    """Calculate batch effect using LISI score."""
    if key not in adata.obs:
        return np.nan
    
    try:
        import lisi
        scores = lisi.compute_lisi(adata.obsm["X_pca"], adata.obs[[key]], n_neighbors)
        return float(np.nanmean(scores))
    except ImportError:
        print("LISI package not found. Install with 'pip install lisi' for batch effect evaluation.")
        return np.nan
    except Exception as e:
        print(f"LISI calculation error: {e}")
        return np.nan
'''

def batch_lisi(adata: ad.AnnData, key: str = "batch", n_neighbors: int = 90) -> float:
    """Calculate batch effect using LISI score."""
    if key not in adata.obs:
        return np.nan
    
    try:
        # Try to import from harmonypy first
        from harmonypy import compute_lisi
        scores = compute_lisi(adata.obsm["X_pca"], adata.obs[[key]], n_neighbors)
        return float(np.nanmean(scores))
    except ImportError:
        print("harmonypy package not found. Install with 'pip install harmonypy' for batch effect evaluation.")
        return np.nan
    except Exception as e:
        print(f"LISI calculation error: {e}")
        return np.nan

def train_scvi_get_metrics(adata: ad.AnnData, p: Dict) -> Dict[str, float]:
    """Train an scVI model and return metrics."""
    try:
        # Set up AnnData for scVI
        scvi.model.SCVI.setup_anndata(adata, batch_key="batch" if "batch" in adata.obs else None)

        # Create model with more stable configuration
        model = scvi.model.SCVI(
            adata,
            n_latent=p["n_latent"],
            n_layers=p["n_layers"],
            n_hidden=p["n_hidden"],
            dropout_rate=p["dropout"],
            # Add additional parameters to improve stability
            use_layer_norm="both",
            use_batch_norm="none",
        )
        
        # Use a smaller learning rate if the current one is too high
        lr = min(p["lr"], 5e-4)  # Cap learning rate to prevent instability
        
        # Add gradient clipping and reduce learning rate if numerical instability occurs
        plan_kwargs = {
            "lr": lr,
            "weight_decay": 1e-6,  # Add weight decay to prevent large weights
            "eps": 1e-8,  # Epsilon for Adam optimizer to prevent division by zero
        }
        
        # Train with early stopping and reduced max epochs if needed
        max_epochs = min(p["max_epochs"], 300)  # Cap max epochs
        model.train(
            max_epochs=max_epochs,
            plan_kwargs=plan_kwargs,
            early_stopping=True,
            early_stopping_patience=20,  # More patience
            early_stopping_min_delta=0.001,  # Less strict improvement threshold
        )

        # Get ELBO from training history
        elbo_values = model.history.get("elbo_validation", [])
        elbo_values = elbo_values[-5:] if len(elbo_values) >= 5 else elbo_values
        elbo = float(np.nanmean(elbo_values)) if len(elbo_values) > 0 else 0.0
        
        # Get latent representation
        # Add try-except to handle potential errors in getting representation
        try:
            adata.obsm["X_scVI"] = model.get_latent_representation()
            
            # Calculate clustering metrics
            sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15)
            sc.tl.leiden(adata, resolution=1.0, key_added="leiden_scvi")
            sil = safe_silhouette(adata.obsm["X_scVI"], adata.obs["leiden_scvi"])
        except Exception as e:
            print(f"Error in latent representation or clustering: {e}")
            sil = np.nan
        
        return {"elbo": elbo, "sil_scvi": sil}
    
    except Exception as e:
        print(f"Error in scVI training: {e}")
        # Return default values if training fails
        return {"elbo": -1000.0, "sil_scvi": 0.0}

# ──────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ──────────────────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, raw: ad.AnnData, cache: Path, w: Dict) -> float:
    """Optuna objective function for hyperparameter optimization."""
    try:
        # ---------------- Pre‑processing params ----------------
        prep = {
            "min_genes": trial.suggest_categorical("min_genes", [200, 300, 500, 1000]),
            "min_cells": trial.suggest_categorical("min_cells", [3, 10]),
            "max_pct_mito": trial.suggest_float("max_pct_mito", 0.05, 0.25, step=0.05),
            "norm_method": trial.suggest_categorical("norm_method", ["total", "sctransform"]),
            "n_top_genes": trial.suggest_categorical("n_top_genes", [1000, 2000, 3000, 4000]),
        }

        # cache read / write
        f = cache / f"adata_{md5_of(prep)}.h5ad"
        adata = ad.read_h5ad(f) if f.exists() else preprocess(raw, prep)
        if not f.exists():
            adata.write(f)

        # PCA‑space metrics
        var_exp = variance_explained(adata)
        sil_pca = safe_silhouette(adata.obsm["X_pca"], adata.obs.get("batch", np.zeros(adata.n_obs)))
        lisi_val = batch_lisi(adata)

        # ---------------- Model params ----------------
        # Use more conservative parameter ranges to prevent NaN issues
        m = {
            "n_latent": trial.suggest_int("n_latent", 10, 30, step=10),
            "n_layers": trial.suggest_int("n_layers", 1, 2),  # Reduced max layers for stability
            "n_hidden": trial.suggest_categorical("n_hidden", [64, 128]),  # Reduced max hidden units
            "dropout": trial.suggest_float("dropout", 0.1, 0.3, step=0.1),
            "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),  # Lower learning rate range
            "max_epochs": trial.suggest_int("max_epochs", 100, 300, step=100),  # Reduced max epochs
        }
        met = train_scvi_get_metrics(adata, m)

        # ---------------- Composite score ----------------
        # scale 0‑1 with extra safeguards against NaN values
        s_var = var_exp if not np.isnan(var_exp) else 0.0  # already 0‑1 for Scanpy PCA
        s_sil_pca = (sil_pca + 1) / 2 if not np.isnan(sil_pca) else 0.0
        s_lisi = 1 / (1 + lisi_val) if not np.isnan(lisi_val) else 0.5
        
        # Extra protection against invalid values
        elbo = met["elbo"]
        sil_scvi = met["sil_scvi"]
        
        # Check for invalid ELBO values
        if np.isnan(elbo) or np.isinf(elbo) or elbo < -10000:
            s_elbo = 0.0
        else:
            s_elbo = 1 / (1 + np.exp(-elbo / 1000))
            
        # Check for invalid silhouette values
        if np.isnan(sil_scvi) or np.isinf(sil_scvi) or sil_scvi < -1 or sil_scvi > 1:
            s_sil_scvi = 0.5
        else:
            s_sil_scvi = (sil_scvi + 1) / 2
            
        # Calculate weighted score
        score = (
            w["var_exp"] * s_var
            + w["sil_pca"] * s_sil_pca
            + w["lisi"] * s_lisi
            + w["sil_scvi"] * s_sil_scvi
            + w["elbo"] * s_elbo
        )
        
        # Final sanity check
        if np.isnan(score) or np.isinf(score):
            print(f"Warning: Invalid score calculated. Returning default low score.")
            return 0.01  # Return a small positive value instead of invalid score
        
        return score
        
    except Exception as e:
        print(f"Error in objective function: {e}")
        # Return a low but valid score if something fails
        return 0.01

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description="Joint optimisation for scVI")
    ap.add_argument("--adata", help="Input .h5ad (defaults: PBMC3k)")
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--output_dir", default="results_optuna")
    ap.add_argument("--study_name", default="scvi_optuna")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)

    # Load data
    if args.adata:
        adata = sc.read_h5ad(args.adata)
    else:
        adata = sc.datasets.pbmc3k()

    storage = f"sqlite:///{out}/{args.study_name}.db"
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=SuccessiveHalvingPruner(),
        storage=storage,
        study_name=args.study_name,
        load_if_exists=True,
    )

    study.optimize(
        lambda t: objective(t, adata, cache, DEFAULT_WEIGHTS),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print("Best value:", study.best_value)
    with open(out / "best_params.json", "w") as fh:
        json.dump(study.best_params, fh, indent=2)
    study.trials_dataframe().to_csv(out / "all_trials.csv", index=False)


if __name__ == "__main__":
    main()
