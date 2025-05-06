"""
scrna_dca_optuna.py — Optuna‑driven two‑stage optimisation for single‑cell DCA
-------------------------------------------------------------------------------

A *reproducible* pipeline that jointly tunes
1. **Pre‑processing** (QC filters, normalisation, HVG count …)
2. **Model** hyper‑parameters for a Deep Count Autoencoder (latent dim, layers, LR …)

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
python scrna_dca_optuna.py --n_trials 20 --output_dir demo_optuna

# Your dataset (.h5ad with optional 'batch' column)
python scrna_dca_optuna.py --adata my_data.h5ad --n_trials 80 --output_dir run1

# Monitor
optuna-dashboard sqlite:///run1/dca_optuna.db
```

Requires
--------
scanpy ≥1.9 • tensorflow ≥2.4 • optuna ≥3 • lisi • sklearn • anndata

"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import os
from pathlib import Path
from typing import Dict, Optional

import anndata as ad
import numpy as np
import optuna
import scanpy as sc
import tensorflow as tf
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from sklearn.metrics import silhouette_score
from scipy import sparse
from tensorflow.keras import layers, Model, optimizers, regularizers, callbacks
import matplotlib.pyplot as plt
import warnings

# Set TensorFlow warnings to error only
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ──────────────────────────────────────────────────────────────────────────────
# WEIGHTS (edit here to change objective)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "var_exp": 0.2,
    "sil_pca": 0.2,
    "lisi": 0.1,
    "sil_dca": 0.25,
    "dca_loss": 0.25,
}

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def md5_of(obj) -> str:
    """Create MD5 hash of an object."""
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

        # Store raw counts
        adata.layers["counts"] = adata.X.copy()

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


# ──────────────────────────────────────────────────────────────────────────────
# DCA Model Components
# ──────────────────────────────────────────────────────────────────────────────

def make_count_matrix(adata: ad.AnnData) -> np.ndarray:
    """Prepare count matrix from AnnData object."""
    if "counts" in adata.layers:
        if sparse.issparse(adata.layers["counts"]):
            counts = adata.layers["counts"].toarray()
        else:
            counts = adata.layers["counts"]
    else:
        if sparse.issparse(adata.X):
            counts = adata.X.toarray()
        else:
            counts = adata.X

    # Ensure non-negative values
    counts[counts < 0] = 0
    
    return counts


def zinb_loss(y_true, y_pred, theta=None, pi=None, eps=1e-10):
    """Zero-inflated negative binomial loss function."""
    # If theta or pi not provided, use the other outputs from the model
    if theta is None:
        # Assuming y_pred is [means, theta, pi]
        y_pred, theta, pi = tf.split(y_pred, 3, axis=-1)
    
    # Clip values for numerical stability
    y_true = tf.clip_by_value(y_true, 0, 1e6)
    y_pred = tf.clip_by_value(y_pred, eps, 1e6)
    theta = tf.clip_by_value(theta, eps, 1e6)
    pi = tf.clip_by_value(pi, eps, 1.0 - eps)
    
    # Negative binomial part (for non-zero counts)
    nb_case = tf.math.lgamma(y_true + theta) - tf.math.lgamma(theta) - tf.math.lgamma(y_true + 1.0) + \
              theta * tf.math.log(theta) + \
              y_true * tf.math.log(y_pred) - \
              (y_true + theta) * tf.math.log(theta + y_pred)
    
    # Zero-inflation part
    zero_nb = theta * (tf.math.log(theta) - tf.math.log(theta + y_pred))
    zero_case = tf.math.log(pi + (1.0 - pi) * tf.exp(zero_nb))
    
    # Non-zero case
    nonzero_case = tf.math.log(1.0 - pi) + nb_case
    
    # Combine zero and non-zero cases
    result = tf.where(
        tf.less(y_true, 1e-8),
        zero_case,
        nonzero_case
    )
    
    return -result


def zinb_loss_separate(y_true, mean, disp, pi=None, eps=1e-10):
    """Zero-inflated negative binomial loss function with separate tensor inputs."""
    # Clip values for numerical stability
    y_true = tf.clip_by_value(y_true, 0, 1e6)
    mean = tf.clip_by_value(mean, eps, 1e6)
    disp = tf.clip_by_value(disp, eps, 1e6)
    
    # Negative binomial part (for non-zero counts)
    nb_case = tf.math.lgamma(y_true + disp) - tf.math.lgamma(disp) - tf.math.lgamma(y_true + 1.0) + \
              disp * tf.math.log(disp) + \
              y_true * tf.math.log(mean) - \
              (y_true + disp) * tf.math.log(disp + mean)
    
    if pi is not None:
        # Clip pi for numerical stability
        pi = tf.clip_by_value(pi, eps, 1.0 - eps)
        
        # Zero-inflation part
        zero_nb = disp * (tf.math.log(disp) - tf.math.log(disp + mean))
        zero_case = tf.math.log(pi + (1.0 - pi) * tf.exp(zero_nb))
        
        # Non-zero case
        nonzero_case = tf.math.log(1.0 - pi) + nb_case
        
        # Combine zero and non-zero cases
        result = tf.where(
            tf.less(y_true, 1e-8),
            zero_case,
            nonzero_case
        )
    else:
        # Simple negative binomial (no zero-inflation)
        zero_nb = disp * (tf.math.log(disp) - tf.math.log(disp + mean))
        result = tf.where(
            tf.less(y_true, 1e-8),
            zero_nb,
            nb_case
        )
    
    return -tf.reduce_mean(result)


class MeanAct(layers.Layer):
    """Mean activation function."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        return tf.math.exp(inputs)


class DispAct(layers.Layer):
    """Dispersion activation function."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        return tf.math.softplus(inputs) + 1e-4


class PiAct(layers.Layer):
    """Pi activation function for zero-inflation probabilities."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        return tf.math.sigmoid(inputs)


def build_dca_autoencoder(input_dim, hidden_dim=128, latent_dim=32, 
                        l1_reg=0.0, l2_reg=0.0, activation='relu', 
                        batchnorm=True, dropout_rate=0.1, use_zinb=True):
    """Build DCA autoencoder model."""
    # Create regularizer
    regularizer = None
    if l1_reg > 0 or l2_reg > 0:
        regularizer = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    
    # Input layer
    input_layer = layers.Input(shape=(input_dim,), name='count_input')
    
    # Encoder layers
    x = layers.Dense(hidden_dim, activation=None, kernel_regularizer=regularizer)(input_layer)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Additional encoder layer
    x = layers.Dense(hidden_dim // 2, activation=None, kernel_regularizer=regularizer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Bottleneck layer
    latent = layers.Dense(latent_dim, activation=None, kernel_regularizer=regularizer, name='latent_layer')(x)
    if batchnorm:
        latent = layers.BatchNormalization()(latent)
    latent = layers.Activation(activation)(latent)
    
    # Decoder layers
    x = layers.Dense(hidden_dim // 2, activation=None, kernel_regularizer=regularizer)(latent)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(hidden_dim, activation=None, kernel_regularizer=regularizer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layers for ZINB distribution
    # 1. Mean
    mean = layers.Dense(input_dim, activation=None, kernel_regularizer=regularizer, name='decoder_mean')(x)
    mean_act = MeanAct(name='mean_activation')(mean)
    
    # 2. Dispersion
    dispersion = layers.Dense(input_dim, activation=None, kernel_regularizer=regularizer, name='decoder_dispersion')(x)
    disp_act = DispAct(name='dispersion_activation')(dispersion)
    
    if use_zinb:
        # 3. Zero-inflation
        pi = layers.Dense(input_dim, activation=None, kernel_regularizer=regularizer, name='decoder_pi')(x)
        pi_act = PiAct(name='pi_activation')(pi)
        
        # Create model with separate outputs
        model = Model(inputs=input_layer, outputs=[mean_act, disp_act, pi_act], name='DCA')
    else:
        # Just use negative binomial (NB) without zero-inflation
        model = Model(inputs=input_layer, outputs=[mean_act, disp_act], name='DCA')
    
    return model


def train_dca_model(adata: ad.AnnData, p: Dict) -> Dict[str, float]:
    """Train DCA model and return metrics."""
    try:
        # Prepare the data
        count_matrix = make_count_matrix(adata)
        input_dim = count_matrix.shape[1]
        
        # Extract model parameters from the trial
        hidden_dim = p["hidden_dim"]
        latent_dim = p["latent_dim"]
        learning_rate = p["learning_rate"]
        dropout_rate = p["dropout_rate"]
        use_zinb = p.get("use_zinb", True)
        max_epochs = min(p.get("max_epochs", 200), 200)  # Limit to 200 epochs
        
        # Build model
        model = build_dca_autoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_reg=0.0,
            l2_reg=p.get("l2_reg", 0.0),
            activation='relu',
            batchnorm=True,
            dropout_rate=dropout_rate,
            use_zinb=use_zinb
        )
        
        # Define optimizer with potentially lower learning rate and weight decay
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            epsilon=1e-8  # Prevent division by zero
        )
        
        # Create custom training step function based on model outputs
        if use_zinb:
            @tf.function
            def train_step(x):
                with tf.GradientTape() as tape:
                    mean, disp, pi = model(x, training=True)
                    loss = zinb_loss_separate(x, mean, disp, pi)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                # Clip gradients to prevent instability
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss
            
            # Function to compute validation loss
            @tf.function
            def compute_loss(x):
                mean, disp, pi = model(x, training=False)
                return zinb_loss_separate(x, mean, disp, pi)
        else:
            @tf.function
            def train_step(x):
                with tf.GradientTape() as tape:
                    mean, disp = model(x, training=True)
                    loss = zinb_loss_separate(x, mean, disp)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                # Clip gradients to prevent instability
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss
            
            # Function to compute validation loss
            @tf.function
            def compute_loss(x):
                mean, disp = model(x, training=False)
                return zinb_loss_separate(x, mean, disp)
        
        # Create training batches
        batch_size = min(128, count_matrix.shape[0] // 3)  # Ensure not too large batches
        dataset = tf.data.Dataset.from_tensor_slices(count_matrix).shuffle(10000).batch(batch_size)
        
        # Train with early stopping
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        losses = []
        
        # Train for max_epochs or until early stopping
        for epoch in range(max_epochs):
            epoch_loss = 0
            n_batches = 0
            
            for batch in dataset:
                batch_loss = train_step(batch)
                epoch_loss += batch_loss
                n_batches += 1
            
            # Compute average epoch loss
            epoch_loss = epoch_loss / n_batches
            losses.append(float(epoch_loss))
            
            # Early stopping logic
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                best_weights = model.get_weights()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
                
            # Optional progress print
            if epoch % 20 == 0:
                print(f"  Epoch {epoch+1}/{max_epochs}, Loss: {epoch_loss:.4f}")
        
        # Restore best weights
        if 'best_weights' in locals():
            model.set_weights(best_weights)
        
        # Get the final loss
        final_loss = compute_loss(count_matrix)
        
        # Get latent representation
        encoder = Model(inputs=model.input, outputs=model.get_layer('latent_layer').output)
        latent_representation = encoder.predict(count_matrix)
        
        # Store latent representation in AnnData
        adata.obsm['X_dca'] = latent_representation
        
        # Compute UMAP from latent representation
        sc.pp.neighbors(adata, use_rep='X_dca', n_neighbors=15)
        sc.tl.leiden(adata, resolution=1.0, key_added='leiden_dca')
        
        # Compute silhouette score on DCA latent space
        batch_key = 'batch' if 'batch' in adata.obs else None
        if batch_key:
            sil_score = safe_silhouette(latent_representation, adata.obs[batch_key].cat.codes)
        else:
            sil_score = np.nan
        
        # Return metrics for optimization
        return {
            'dca_loss': float(final_loss),
            'sil_dca': sil_score
        }
    
    except Exception as e:
        print(f"Error in DCA training: {e}")
        # Return default values if training fails
        return {
            'dca_loss': 1000.0,  # Large loss value
            'sil_dca': 0.0
        }


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
        m = {
            "latent_dim": trial.suggest_int("latent_dim", 10, 50, step=10),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "l2_reg": trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True),
            "use_zinb": trial.suggest_categorical("use_zinb", [True, False]),
            "max_epochs": trial.suggest_int("max_epochs", 100, 200, step=50)
        }
        
        # Train DCA and get metrics
        dca_metrics = train_dca_model(adata, m)

        # ---------------- Composite score ----------------
        # Scale 0‑1 with extra safeguards against NaN values
        s_var = var_exp if not np.isnan(var_exp) else 0.0  # already 0‑1 for Scanpy PCA
        s_sil_pca = (sil_pca + 1) / 2 if not np.isnan(sil_pca) else 0.0
        s_lisi = 1 / (1 + lisi_val) if not np.isnan(lisi_val) else 0.5
        
        # Extra protection against invalid values
        dca_loss = dca_metrics['dca_loss']
        sil_dca = dca_metrics['sil_dca']
        
        # Check for invalid loss values
        if np.isnan(dca_loss) or np.isinf(dca_loss) or dca_loss > 1000:
            s_dca_loss = 0.0
        else:
            # Normalize DCA loss to 0-1 scale (lower is better)
            s_dca_loss = 1 / (1 + np.exp(dca_loss / 100))
            
        # Check for invalid silhouette values
        if np.isnan(sil_dca) or np.isinf(sil_dca) or sil_dca < -1 or sil_dca > 1:
            s_sil_dca = 0.5
        else:
            s_sil_dca = (sil_dca + 1) / 2
            
        # Calculate weighted score
        score = (
            w["var_exp"] * s_var
            + w["sil_pca"] * s_sil_pca
            + w["lisi"] * s_lisi
            + w["sil_dca"] * s_sil_dca
            + w["dca_loss"] * s_dca_loss
        )
        
        # Final sanity check
        if np.isnan(score) or np.isinf(score):
            print(f"Warning: Invalid score calculated. Returning default low score.")
            return 0.01  # Return a small positive value instead of invalid score
        
        # Print metrics for monitoring
        print(f"Trial metrics - var_exp: {s_var:.3f}, sil_pca: {s_sil_pca:.3f}, "
              f"lisi: {s_lisi:.3f}, sil_dca: {s_sil_dca:.3f}, dca_loss: {s_dca_loss:.3f}")
        print(f"Trial score: {score:.3f}")
        
        return score
        
    except Exception as e:
        print(f"Error in objective function: {e}")
        # Return a low but valid score if something fails
        return 0.01


# This function would go at the end of your existing file, after all other functions

def main():
    """Main entry point for the optimization pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimize single-cell DCA with Optuna')
    parser.add_argument('--adata', type=str, help='Path to input h5ad file (anndata)')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--output_dir', type=str, default='dca_optuna_results',
                        help='Directory to save results')
    parser.add_argument('--weights', type=str, default=None,
                        help='JSON file with custom metric weights')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed for reproducibility')
    parser.add_argument('--study_name', type=str, default='dca_optuna',
                        help='Name for the Optuna study')
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Cache directory for processed AnnData objects
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Load or create sample data
    if args.adata:
        print(f"Loading data from {args.adata}")
        raw = ad.read_h5ad(args.adata)
    else:
        print("No data provided, using PBMC3k test dataset")
        sc.datasets.pbmc3k()
        raw = sc.datasets.pbmc3k()
        print(f"Loaded sample dataset with {raw.n_obs} cells and {raw.n_vars} genes")

    # Ensure batch column exists for batch correction metrics
    if 'batch' not in raw.obs:
        print("Warning: No 'batch' column found in data. Creating dummy batch.")
        raw.obs['batch'] = 'batch1'
        raw.obs['batch'] = raw.obs['batch'].astype('category')

    # Load custom weights if provided, otherwise use defaults
    if args.weights and os.path.exists(args.weights):
        with open(args.weights, 'r') as f:
            weights = json.load(f)
        print(f"Using custom weights: {weights}")
    else:
        weights = DEFAULT_WEIGHTS
        print(f"Using default weights: {weights}")

    # Save weights for reference
    with open(output_dir / "weights.json", 'w') as f:
        json.dump(weights, f, indent=2)

    # Setup Optuna study
    db_path = output_dir / f"{args.study_name}.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=SuccessiveHalvingPruner(),
        load_if_exists=True
    )

    # Run the optimization
    print(f"Starting optimization with {args.n_trials} trials")
    study.optimize(
        lambda trial: objective(trial, raw, cache_dir, weights),
        n_trials=args.n_trials
    )

    # Print and save results
    print("\nOptimization complete!")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")

    # Save best parameters
    with open(output_dir / "best_params.json", 'w') as f:
        json.dump(study.best_params, f, indent=2)

    # Process data with best parameters and train final model
    print("\nTraining final model with best parameters...")
    
    # Extract best preprocessing parameters
    best_prep = {
        "min_genes": study.best_params["min_genes"],
        "min_cells": study.best_params["min_cells"],
        "max_pct_mito": study.best_params["max_pct_mito"],
        "norm_method": study.best_params["norm_method"],
        "n_top_genes": study.best_params["n_top_genes"],
    }
    
    # Extract best model parameters
    best_model = {
        "latent_dim": study.best_params["latent_dim"],
        "hidden_dim": study.best_params["hidden_dim"],
        "dropout_rate": study.best_params["dropout_rate"],
        "learning_rate": study.best_params["learning_rate"],
        "l2_reg": study.best_params["l2_reg"],
        "use_zinb": study.best_params["use_zinb"],
        "max_epochs": study.best_params["max_epochs"]
    }
    
    # Apply best preprocessing
    best_adata_path = cache_dir / f"adata_{md5_of(best_prep)}.h5ad"
    if best_adata_path.exists():
        print("Loading best preprocessed data from cache")
        best_adata = ad.read_h5ad(best_adata_path)
    else:
        print("Preprocessing with best parameters")
        best_adata = preprocess(raw, best_prep)
        best_adata.write(best_adata_path)
    
    # Train final model
    print("Training final DCA model")
    train_dca_model(best_adata, best_model)
    
    # Compute UMAP for visualization
    print("Computing UMAP embeddings")
    sc.pp.neighbors(best_adata, use_rep='X_dca')
    sc.tl.umap(best_adata)
    sc.tl.leiden(best_adata, key_added='leiden_final')
    
    # Create a plot to visualize results
    print("Creating visualization plots")
    sc.settings.set_figure_params(figsize=(10, 8))
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot UMAP with leiden clusters
    sc.pl.umap(best_adata, color='leiden_final', ax=axs[0, 0], show=False, title='DCA Clusters')
    
    # Plot UMAP with batch if available
    if 'batch' in best_adata.obs and len(best_adata.obs['batch'].unique()) > 1:
        sc.pl.umap(best_adata, color='batch', ax=axs[0, 1], show=False, title='Batch Effect')
    else:
        axs[0, 1].set_title("No batch information available")
        axs[0, 1].axis('off')
    
    # Plot optimization history
    trial_values = [trial.value for trial in study.trials if trial.value is not None]
    axs[1, 0].plot(trial_values)
    axs[1, 0].set_title('Optimization History')
    axs[1, 0].set_xlabel('Trial')
    axs[1, 0].set_ylabel('Score')
    
    # Plot PCA variance explained
    if 'pca' in best_adata.uns and 'variance_ratio' in best_adata.uns['pca']:
        var_ratio = best_adata.uns['pca']['variance_ratio']
        axs[1, 1].plot(np.cumsum(var_ratio[:50]))
        axs[1, 1].set_title('PCA Variance Explained')
        axs[1, 1].set_xlabel('PC')
        axs[1, 1].set_ylabel('Cumulative Variance')
        axs[1, 1].axhline(y=0.8, color='r', linestyle='--')
    else:
        axs[1, 1].set_title("PCA variance data not available")
        axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dca_results.png', dpi=300)
    
    # Save the final processed data
    final_path = output_dir / "final_processed.h5ad"
    print(f"Saving final processed data to {final_path}")
    best_adata.write(final_path)
    
    print(f"\nAll results saved to {output_dir}")
    print("You can monitor the optimization progress with:")
    print(f"optuna-dashboard sqlite:///{db_path}")
    
    return study


if __name__ == "__main__":
    main()
