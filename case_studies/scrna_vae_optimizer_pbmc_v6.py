"""
Hyperparameter optimization for VAE-based batch correction on scRNA-seq data using Optuna
with data saving for Streamlit visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from sklearn.metrics import silhouette_score
import optuna
import scvi
import torch

# Set random seed for reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Base directory for saving results
RESULTS_DIR = "optuna_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Directory for saving data files
DATA_DIR = os.path.join(RESULTS_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def load_pbmc_dataset():
    """
    Load the PBMC dataset from scvi-tools.
    
    Returns:
        AnnData: Combined PBMC dataset with batch information
    """
    print("Loading PBMC dataset...")
    pbmc_data = scvi.data.pbmc_dataset()
    print(f"PBMC dataset: {pbmc_data.shape[0]} cells, {pbmc_data.shape[1]} genes")
    
    # Convert batch to categorical
    pbmc_data.obs['batch'] = pbmc_data.obs['batch'].astype('category')
    
    # Save the raw dataset for Streamlit visualization
    raw_data_path = os.path.join(DATA_DIR, "pbmc_raw.h5ad")
    if not os.path.exists(raw_data_path):
        print(f"Saving raw PBMC dataset to {raw_data_path}")
        pbmc_data.write(raw_data_path)
    
    return pbmc_data


def run_preprocessing(adata, n_top_genes=2000, min_genes=200, min_cells=3, normalize_target=1e4, save_path=None):
    """
    Perform preprocessing steps on the AnnData object
    
    Parameters:
        adata (AnnData): The annotated data matrix
        n_top_genes (int): Number of highly variable genes to select
        min_genes (int): Minimum number of genes per cell
        min_cells (int): Minimum number of cells per gene
        normalize_target (float): Target sum for normalization
        save_path (str): Path to save the preprocessed data
        
    Returns:
        AnnData: Preprocessed data
    """
    print(f"Preprocessing with parameters: n_top_genes={n_top_genes}, min_genes={min_genes}, min_cells={min_cells}, normalize_target={normalize_target}")
    
    # Create a proper copy to avoid ImplicitModificationWarning
    adata = adata.copy()
    
    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # Store raw counts for scVI
    adata.layers["counts"] = adata.X.copy()
    
    # Normalize the data
    sc.pp.normalize_total(adata, target_sum=normalize_target)
    sc.pp.log1p(adata)
    
    # Make sure batch is categorical
    if 'batch' in adata.obs.columns:
        adata.obs['batch'] = adata.obs['batch'].astype('category')
    
    # Select highly variable genes
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key='batch')
        highly_variable_genes = adata.var.highly_variable
        adata_subset = adata[:, highly_variable_genes].copy()
    except Exception as e:
        print(f"Error finding highly variable genes with batch correction: {e}")
        print("Falling back to standard highly variable genes without batch correction")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        highly_variable_genes = adata.var.highly_variable
        adata_subset = adata[:, highly_variable_genes].copy()
    
    print(f"After preprocessing: {adata_subset.shape[0]} cells, {adata_subset.shape[1]} genes")
    
    # Save the preprocessed data if path is provided
    if save_path:
        print(f"Saving preprocessed data to {save_path}")
        adata_subset.write(save_path)
    
    return adata_subset


def run_vae_batch_correction(adata, batch_key='batch', n_latent=10, n_hidden=128, n_layers=2, max_epochs=200, 
                             learning_rate=1e-3, dropout_rate=0.1, use_layer_norm=True, use_batch_norm=True, 
                             save_path=None, save_model_path=None):
    """
    Run scVI VAE model for batch correction with specified parameters
    
    Parameters:
        adata (AnnData): The annotated data matrix
        batch_key (str): The key in adata.obs for batch information
        n_latent (int): Dimensionality of the latent space
        n_hidden (int): Number of nodes per hidden layer
        n_layers (int): Number of hidden layers
        max_epochs (int): Maximum number of training epochs
        learning_rate (float): Learning rate for the optimizer (NOTE: may not be used in all scVI versions)
        dropout_rate (float): Dropout rate for regularization
        use_layer_norm (bool): Whether to use layer normalization
        use_batch_norm (bool): Whether to use batch normalization
        save_path (str): Path to save the corrected data
        save_model_path (str): Path to save the trained model
        
    Returns:
        tuple: (model, adata_with_latent)
    """
    print(f"Training VAE with parameters: n_latent={n_latent}, n_hidden={n_hidden}, n_layers={n_layers}, " +
          f"learning_rate={learning_rate}, dropout_rate={dropout_rate}, layer_norm={use_layer_norm}, batch_norm={use_batch_norm}")
    
    # Ensure batch is categorical
    adata.obs[batch_key] = adata.obs[batch_key].astype('category')
    
    # Setup the model
    try:
        scvi.model.SCVI.setup_anndata(
            adata,
            batch_key=batch_key,
            labels_key='labels' if 'labels' in adata.obs.columns else None,
            layer="counts" if "counts" in adata.layers else None
        )
    except Exception as e:
        print(f"Error in setup_anndata: {e}")
        scvi.model.SCVI.setup_anndata(
            adata,
            batch_key=batch_key,
            labels_key='labels' if 'labels' in adata.obs.columns else None
        )
    
    # Define the model with all parameters
    model = scvi.model.SCVI(
        adata,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        use_layer_norm=use_layer_norm,
        use_batch_norm=use_batch_norm,
    )
    
    # Try to set the learning rate via PyTorch optimizer if supported by this version
    try:
        # Get the current optimizer
        optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate)
        
        # Set model's optimizer with desired learning rate
        model._model.optimizer = optimizer
        
        print(f"Successfully set learning rate to {learning_rate} via PyTorch optimizer")
    except Exception as e:
        print(f"Note: Could not set learning rate directly. Using default: {e}")
    
    # Train the model with early stopping
    try:
        # First try with validation split and early stopping
        model.train(
            max_epochs=max_epochs,
            check_val_every_n_epoch=10,
            early_stopping=True,
            early_stopping_patience=15,
            train_size=0.9  # Use 90% of data for training, 10% for validation
        )
    except Exception as e:
        print(f"Warning: Could not train with validation split: {e}")
        print("Trying simplified training...")
        
        # Fallback to simpler training parameters
        model.train(
            max_epochs=max_epochs,
            check_val_every_n_epoch=10
        )
    
    # Get latent representation
    adata.obsm['X_scVI'] = model.get_latent_representation()
    
    # Compute UMAP on latent representation for visualization
    sc.pp.neighbors(adata, use_rep='X_scVI')
    sc.tl.umap(adata)
    
    # Save the corrected data if path is provided
    if save_path:
        print(f"Saving batch-corrected data to {save_path}")
        adata.write(save_path)
    
    # Save the model if path is provided
    if save_model_path:
        print(f"Saving model to {save_model_path}")
        model.save(save_model_path)
    
    return model, adata


def evaluate_batch_correction(adata, batch_key='batch', label_key=None):
    """
    Evaluate the effectiveness of batch correction using silhouette scores
    
    Parameters:
        adata (AnnData): The annotated data matrix with latent representation
        batch_key (str): The key in adata.obs for batch information
        label_key (str): The key in adata.obs for cell type labels
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    results = {}
    
    # Calculate silhouette scores on PCA embedding
    if 'X_pca' not in adata.obsm:
        sc.pp.pca(adata)
    
    batch_sil_pca = silhouette_score(
        adata.obsm['X_pca'], 
        adata.obs[batch_key].cat.codes, 
        metric='euclidean'
    )
    results['batch_silhouette_pca'] = batch_sil_pca
    
    if label_key and label_key in adata.obs.columns:
        cell_sil_pca = silhouette_score(
            adata.obsm['X_pca'], 
            adata.obs[label_key].cat.codes, 
            metric='euclidean'
        )
        results['celltype_silhouette_pca'] = cell_sil_pca
    
    # Calculate silhouette scores on scVI latent space
    batch_sil_scvi = silhouette_score(
        adata.obsm['X_scVI'], 
        adata.obs[batch_key].cat.codes, 
        metric='euclidean'
    )
    results['batch_silhouette_scvi'] = batch_sil_scvi
    
    if label_key and label_key in adata.obs.columns:
        cell_sil_scvi = silhouette_score(
            adata.obsm['X_scVI'], 
            adata.obs[label_key].cat.codes, 
            metric='euclidean'
        )
        results['celltype_silhouette_scvi'] = cell_sil_scvi
    
    return results


def objective(trial):
    """
    Optuna objective function for optimizing the VAE batch correction pipeline
    
    Parameters:
        trial (optuna.Trial): The trial object that manages hyperparameters

    Returns:
        float: A metric to be minimized or maximized
    """
    # Create a unique ID for this trial
    trial_id = trial.number
    
    # Define paths for saving this trial's data
    preprocessed_path = os.path.join(DATA_DIR, f"trial_{trial_id}_preprocessed.h5ad")
    corrected_path = os.path.join(DATA_DIR, f"trial_{trial_id}_corrected.h5ad")
    model_path = os.path.join(DATA_DIR, f"trial_{trial_id}_model")
    
    # Load data once for each trial
    pbmc_data = load_pbmc_dataset()
    
    # Define preprocessing hyperparameters to optimize
    preprocessing_params = {
        'n_top_genes': trial.suggest_int('n_top_genes', 1000, 4000, step=500),
        'min_genes': trial.suggest_int('min_genes', 150, 300, step=50),
        'min_cells': trial.suggest_int('min_cells', 2, 10, step=1),
        'normalize_target': trial.suggest_float('normalize_target', 1e3, 1e5, log=True),
        'save_path': preprocessed_path if trial.number < 3 else None  # Save only first few trials to save disk space
    }
    
    # Run preprocessing with suggested hyperparameters
    pbmc_preprocessed = run_preprocessing(pbmc_data, **preprocessing_params)
    
    # Define VAE hyperparameters to optimize
    vae_params = {
        'n_latent': trial.suggest_int('n_latent', 8, 30, step=2),
        'n_hidden': trial.suggest_int('n_hidden', 64, 256, step=32),
        'n_layers': trial.suggest_int('n_layers', 1, 4, step=1),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1),
        'use_layer_norm': trial.suggest_categorical('use_layer_norm', [True, False]),
        'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
        'max_epochs': 100,  # Fixed to a lower value for faster trials
        'save_path': corrected_path if trial.number < 3 else None,  # Save only first few trials
        'save_model_path': model_path if trial.number < 3 else None  # Save only first few trials
    }
    
    # Run VAE batch correction with suggested hyperparameters
    _, pbmc_corrected = run_vae_batch_correction(pbmc_preprocessed, **vae_params)
    
    # Get appropriate label key
    label_key = 'str_labels' if 'str_labels' in pbmc_corrected.obs.columns else 'labels'
    
    # Evaluate batch correction
    evaluation_results = evaluate_batch_correction(
        pbmc_corrected,
        batch_key='batch',
        label_key=label_key
    )
    
    # Calculate optimization metric:
    # 1. We want to MINIMIZE batch silhouette score in the latent space (batches should mix)
    # 2. We want to MAXIMIZE cell type silhouette score in the latent space (biology preserved)
    # 3. We want batch effect to be reduced compared to PCA space
    
    batch_effect_reduction = evaluation_results['batch_silhouette_pca'] - evaluation_results['batch_silhouette_scvi']
    
    if label_key and 'celltype_silhouette_scvi' in evaluation_results:
        bio_preservation = evaluation_results['celltype_silhouette_scvi']
    else:
        bio_preservation = 0.0
    
    # For Optuna to minimize: lower is better, so we negate bio_preservation
    # We also add a small penalty for batch silhouette score that's still high
    optimization_score = -bio_preservation + 0.5 * evaluation_results['batch_silhouette_scvi'] - 0.5 * batch_effect_reduction
    
    # Save trial results for later analysis
    trial_results = {
        **preprocessing_params,
        **vae_params,
        **evaluation_results,
        'batch_effect_reduction': batch_effect_reduction,
        'bio_preservation': bio_preservation,
        'optimization_score': optimization_score
    }
    
    # Remove save paths from the results
    if 'save_path' in trial_results:
        trial_results.pop('save_path')
    if 'save_model_path' in trial_results:
        trial_results.pop('save_model_path')
    
    # Save the current trial results to a CSV
    results_df = pd.DataFrame([trial_results])
    results_df.to_csv(os.path.join(RESULTS_DIR, f"trial_{trial.number}.csv"), index=False)
    
    # Also append to a running log of all trials
    all_trials_path = os.path.join(RESULTS_DIR, "all_trials.csv")
    if os.path.exists(all_trials_path):
        all_trials_df = pd.read_csv(all_trials_path)
        all_trials_df = pd.concat([all_trials_df, results_df], ignore_index=True)
    else:
        all_trials_df = results_df
    
    all_trials_df.to_csv(all_trials_path, index=False)
    
    # Print current trial results
    print(f"\nTrial {trial.number} completed")
    print(f"Batch silhouette (PCA): {evaluation_results['batch_silhouette_pca']:.4f}")
    print(f"Batch silhouette (scVI): {evaluation_results['batch_silhouette_scvi']:.4f}")
    if 'celltype_silhouette_scvi' in evaluation_results:
        print(f"Cell type silhouette (scVI): {evaluation_results['celltype_silhouette_scvi']:.4f}")
    print(f"Batch effect reduction: {batch_effect_reduction:.4f}")
    print(f"Optimization score: {optimization_score:.4f}")
    
    return optimization_score


def visualize_optimization_results(study):
    """
    Visualize the optimization results from Optuna study
    
    Parameters:
        study (optuna.Study): The completed Optuna study
    """
    # Create directory for figures
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Visualization 1: Parameter importance
    try:
        param_importance = optuna.visualization.plot_param_importances(study)
        param_importance.write_image(os.path.join(figures_dir, "param_importance.png"))
    except Exception as e:
        print(f"Could not create parameter importance plot: {e}")
    
    # Visualization 2: Optimization history
    try:
        opt_history = optuna.visualization.plot_optimization_history(study)
        opt_history.write_image(os.path.join(figures_dir, "optimization_history.png"))
    except Exception as e:
        print(f"Could not create optimization history plot: {e}")
    
    # Visualization 3: Parallel coordinate plot
    try:
        parallel_plot = optuna.visualization.plot_parallel_coordinate(study)
        parallel_plot.write_image(os.path.join(figures_dir, "parallel_coordinate.png"))
    except Exception as e:
        print(f"Could not create parallel coordinate plot: {e}")
    
    # Visualization 4: Slice plot
    try:
        slice_plot = optuna.visualization.plot_slice(study)
        slice_plot.write_image(os.path.join(figures_dir, "slice_plot.png"))
    except Exception as e:
        print(f"Could not create slice plot: {e}")


def run_optimization(n_trials=20, study_name="vae_batch_correction"):
    """
    Run the hyperparameter optimization study
    
    Parameters:
        n_trials (int): Number of trials to run
        study_name (str): Name of the study
        
    Returns:
        optuna.Study: The completed study
    """
    # Create a new study or load an existing one
    storage_path = f"sqlite:///{os.path.join(RESULTS_DIR, f'{study_name}.db')}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        direction="minimize",  # We're minimizing the optimization score
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Run the optimization
    print(f"Starting optimization with {n_trials} trials")
    study.optimize(objective, n_trials=n_trials)
    
    # Get the best trial
    best_trial = study.best_trial
    print("\nBest trial:")
    print(f"  Value: {best_trial.value:.4f}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Visualize results
    visualize_optimization_results(study)
    
    # Save the best parameters
    best_params = best_trial.params
    pd.DataFrame([best_params]).to_csv(os.path.join(RESULTS_DIR, "best_params.csv"), index=False)
    
    return study


def apply_best_parameters(best_params):
    """
    Apply the best parameters to the full dataset and evaluate
    
    Parameters:
        best_params (dict): Dictionary of best parameters
    """
    print("\nApplying best parameters to full dataset...")
    
    # Extract preprocessing and VAE parameters
    preprocessing_params = {
        'n_top_genes': best_params['n_top_genes'],
        'min_genes': best_params['min_genes'],
        'min_cells': best_params['min_cells'],
        'normalize_target': best_params['normalize_target'],
        'save_path': os.path.join(DATA_DIR, "best_preprocessed.h5ad")
    }
    
    vae_params = {
        'n_latent': best_params['n_latent'],
        'n_hidden': best_params['n_hidden'], 
        'n_layers': best_params['n_layers'],
        'learning_rate': best_params['learning_rate'],
        'dropout_rate': best_params['dropout_rate'],
        'use_layer_norm': best_params['use_layer_norm'],
        'use_batch_norm': best_params['use_batch_norm'],
        'max_epochs': 400,  # Use more epochs for the final model
        'save_path': os.path.join(DATA_DIR, "best_corrected.h5ad"),
        'save_model_path': os.path.join(RESULTS_DIR, "best_model")
    }
    
    # Load dataset
    pbmc_data = load_pbmc_dataset()
    
    # Preprocess with best parameters
    pbmc_preprocessed = run_preprocessing(pbmc_data, **preprocessing_params)
    
    # Train VAE with best parameters and save the model
    model, pbmc_corrected = run_vae_batch_correction(pbmc_preprocessed, **vae_params)
    
    # Get appropriate label key
    label_key = 'str_labels' if 'str_labels' in pbmc_corrected.obs.columns else 'labels'
    
    # Evaluate the model
    evaluation_results = evaluate_batch_correction(
        pbmc_corrected,
        batch_key='batch',
        label_key=label_key
    )
    
    # Print results
    print("\nFinal model evaluation:")
    print(f"Batch silhouette (PCA): {evaluation_results['batch_silhouette_pca']:.4f}")
    print(f"Batch silhouette (scVI): {evaluation_results['batch_silhouette_scvi']:.4f}")
    
    if 'celltype_silhouette_pca' in evaluation_results:
        print(f"Cell type silhouette (PCA): {evaluation_results['celltype_silhouette_pca']:.4f}")
    
    if 'celltype_silhouette_scvi' in evaluation_results:
        print(f"Cell type silhouette (scVI): {evaluation_results['celltype_silhouette_scvi']:.4f}")
    
    batch_effect_reduction = evaluation_results['batch_silhouette_pca'] - evaluation_results['batch_silhouette_scvi']
    print(f"Batch effect reduction: {batch_effect_reduction:.4f}")
    
    # Save evaluation results
    pd.DataFrame([evaluation_results]).to_csv(os.path.join(RESULTS_DIR, "final_evaluation.csv"), index=False)
    
    # Visualize final results
    try:
        # Create UMAP visualizations for batch and cell type
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Batch visualization
        sc.pl.umap(pbmc_corrected, color='batch', ax=axes[0], show=False, title="UMAP of corrected data (by batch)")
        
        # Cell type visualization if available
        if label_key in pbmc_corrected.obs.columns:
            sc.pl.umap(pbmc_corrected, color=label_key, ax=axes[1], show=False, title="UMAP of corrected data (by cell type)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "figures", "final_umap.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    print(f"\nOptimization complete! Results are saved in the '{RESULTS_DIR}' directory.")
    print(f"All data files for Streamlit visualization are saved in '{DATA_DIR}'.")


def main():
    """
    Main function to run the parameter optimization
    """
    # Set the number of trials for the optimization
    n_trials = 5  # Start with a small number for testing, increase for better results
    
    # Run the optimization
    study = run_optimization(n_trials=n_trials)
    
    # Apply the best parameters
    apply_best_parameters(study.best_trial.params)


if __name__ == "__main__":
    main()
