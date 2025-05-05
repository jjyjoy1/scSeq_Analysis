"""
VAE-based batch correction for single-cell RNA-seq data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
from scipy import sparse
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

import scvi
import torch
import pickle


def load_pbmc_dataset():
    """
    Load the PBMC dataset from scvi-tools.
    
    This dataset contains cells from two batches (PBMC4K and PBMC8K),
    making it suitable for demonstrating batch correction methods.
    
    Returns:
        AnnData: Combined PBMC dataset with batch information
    """
    print("Loading PBMC dataset...")
    
    # Load the dataset
    pbmc_data = scvi.data.pbmc_dataset()
    
    print(f"PBMC dataset: {pbmc_data.shape[0]} cells, {pbmc_data.shape[1]} genes")
    
    # Print batch information
    print("\nBatch distribution:")
    print(pbmc_data.obs['batch'].value_counts().to_string())
    
    # Convert batch to categorical to ensure compatibility
    pbmc_data.obs['batch'] = pbmc_data.obs['batch'].astype('category')
    
    # Print cell type information if available
    if 'labels' in pbmc_data.obs.columns:
        print("\nCell types in PBMC dataset:")
        print(pbmc_data.obs['labels'].value_counts().to_string())
        
        # Also print the string labels if available
        if 'str_labels' in pbmc_data.obs.columns:
            print("\nCell type names:")
            for i, label in enumerate(pbmc_data.obs['str_labels'].unique()):
                print(f"{i}: {label}")
    
    return pbmc_data


def load_pbmc_seurat_dataset():
    """
    Load the PBMC Seurat v4 CITE-seq dataset from scvi-tools.
    
    This dataset contains cells from 24 batches across different donors and timepoints,
    making it suitable for more complex batch correction scenarios.
    
    Returns:
        AnnData: PBMC Seurat dataset with batch information
    """
    print("Loading PBMC Seurat v4 CITE-seq dataset...")
    
    # Load the dataset
    pbmc_seurat = scvi.data.pbmc_seurat_v4_cite_seq()
    
    print(f"PBMC Seurat dataset: {pbmc_seurat.shape[0]} cells, {pbmc_seurat.shape[1]} genes")
    
    # Print batch distribution (using donor and time as batch factors)
    print("\nDonor distribution:")
    print(pbmc_seurat.obs['donor'].value_counts().to_string())
    
    print("\nTimepoint distribution:")
    print(pbmc_seurat.obs['time'].value_counts().to_string())
    
    # Print cell type information
    print("\nCell types (level 1):")
    print(pbmc_seurat.obs['celltype.l1'].value_counts().to_string())
    
    # Create a combined batch column if needed for batch correction
    pbmc_seurat.obs['batch'] = pbmc_seurat.obs['donor'] + '_' + pbmc_seurat.obs['time']
    pbmc_seurat.obs['batch'] = pbmc_seurat.obs['batch'].astype('category')
    print(f"\nTotal number of unique batches: {pbmc_seurat.obs['batch'].nunique()}")
    
    return pbmc_seurat


def run_preprocessing(adata):
    """
    Perform preprocessing steps on the AnnData object
    
    Parameters:
        adata (AnnData): The annotated data matrix
        
    Returns:
        AnnData: Preprocessed data
    """
    print("Performing basic preprocessing...")
    
    # Create a proper copy to avoid ImplicitModificationWarning
    adata = adata.copy()
    
    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Store raw counts for scVI
    print("Storing raw counts...")
    adata.layers["counts"] = adata.X.copy()
    
    # Normalize the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Make sure batch is categorical
    if 'batch' in adata.obs.columns:
        print("Converting batch to categorical type...")
        adata.obs['batch'] = adata.obs['batch'].astype('category')
    
    # Select highly variable genes
    print("Finding highly variable genes...")
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='batch')
        # Create a proper subset to avoid warnings
        highly_variable_genes = adata.var.highly_variable
        adata_subset = adata[:, highly_variable_genes].copy()
    except Exception as e:
        print(f"Error finding highly variable genes with batch correction: {e}")
        print("Falling back to standard highly variable genes without batch correction")
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        highly_variable_genes = adata.var.highly_variable
        adata_subset = adata[:, highly_variable_genes].copy()
    
    print(f"After preprocessing: {adata_subset.shape[0]} cells, {adata_subset.shape[1]} genes")
    return adata_subset


def run_vae_batch_correction(adata, batch_key='batch', n_latent=10, n_hidden=128, n_layers=2, max_epochs=400, model_path=None):
    """
    Run scVI VAE model for batch correction on the provided dataset.
    
    Parameters:
        adata (AnnData): The annotated data matrix.
        batch_key (str): The key in adata.obs for batch information.
        n_latent (int): Dimensionality of the latent space.
        n_hidden (int): Number of nodes per hidden layer.
        n_layers (int): Number of hidden layers.
        max_epochs (int): Maximum number of training epochs.
        model_path (str): Path to save/load the model.
        
    Returns:
        tuple: (model, adata_with_latent), where model is the trained scVI model and
               adata_with_latent has the latent representation added.
    """
    # Check if we can load a pre-trained model
    if model_path and os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        try:
            # Setup anndata to ensure compatibility
            scvi.model.SCVI.setup_anndata(
                adata,
                batch_key=batch_key,
                labels_key='labels' if 'labels' in adata.obs.columns else None,
                layer="counts" if "counts" in adata.layers else None
            )
            # Load the model
            model = scvi.model.SCVI.load(model_path, adata=adata)
            # Get latent representation
            print("Extracting latent representation from loaded model...")
            adata.obsm['X_scVI'] = model.get_latent_representation()
            return model, adata
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training a new model instead...")
    
    # Ensure batch is categorical
    adata.obs[batch_key] = adata.obs[batch_key].astype('category')
    
    # Setup the model
    print("Setting up scVI model...")
    try:
        scvi.model.SCVI.setup_anndata(
            adata,
            batch_key=batch_key,
            labels_key='labels' if 'labels' in adata.obs.columns else None,
            layer="counts" if "counts" in adata.layers else None
        )
    except Exception as e:
        print(f"Error in setup_anndata: {e}")
        print("Trying alternative setup...")
        # If "counts" layer doesn't work, try with raw data
        scvi.model.SCVI.setup_anndata(
            adata,
            batch_key=batch_key,
            labels_key='labels' if 'labels' in adata.obs.columns else None
        )
    
    # Define the model
    model = scvi.model.SCVI(
        adata,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers
    )
    
    # Train the model - using max_epochs instead of n_epochs
    print(f"Training VAE for {max_epochs} epochs...")
    model.train(max_epochs=max_epochs, check_val_every_n_epoch=10)
    
    # Get latent representation
    print("Extracting latent representation...")
    adata.obsm['X_scVI'] = model.get_latent_representation()
    
    # Save the model if path is provided
    if model_path:
        print(f"Saving model to {model_path}")
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save(model_path)
    
    # Return the model and updated anndata
    return model, adata


def visualize_batch_correction(adata, batch_key='batch', label_key=None, save_path=None):
    """
    Visualize the batch correction results using UMAP.
    
    Parameters:
        adata (AnnData): The annotated data matrix with latent representation.
        batch_key (str): The key in adata.obs for batch information.
        label_key (str): The key in adata.obs for cell type labels.
        save_path (str): Path to save the figures (optional).
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    
    # Create a copy of the AnnData object
    adata_copy = adata.copy()
    
    # Compute UMAP on the original data
    print("Computing UMAP on original data...")
    sc.pp.pca(adata_copy)
    sc.pp.neighbors(adata_copy)
    sc.tl.umap(adata_copy)
    
    # Plot UMAP colored by batch
    print("Plotting UMAP by batch...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sc.pl.umap(adata_copy, color=batch_key, ax=axes[0], show=False, title='Before correction (PCA)')
    
    # Compute UMAP on the latent representation
    print("Computing UMAP on latent representation...")
    sc.pp.neighbors(adata, use_rep='X_scVI')
    sc.tl.umap(adata)
    
    sc.pl.umap(adata, color=batch_key, ax=axes[1], show=False, title='After correction (scVI)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_batch.png", dpi=300)
    plt.show()
    
    # If label_key is provided, visualize by cell type
    if label_key and label_key in adata.obs.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sc.pl.umap(adata_copy, color=label_key, ax=axes[0], show=False, title='Before correction (PCA)')
        sc.pl.umap(adata, color=label_key, ax=axes[1], show=False, title='After correction (scVI)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_celltype.png", dpi=300)
        plt.show()


def evaluate_batch_correction(adata, batch_key='batch', label_key=None):
    """
    Evaluate the effectiveness of batch correction using:
    1. Silhouette score on batches (should decrease after correction)
    2. Silhouette score on cell types (should increase or stay similar after correction)
    
    Parameters:
        adata (AnnData): The annotated data matrix with latent representation
        batch_key (str): The key in adata.obs for batch information
        label_key (str): The key in adata.obs for cell type labels
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    from sklearn.metrics import silhouette_score
    import numpy as np
    
    results = {}
    
    # Ensure batch and label columns are categorical
    adata.obs[batch_key] = adata.obs[batch_key].astype('category')
    if label_key and label_key in adata.obs.columns:
        adata.obs[label_key] = adata.obs[label_key].astype('category')
    
    # Calculate silhouette scores on PCA embedding
    print("Calculating silhouette scores on PCA embedding...")
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
    print("Calculating silhouette scores on scVI latent space...")
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
    
    # Print results
    print("\nBatch correction evaluation:")
    print(f"Batch silhouette score (PCA): {batch_sil_pca:.4f}")
    print(f"Batch silhouette score (scVI): {batch_sil_scvi:.4f}")
    if label_key and label_key in adata.obs.columns:
        print(f"Cell type silhouette score (PCA): {cell_sil_pca:.4f}")
        print(f"Cell type silhouette score (scVI): {cell_sil_scvi:.4f}")
    
    # Ideal results: batch silhouette should decrease (batches mixed)
    # Cell type silhouette should increase or stay similar (biology preserved)
    if batch_sil_scvi < batch_sil_pca:
        print("\n✓ Batch effect reduced (lower batch silhouette score)")
    else:
        print("\n✗ Batch effect may not be sufficiently reduced")
    
    if label_key and label_key in adata.obs.columns:
        if cell_sil_scvi >= cell_sil_pca:
            print("✓ Cell type separation preserved or improved")
        else:
            print("✗ Cell type separation may be compromised")
    
    return results


def visualize_latent_space(adata, batch_key='batch', label_key=None, n_dims=2, method='umap', save_path=None):
    """
    Visualize the latent space using dimensionality reduction (UMAP or t-SNE)
    
    Parameters:
        adata (AnnData): The annotated data matrix with latent representation
        batch_key (str): The key in adata.obs for batch information
        label_key (str): The key in adata.obs for cell type labels
        n_dims (int): Number of dimensions for the visualization (2 or 3)
        method (str): Method for dimensionality reduction ('umap' or 'tsne')
        save_path (str): Path to save the figures (optional)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import scanpy as sc
    from sklearn.manifold import TSNE  # Import sklearn's TSNE directly
    
    # Create figure
    if n_dims == 3:
        fig = plt.figure(figsize=(20, 10))
        if batch_key and label_key:
            batch_ax = fig.add_subplot(121, projection='3d')
            label_ax = fig.add_subplot(122, projection='3d')
        else:
            batch_ax = fig.add_subplot(111, projection='3d')
            label_ax = None
    else:  # 2D
        fig, axes = plt.subplots(1, 2 if label_key else 1, figsize=(20 if label_key else 10, 8))
        batch_ax = axes[0] if isinstance(axes, np.ndarray) else axes
        label_ax = axes[1] if isinstance(axes, np.ndarray) and len(axes) > 1 else None
    
    # Get latent representation and perform dimensionality reduction if needed
    if method == 'umap':
        if 'X_umap' not in adata.obsm:
            print("Computing UMAP...")
            sc.pp.neighbors(adata, use_rep='X_scVI')
            
            # Handle both 2D and 3D UMAP
            try:
                if n_dims == 3:
                    # Try to use n_components if supported
                    try:
                        sc.tl.umap(adata, n_components=n_dims)
                    except TypeError:
                        print("Your scanpy version doesn't support n_components in umap.")
                        print("Using default 2D UMAP instead.")
                        sc.tl.umap(adata)
                else:
                    sc.tl.umap(adata)
            except Exception as e:
                print(f"Error computing UMAP: {e}")
                print("Falling back to PCA for visualization.")
                if 'X_pca' not in adata.obsm:
                    sc.pp.pca(adata)
                method = 'pca'
                embed_key = 'X_pca'
        
        embed_key = 'X_umap'
    
    elif method == 'tsne':
        embed_key = f'X_tsne_{n_dims}d'
        
        if embed_key not in adata.obsm:
            print(f"Computing t-SNE ({n_dims}D) using scikit-learn...")
            # Use scikit-learn's TSNE implementation directly
            tsne = TSNE(
                n_components=n_dims,
                learning_rate='auto',
                init='pca',
                random_state=42
            )
            
            # Apply t-SNE to the latent representation
            tsne_result = tsne.fit_transform(adata.obsm['X_scVI'])
            
            # Store the result in AnnData
            adata.obsm[embed_key] = tsne_result
    
    elif method == 'pca':
        if 'X_pca' not in adata.obsm:
            print("Computing PCA...")
            sc.pp.pca(adata)
        embed_key = 'X_pca'
    
    # Get coordinates
    x = adata.obsm[embed_key][:, 0]
    y = adata.obsm[embed_key][:, 1]
    z = adata.obsm[embed_key][:, 2] if n_dims == 3 and adata.obsm[embed_key].shape[1] > 2 else None
    
    # Get batch and label information
    batches = adata.obs[batch_key].astype('category').cat.categories
    batch_colors = plt.cm.tab10(np.linspace(0, 1, len(batches)))
    batch_indices = adata.obs[batch_key].astype('category').cat.codes
    
    # Plot by batch
    if n_dims == 3 and z is not None:
        batch_scatter = batch_ax.scatter(x, y, z, c=batch_colors[batch_indices], s=5, alpha=0.7)
        batch_ax.set_title(f'Latent space colored by {batch_key} ({method.upper()})', fontsize=14)
        batch_ax.set_xlabel('Dimension 1', fontsize=12)
        batch_ax.set_ylabel('Dimension 2', fontsize=12)
        batch_ax.set_zlabel('Dimension 3', fontsize=12)
    else:
        batch_scatter = batch_ax.scatter(x, y, c=batch_colors[batch_indices], s=5, alpha=0.7)
        batch_ax.set_title(f'Latent space colored by {batch_key} ({method.upper()})', fontsize=14)
        batch_ax.set_xlabel('Dimension 1', fontsize=12)
        batch_ax.set_ylabel('Dimension 2', fontsize=12)
    
    # Add legend for batches
    batch_legends = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=batch_colors[i], markersize=10) 
                    for i in range(len(batches))]
    batch_ax.legend(batch_legends, batches, loc='upper right', title=batch_key)
    
    # Plot by cell type if available
    if label_key and label_key in adata.obs.columns and label_ax is not None:
        labels = adata.obs[label_key].astype('category').cat.categories
        label_colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        label_indices = adata.obs[label_key].astype('category').cat.codes
        
        if n_dims == 3 and z is not None:
            label_scatter = label_ax.scatter(x, y, z, c=label_colors[label_indices], s=5, alpha=0.7)
            label_ax.set_title(f'Latent space colored by {label_key} ({method.upper()})', fontsize=14)
            label_ax.set_xlabel('Dimension 1', fontsize=12)
            label_ax.set_ylabel('Dimension 2', fontsize=12)
            label_ax.set_zlabel('Dimension 3', fontsize=12)
        else:
            label_scatter = label_ax.scatter(x, y, c=label_colors[label_indices], s=5, alpha=0.7)
            label_ax.set_title(f'Latent space colored by {label_key} ({method.upper()})', fontsize=14)
            label_ax.set_xlabel('Dimension 1', fontsize=12)
            label_ax.set_ylabel('Dimension 2', fontsize=12)
        
        # Add legend for cell types
        label_legends = [plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=label_colors[i], markersize=10) 
                        for i in range(len(labels))]
        label_ax.legend(label_legends, labels, loc='upper right', title=label_key)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_{method}_{n_dims}d.png", dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """
    Main function to demonstrate VAE-based batch correction on PBMC data.
    """
    # Set random seed for reproducibility
    import random
    import numpy as np
    import torch
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Define model path for saving/loading
    model_path = "models/pbmc_vae_model"
    
    # Load the PBMC dataset
    pbmc_data = load_pbmc_dataset()
    
    # Print data types of columns to help with debugging
    print("\nColumn dtypes:")
    for col in pbmc_data.obs.columns:
        print(f"{col}: {pbmc_data.obs[col].dtype}")
    
    # Run preprocessing
    pbmc_data = run_preprocessing(pbmc_data)
    
    # Run VAE batch correction
    model, pbmc_data = run_vae_batch_correction(
        pbmc_data,
        batch_key='batch',
        n_latent=15,
        n_hidden=128,
        n_layers=2,
        max_epochs=200,
        model_path=model_path  # Save/load the model
    )
    
    # Get the appropriate label key
    label_key = 'str_labels' if 'str_labels' in pbmc_data.obs.columns else 'labels'
    
    # Visualize the results using UMAP
    try:
        visualize_batch_correction(
            pbmc_data,
            batch_key='batch',
            label_key=label_key,
            save_path='pbmc_batch_correction'
        )
    except Exception as e:
        print(f"Error in batch correction visualization: {e}")
        print("Continuing with other visualizations...")
    
    # For 2D visualizations, use PCA as a fallback option
    try:
        visualize_latent_space(
            pbmc_data,
            batch_key='batch',
            label_key=label_key,
            method='pca',  # Use PCA which is more stable
            n_dims=2,
            save_path='pbmc_latent_pca'
        )
    except Exception as e:
        print(f"Error in PCA visualization: {e}")
    
    # Use sklearn's TSNE implementation directly in the updated function
    try:
        visualize_latent_space(
            pbmc_data,
            batch_key='batch',
            label_key=label_key,
            method='tsne',
            n_dims=2,
            save_path='pbmc_latent_tsne'
        )
    except Exception as e:
        print(f"Error in t-SNE visualization: {e}")
    
    # Evaluate the batch correction
    try:
        evaluation_results = evaluate_batch_correction(
            pbmc_data,
            batch_key='batch',
            label_key=label_key
        )
    except Exception as e:
        print(f"Error in batch correction evaluation: {e}")
    
    print("Batch correction analysis complete!")


if __name__ == "__main__":
    main()
