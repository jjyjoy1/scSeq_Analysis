import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
from scipy import sparse
import scvi
import torch
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sc.settings.set_figure_params(dpi=100, figsize=(8, 8), frameon=False)
sc.settings.verbosity = 1

class ScRNASeqPipeline:
    """
    A comprehensive pipeline for single-cell RNA sequencing data analysis,
    with batch integration, normalization, clustering and differential expression.
    """
    
    def __init__(self, output_dir="./results"):
        """
        Initialize the scRNA-seq pipeline.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.adata = None
        self.scvi_model = None
        
    def load_data(self, file_paths, batch_key="batch", sample_key=None):
        """
        Load and combine multiple datasets.
        
        Parameters:
        -----------
        file_paths : dict
            Dictionary with batch names as keys and file paths as values
        batch_key : str
            Key to use for batch information
        sample_key : str, optional
            Key to use for sample information if available
        """
        adatas = []
        
        for batch_name, file_path in file_paths.items():
            print(f"Loading {batch_name} from {file_path}")
            
            # Determine file format and load accordingly
            if file_path.endswith('.h5ad'):
                adata = sc.read_h5ad(file_path)
            elif file_path.endswith('.h5'):
                adata = sc.read_10x_h5(file_path)
            elif file_path.endswith('.mtx'):
                adata = sc.read_mtx(file_path)
            elif file_path.endswith('.csv'):
                adata = sc.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format for {file_path}")
            
            # Add batch annotation
            adata.obs[batch_key] = batch_name
            adatas.append(adata)
        
        # Concatenate all datasets
        if len(adatas) > 1:
            self.adata = adatas[0].concatenate(adatas[1:], batch_key=batch_key)
        else:
            self.adata = adatas[0]
            
        print(f"Combined dataset: {self.adata.shape[0]} cells and {self.adata.shape[1]} genes")
        return self
        
    def qc_and_filtering(self, min_genes=200, min_cells=3, max_genes=6000, 
                         max_mt_percent=20, min_counts=1000):
        """
        Perform quality control and filter cells and genes.
        
        Parameters:
        -----------
        min_genes : int
            Minimum number of genes expressed per cell
        min_cells : int
            Minimum number of cells expressing a gene
        max_genes : int
            Maximum number of genes expressed per cell
        max_mt_percent : float
            Maximum percentage of mitochondrial genes
        min_counts : int
            Minimum number of counts per cell
        """
        if self.adata is None:
            raise ValueError("No data loaded. Call load_data first.")
            
        print("Performing QC and filtering...")
        
        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(self.adata, inplace=True, percent_top=None, log1p=False, qc_vars=['mt'])
        
        # Annotate mitochondrial genes (if not already done)
        if 'mt' not in self.adata.var_names.str.lower():
            self.adata.var['mt'] = self.adata.var_names.str.startswith(('MT-', 'mt-'))
            sc.pp.calculate_qc_metrics(self.adata, qc_vars=['mt'], inplace=True)
        
        # Plot QC metrics before filtering
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        
        sns.distplot(self.adata.obs['n_genes_by_counts'], kde=False, bins=60, ax=axs[0])
        axs[0].set_xlabel('Number of genes')
        axs[0].set_ylabel('Number of cells')
        
        sns.distplot(self.adata.obs['total_counts'], kde=False, bins=60, ax=axs[1])
        axs[1].set_xlabel('Total counts')
        axs[1].set_ylabel('Number of cells')
        
        sns.distplot(self.adata.obs['pct_counts_mt'], kde=False, bins=60, ax=axs[2])
        axs[2].set_xlabel('Percent mitochondrial')
        axs[2].set_ylabel('Number of cells')
        
        sc.pl.scatter(self.adata, 'n_genes_by_counts', 'pct_counts_mt', ax=axs[3])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'qc_before_filtering.png'))
        plt.close()
        
        # Filter cells
        print("Before filtering:", self.adata.shape)
        self.adata = self.adata[self.adata.obs['n_genes_by_counts'] > min_genes]
        self.adata = self.adata[self.adata.obs['n_genes_by_counts'] < max_genes]
        self.adata = self.adata[self.adata.obs['pct_counts_mt'] < max_mt_percent]
        self.adata = self.adata[self.adata.obs['total_counts'] > min_counts]
        
        # Filter genes
        sc.pp.filter_genes(self.adata, min_cells=min_cells)
        
        print("After filtering:", self.adata.shape)
        
        # Plot QC metrics after filtering
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        
        sns.distplot(self.adata.obs['n_genes_by_counts'], kde=False, bins=60, ax=axs[0])
        axs[0].set_xlabel('Number of genes')
        axs[0].set_ylabel('Number of cells')
        
        sns.distplot(self.adata.obs['total_counts'], kde=False, bins=60, ax=axs[1])
        axs[1].set_xlabel('Total counts')
        axs[1].set_ylabel('Number of cells')
        
        sns.distplot(self.adata.obs['pct_counts_mt'], kde=False, bins=60, ax=axs[2])
        axs[2].set_xlabel('Percent mitochondrial')
        axs[2].set_ylabel('Number of cells')
        
        sc.pl.scatter(self.adata, 'n_genes_by_counts', 'pct_counts_mt', ax=axs[3])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'qc_after_filtering.png'))
        plt.close()
        
        return self
    
    def normalize_and_scale(self, n_top_genes=2000, target_sum=1e4, log=True):
        """
        Normalize, log-transform, and select highly variable genes.
        
        Parameters:
        -----------
        n_top_genes : int
            Number of highly variable genes to keep
        target_sum : float
            Target sum for normalization
        log : bool
            Whether to log-transform the data
        """
        print("Normalizing and scaling data...")
        
        # Normalize
        sc.pp.normalize_total(self.adata, target_sum=target_sum)
        
        # Log transform
        if log:
            sc.pp.log1p(self.adata)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(self.adata, n_top_genes=n_top_genes, flavor="seurat")
        
        # Plot highly variable genes
        plt.figure(figsize=(10, 7))
        sc.pl.highly_variable_genes(self.adata)
        plt.savefig(os.path.join(self.output_dir, 'highly_variable_genes.png'))
        plt.close()
        
        # Store raw counts in .raw attribute
        self.adata.raw = self.adata.copy()
        
        # Subset to highly variable genes for dimensionality reduction
        self.adata = self.adata[:, self.adata.var.highly_variable]
        
        # Scale data
        sc.pp.scale(self.adata, max_value=10)
        
        return self
    
    def run_pca(self, n_comps=50, plot=True):
        """
        Run PCA for initial dimensionality reduction.
        
        Parameters:
        -----------
        n_comps : int
            Number of principal components
        plot : bool
            Whether to plot PCA results
        """
        print("Running PCA...")
        sc.tl.pca(self.adata, svd_solver='arpack', n_comps=n_comps)
        
        if plot:
            # Plot variance ratio
            plt.figure(figsize=(10, 5))
            sc.pl.pca_variance_ratio(self.adata, log=True, n_pcs=n_comps)
            plt.savefig(os.path.join(self.output_dir, 'pca_variance_ratio.png'))
            plt.close()
            
            # Plot PCA
            plt.figure(figsize=(10, 10))
            sc.pl.pca(self.adata, color='batch')
            plt.savefig(os.path.join(self.output_dir, 'pca.png'))
            plt.close()
        
        return self
    
    def prepare_scvi(self, batch_key="batch", categorical_covariate_keys=None, continuous_covariate_keys=None):
        """
        Prepare data for scVI model fitting.
        
        Parameters:
        -----------
        batch_key : str
            Key for batch information
        categorical_covariate_keys : list, optional
            List of categorical covariates to include
        continuous_covariate_keys : list, optional
            List of continuous covariates to include
        """
        print("Preparing data for scVI...")
        
        # Make sure we're working with raw counts for scVI
        if self.adata.raw is not None:
            adata_for_scvi = self.adata.raw.to_adata()
        else:
            # If raw isn't available, use the original data
            adata_for_scvi = self.adata.copy()
        
        # Keep only highly variable genes if they were computed
        if 'highly_variable' in self.adata.var:
            adata_for_scvi = adata_for_scvi[:, self.adata.var_names]
        
        # Set up the AnnData object for scVI
        scvi.model.SCVI.setup_anndata(
            adata_for_scvi,
            layer=None,  # If counts are in a layer, specify the layer
            batch_key=batch_key,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys
        )
        
        # Store the prepared anndata
        self.adata_for_scvi = adata_for_scvi
        
        return self
    
    def run_scvi(self, n_latent=30, n_layers=2, n_epochs=200, lr=0.001, early_stopping=True, 
                 batch_size=256, use_gpu=True):
        """
        Run scVI for batch integration and dimensionality reduction.
        
        Parameters:
        -----------
        n_latent : int
            Dimension of the latent space
        n_layers : int
            Number of hidden layers used for encoder and decoder NNs
        n_epochs : int
            Number of epochs to train for
        lr : float
            Learning rate for optimizer
        early_stopping : bool
            Whether to use early stopping
        batch_size : int
            Batch size for training
        use_gpu : bool
            Whether to use GPU for training
        """
        print("Training scVI model...")
        
        if self.adata_for_scvi is None:
            raise ValueError("Data not prepared for scVI. Call prepare_scvi first.")
        
        # Configure GPU usage
        if use_gpu and torch.cuda.is_available():
            gpu_device = torch.device("cuda")
            print("Using GPU:", torch.cuda.get_device_name(0))
        else:
            gpu_device = None
            if use_gpu:
                print("GPU requested but not available. Using CPU instead.")
            else:
                print("Using CPU for training.")
        
        # Create and train the scVI model
        self.scvi_model = scvi.model.SCVI(
            self.adata_for_scvi,
            n_hidden=128,
            n_latent=n_latent,
            n_layers=n_layers
        )
        
        self.scvi_model.train(
            max_epochs=n_epochs,
            lr=lr,
            use_gpu=gpu_device is not None,
            early_stopping=early_stopping,
            batch_size=batch_size,
            plan_kwargs={'weight_decay': 0.001}
        )
        
        # Get the latent representation
        self.adata_for_scvi.obsm["X_scVI"] = self.scvi_model.get_latent_representation()
        
        # Copy latent representation to original adata
        self.adata.obsm["X_scVI"] = self.adata_for_scvi[self.adata.obs_names].obsm["X_scVI"]
        
        return self
    
    def run_umap(self, use_scvi=True, n_neighbors=15, min_dist=0.3, plot=True):
        """
        Run UMAP for visualization.
        
        Parameters:
        -----------
        use_scvi : bool
            Whether to use scVI latent space for UMAP
        n_neighbors : int
            Number of neighbors for UMAP
        min_dist : float
            Minimum distance for UMAP
        plot : bool
            Whether to plot UMAP results
        """
        print("Running UMAP...")
        
        if use_scvi:
            # Use scVI latent space
            if "X_scVI" not in self.adata.obsm:
                raise ValueError("scVI latent space not available. Run run_scvi first.")
            sc.pp.neighbors(self.adata, use_rep='X_scVI', n_neighbors=n_neighbors)
        else:
            # Use PCA
            if "X_pca" not in self.adata.obsm:
                raise ValueError("PCA not available. Run run_pca first.")
            sc.pp.neighbors(self.adata, use_rep='X_pca', n_neighbors=n_neighbors)
        
        # Run UMAP
        sc.tl.umap(self.adata, min_dist=min_dist)
        
        if plot:
            plt.figure(figsize=(10, 10))
            sc.pl.umap(self.adata, color='batch')
            plt.savefig(os.path.join(self.output_dir, 'umap_batch.png'))
            plt.close()
        
        return self
    
    def run_clustering(self, resolution=0.8, plot=True):
        """
        Run Leiden clustering.
        
        Parameters:
        -----------
        resolution : float
            Resolution parameter for clustering
        plot : bool
            Whether to plot clustering results
        """
        print("Running clustering...")
        
        if "neighbors" not in self.adata.uns:
            raise ValueError("Neighbors graph not available. Run run_umap first.")
        
        # Run Leiden clustering
        sc.tl.leiden(self.adata, resolution=resolution)
        
        if plot:
            plt.figure(figsize=(10, 10))
            sc.pl.umap(self.adata, color='leiden')
            plt.savefig(os.path.join(self.output_dir, 'umap_clusters.png'))
            plt.close()
            
            # Plot both batch and clusters
            plt.figure(figsize=(15, 7))
            ax1 = plt.subplot(1, 2, 1)
            sc.pl.umap(self.adata, color='batch', ax=ax1, show=False, title='Batch')
            ax2 = plt.subplot(1, 2, 2)
            sc.pl.umap(self.adata, color='leiden', ax=ax2, show=False, title='Clusters')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'umap_batch_vs_clusters.png'))
            plt.close()
        
        return self
    
    def find_markers(self, groupby='leiden', n_genes=25, method='wilcoxon'):
        """
        Find marker genes for clusters.
        
        Parameters:
        -----------
        groupby : str
            Key for grouping cells
        n_genes : int
            Number of top genes to report
        method : str
            Method for differential expression analysis
        """
        print(f"Finding marker genes with {method}...")
        
        if groupby not in self.adata.obs:
            raise ValueError(f"{groupby} not in adata.obs")
        
        # Make sure we're using raw data for DE
        if self.adata.raw is not None:
            sc.pp.log1p(self.adata.raw.X)
            
        # Run marker gene detection
        sc.tl.rank_genes_groups(self.adata, groupby=groupby, method=method, pts=True, use_raw=True)
        
        # Get results into a dataframe
        self.markers = sc.get.rank_genes_groups_df(self.adata, group=None)
        
        # Save markers to csv
        self.markers.to_csv(os.path.join(self.output_dir, 'marker_genes.csv'))
        
        # Plot top marker genes
        plt.figure(figsize=(10, n_genes // 2))
        sc.pl.rank_genes_groups_dotplot(self.adata, n_genes=n_genes, groupby=groupby)
        plt.savefig(os.path.join(self.output_dir, 'marker_genes_dotplot.png'))
        plt.close()
        
        plt.figure(figsize=(18, 10))
        sc.pl.rank_genes_groups_heatmap(self.adata, n_genes=n_genes, groupby=groupby)
        plt.savefig(os.path.join(self.output_dir, 'marker_genes_heatmap.png'))
        plt.close()
        
        return self
    
    def run_bayesian_de(self, groupby='leiden', condition_key=None, batch_key='batch'):
        """
        Run Bayesian differential expression analysis using scVI.
        
        Parameters:
        -----------
        groupby : str
            Key for grouping cells
        condition_key : str, optional
            Key for experimental condition
        batch_key : str
            Key for batch information
        """
        print("Running Bayesian differential expression with scVI...")
        
        if self.scvi_model is None:
            raise ValueError("scVI model not available. Run run_scvi first.")
        
        if groupby not in self.adata.obs:
            raise ValueError(f"{groupby} not in adata.obs")
        
        # Get unique groups
        groups = self.adata.obs[groupby].unique().tolist()
        
        # For each group, compare against all others
        bayesian_de_results = {}
        for group in groups:
            print(f"Finding DE genes for {groupby}={group}...")
            cell_idx1 = self.adata.obs[groupby] == group
            cell_idx2 = self.adata.obs[groupby] != group
            
            # Get cell indices in the scVI model
            idx1 = self.adata_for_scvi.obs_names.isin(self.adata.obs_names[cell_idx1])
            idx2 = self.adata_for_scvi.obs_names.isin(self.adata.obs_names[cell_idx2])
            
            de_genes = self.scvi_model.differential_expression(
                idx1=idx1,
                idx2=idx2,
                mode="change",
                delta=0.25,  # Effect size threshold
                batch_correction=True,
                batchid1=None,  # Use all batches
                batchid2=None
            )
            
            # Save DE results
            de_genes.to_csv(os.path.join(self.output_dir, f'bayesian_de_{groupby}_{group}.csv'))
            bayesian_de_results[group] = de_genes
        
        self.bayesian_de_results = bayesian_de_results
        
        return self
    
    def annotate_clusters(self, markers_dict=None, groupby='leiden', threshold=0.5):
        """
        Annotate clusters based on marker genes.
        
        Parameters:
        -----------
        markers_dict : dict
            Dictionary with cell types as keys and lists of marker genes as values
        groupby : str
            Key for grouping cells
        threshold : float
            Threshold for marker gene enrichment score
        """
        if markers_dict is None:
            print("No markers provided. Skipping annotation.")
            return self
        
        print("Annotating clusters...")
        
        if groupby not in self.adata.obs:
            raise ValueError(f"{groupby} not in adata.obs")
        
        # Calculate enrichment score for each cell type in each cluster
        scores = {}
        for cell_type, markers in markers_dict.items():
            # Filter to markers that exist in our dataset
            valid_markers = [m for m in markers if m in self.adata.var_names]
            
            if not valid_markers:
                print(f"Warning: No markers for {cell_type} found in the dataset")
                continue
                
            # Calculate score for each cluster
            for cluster in self.adata.obs[groupby].unique():
                # Get top markers for this cluster
                if hasattr(self, 'markers'):
                    cluster_markers = self.markers[self.markers['group'] == cluster]
                    top_cluster_markers = set(cluster_markers.head(100)['names'].tolist())
                    
                    # Calculate overlap
                    overlap = len(set(valid_markers).intersection(top_cluster_markers))
                    score = overlap / len(valid_markers)
                    
                    if cluster not in scores:
                        scores[cluster] = {}
                    scores[cluster][cell_type] = score
        
        # Assign cell type to each cluster
        annotations = {}
        for cluster, cell_type_scores in scores.items():
            if not cell_type_scores:
                annotations[cluster] = "Unknown"
                continue
                
            best_cell_type = max(cell_type_scores.items(), key=lambda x: x[1])
            
            if best_cell_type[1] >= threshold:
                annotations[cluster] = best_cell_type[0]
            else:
                annotations[cluster] = "Unknown"
        
        # Add annotations to AnnData
        self.adata.obs['cell_type'] = self.adata.obs[groupby].map(annotations).astype('category')
        
        # Plot UMAP with cell type annotations
        plt.figure(figsize=(10, 10))
        sc.pl.umap(self.adata, color='cell_type')
        plt.savefig(os.path.join(self.output_dir, 'umap_cell_types.png'))
        plt.close()
        
        return self
    
    def save_results(self, filename='scrnaseq_results.h5ad'):
        """
        Save the AnnData object with all results.
        
        Parameters:
        -----------
        filename : str
            Filename to save results
        """
        output_path = os.path.join(self.output_dir, filename)
        print(f"Saving results to {output_path}")
        
        # Save the AnnData object
        self.adata.write(output_path)
        
        # Save the scVI model if available
        if self.scvi_model is not None:
            self.scvi_model.save(os.path.join(self.output_dir, "scvi_model"))
            
        return self

# Example usage
def run_pipeline_example():
    """
    Example of how to use the pipeline.
    """
    # Initialize the pipeline
    pipeline = ScRNASeqPipeline(output_dir="./scrnaseq_results")
    
    # Define data paths (example)
    data_paths = {
        "sample1": "path/to/sample1.h5ad",
        "sample2": "path/to/sample2.h5ad"
    }
    
    # Run the pipeline
    pipeline.load_data(data_paths, batch_key="batch") \
        .qc_and_filtering(min_genes=200, max_genes=6000, max_mt_percent=20) \
        .normalize_and_scale(n_top_genes=2000) \
        .run_pca(n_comps=50) \
        .prepare_scvi(batch_key="batch") \
        .run_scvi(n_latent=30, n_epochs=200) \
        .run_umap(use_scvi=True) \
        .run_clustering(resolution=0.8) \
        .find_markers() \
        .run_bayesian_de() \
        .save_results()
    
    return pipeline

# Example for immune cell annotation
def immune_cell_annotation_example():
    """
    Example of cell type annotation with immune cell markers.
    """
    # Dictionary of immune cell markers
    immune_markers = {
        "T_cells": ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4"],
        "B_cells": ["CD19", "MS4A1", "CD79A", "CD79B"],
        "NK_cells": ["NCAM1", "NKG7", "KLRD1", "KLRF1"],
        "Monocytes": ["CD14", "LYZ", "FCGR3A", "MS4A7"],
        "Dendritic_cells": ["FCER1A", "CLEC10A", "ITGAX", "ITGAM"],
        "Neutrophils": ["S100A8", "S100A9", "FCGR3B", "CSF3R"],
        "Platelets": ["PPBP", "PF4", "GP1BA", "ITGA2B"]
    }
    
    # Initialize and run pipeline
    pipeline = ScRNASeqPipeline(output_dir="./immune_analysis")
    # ... load data and run pipeline steps ...
    
    # Annotate clusters with immune markers
    pipeline.annotate_clusters(markers_dict=immune_markers)
    
    return pipeline



