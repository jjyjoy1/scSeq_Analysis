#!/usr/bin/env python
# Complete scRNA-seq pipeline for CNV analysis: FASTQ to final results

# This workflow includes:
# 1. Cell Ranger execution (via subprocess)
# 2. Quality control and preprocessing
# 3. Integration of cancer and normal samples with scVI
# 4. CNV inference with inferCNVpy
# 5. Visualization and export of results

import os
import subprocess
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# For scVI integration
import scvi
import torch

# For CNV inference
import infercnvpy as cnv

# Set seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Set up plotting defaults
sc.settings.set_figure_params(dpi=100, figsize=(8, 8))
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)

#######################
# 1. Cell Ranger Processing
#######################

def run_cellranger(fastq_dir, transcriptome_ref, sample_id, output_dir):
    """
    Run Cell Ranger count on FASTQ files
    
    Parameters:
    -----------
    fastq_dir : str
        Directory containing FASTQ files
    transcriptome_ref : str
        Path to Cell Ranger compatible reference genome
    sample_id : str
        Sample name
    output_dir : str
        Output directory
    """
    cmd = [
        "cellranger", "count",
        f"--id={sample_id}",
        f"--transcriptome={transcriptome_ref}",
        f"--fastqs={fastq_dir}",
        f"--sample={sample_id}",
        f"--localcores=16",  # Adjust based on your system
        f"--localmem=64",    # Adjust based on your system
        "--include-introns=true"
    ]
    
    print(f"Running Cell Ranger for sample {sample_id}")
    print(" ".join(cmd))
    
    # Execute Cell Ranger
    subprocess.run(cmd, check=True, cwd=output_dir)
    
    return os.path.join(output_dir, sample_id, "outs", "filtered_feature_bc_matrix.h5")


def process_all_samples(sample_info_df, transcriptome_ref, output_base_dir):
    """
    Process all samples with Cell Ranger
    
    Parameters:
    -----------
    sample_info_df : pandas.DataFrame
        DataFrame with columns: sample_id, fastq_dir, condition (tumor/normal)
    transcriptome_ref : str
        Path to Cell Ranger compatible reference genome
    output_base_dir : str
        Base output directory
    
    Returns:
    --------
    dict
        Dictionary with sample_ids as keys and paths to h5 files as values
    """
    h5_files = {}
    
    for _, row in sample_info_df.iterrows():
        sample_id = row['sample_id']
        fastq_dir = row['fastq_dir']
        
        # Create output directory
        output_dir = os.path.join(output_base_dir, "cellranger")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run Cell Ranger
        h5_path = run_cellranger(fastq_dir, transcriptome_ref, sample_id, output_dir)
        h5_files[sample_id] = h5_path
    
    return h5_files


#######################
# 2. Quality Control and Preprocessing
#######################

def perform_qc(adata, mt_pattern="^MT-", min_genes=200, min_cells=3, 
               max_genes=5000, max_pct_mt=20, n_hvgs=2000):
    """
    Perform quality control on AnnData object
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object
    mt_pattern : str
        Regex pattern to identify mitochondrial genes
    min_genes : int
        Minimum number of genes per cell
    min_cells : int
        Minimum number of cells per gene
    max_genes : int
        Maximum number of genes per cell
    max_pct_mt : float
        Maximum percentage of mitochondrial genes
    n_hvgs : int
        Number of highly variable genes to keep
    
    Returns:
    --------
    AnnData
        Filtered AnnData object
    """
    print(f"Original shape: {adata.shape}")
    
    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=[mt_pattern], inplace=True)
    
    # Filter cells
    adata = adata[adata.obs.n_genes_by_counts >= min_genes, :]
    adata = adata[adata.obs.n_genes_by_counts <= max_genes, :]
    adata = adata[adata.obs.pct_counts_mt <= max_pct_mt, :]
    
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    print(f"Shape after filtering: {adata.shape}")
    
    # Normalize and log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_hvgs)
    
    # Create a raw slot for future reference
    adata.raw = adata
    
    # Keep only highly variable genes for downstream analysis
    adata = adata[:, adata.var.highly_variable]
    
    return adata


def load_and_preprocess_samples(h5_files, sample_info_df):
    """
    Load and preprocess all samples
    
    Parameters:
    -----------
    h5_files : dict
        Dictionary with sample_ids as keys and paths to h5 files as values
    sample_info_df : pandas.DataFrame
        DataFrame with sample information
    
    Returns:
    --------
    dict
        Dictionary with 'tumor' and 'normal' AnnData objects
    """
    adatas = {}
    
    tumor_samples = []
    normal_samples = []
    
    # Process each sample
    for sample_id, h5_path in h5_files.items():
        print(f"Processing sample: {sample_id}")
        
        # Load data
        adata = sc.read_10x_h5(h5_path)
        adata.var_names_make_unique()
        
        # Add sample information
        sample_info = sample_info_df[sample_info_df.sample_id == sample_id].iloc[0]
        adata.obs['sample_id'] = sample_id
        adata.obs['condition'] = sample_info['condition']
        
        # Perform QC
        adata = perform_qc(adata)
        
        # Store according to condition
        if sample_info['condition'] == 'tumor':
            tumor_samples.append(adata)
        else:  # normal
            normal_samples.append(adata)
    
    # Concatenate samples by condition
    if tumor_samples:
        adatas['tumor'] = ad.concat(tumor_samples, label="sample_id", join="outer", 
                                    merge="same", index_unique="_")
    
    if normal_samples:
        adatas['normal'] = ad.concat(normal_samples, label="sample_id", join="outer", 
                                     merge="same", index_unique="_")
    
    return adatas


def cluster_and_annotate(adata, resolution=0.8):
    """
    Perform dimensionality reduction, clustering, and visualization
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object
    resolution : float
        Resolution for Leiden clustering
    
    Returns:
    --------
    AnnData
        AnnData object with clustering and UMAP results
    """
    # PCA
    sc.pp.pca(adata)
    
    # Neighborhood graph
    sc.pp.neighbors(adata)
    
    # UMAP for visualization
    sc.tl.umap(adata)
    
    # Clustering
    sc.tl.leiden(adata, resolution=resolution)
    
    # Optional: Run marker genes to aid in annotation
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    
    return adata


#######################
# 3. Integration with scVI
#######################

def integrate_with_scvi(adata_tumor, adata_normal, batch_key='sample_id', 
                        n_latent=30, n_layers=2, n_epochs=400):
    """
    Integrate tumor and normal samples using scVI
    
    Parameters:
    -----------
    adata_tumor : AnnData
        Tumor AnnData object
    adata_normal : AnnData
        Normal AnnData object
    batch_key : str
        Batch key for integration
    n_latent : int
        Number of latent dimensions
    n_layers : int
        Number of layers in the neural network
    n_epochs : int
        Number of training epochs
    
    Returns:
    --------
    AnnData
        Integrated AnnData object
    """
    # Concatenate tumor and normal data
    adata_integrated = ad.concat([adata_tumor, adata_normal], label="condition", 
                                join="outer", merge="same", index_unique="_")
    
    # Reset the AnnData.raw attribute
    adata_integrated.raw = adata_integrated
    
    # Keep only highly variable genes for integration
    sc.pp.highly_variable_genes(adata_integrated, flavor="seurat", n_top_genes=3000, 
                               batch_key=batch_key)
    
    adata_integrated = adata_integrated[:, adata_integrated.var.highly_variable]
    
    # Setup scVI model
    scvi.model.SCVI.setup_anndata(adata_integrated, batch_key=batch_key, 
                                  layer=None)
    
    # Create and train the model
    model = scvi.model.SCVI(adata_integrated, n_layers=n_layers, n_latent=n_latent)
    model.train(max_epochs=n_epochs, plan_kwargs={'lr': 5e-4})
    
    # Get integrated latent representation
    adata_integrated.obsm["X_scVI"] = model.get_latent_representation()
    
    # Recompute neighborhood graph and clustering on integrated data
    sc.pp.neighbors(adata_integrated, use_rep="X_scVI")
    sc.tl.umap(adata_integrated)
    sc.tl.leiden(adata_integrated, resolution=0.8)
    
    # Save the model for future use
    model.save("scvi_model/", overwrite=True)
    
    return adata_integrated, model


#######################
# 4. CNV Analysis with inferCNVpy
#######################

def prepare_for_cnv(adata, gene_pos_file):
    """
    Prepare AnnData object for CNV analysis
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object
    gene_pos_file : str
        Path to gene position file
    
    Returns:
    --------
    AnnData
        AnnData object ready for CNV analysis
    """
    # Load gene positions
    gene_pos = pd.read_csv(gene_pos_file, sep='\t')
    
    # Add gene positions to AnnData object
    cnv.io.add_positions(adata, gene_pos)
    
    # Use raw counts for CNV inference
    if adata.raw is not None:
        adata_raw = adata.raw.to_adata()
    else:
        adata_raw = adata.copy()
        sc.pp.log1p(adata_raw)  # Log transform if not already done
    
    # Ensure positions are present
    adata_raw.var = adata.var[['chromosome', 'start', 'end']]
    
    return adata_raw


def run_infercnv(adata, reference_key='cell_type', reference_cat=['B cells', 'T cells'], 
                window_size=100):
    """
    Run inferCNVpy to detect copy number variations
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with gene positions
    reference_key : str
        Column in adata.obs for reference cell types
    reference_cat : list
        Categories in reference_key to use as reference cells
    window_size : int
        Window size for CNV inference
    
    Returns:
    --------
    AnnData
        AnnData object with CNV results
    """
    # Create reference mask
    adata.obs['reference'] = adata.obs[reference_key].isin(reference_cat)
    
    # Run CNV inference
    cnv.tl.infercnv(adata, reference_key='reference', 
                   window_size=window_size, 
                   step=window_size // 2)
    
    # Run UMAP on CNV data
    sc.pp.pca(adata, n_comps=30, use_highly_variable=False, svd_solver='arpack')
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    # Cluster cells based on CNV profiles
    sc.tl.leiden(adata, key_added='cnv_clusters', resolution=0.8)
    
    return adata


def visualize_cnv_results(adata, output_dir):
    """
    Visualize CNV results
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with CNV results
    output_dir : str
        Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot CNV heatmap
    plt.figure(figsize=(20, 10))
    cnv.pl.chromosome_heatmap(adata, groupby='condition')
    plt.savefig(os.path.join(output_dir, 'cnv_heatmap_by_condition.png'), dpi=300, bbox_inches='tight')
    
    # Plot CNV heatmap by original cluster
    if 'leiden' in adata.obs.columns:
        plt.figure(figsize=(20, 10))
        cnv.pl.chromosome_heatmap(adata, groupby='leiden')
        plt.savefig(os.path.join(output_dir, 'cnv_heatmap_by_cluster.png'), dpi=300, bbox_inches='tight')
    
    # Plot CNV heatmap by CNV-based cluster
    if 'cnv_clusters' in adata.obs.columns:
        plt.figure(figsize=(20, 10))
        cnv.pl.chromosome_heatmap(adata, groupby='cnv_clusters')
        plt.savefig(os.path.join(output_dir, 'cnv_heatmap_by_cnv_cluster.png'), dpi=300, bbox_inches='tight')
    
    # Plot CNV UMAP
    plt.figure(figsize=(10, 10))
    sc.pl.umap(adata, color=['condition', 'reference', 'cnv_clusters'], ncols=1)
    plt.savefig(os.path.join(output_dir, 'cnv_umap.png'), dpi=300, bbox_inches='tight')
    
    # Plot specific chromosomes of interest (common in cancer)
    chromosomes_of_interest = ['1', '7', '8', '17']
    for chrom in chromosomes_of_interest:
        plt.figure(figsize=(15, 5))
        cnv.pl.chromosome(adata, chromosome=chrom, groupby='condition')
        plt.savefig(os.path.join(output_dir, f'cnv_chromosome_{chrom}.png'), dpi=300, bbox_inches='tight')


#######################
# Main Execution Function
#######################

def run_full_pipeline(sample_info_csv, transcriptome_ref, gene_pos_file, output_dir):
    """
    Run the full pipeline from FASTQ to CNV analysis
    
    Parameters:
    -----------
    sample_info_csv : str
        Path to CSV with sample information (sample_id, fastq_dir, condition)
    transcriptome_ref : str
        Path to Cell Ranger compatible reference genome
    gene_pos_file : str
        Path to gene position file
    output_dir : str
        Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sample information
    sample_info_df = pd.read_csv(sample_info_csv)
    
    # 1. Run Cell Ranger for all samples
    h5_files = process_all_samples(sample_info_df, transcriptome_ref, output_dir)
    
    # 2. Load and preprocess samples
    adatas = load_and_preprocess_samples(h5_files, sample_info_df)
    
    # 3. Perform clustering and basic annotation on each sample group
    if 'tumor' in adatas:
        adatas['tumor'] = cluster_and_annotate(adatas['tumor'])
    if 'normal' in adatas:
        adatas['normal'] = cluster_and_annotate(adatas['normal'])
    
    # Save preprocessed data
    for condition, adata in adatas.items():
        adata.write(os.path.join(output_dir, f"{condition}_preprocessed.h5ad"))
    
    # 4. Integrate tumor and normal samples with scVI
    if 'tumor' in adatas and 'normal' in adatas:
        adata_integrated, scvi_model = integrate_with_scvi(adatas['tumor'], adatas['normal'])
        adata_integrated.write(os.path.join(output_dir, "integrated_data.h5ad"))
    else:
        adata_integrated = None
        print("Cannot perform integration: need both tumor and normal samples")
    
    # 5. Prepare for CNV analysis
    if adata_integrated is not None:
        adata_cnv = prepare_for_cnv(adata_integrated, gene_pos_file)
        
        # 6. Run CNV inference
        # Identify the key for cell types: might be 'leiden' or a specific annotation
        reference_key = 'leiden'  # This could be a specific annotation column too
        
        # Determine reference categories (typically immune cells or stromal cells)
        # This is an example - you should identify the actual clusters that correspond to normal cells
        reference_clusters = ['0', '5', '8']  # Example clusters that might represent normal cells
        
        adata_cnv = run_infercnv(adata_cnv, reference_key=reference_key, 
                                reference_cat=reference_clusters)
        
        # Save CNV results
        adata_cnv.write(os.path.join(output_dir, "cnv_results.h5ad"))
        
        # 7. Visualize results
        visualize_cnv_results(adata_cnv, os.path.join(output_dir, "cnv_plots"))
    
    print("Pipeline completed successfully!")


# Example usage
if __name__ == "__main__":
    # Create a sample information CSV with the following columns:
    # sample_id,fastq_dir,condition
    # tumor_sample,/path/to/tumor_fastqs,tumor
    # normal_sample1,/path/to/normal1_fastqs,normal
    # normal_sample2,/path/to/normal2_fastqs,normal
    
    # Sample configuration
    sample_info_csv = "sample_info.csv"
    transcriptome_ref = "/path/to/refdata-gex-GRCh38-2020-A"
    gene_pos_file = "/path/to/gene_positions.tsv"
    output_dir = "sc_cnv_analysis"
    
    run_full_pipeline(sample_info_csv, transcriptome_ref, gene_pos_file, output_dir)


