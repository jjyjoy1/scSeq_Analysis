# Import necessary libraries
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import anndata

# For GLUER specific imports
from gluer import GLUER, GluerDataset
from gluer.utils import create_graph, preprocess_data

# Load example data (in a real scenario, these would be your actual datasets)
# Single-cell RNA-seq data
rna_data = sc.read_h5ad('path/to/scRNA_data.h5ad')

# Spatial transcriptomics data
spatial_data = sc.read_h5ad('path/to/spatial_data.h5ad')

# Preprocess the data
# 1. Normalize and scale the RNA-seq data
sc.pp.normalize_total(rna_data, target_sum=1e4)
sc.pp.log1p(rna_data)
sc.pp.highly_variable_genes(rna_data, n_top_genes=2000)
rna_data = rna_data[:, rna_data.var.highly_variable]

# 2. Normalize and scale the spatial data
sc.pp.normalize_total(spatial_data, target_sum=1e4)
sc.pp.log1p(spatial_data)

# 3. Find common genes between the two datasets
common_genes = list(set(rna_data.var_names).intersection(set(spatial_data.var_names)))
print(f"Number of common genes: {len(common_genes)}")

# Subset both datasets to include only common genes
rna_data = rna_data[:, common_genes]
spatial_data = spatial_data[:, common_genes]

# Create GLUER datasets
rna_dataset = GluerDataset(rna_data.X, modality='rna')
spatial_dataset = GluerDataset(spatial_data.X, modality='spatial', 
                              spatial_coords=spatial_data.obsm['spatial'])

# Initialize GLUER model
gluer_model = GLUER(
    input_dims={'rna': rna_dataset.n_features, 
                'spatial': spatial_dataset.n_features},
    hidden_dims=[512, 256],
    latent_dim=64,
    modalities=['rna', 'spatial'],
    n_factors=20  # Number of shared factors to identify
)

# Train the model
gluer_model.fit(
    datasets={'rna': rna_dataset, 'spatial': spatial_dataset},
    batch_size=128,
    epochs=50,
    learning_rate=1e-3
)

# Get the integrated embeddings
integrated_embeddings = gluer_model.get_latent_embeddings({
    'rna': rna_dataset,
    'spatial': spatial_dataset
})

# The integrated embeddings can now be used for downstream analysis
# For example, clustering the integrated data
from sklearn.cluster import KMeans

# Perform clustering on the integrated embeddings
kmeans = KMeans(n_clusters=8, random_state=0).fit(integrated_embeddings)
cluster_labels = kmeans.labels_

# Add cluster labels to the original datasets
rna_data.obs['integrated_cluster'] = cluster_labels[:rna_data.n_obs]
spatial_data.obs['integrated_cluster'] = cluster_labels[rna_data.n_obs:]

# Visualize the integrated clusters
# For RNA-seq data
sc.pp.neighbors(rna_data, use_rep='X')
sc.tl.umap(rna_data)
sc.pl.umap(rna_data, color='integrated_cluster', title='RNA-seq data - Integrated clusters')

# For spatial data
plt.figure(figsize=(10, 8))
plt.scatter(spatial_data.obsm['spatial'][:, 0], 
            spatial_data.obsm['spatial'][:, 1], 
            c=spatial_data.obs['integrated_cluster'], 
            cmap='tab20', s=5)
plt.colorbar(label='Integrated cluster')
plt.title('Spatial data - Integrated clusters')
plt.xlabel('Spatial X')
plt.ylabel('Spatial Y')
plt.show()

# Perform differential expression analysis between clusters
sc.tl.rank_genes_groups(rna_data, 'integrated_cluster', method='wilcoxon')
sc.pl.rank_genes_groups(rna_data, n_genes=10, sharey=False)

# Visualize factor loadings to interpret shared factors
factor_loadings = gluer_model.get_factor_loadings()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(factor_loadings['rna'], aspect='auto', cmap='viridis')
plt.title('RNA-seq factor loadings')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(factor_loadings['spatial'], aspect='auto', cmap='viridis')
plt.title('Spatial factor loadings')
plt.colorbar()
plt.tight_layout()
plt.show()

