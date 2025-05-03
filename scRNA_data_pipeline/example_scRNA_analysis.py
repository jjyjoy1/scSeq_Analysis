import scanpy as sc
from scrna_pipeline import ScRNASeqPipeline

# 1. Download a sample dataset (PBMC dataset from 10X Genomics)
adata_pbmc = sc.datasets.pbmc3k()
adata_pbmc.obs['batch'] = 'pbmc3k'

# Create a synthetic second batch by subsampling
adata_pbmc2 = adata_pbmc[::2].copy()
adata_pbmc2.obs['batch'] = 'pbmc3k_batch2'

# Save datasets
adata_pbmc.write('./data/pbmc_batch1.h5ad')
adata_pbmc2.write('./data/pbmc_batch2.h5ad')

# 2. Initialize the pipeline and specify data paths
pipeline = ScRNASeqPipeline(output_dir="./pbmc_results")

data_paths = {
    "batch1": "./data/pbmc_batch1.h5ad",
    "batch2": "./data/pbmc_batch2.h5ad"
}

pipeline.annotate_clusters(markers_dict=immune_markers)

# 3. Save the results
pipeline.save_results(filename='pbmc_analysis.h5ad')

print("Analysis complete!")
