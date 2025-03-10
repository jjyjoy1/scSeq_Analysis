# Comprehensive Guide to Single-Cell RNA-Seq Analysis

This guide provides a complete workflow for analyzing single-cell RNA sequencing data, with implementations in both Python and R. The pipeline includes data integration, normalization, dimensionality reduction (including VAE-based approaches), clustering, and differential expression analysis with uncertainty quantification.

## Choosing Between Python and R Implementations

Both implementations provide similar functionality but with different strengths:

### Python Implementation (Scanpy + scVI)
- Uses Scanpy for preprocessing and basic analysis
- Leverages scVI for VAE-based integration and Bayesian differential expression
- Provides uncertainty quantification in differential expression
- Better for large datasets due to GPU acceleration
- More customizable deep learning models

### R Implementation (Seurat + Harmony)
- Uses Seurat for preprocessing and analysis
- Leverages Harmony for batch integration
- Provides robust statistical testing with MAST
- Familiar to many biologists
- Excellent visualization capabilities

## General Workflow

Both implementations follow this general workflow:

1. **Data Loading and QC**
   - Load multiple datasets/batches
   - Calculate QC metrics (gene count, UMI count, mitochondrial percentage)
   - Filter low-quality cells and genes

2. **Normalization and Feature Selection**
   - Normalize for sequencing depth
   - Find highly variable genes
   - Scale data

3. **Batch Integration**
   - Correct for technical variations between batches
   - Create an integrated representation

4. **Dimensionality Reduction**
   - PCA for initial reduction
   - VAE (Python) or Harmony (R) for integration
   - UMAP for visualization

5. **Clustering**
   - Graph-based community detection
   - Flexible resolution parameter to control granularity

6. **Marker Gene Identification**
   - Find genes that distinguish clusters
   - Visualize with heatmaps and feature plots

7. **Differential Expression Analysis**
   - Compare expression between conditions or clusters
   - Visualize with volcano plots and heatmaps

## Python Implementation Example

Here's how to use the Python implementation for a complete analysis:

```python
from scrna_pipeline import ScRNASeqPipeline

# 1. Initialize pipeline and load data
pipeline = ScRNASeqPipeline(output_dir="./results")
data_paths = {
    "sample1": "path/to/sample1.h5ad",
    "sample2": "path/to/sample2.h5ad"
}
pipeline.load_data(data_paths, batch_key="batch")

# 2. Run the complete pipeline
pipeline.qc_and_filtering(min_genes=200, max_genes=6000, max_mt_percent=20) \
    .normalize_and_scale(n_top_genes=2000) \
    .run_pca(n_comps=50) \
    .prepare_scvi(batch_key="batch") \
    .run_scvi(n_latent=30, n_epochs=200) \
    .run_umap(use_scvi=True) \
    .run_clustering(resolution=0.8) \
    .find_markers() \
    .run_bayesian_de() \
    .save_results()

# 3. Annotate clusters with known cell type markers
immune_markers = {
    "T_cells": ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4"],
    "B_cells": ["CD19", "MS4A1", "CD79A", "CD79B"],
    "NK_cells": ["NCAM1", "NKG7", "KLRD1", "KLRF1"],
    "Monocytes": ["CD14", "LYZ", "FCGR3A", "MS4A7"],
    "Dendritic_cells": ["FCER1A", "CLEC10A", "ITGAX", "ITGAM"],
    "Neutrophils": ["S100A8", "S100A9", "FCGR3B", "CSF3R"]
}
pipeline.annotate_clusters(markers_dict=immune_markers)
```

### For Custom VAE Analysis (More Advanced)

```python
from vae_de_implementation import prepare_data_from_anndata, scRNAVAE, train_vae, bayesian_differential_expression

# Prepare data from AnnData object
train_dl, val_dl, data_info = prepare_data_from_anndata(
    adata,
    batch_key="batch",
    use_highly_variable=True
)

# Create and train VAE model
model = scRNAVAE(
    n_genes=data_info['n_genes'],
    n_batches=data_info['n_batches'],
    latent_dim=20
)
metrics = train_vae(model, train_dl, val_dl, n_epochs=100)

# Run Bayesian differential expression
de_results = bayesian_differential_expression(
    model,
    adata,
    groupby="condition",
    group1="control",
    group2="treatment",
    batch_key="batch",
    n_samples=50
)

# Visualize results
from vae_de_implementation import plot_de_results, visualize_uncertainty
fig = plot_de_results(de_results, n_top_genes=20)

# Visualize uncertainty for top genes
top_genes = de_results.head(5)['gene'].tolist()
fig = visualize_uncertainty(model, adata, gene_names=top_genes, groupby="condition", n_samples=20)
```

## R Implementation Example

Here's how to use the R implementation for a complete analysis:

```r
source("r_scrna_pipeline.R")

# 1. Set up file paths and parameters
file_paths <- c(
  "path/to/sample1/",
  "path/to/sample2/"
)
batch_ids <- c("batch1", "batch2")

# 2. Define cell type markers for annotation
immune_markers <- list(
  "T_cells" = c("CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4"),
  "B_cells" = c("CD19", "MS4A1", "CD79A", "CD79B"),
  "NK_cells" = c("NCAM1", "NKG7", "KLRD1", "KLRF1"),
  "Monocytes" = c("CD14", "LYZ", "FCGR3A", "MS4A7"),
  "Dendritic_cells" = c("FCER1A", "CLEC10A", "ITGAX", "ITGAM"),
  "Neutrophils" = c("S100A8", "S100A9", "FCGR3B", "CSF3R")
)

# 3. Run the complete pipeline
results <- run_scrna_seq_pipeline(
  file_paths = file_paths,
  batch_ids = batch_ids,
  file_format = "10X_mtx",
  output_dir = "./scrna_analysis",
  markers_dict = immune_markers,
  normalization_method = "SCT",
  n_variable_features = 2000,
  n_pcs = 30,
  resolution = 0.8
)

# 4. Access the resulting Seurat object
seurat_obj <- results$seurat_obj

# 5. Run differential expression between conditions
if ("condition" %in% colnames(seurat_obj@meta.data)) {
  de_results <- condition_differential_expression(
    seurat_obj = seurat_obj,
    condition_col = "condition",
    condition1 = "control",
    condition2 = "treatment",
    output_dir = "./scrna_analysis"
  )
}
```

## Comparing Python vs. R Methods for Key Steps

### Batch Integration

**Python (scVI)**:
- Uses variational autoencoders to learn a batch-corrected latent representation
- Models count distribution probabilistically
- Provides uncertainty estimates
- Learns non-linear relationships

**R (Harmony)**:
- Iterative clustering and correction algorithm
- Fast and memory-efficient
- Works directly on PCA embeddings
- Linear correction approach

### Normalization

**Python**:
- Library size normalization followed by log transformation
- Supports scVI's built-in normalization via negative binomial model

**R**:
- Standard log-normalization
- SCTransform, which uses regularized negative binomial regression

### Differential Expression

**Python**:
- Bayesian differential expression via VAE sampling
- Uncertainty quantification through multiple posterior samples
- Accounts for batch effects in latent space

**R**:
- MAST: hurdle model for zero-inflated data
- Robust statistical testing with multiple comparison correction
- Condition-specific and cluster-specific comparisons

## Tips for Analysis

### Choosing the Right Resolution for Clustering

The resolution parameter controls the granularity of clustering. Higher values lead to more clusters:

```python
# Python - try multiple resolutions
for res in [0.4, 0.6, 0.8, 1.0, 1.2]:
    pipeline.run_clustering(resolution=res)
```

```r
# R - try multiple resolutions
resolution_range <- seq(0.4, 1.2, by=0.2)
for (res in resolution_range) {
  seurat_obj <- FindClusters(seurat_obj, resolution = res)
}
```

### Evaluating Batch Integration Quality

Check if cells from different batches mix well in the UMAP:

```python
# Python - plot batches vs clusters
sc.pl.umap(adata, color=['batch', 'leiden'], wspace=0.5)
```

```r
# R - plot batches vs clusters
p1 <- DimPlot(seurat_obj, reduction = "umap", group.by = "batch")
p2 <- DimPlot(seurat_obj, reduction = "umap", group.by = "seurat_clusters", label = TRUE)
p1 | p2
```

### Handling Ambient RNA Contamination

Consider using SoupX (R) or removing ambient RNA signatures:

```r
# R - use SoupX for ambient RNA removal
library(SoupX)
sc <- SoupChannel(TObjImport, soupImport)
sc <- estimateSoup(sc)
sc <- setClusters(sc, seurat_obj$seurat_clusters)
corrected_counts <- adjustCounts(sc)
```

## Advanced Analysis Capabilities

### Trajectory Analysis

After clustering, you might want to infer developmental trajectories:

**Python**:
```python
# Using PAGA in Scanpy
sc.tl.paga(adata, groups='leiden')
sc.pl.paga(adata, color='leiden')
sc.tl.umap(adata, init_pos='paga')
```

**R**:
```r
# Using Slingshot in R
library(slingshot)
sce <- as.SingleCellExperiment(seurat_obj)
sce <- slingshot(sce, clusterLabels = 'seurat_clusters', reducedDim = 'UMAP')
```

### Cell-Cell Communication Analysis

**Python**:
```python
# Using CellChat
import cellchat
adata_chat = cellchat.tl.prepare_adata(adata, cluster_key='leiden', cell_type_key='cell_type')
cellchat.tl.run_cellchat(adata_chat)
```

**R**:
```r
# Using CellChat
library(CellChat)
cellchat <- createCellChat(seurat_obj, group.by = "cell_type")
cellchat <- addMeta(cellchat, meta = seurat_obj@meta.data)
cellchat <- setIdent(cellchat, ident.use = "cell_type")
cellchat <- subsetData(cellchat)
```

### RNA Velocity Analysis

To analyze future cell states based on spliced/unspliced ratios:

**Python**:
```python
# Using scVelo
import scvelo as sv
adata_vel = sv.read_loom("velocyto_output.loom")
adata_vel = sv.utils.merge(adata, adata_vel)
sv.tl.velocity(adata_vel, mode='stochastic')
sv.tl.velocity_graph(adata_vel)
sv.pl.velocity_embedding_stream(adata_vel, basis='umap')
```

## Interpreting Results

### Key Visualizations to Generate

1. **QC Plots**: Violin plots of UMI count, gene count, and mitochondrial percentage
2. **UMAP by Batch**: Check integration success
3. **UMAP by Cluster**: Identify cellular populations
4. **UMAP by Cell Type**: After annotation
5. **Marker Heatmap**: Top genes per cluster
6. **Volcano Plots**: For differential expression
7. **Feature Plots**: Key genes of interest

### Important Metrics to Report

1. **Number of cells** before and after filtering
2. **Number of clusters** identified
3. **Marker genes** for each cluster
4. **Differential expression statistics** between conditions
5. **Integration metrics** (mixing entropy, silhouette scores)

## Conclusion

The pipelines provided offer comprehensive frameworks for single-cell RNA-seq analysis. The Python implementation excels at scalability and uncertainty quantification through its VAE-based approach, while the R implementation provides robust statistical testing and visualization through Seurat and Harmony.

Choose the implementation that best matches your biological question, computational resources, and familiarity with the programming languages.

## References

1. Wolf FA, Angerer P, Theis FJ. (2018). "SCANPY: large-scale single-cell gene expression data analysis." Genome Biology.
2. Lopez R, Regier J, Cole MB, Jordan MI, Yosef N. (2018). "Deep generative modeling for single-cell transcriptomics." Nature Methods.
3. Stuart T, Butler A, Hoffman P, et al. (2019). "Comprehensive Integration of Single-Cell Data." Cell.
4. Korsunsky I, Millard N, Fan J, et al. (2019). "Fast, sensitive and accurate integration of single-cell data with Harmony." Nature Methods.
5. Finak G, McDavid A, Yajima M, et al. (2015). "MAST: a flexible statistical framework for assessing transcriptional changes and characterizing heterogeneity in single-cell RNA sequencing data." Genome Biology.
