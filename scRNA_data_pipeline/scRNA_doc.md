# Single-Cell RNA-seq Analysis Pipeline Documentation

## Pipeline Components

### 1. Quality Control and Filtering
The QC step calculates metrics and filters cells/genes based on:
- **Cell filtering criteria**:
  - Too few genes → potential empty droplets
  - Too many genes → potential doublets
  - High mitochondrial percentage → cell stress/death
  - Low-count cells → poor quality RNA

### 2. Normalization and Feature Selection
- Normalizes data to account for sequencing depth differences
- Identifies **highly variable genes (HVGs)** showing biological variation
- Reduces dimensionality and computational cost

### 3. VAE-based Integration (scVI)
Core pipeline component using [scVI](https://scvi-tools.org/):
- Learns latent representation accounting for batch effects
- Models count distribution of scRNA-seq data
- Provides uncertainty estimates

**Advantages over traditional PCA**:
✔ Non-linear (captures complex relationships)  
✔ Probabilistic (models uncertainty)  
✔ Specifically designed for scRNA-seq count distributions  

### 4. Clustering and Marker Gene Identification
Post-reduction steps:
1. Build nearest-neighbor graph
2. Cluster cells using [Leiden algorithm](https://www.nature.com/articles/s41598-019-41695-z)
3. Identify cluster-specific marker genes (Wilcoxon rank-sum test)
4. Run Bayesian differential expression with trained VAE

### 5. Cell Type Annotation
- Labels clusters by comparing marker genes with known signatures
- Particularly useful for well-characterized tissues (e.g., blood)

## Visualization Outputs
- QC metric distributions
- UMAP plots (colored by batch/cluster/cell type)
- Marker gene heatmaps and dot plots
- Differential expression results

## Advanced Analysis Options
Using the VAE latent space enables:
- **Trajectory inference**: [PAGA](https://doi.org/10.1186/s13059-019-1663-x), [Palantir](https://www.nature.com/articles/s41587-019-0068-4)
- **RNA velocity**: [scVelo](https://scvelo.readthedocs.io/)
- Multi-modal integration (CITE-seq, spatial)
- Transfer learning for new datasets

## Implementation Guide

### Data Preparation
```python
pipeline.load_data(data_paths, batch_key="batch")

