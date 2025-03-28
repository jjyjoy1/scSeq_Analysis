# Single-Cell RNA Sequencing Analysis Pipeline

A comprehensive pipeline for single-cell RNA sequencing analysis with implementations in both Python and R, focusing on batch integration, normalization, clustering, and differential expression analysis.

## Features

- **Dual Implementation**: Available in both Python (Scanpy + scVI) and R (Seurat + Harmony)
- **Extended Analyses**: Includes TCR/BCR analysis and spatial transcriptomics
- **Deep Learning Integration**: Summaries and implementations of three scData integration models
- **Modular Design**: Easy to adapt for different research needs

## Implementations

### Python (Scanpy + scVI)
✔ Complete modular pipeline built with Scanpy and scVI  
✔ Handles data loading, QC, normalization, dimensionality reduction, clustering, and marker identification  
✔ Implements VAE-based batch correction and Bayesian differential expression  
✔ Processes multiple datasets with batch effect removal  
✔ Quantifies uncertainty in gene expression through VAE modeling  

### R (Seurat + Harmony)
✔ Complementary pipeline using Seurat and Harmony  
✔ Robust statistical methods for single-cell analysis  
✔ Includes cell type annotation, visualization, and differential expression using MAST  

### Custom VAE Implementation
✔ PyTorch implementation of variational autoencoder for scRNA-seq  
✔ Models gene expression as negative binomial distributions  
✔ Provides uncertainty quantification for differential expression  
✔ Comprehensive visualization functions  

## Analysis Capabilities

### Data Integration
| Feature           | Python Implementation | R Implementation |
|-------------------|----------------------|------------------|
| Batch Correction  | scVI's VAE           | Harmony          |
| Data Normalization| Model-based (VAE)    | SCTransform      |

### Core Analysis Steps
1. **Normalization & Denoising**
   - Library size normalization with log transformation
   - SCTransform (R pipeline)
   - Model-based normalization with VAE (Python)

2. **Dimensionality Reduction**
   - PCA for initial reduction
   - VAE latent space (Python)
   - Harmony-corrected PCA (R)
   - UMAP for visualization

3. **Clustering**
   - Graph-based clustering with Leiden algorithm (Python)
   - Shared nearest neighbor clustering (R)
   - Adjustable resolution parameter

4. **Differential Expression**
   - Wilcoxon tests
   - Bayesian differential expression through VAE sampling (Python)
   - MAST for zero-inflated models (R)
   - Uncertainty quantification (Python)

## Documentation
- Comprehensive tutorial for both implementations
- Parameter selection guidelines
- Interpretation guidelines
- Comparison of integration and analysis methods

## Additional Resources
- TCR/BCR single cell data analysis scripts
- Spatial transcriptomics data analysis scripts
- Three scData integration deep learning models with:
  - Basic usages
  - Model performance evaluation
  - Visualization techniques

