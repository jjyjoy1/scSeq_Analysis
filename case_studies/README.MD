# scRNA-seq Analysis with VAEs and Deep Count Autoencoders

A comprehensive toolkit for analyzing single-cell RNA sequencing (scRNA-seq) data using deep learning approaches, featuring Variational Autoencoders (VAEs) and Deep Count Autoencoders (DCA) with hyperparameter optimization.

## Overview

This repository provides implementations of two powerful deep learning models for denoising, dimension reduction, and batch effect correction in scRNA-seq data:

1. **Variational Autoencoders (VAEs)** - Generative models that learn a probabilistic latent representation of gene expression data
2. **Deep Count Autoencoders (DCA)** - Specialized autoencoders designed specifically for the count-based nature of scRNA-seq data

Both approaches are enhanced with Optuna-based hyperparameter optimization to find the best model configurations for your specific datasets.

## Features

- 🧬 Load and preprocess scRNA-seq data from various formats
- 🧹 Denoise and impute gene expression data
- 🔄 Correct batch effects across multiple experiments
- 📊 Reduce dimensionality for visualization and analysis
- 🤖 Automatically optimize model parameters with Optuna
- 📈 Evaluate model performance with silhouette scores and visualization
- 📱 Interactive visualization dashboard (optional)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scRNA-analysis.git
cd scRNA-analysis

# Create a conda environment
conda create -n scseq python=3.9
conda activate scseq

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Running DCA with default parameters

```bash
python scrna_dca_pbmc_v1.py --no-optuna
```

### Running DCA with Optuna optimization

```bash
python scrna_dca_pbmc_v1.py --trials 30 --timeout 3600
```

### Running VAE with Optuna optimization

```bash
python scrna_vae_pbmc_v1.py --trials 30 --timeout 3600
```

## Model Explanations

### Variational Autoencoders (VAEs)

Variational Autoencoders are generative models that learn to encode data into a latent space with a probabilistic distribution, usually a Gaussian. The VAE architecture consists of:

1. **Encoder**: Maps high-dimensional gene expression data to a low-dimensional latent space, representing means (μ) and variances (σ²) of the latent variables.
2. **Latent Space**: A compressed representation where similar cells are positioned close to each other.
3. **Decoder**: Reconstructs the original gene expression data from the latent representation.

The VAE is trained by optimizing two components:
- **Reconstruction Loss**: Measures how well the model reconstructs the input data
- **KL Divergence**: Ensures the latent space has a well-behaved distribution

For scRNA-seq data, the VAE can be adapted with specialized loss functions like Negative Binomial or Zero-Inflated Negative Binomial to better handle count data with dropout events.

### Deep Count Autoencoders (DCA)

Deep Count Autoencoders are specialized neural networks designed specifically for scRNA-seq data. Key features include:

1. **Zero-Inflated Negative Binomial (ZINB) Loss**: Models both dropout events (zero inflation) and the count distribution of gene expression data, addressing the unique characteristics of scRNA-seq data.
2. **Deep Architecture**: Multiple hidden layers allow capturing complex relationships between genes.
3. **Regularization**: Techniques like dropout and L1/L2 regularization prevent overfitting.

DCA outperforms standard autoencoders for scRNA-seq by:
- Explicitly modeling technical noise and dropout events
- Accounting for overdispersion in count data
- Preserving biological signal while removing technical noise

### Hyperparameter Optimization with Optuna

Both models use Optuna for automatic hyperparameter optimization:

1. **Search Space**: Parameters like hidden layer dimensions, latent space dimensions, learning rate, batch size, and regularization strength.
2. **Objective**: Minimizing validation loss
3. **Pruning**: Early stopping of unpromising trials to improve efficiency
4. **Visualization**: Optimization history and parameter importance analysis

## Usage Examples

### Preprocessing Data

```python
# Load and preprocess data
adata = load_dataset()
adata_preprocessed = preprocess_data(
    adata,
    n_top_genes=2000,
    min_genes=200,
    min_cells=3
)
```

### Running Hyperparameter Optimization

```python
# Optimize DCA hyperparameters
best_params = optimize_dca_hyperparameters(
    adata_preprocessed,
    n_trials=30,
    timeout=3600,  # 1 hour timeout
    batch_key='batch'
)

# Train with optimized parameters
model, adata_denoised = train_dca_with_best_params(
    adata_preprocessed,
    best_params,
    batch_key='batch'
)
```

### Analysis and Visualization

```python
# Evaluate batch correction
evaluation_results = evaluate_dca_correction(
    adata_denoised,
    batch_key='batch',
    cell_type_key='cell_type'
)

# Visualize results
visualize_dca_results(
    adata_denoised,
    batch_key='batch',
    cell_type_key='cell_type'
)
```

## Model Comparison

| Feature | VAE | DCA |
|---------|-----|-----|
| Loss Function | MSE, BCE, or NB | ZINB |
| Latent Space | Probabilistic | Deterministic |
| Specialized for scRNA-seq | Adaptable | Specifically designed |
| Dropout Modeling | Optional | Integrated |
| Count Distribution Modeling | Optional | Integrated |
| Training Speed | Faster | Slower |
| Memory Usage | Lower | Higher |
| Performance on Sparse Data | Good | Excellent |

## Advanced Configuration

### Command-line Arguments

```
VAE Arguments:
  --latent-dim INT       Latent space dimensionality
  --hidden-dims INT,...  Hidden layer dimensions
  --beta FLOAT           Weight for KL divergence term
  --learning-rate FLOAT  Learning rate for optimizer
  --batch-size INT       Training batch size
  --no-optuna            Disable Optuna optimization
  --trials INT           Number of Optuna trials
  --timeout INT          Timeout for Optuna in seconds

DCA Arguments:
  --latent-dim INT       Latent space dimensionality
  --hidden-dim INT       Hidden layer dimensions
  --learning-rate FLOAT  Learning rate for optimizer
  --batch-size INT       Training batch size
  --no-zinb              Disable Zero-Inflated Negative Binomial loss
  --no-optuna            Disable Optuna optimization
  --trials INT           Number of Optuna trials
  --timeout INT          Timeout for Optuna in seconds
```

## References

1. Eraslan, G., Simon, L.M., Mircea, M. et al. Single-cell RNA-seq denoising using a deep count autoencoder. Nat Commun 10, 390 (2019).
2. Lopez, R., Regier, J., Cole, M.B. et al. Deep generative modeling for single-cell transcriptomics. Nat Methods 15, 1053–1058 (2018).
3. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.


