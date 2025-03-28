Graph-Linked Unified Embedding Representation(GLUER), integrates single-cell multi-omics and multiplex imaging data by first finding shared factors across modalities (via matrix factorization) and then using a neural network to project different assays into a common nonlinear space. 

GLUER is a computational method designed to integrate multiple types of single-cell data, particularly focusing on single-cell multi-omics and multiplex imaging data.

GLUER works through a two-step process:

First, it performs matrix factorization across different data modalities to identify shared factors or patterns. This step helps extract common biological signals that exist across the different measurement types.
Then, it uses a neural network to project these different assays into a common nonlinear embedding space, allowing for integrated analysis of heterogeneous data types.

The approach is particularly valuable for analyzing complex biological systems where you have measurements from different technologies (like scRNA-seq, ATAC-seq, and imaging data) for the same biological system. By creating this unified representation, GLUER allows researchers to leverage complementary information across modalities and gain more comprehensive insights into cellular states and functions.
GLUER is part of the broader field of multi-modal data integration methods in computational biology, which has been advancing rapidly to handle the increasing diversity of single-cell measurement technologies.

GLUER_RNASeq_Spaatial_step1.py
Load and preprocess both scRNA-seq and spatial transcriptomics data
Find common genes between the two datasets
Create GLUER datasets for each modality
Initialize and train a GLUER model
Extract integrated embeddings that combine information from both modalities
Use the integrated embeddings for downstream analysis like clustering
Visualize the clusters in both the RNA-seq UMAP and spatial coordinates
Perform differential expression analysis using the integrated clusters
Visualize factor loadings to interpret the shared factors identified by GLUER

GLUER_RNASeq_Spaatial_step2.py
evaluate the performance of GLUER integration model, which handles
Model training with multiple hyperparameter configurations
Quantitative performance metrics
Visual representation of results
Cross-modality prediction assessment

Key Performance Metrics Included
The evaluation framework measures GLUER's performance using several metrics:
Integration Quality Metrics

Modality Mixing Score: Measures how well RNA-seq and spatial data points intermingle in the latent space (higher is better)
Reconstruction Loss: Evaluates how well the model reconstructs the original data

Clustering Quality Metrics

Adjusted Rand Index (ARI): Compares clustering results with ground truth labels
Normalized Mutual Information (NMI): Measures shared information between predicted and true labels
Silhouette Score: Evaluates cluster separation and cohesion

Transfer Learning Evaluation

Cross-Modal Prediction Accuracy: Assesses how well cell type labels transfer between modalities

Visualization Components
The evaluation code generates multiple visualizations:

Training History: Loss curves showing model convergence
Hyperparameter Comparison: Bar charts comparing performance across different model configurations
Embedding Visualizations:

t-SNE and UMAP projections colored by modality and cell type
Spatial plots with transferred cell type labels


Factor Analysis: Heatmaps of factor loadings and lists of top genes per factor





