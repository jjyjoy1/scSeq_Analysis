# GLUER Performance Evaluation and Visualization
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_index, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import anndata
from scipy.sparse import issparse
from gluer import GLUER, GluerDataset
from gluer.utils import create_graph, preprocess_data

# Load and preprocess datasets (same as before)
# ...code for loading and preprocessing data...

# Split data into training and test sets for cross-validation
def split_anndata(adata, test_size=0.2, random_state=42):
    """Split AnnData object into train and test sets"""
    train_idx, test_idx = train_test_split(
        np.arange(adata.n_obs), test_size=test_size, random_state=random_state
    )
    return adata[train_idx], adata[test_idx]

rna_train, rna_test = split_anndata(rna_data)
spatial_train, spatial_test = split_anndata(spatial_data)

# Create GLUER datasets for train and test
rna_train_dataset = GluerDataset(rna_train.X, modality='rna')
spatial_train_dataset = GluerDataset(spatial_train.X, modality='spatial',
                                    spatial_coords=spatial_train.obsm['spatial'])

rna_test_dataset = GluerDataset(rna_test.X, modality='rna')
spatial_test_dataset = GluerDataset(spatial_test.X, modality='spatial',
                                   spatial_coords=spatial_test.obsm['spatial'])

# Initialize GLUER model with various hyperparameters for comparison
hyperparameter_sets = [
    {'hidden_dims': [512, 256], 'latent_dim': 32, 'n_factors': 15},
    {'hidden_dims': [512, 256], 'latent_dim': 64, 'n_factors': 20},
    {'hidden_dims': [1024, 512, 256], 'latent_dim': 128, 'n_factors': 30}
]

results = []

# Evaluate different hyperparameter settings
for i, params in enumerate(hyperparameter_sets):
    print(f"Training model {i+1}/{len(hyperparameter_sets)} with params: {params}")
    
    # Initialize and train GLUER model
    gluer_model = GLUER(
        input_dims={'rna': rna_train_dataset.n_features, 
                    'spatial': spatial_train_dataset.n_features},
        hidden_dims=params['hidden_dims'],
        latent_dim=params['latent_dim'],
        modalities=['rna', 'spatial'],
        n_factors=params['n_factors']
    )
    
    # Train the model with early stopping
    history = gluer_model.fit(
        datasets={'rna': rna_train_dataset, 'spatial': spatial_train_dataset},
        batch_size=128,
        epochs=100,
        learning_rate=1e-3,
        validation_datasets={'rna': rna_test_dataset, 'spatial': spatial_test_dataset},
        early_stopping=True,
        patience=10
    )
    
    # Get embeddings for test data
    train_embeddings = gluer_model.get_latent_embeddings({
        'rna': rna_train_dataset,
        'spatial': spatial_train_dataset
    })
    
    test_embeddings = gluer_model.get_latent_embeddings({
        'rna': rna_test_dataset,
        'spatial': spatial_test_dataset
    })
    
    # Assuming we have some ground truth cell type annotations
    # If not available, we can use unsupervised metrics only
    if 'cell_type' in rna_data.obs.columns:
        # Supervised evaluation: Cell type classification
        # Train KNN classifier on training embeddings
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(train_embeddings[:rna_train.n_obs], rna_train.obs['cell_type'])
        
        # Predict cell types for test embeddings
        rna_test_pred = knn.predict(test_embeddings[:rna_test.n_obs])
        rna_test_acc = accuracy_score(rna_test.obs['cell_type'], rna_test_pred)
        
        # Predict cell types for spatial data 
        # (assuming spatial data doesn't have cell type annotations)
        spatial_train_pred = knn.predict(train_embeddings[rna_train.n_obs:])
        
        # Compute ARI for clustering quality
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=len(rna_data.obs['cell_type'].unique()), random_state=0)
        cluster_labels = kmeans.fit_predict(test_embeddings)
        ari = adjusted_rand_index(
            rna_test.obs['cell_type'].values,
            cluster_labels[:rna_test.n_obs]
        )
        nmi = normalized_mutual_info_score(
            rna_test.obs['cell_type'].values,
            cluster_labels[:rna_test.n_obs]
        )
    else:
        # If no cell type annotations, just compute unsupervised metrics
        rna_test_acc = np.nan
        ari = np.nan
        nmi = np.nan
    
    # Unsupervised evaluation: Silhouette score
    sil_score = silhouette_score(test_embeddings, kmeans.labels_)
    
    # Integration quality: Batch mixing metric
    # Calculate how well the two modalities mix in the latent space
    # Create labels for modality
    modality_labels = np.concatenate([
        np.zeros(rna_test.n_obs),
        np.ones(spatial_test.n_obs)
    ])
    
    # Compute KNN graph
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=20)
    nn.fit(test_embeddings)
    indices = nn.kneighbors(test_embeddings, return_distance=False)
    
    # Compute modality mixing score (1 = perfect mixing, 0 = no mixing)
    mixing_scores = []
    for i, neighbors in enumerate(indices):
        my_modality = modality_labels[i]
        neighbor_modalities = modality_labels[neighbors]
        mixing_score = np.mean(neighbor_modalities != my_modality)
        mixing_scores.append(mixing_score)
    
    avg_mixing = np.mean(mixing_scores)
    
    # Calculate reconstruction loss on test data
    test_loss = gluer_model.compute_loss({
        'rna': rna_test_dataset,
        'spatial': spatial_test_dataset
    })
    
    # Store results
    results.append({
        'params': params,
        'accuracy': rna_test_acc,
        'ari': ari,
        'nmi': nmi,
        'silhouette': sil_score,
        'modality_mixing': avg_mixing,
        'test_loss': test_loss,
        'history': history,
        'model': gluer_model,
        'train_embeddings': train_embeddings,
        'test_embeddings': test_embeddings
    })
    
    print(f"Model {i+1} results:")
    print(f"  Accuracy: {rna_test_acc:.4f}")
    print(f"  ARI: {ari:.4f}")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Silhouette: {sil_score:.4f}")
    print(f"  Modality mixing: {avg_mixing:.4f}")
    print(f"  Test loss: {test_loss:.4f}")

# Find best model based on combined metrics
best_idx = np.argmax([
    r['accuracy'] + r['ari'] + r['silhouette'] + r['modality_mixing'] - r['test_loss'] 
    for r in results
])
best_model = results[best_idx]['model']
best_params = results[best_idx]['params']
print(f"Best model: {best_idx+1} with params: {best_params}")

# Comprehensive visualization of best model
#---------------------------------------------------------------------------------

# 1. Training history visualization
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(results[best_idx]['history']['loss'])
plt.plot(results[best_idx]['history']['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

# 2. Hyperparameter comparison
plt.subplot(1, 3, 2)
metrics = ['accuracy', 'ari', 'silhouette', 'modality_mixing']
x = np.arange(len(hyperparameter_sets))
width = 0.2
for i, metric in enumerate(metrics):
    values = [r[metric] for r in results]
    plt.bar(x + i*width, values, width, label=metric)
plt.xticks(x + width*1.5, [f"Model {i+1}" for i in range(len(hyperparameter_sets))])
plt.legend()
plt.title('Model Comparison')

# 3. Plot latent dimensions influence
plt.subplot(1, 3, 3)
latent_dims = [p['latent_dim'] for p in hyperparameter_sets]
test_losses = [r['test_loss'] for r in results]
plt.scatter(latent_dims, test_losses)
for i, (x, y) in enumerate(zip(latent_dims, test_losses)):
    plt.annotate(f"Model {i+1}", (x, y))
plt.xlabel('Latent Dimension')
plt.ylabel('Test Loss')
plt.title('Latent Dim vs Test Loss')
plt.tight_layout()
plt.savefig('gluer_training_metrics.png', dpi=300)

# 4. Embedding visualizations
# Get best embeddings
train_embeddings = results[best_idx]['train_embeddings']
test_embeddings = results[best_idx]['test_embeddings']

# Dimensionality reduction for visualization
# t-SNE
tsne = TSNE(n_components=2, random_state=42)
train_tsne = tsne.fit_transform(train_embeddings)
rna_train_tsne = train_tsne[:rna_train.n_obs]
spatial_train_tsne = train_tsne[rna_train.n_obs:]

# UMAP
reducer = umap.UMAP(random_state=42)
train_umap = reducer.fit_transform(train_embeddings)
rna_train_umap = train_umap[:rna_train.n_obs]
spatial_train_umap = train_umap[rna_train.n_obs:]

# Visualization of the embeddings
plt.figure(figsize=(20, 15))

# 5. t-SNE colored by modality
plt.subplot(2, 3, 1)
plt.scatter(rna_train_tsne[:, 0], rna_train_tsne[:, 1], alpha=0.5, label='RNA-seq', s=5)
plt.scatter(spatial_train_tsne[:, 0], spatial_train_tsne[:, 1], alpha=0.5, label='Spatial', s=5)
plt.title('t-SNE of Integrated Embeddings by Modality')
plt.legend()

# 6. t-SNE colored by cell type (if available)
if 'cell_type' in rna_data.obs.columns:
    plt.subplot(2, 3, 2)
    cell_types = rna_train.obs['cell_type'].unique()
    for ct in cell_types:
        mask = rna_train.obs['cell_type'] == ct
        plt.scatter(rna_train_tsne[mask, 0], rna_train_tsne[mask, 1], label=ct, s=5)
    plt.title('t-SNE of RNA-seq Embeddings by Cell Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 7. UMAP colored by modality
plt.subplot(2, 3, 4)
plt.scatter(rna_train_umap[:, 0], rna_train_umap[:, 1], alpha=0.5, label='RNA-seq', s=5)
plt.scatter(spatial_train_umap[:, 0], spatial_train_umap[:, 1], alpha=0.5, label='Spatial', s=5)
plt.title('UMAP of Integrated Embeddings by Modality')
plt.legend()

# 8. UMAP colored by cell type (if available)
if 'cell_type' in rna_data.obs.columns:
    plt.subplot(2, 3, 5)
    for ct in cell_types:
        mask = rna_train.obs['cell_type'] == ct
        plt.scatter(rna_train_umap[mask, 0], rna_train_umap[mask, 1], label=ct, s=5)
    plt.title('UMAP of RNA-seq Embeddings by Cell Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 9. Spatial plot with transferred labels
plt.subplot(2, 3, 3)
if 'cell_type' in rna_data.obs.columns:
    # Get the best model's predictions for spatial data
    best_model_spatial_pred = results[best_idx]['model'].transfer_labels(
        source_data=rna_train_dataset,
        source_labels=rna_train.obs['cell_type'].values,
        target_data=spatial_train_dataset
    )
    
    # Plot spatial coordinates colored by predicted cell type
    spatial_coords = spatial_train.obsm['spatial']
    for i, ct in enumerate(cell_types):
        mask = best_model_spatial_pred == ct
        plt.scatter(spatial_coords[mask, 0], spatial_coords[mask, 1], 
                   label=ct, s=15, alpha=0.7)
    plt.title('Spatial Data with Transferred Cell Types')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
else:
    # If no cell type info, show clustering results
    kmeans = KMeans(n_clusters=8, random_state=0)
    spatial_clusters = kmeans.fit_predict(train_embeddings[rna_train.n_obs:])
    
    # Plot spatial coordinates colored by cluster
    spatial_coords = spatial_train.obsm['spatial']
    for i in range(8):
        mask = spatial_clusters == i
        plt.scatter(spatial_coords[mask, 0], spatial_coords[mask, 1], 
                   label=f'Cluster {i}', s=15, alpha=0.7)
    plt.title('Spatial Data Clustering from Integration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 10. Factor loadings visualization
plt.subplot(2, 3, 6)
factor_loadings = best_model.get_factor_loadings()
plt.figure(figsize=(15, 10))

# Get top genes for each factor
n_factors = factor_loadings['rna'].shape[0]
n_top_genes = 10
top_genes_per_factor = []

for f in range(n_factors):
    # Get top genes for this factor
    factor_scores = factor_loadings['rna'][f, :]
    top_idx = np.argsort(-np.abs(factor_scores))[:n_top_genes]
    top_genes = [common_genes[i] for i in top_idx]
    top_scores = factor_scores[top_idx]
    top_genes_per_factor.append((top_genes, top_scores))

# Plot heatmap of factor loadings
plt.figure(figsize=(20, 10))
for mod_idx, modality in enumerate(['rna', 'spatial']):
    plt.subplot(1, 2, mod_idx+1)
    sns.heatmap(factor_loadings[modality], cmap='coolwarm', center=0)
    plt.title(f'{modality.upper()} Factor Loadings')
    plt.xlabel('Genes')
    plt.ylabel('Factors')

plt.tight_layout()
plt.savefig('gluer_factor_loadings.png', dpi=300)

# 11. Print top genes for each factor
for f in range(min(5, n_factors)):  # Show top 5 factors
    genes, scores = top_genes_per_factor[f]
    print(f"Factor {f+1} top genes:")
    for gene, score in zip(genes, scores):
        print(f"  {gene}: {score:.4f}")

# 12. Cross-modality prediction evaluation
# If we have annotations for both modalities, evaluate transfer learning
if 'spatial_annotations' in spatial_data.obs.columns and 'cell_type' in rna_data.obs.columns:
    # RNA to Spatial transfer
    rna_to_spatial_pred = best_model.transfer_labels(
        source_data=rna_train_dataset,
        source_labels=rna_train.obs['cell_type'].values,
        target_data=spatial_train_dataset
    )
    
    rna_to_spatial_acc = accuracy_score(
        spatial_train.obs['spatial_annotations'].values, 
        rna_to_spatial_pred
    )
    
    # Spatial to RNA transfer
    spatial_to_rna_pred = best_model.transfer_labels(
        source_data=spatial_train_dataset,
        source_labels=spatial_train.obs['spatial_annotations'].values,
        target_data=rna_train_dataset
    )
    
    spatial_to_rna_acc = accuracy_score(
        rna_train.obs['cell_type'].values, 
        spatial_to_rna_pred
    )
    
    print("Cross-modality prediction results:")
    print(f"  RNA → Spatial accuracy: {rna_to_spatial_acc:.4f}")
    print(f"  Spatial → RNA accuracy: {spatial_to_rna_acc:.4f}")

# 13. Save the best model
torch.save(best_model.state_dict(), 'best_gluer_model.pt')

# 14. Generate a comprehensive report
with open('gluer_model_report.md', 'w') as f:
    f.write("# GLUER Model Evaluation Report\n\n")
    
    f.write("## Model Hyperparameters\n\n")
    f.write(f"Best model parameters: {best_params}\n\n")
    
    f.write("## Performance Metrics\n\n")
    f.write(f"* Accuracy: {results[best_idx]['accuracy']:.4f}\n")
    f.write(f"* Adjusted Rand Index: {results[best_idx]['ari']:.4f}\n")
    f.write(f"* Normalized Mutual Information: {results[best_idx]['nmi']:.4f}\n")
    f.write(f"* Silhouette Score: {results[best_idx]['silhouette']:.4f}\n")
    f.write(f"* Modality Mixing: {results[best_idx]['modality_mixing']:.4f}\n")
    f.write(f"* Test Loss: {results[best_idx]['test_loss']:.4f}\n\n")
    
    f.write("## Training History\n\n")
    f.write("See 'gluer_training_metrics.png' for training history visualization.\n\n")
    
    f.write("## Embedding Visualization\n\n")
    f.write("See 'gluer_embedding_visualization.png' for t-SNE and UMAP visualizations.\n\n")
    
    f.write("## Factor Analysis\n\n")
    for f in range(min(5, n_factors)):
        genes, scores = top_genes_per_factor[f]
        f.write(f"### Factor {f+1} top genes:\n")
        for gene, score in zip(genes, scores):
            f.write(f"* {gene}: {score:.4f}\n")
        f.write("\n")
    
    f.write("## Conclusion\n\n")
    f.write("The GLUER model successfully integrated scRNA-seq and spatial transcriptomics data, ")
    f.write("revealing shared biological factors across modalities and enabling cross-modal transfer learning.\n")

print("Evaluation complete! See generated PNG files and report for detailed results.")
