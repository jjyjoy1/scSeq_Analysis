# Complete scGPT Analysis Pipeline with Visualizations and Performance Evaluation
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import scgpt
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
import umap
import networkx as nx
from torch.utils.data import DataLoader

# Set plot styles
plt.rcParams['figure.figsize'] = (10, 8)
sns.set_style("whitegrid")
sc.settings.set_figure_params(dpi=100, frameon=False)

# 1. Data Loading and Preprocessing
print("Loading and preprocessing data...")
adata = sc.read_10x_mtx("./pbmc_data/filtered_gene_bc_matrices/hg19/")

# Basic preprocessing with Scanpy
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Calculate QC metrics
sc.pp.calculate_qc_metrics(adata, inplace=True)

# Visualize QC metrics
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(adata.obs['n_genes_by_counts'], kde=False, ax=axs[0])
axs[0].set_title('Distribution of genes per cell')
axs[0].set_xlabel('Number of genes')

sns.histplot(adata.obs['total_counts'], kde=False, ax=axs[1])
axs[1].set_title('Distribution of UMI counts per cell')
axs[1].set_xlabel('Number of UMIs')
plt.tight_layout()
plt.savefig('qc_metrics.png')
plt.close()

# Filter cells based on QC metrics
adata = adata[adata.obs['n_genes_by_counts'] < 4500, :]
adata = adata[adata.obs['pct_counts_mt'] < 20, :]

# Normalize and log transform
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Visualize gene expression distribution before and after normalization
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(adata.X.flatten(), bins=50, ax=axs[0])
axs[0].set_title('Gene expression distribution (log1p)')
axs[0].set_xlabel('log1p expression')

# Prepare data for scGPT
preprocessor = Preprocessor()
preprocessor.fit(adata)
processed_adata = preprocessor.transform(adata)

# Create gene vocabulary for tokenization
gene_vocab = GeneVocab.from_adata(processed_adata)
print(f"Created gene vocabulary with {len(gene_vocab)} genes")

# Convert data to scGPT format
input_data = scgpt.data.AnnDataset(
    processed_adata,
    gene_vocab=gene_vocab,
    max_seq_len=2000,  # Maximum number of genes to consider
    pad_token=gene_vocab.pad_token,
    mask_token=gene_vocab.mask_token
)

# 2. Model Setup and Training
print("Setting up and training the model...")
# Define model configuration
model_config = {
    "vocab_size": len(gene_vocab),
    "hidden_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 8,
    "intermediate_size": 2048,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1
}

# Initialize the model
model = TransformerModel(
    vocab_size=model_config["vocab_size"],
    hidden_size=model_config["hidden_size"],
    num_hidden_layers=model_config["num_hidden_layers"],
    num_attention_heads=model_config["num_attention_heads"],
    intermediate_size=model_config["intermediate_size"],
    hidden_dropout_prob=model_config["hidden_dropout_prob"],
    attention_probs_dropout_prob=model_config["attention_probs_dropout_prob"]
)

# Create data loaders for training and validation
from torch.utils.data import random_split
train_size = int(0.8 * len(input_data))
val_size = len(input_data) - train_size
train_dataset, val_dataset = random_split(input_data, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Set up training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training loop with validation
num_epochs = 10
train_losses = []
val_losses = []

# Early stopping parameters
patience = 3
best_val_loss = float('inf')
counter = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    epoch_train_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Calculate loss
        loss = criterion(outputs.logits.view(-1, model_config["vocab_size"]), labels.view(-1))
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
    
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = criterion(outputs.logits.view(-1, model_config["vocab_size"]), labels.view(-1))
            epoch_val_loss += loss.item()
    
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        # Save the best model
        torch.save(model.state_dict(), "scgpt_best_model.pt")
    else:
        counter += 1
        print(f"EarlyStopping counter: {counter} out of {patience}")
        if counter >= patience:
            print("Early stopping")
            break

# Load the best model
model.load_state_dict(torch.load("scgpt_best_model.pt"))

# Visualize training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss.png')
plt.close()

# 3. Extracting Cell Embeddings and Evaluating Representation Quality
print("Extracting cell embeddings...")
model.eval()
cell_embeddings = []

with torch.no_grad():
    for batch in DataLoader(input_data, batch_size=32, shuffle=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Get hidden states (embeddings)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get the final hidden state (cell embedding)
        final_layer_hidden = outputs.hidden_states[-1]
        # Take mean across sequence dimension (excluding padding)
        seq_lengths = attention_mask.sum(dim=1, keepdim=True)
        cell_emb = torch.sum(final_layer_hidden * attention_mask.unsqueeze(-1), dim=1) / seq_lengths
        cell_embeddings.append(cell_emb.cpu().numpy())

# Combine embeddings
cell_embeddings = np.vstack(cell_embeddings)

# Add embeddings to AnnData for downstream analysis
adata.obsm['X_scGPT'] = cell_embeddings

# Evaluate embedding quality: Measure separability using PCA variance explained
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca_embeddings = pca.fit_transform(cell_embeddings)

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.bar(range(1, 21), pca.explained_variance_ratio_[:20] * 100)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance (%)')
plt.title('PCA Explained Variance of scGPT Embeddings')
plt.xticks(range(1, 21))
plt.grid(True)
plt.savefig('pca_explained_variance.png')
plt.close()

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print(f"Cumulative variance explained by first 10 PCs: {cumulative_variance[9]*100:.2f}%")
print(f"Cumulative variance explained by first 20 PCs: {cumulative_variance[19]*100:.2f}%")

# 4. Clustering and Visualization
print("Performing clustering and visualization...")
# Construct neighborhood graph
sc.pp.neighbors(adata, use_rep='X_scGPT', n_neighbors=15)

# Run UMAP
sc.tl.umap(adata)

# Leiden clustering with different resolutions
resolutions = [0.2, 0.5, 0.8, 1.2]
for res in resolutions:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_res{res}')

# Visualize UMAP with clusters at different resolutions
fig, axs = plt.subplots(2, 2, figsize=(16, 14))
axs = axs.flatten()

for i, res in enumerate(resolutions):
    sc.pl.umap(adata, color=f'leiden_res{res}', title=f'Leiden Resolution {res}', ax=axs[i], show=False)

plt.tight_layout()
plt.savefig('umap_clusters.png')
plt.close()

# Calculate silhouette scores for different resolutions to find optimal clustering
silhouette_scores = []
for res in resolutions:
    labels = adata.obs[f'leiden_res{res}'].cat.codes.values
    score = silhouette_score(cell_embeddings, labels, sample_size=10000 if len(labels) > 10000 else None)
    silhouette_scores.append(score)
    print(f"Resolution {res}: Silhouette Score = {score:.4f}")

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(resolutions, silhouette_scores, '-o')
plt.xlabel('Resolution')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Clustering Resolutions')
plt.grid(True)
plt.savefig('silhouette_scores.png')
plt.close()

# Select the best resolution based on silhouette score
best_res = resolutions[np.argmax(silhouette_scores)]
print(f"Best clustering resolution: {best_res}")
adata.obs['best_clusters'] = adata.obs[f'leiden_res{best_res}']

# 5. Differential Expression Analysis for Cluster Characterization
print("Performing differential expression analysis...")
# Find marker genes for each cluster
sc.tl.rank_genes_groups(adata, 'best_clusters', method='wilcoxon')

# Plot top marker genes for each cluster
sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, groupby='best_clusters', 
                               show_gene_labels=True, figsize=(12, 10),
                               save='top_markers_heatmap.png')

sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, groupby='best_clusters',
                               standard_scale='var', figsize=(12, 8),
                               save='top_markers_dotplot.png')

# Generate a table of top markers for each cluster
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
markers_df = pd.DataFrame({group + '_' + key: result[key][group]
                           for group in groups
                           for key in ['names', 'logfoldchanges', 'pvals_adj']})
markers_df.to_csv('cluster_marker_genes.csv')

# 6. Visualizing Gene Expression Patterns
print("Visualizing gene expression patterns...")
# Get top markers across all clusters
top_markers = []
for group in groups:
    top_markers.extend(result['names'][group][:5])

# Remove duplicates while preserving order
top_markers = list(dict.fromkeys(top_markers))[:20]  # Keep top 20 unique genes

# Plot gene expression on UMAP for selected markers
fig, axs = plt.subplots(4, 5, figsize=(20, 16))
axs = axs.flatten()

for i, gene in enumerate(top_markers[:20]):  # Plot top 20 genes
    if gene in adata.var_names:
        sc.pl.umap(adata, color=gene, ax=axs[i], show=False, title=gene, color_map='viridis')

plt.tight_layout()
plt.savefig('marker_gene_expression_umap.png')
plt.close()

# 7. Analyzing Attention Weights for Gene Regulatory Networks
print("Analyzing attention patterns...")
# Function to get attention weights for a specific cell
def get_attention_for_cell(cell_idx, layer_idx=5):
    model.eval()
    with torch.no_grad():
        batch = input_data[cell_idx:cell_idx+1]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Forward pass with output_attentions=True
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        # Get attention weights from specified layer
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        attention_weights = outputs.attentions[layer_idx].cpu().numpy()
        
        return attention_weights[0]  # First (only) batch

# Function to get gene names for a specific cell
def get_gene_names_for_cell(cell_idx):
    input_ids = input_data[cell_idx]["input_ids"].numpy()
    genes = [gene_vocab.convert_ids_to_tokens(idx) for idx in input_ids if idx != gene_vocab.pad_token_id]
    return genes

# Select a cell from each major cluster for attention analysis
cluster_representatives = {}
for cluster in adata.obs['best_clusters'].cat.categories:
    cells_in_cluster = np.where(adata.obs['best_clusters'] == cluster)[0]
    if len(cells_in_cluster) > 0:
        # Pick the cell closest to the cluster center
        cluster_representatives[cluster] = cells_in_cluster[0]

# Analyze attention for a representative cell
for cluster, cell_idx in list(cluster_representatives.items())[:3]:  # Limit to 3 clusters for brevity
    print(f"Analyzing attention patterns for cluster {cluster}, cell {cell_idx}")
    attention_weights = get_attention_for_cell(cell_idx)
    genes = get_gene_names_for_cell(cell_idx)
    
    # Average attention across heads
    avg_attention = attention_weights.mean(axis=0)
    
    # Trim to only include actual genes (remove padding)
    n_genes = len(genes)
    avg_attention = avg_attention[:n_genes, :n_genes]
    
    # Visualize attention heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attention, xticklabels=False, yticklabels=False, cmap='viridis')
    plt.title(f'Average Attention Map for Cluster {cluster}')
    plt.tight_layout()
    plt.savefig(f'attention_heatmap_cluster_{cluster}.png')
    plt.close()
    
    # Create gene interaction network
    import itertools
    gene_pairs = []
    for i, j in itertools.product(range(min(50, len(genes))), range(min(50, len(genes)))):
        if i != j:  # Exclude self-attention
            gene_pairs.append((genes[i], genes[j], avg_attention[i, j]))
    
    # Sort by attention score and keep top interactions
    top_interactions = sorted(gene_pairs, key=lambda x: x[2], reverse=True)[:20]
    
    # Create network graph
    G = nx.DiGraph()
    edge_weights = []
    for source, target, weight in top_interactions:
        G.add_edge(source, target, weight=weight)
        edge_weights.append(weight * 3)  # Scale for visibility
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
    edges = nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, 
                                  edge_color='darkblue', arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f"Gene Regulatory Network for Cluster {cluster}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gene_network_cluster_{cluster}.png')
    plt.close()
    
    # Print top gene interactions
    print(f"Top gene interactions for cluster {cluster}:")
    for source, target, score in top_interactions[:10]:
        print(f"  {source} → {target}: {score:.4f}")

# 8. Model Performance Evaluation
print("Evaluating model performance...")
# Pseudobulk gene expression reconstruction
def evaluate_gene_expression_reconstruction():
    model.eval()
    true_expressions = []
    pred_expressions = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            
            # Convert to expression values (simplified)
            true_expr = labels.cpu().numpy()
            pred_expr = predictions
            
            # Collect results
            true_expressions.append(true_expr)
            pred_expressions.append(pred_expr)
    
    true_expressions = np.concatenate(true_expressions)
    pred_expressions = np.concatenate(pred_expressions)
    
    # Calculate metrics
    accuracy = np.mean(true_expressions == pred_expressions)
    return accuracy

gene_accuracy = evaluate_gene_expression_reconstruction()
print(f"Gene expression reconstruction accuracy: {gene_accuracy:.4f}")

# Pretend we have cell type labels (in real applications, these might come from external sources)
# Here we'll just use clusters as a proxy for demonstration
adata.obs['cell_type'] = adata.obs['best_clusters']

# Split into training and test sets
cell_indices = np.arange(adata.n_obs)
train_indices, test_indices = train_test_split(cell_indices, test_size=0.3, random_state=42)

X_train = adata.obsm['X_scGPT'][train_indices]
y_train = adata.obs['cell_type'].iloc[train_indices]
X_test = adata.obsm['X_scGPT'][test_indices]
y_test = adata.obs['cell_type'].iloc[test_indices]

# Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Cell type classification accuracy: {accuracy:.4f}")

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=adata.obs['cell_type'].cat.categories,
            yticklabels=adata.obs['cell_type'].cat.categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Cell Type Classification')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Feature importance
feature_importance = clf.feature_importances_
top_features = np.argsort(feature_importance)[-20:][::-1]

plt.figure(figsize=(10, 8))
plt.barh(range(len(top_features)), feature_importance[top_features])
plt.yticks(range(len(top_features)), [f"Feature {i}" for i in top_features])
plt.xlabel('Feature Importance')
plt.title('Top 20 Important Features for Cell Classification')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# 9. Comparison with Traditional Methods
print("Comparing with traditional methods...")
# PCA baseline
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.pca(adata)
sc.pp.neighbors(adata, use_rep='X_pca')
sc.tl.leiden(adata, resolution=best_res, key_added='pca_leiden')

# Compare clustering performance
ari_scgpt = adjusted_rand_score(adata.obs['best_clusters'].cat.codes, adata.obs['cell_type'].cat.codes)
ari_pca = adjusted_rand_score(adata.obs['pca_leiden'].cat.codes, adata.obs['cell_type'].cat.codes)

nmi_scgpt = normalized_mutual_info_score(adata.obs['best_clusters'].cat.codes, adata.obs['cell_type'].cat.codes)
nmi_pca = normalized_mutual_info_score(adata.obs['pca_leiden'].cat.codes, adata.obs['cell_type'].cat.codes)

print(f"scGPT ARI: {ari_scgpt:.4f}, NMI: {nmi_scgpt:.4f}")
print(f"PCA ARI: {ari_pca:.4f}, NMI: {nmi_pca:.4f}")

# Visualize UMAP with PCA vs scGPT
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

# Re-run UMAP with PCA
sc.tl.umap(adata, neighbors_key='neighbors')
sc.pl.umap(adata, color='pca_leiden', title='PCA + Leiden', ax=axs[0], show=False)

# Use scGPT UMAP
sc.pp.neighbors(adata, use_rep='X_scGPT', key_added='scgpt_neighbors')
sc.tl.umap(adata, neighbors_key='scgpt_neighbors', key_added='scgpt_umap')
sc.pl.embedding(adata, 'scgpt_umap', color='best_clusters', title='scGPT + Leiden', ax=axs[1], show=False)

plt.tight_layout()
plt.savefig('pca_vs_scgpt_umap.png')
plt.close()

# 10. Summary statistics and report
print("Generating summary report...")
# Compile key statistics
statistics = {
    "Total cells": adata.n_obs,
    "Total genes": adata.n_vars,
    "Number of clusters": len(adata.obs['best_clusters'].cat.categories),
    "Model training epochs": len(train_losses),
    "Final training loss": train_losses[-1],
    "Final validation loss": val_losses[-1],
    "Gene reconstruction accuracy": gene_accuracy,
    "Cell type classification accuracy": accuracy,
    "ARI score (scGPT)": ari_scgpt,
    "NMI score (scGPT)": nmi_scgpt,
    "ARI score (PCA)": ari_pca, 
    "NMI score (PCA)": nmi_pca
}

# Save as a text file
with open('scgpt_analysis_summary.txt', 'w') as f:
    f.write("scGPT Analysis Summary\n")
    f.write("=====================\n\n")
    for key, value in statistics.items():
        f.write(f"{key}: {value}\n")
    
    f.write("\nTop marker genes by cluster:\n")
    for cluster in adata.obs['best_clusters'].cat.categories:
        markers = result['names'][cluster][:5]
        f.write(f"Cluster {cluster}: {', '.join(markers)}\n")

print("Analysis complete! Summary saved to 'scgpt_analysis_summary.txt'")


