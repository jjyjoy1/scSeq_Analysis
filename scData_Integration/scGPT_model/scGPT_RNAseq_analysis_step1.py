# Install required packages
!pip install scgpt anndata scanpy torch torchvision

# Import libraries
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import torch
import scgpt
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor

# Load a single-cell dataset (e.g., 10X PBMC data)
adata = sc.read_10x_mtx("./pbmc_data/filtered_gene_bc_matrices/hg19/")

# Basic preprocessing with Scanpy
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Prepare data for scGPT
preprocessor = Preprocessor()
preprocessor.fit(adata)
processed_adata = preprocessor.transform(adata)

# Create gene vocabulary for tokenization
gene_vocab = GeneVocab.from_adata(processed_adata)

# Convert data to scGPT format
input_data = scgpt.data.AnnDataset(
    processed_adata,
    gene_vocab=gene_vocab,
    max_seq_len=2000,  # Maximum number of genes to consider
    pad_token=gene_vocab.pad_token,
    mask_token=gene_vocab.mask_token
)

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

# Create data loader
from torch.utils.data import DataLoader
train_loader = DataLoader(input_data, batch_size=32, shuffle=True)

# Set up training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # Get input_ids and labels
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
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "scgpt_pbmc_model.pt")

# Extract cell embeddings
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
        # We take the [CLS] token embedding or average across gene tokens
        final_layer_hidden = outputs.hidden_states[-1]
        # Take mean across sequence dimension (excluding padding)
        seq_lengths = attention_mask.sum(dim=1, keepdim=True)
        cell_emb = torch.sum(final_layer_hidden * attention_mask.unsqueeze(-1), dim=1) / seq_lengths
        cell_embeddings.append(cell_emb.cpu().numpy())

# Combine embeddings
cell_embeddings = np.vstack(cell_embeddings)

# Add embeddings to AnnData for downstream analysis
adata.obsm['X_scGPT'] = cell_embeddings

# Dimensionality reduction and visualization
sc.pp.neighbors(adata, use_rep='X_scGPT')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.umap(adata, color='leiden', title='Cell Clusters from scGPT Embeddings')


# Get attention weights for a specific cell
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

# Extract gene names for this cell
def get_gene_names_for_cell(cell_idx):
    input_ids = input_data[cell_idx]["input_ids"].numpy()
    genes = [gene_vocab.convert_ids_to_tokens(idx) for idx in input_ids if idx != gene_vocab.pad_token_id]
    return genes

# Analyze attention for a specific cell
cell_idx = 42  # Example cell
attention_weights = get_attention_for_cell(cell_idx)
genes = get_gene_names_for_cell(cell_idx)

# Average attention across heads
avg_attention = attention_weights.mean(axis=0)

# Create a dataframe for the attention matrix
attention_df = pd.DataFrame(avg_attention, index=genes, columns=genes)

# Find top gene interactions based on attention scores
import itertools
gene_pairs = []
for i, j in itertools.product(range(len(genes)), range(len(genes))):
    if i != j:  # Exclude self-attention
        gene_pairs.append((genes[i], genes[j], avg_attention[i, j]))

# Sort by attention score
top_interactions = sorted(gene_pairs, key=lambda x: x[2], reverse=True)[:20]
print("Top gene interactions based on attention:")
for source, target, score in top_interactions:
    print(f"{source} → {target}: {score:.4f}")

# Visualize as a network
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
for source, target, weight in top_interactions:
    G.add_edge(source, target, weight=weight)

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
edges = nx.draw_networkx_edges(G, pos, width=[G[u][v]['weight'] * 5 for u, v in G.edges()])
nodes = nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
labels = nx.draw_networkx_labels(G, pos)
plt.title("Gene Regulatory Network Inferred from scGPT Attention")
plt.axis('off')
plt.show()

# Train a classifier on top of scGPT embeddings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Assuming we have some known cell type labels for a subset of cells
labeled_cells = pd.read_csv("cell_annotations.csv")
adata.obs['cell_type'] = 'Unknown'
adata.obs.loc[labeled_cells.index, 'cell_type'] = labeled_cells['cell_type']

# Get indices of labeled cells
labeled_mask = adata.obs['cell_type'] != 'Unknown'
X = adata.obsm['X_scGPT'][labeled_mask]
y = adata.obs['cell_type'][labeled_mask]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Predict on all cells
all_predictions = clf.predict(adata.obsm['X_scGPT'])
adata.obs['predicted_cell_type'] = all_predictions

# Visualize cell type predictions
sc.pl.umap(adata, color='predicted_cell_type', title='Predicted Cell Types using scGPT')

#A powerful capability of scGPT is integrating multi-modal data
# Assuming we also have ATAC-seq data for the same cells
atac_adata = sc.read_h5ad("atac_data.h5ad")

# Process ATAC data
sc.pp.normalize_total(atac_adata)
sc.pp.log1p(atac_adata)

# Create joint embedding with both modalities
from scgpt.multimodal import MultiModalEncoder

# Initialize multi-modal encoder
multimodal_encoder = MultiModalEncoder(
    rna_model=model,
    modalities=["RNA", "ATAC"],
    hidden_size=512
)

# Get embeddings for both modalities
rna_embeddings = adata.obsm['X_scGPT']
atac_embeddings = multimodal_encoder.encode_atac(atac_adata)

# Combine embeddings
joint_embeddings = multimodal_encoder.integrate_embeddings(
    [rna_embeddings, atac_embeddings]
)

# Add to AnnData for visualization
adata.obsm['X_multimodal'] = joint_embeddings

# Analyze joint embeddings
sc.pp.neighbors(adata, use_rep='X_multimodal')
sc.tl.umap(adata)
sc.tl.leiden(adata, key_added='multimodal_clusters')
sc.pl.umap(adata, color=['leiden', 'multimodal_clusters'], 
           title=['RNA-only clusters', 'Multi-modal clusters'])




