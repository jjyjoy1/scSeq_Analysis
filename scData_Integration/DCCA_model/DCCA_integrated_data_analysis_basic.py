import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import normalize

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns


# Simulate 100 cells, 500 RNA features, 300 ATAC peaks
n_cells = 100
rna_dim = 500
atac_dim = 300

# Generate latent factors (shared biology)
latent_dim = 20
z = np.random.normal(size=(n_cells, latent_dim))

# Simulate RNA data (latent -> RNA)
W_rna = np.random.randn(latent_dim, rna_dim)
rna_data = z.dot(W_rna) + np.random.normal(scale=0.1, size=(n_cells, rna_dim))
rna_data = normalize(rna_data, axis=1)  # Normalize

# Simulate ATAC data (latent -> ATAC)
W_atac = np.random.randn(latent_dim, atac_dim)
atac_data = z.dot(W_atac) + np.random.normal(scale=0.1, size=(n_cells, atac_dim))
atac_data = normalize(atac_data, axis=1)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # Output mean and log-variance
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# Initialize VAEs for RNA and ATAC
vae_rna = VAE(rna_dim, latent_dim)
vae_atac = VAE(atac_dim, latent_dim)

class CrossModalAttention(nn.Module):
    def __init__(self, latent_dim, n_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(latent_dim, n_heads)
        
    def forward(self, query, key_value):
        # Reshape for attention (seq_len, batch, features)
        query = query.unsqueeze(0)
        key_value = key_value.unsqueeze(0)
        attn_output, _ = self.attention(query, key_value, key_value)
        return attn_output.squeeze(0)


def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (MSE)
    recon_loss = nn.MSELoss()(recon_x, x)
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def cycle_loss(original, cycled):
    return nn.MSELoss()(original, cycled)


# Convert data to PyTorch tensors
rna_tensor = torch.FloatTensor(rna_data)
atac_tensor = torch.FloatTensor(atac_data)

# Optimizers
optimizer = optim.Adam(
    list(vae_rna.parameters()) + list(vae_atac.parameters()), lr=1e-3
)

# Attention modules
attn_rna_to_atac = CrossModalAttention(latent_dim)
attn_atac_to_rna = CrossModalAttention(latent_dim)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    # Forward pass: RNA -> Latent
    recon_rna, mu_rna, logvar_rna = vae_rna(rna_tensor)
    # Forward pass: ATAC -> Latent
    recon_atac, mu_atac, logvar_atac = vae_atac(atac_tensor)
    
    # Cross-modal translation with attention
    # RNA -> ATAC
    atac_query = mu_rna
    atac_translated = attn_rna_to_atac(atac_query, mu_atac)
    # ATAC -> RNA
    rna_query = mu_atac
    rna_translated = attn_atac_to_rna(rna_query, mu_rna)
    
    # Cycle consistency: A -> B -> A
    cycled_rna, _, _ = vae_rna(rna_translated)
    cycled_atac, _, _ = vae_atac(atac_translated)
    
    # Compute losses
    loss_rna = vae_loss(recon_rna, rna_tensor, mu_rna, logvar_rna)
    loss_atac = vae_loss(recon_atac, atac_tensor, mu_atac, logvar_atac)
    loss_cycle = cycle_loss(rna_tensor, cycled_rna) + cycle_loss(atac_tensor, cycled_atac)
    
    # Alignment loss (force shared latent space)
    alignment_loss = nn.CosineEmbeddingLoss()(mu_rna, mu_atac, torch.ones(n_cells))
    
    # Total loss
    total_loss = loss_rna + loss_atac + loss_cycle + alignment_loss
    
    # Backprop
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss.item():.4f}")


def evaluate_alignment(mu_rna, mu_atac, z_true):
    # Convert tensors to numpy
    mu_rna_np = mu_rna.detach().numpy()
    mu_atac_np = mu_atac.detach().numpy()
    
    # 1. Cosine similarity between paired latent vectors
    cosine_sim = np.diag(cosine_similarity(mu_rna_np, mu_atac_np))
    print(f"Mean cosine similarity: {np.mean(cosine_sim):.3f}")
    
    # 2. Cluster alignment (Adjusted Rand Index)
    # Generate synthetic "true" labels from shared latent factors (z_true)
    kmeans_true = KMeans(n_clusters=3).fit(z_true)
    true_labels = kmeans_true.labels_
    
    # Cluster the aligned latent space
    combined_latent = np.concatenate([mu_rna_np, mu_atac_np], axis=0)
    kmeans_pred = KMeans(n_clusters=3).fit(combined_latent)
    pred_labels = kmeans_pred.labels_
    
    # Split predictions back into RNA and ATAC
    rna_labels = pred_labels[:n_cells]
    atac_labels = pred_labels[n_cells:]
    
    # Compute ARI between RNA/ATAC clusters and true clusters
    ari_rna = adjusted_rand_score(true_labels, rna_labels)
    ari_atac = adjusted_rand_score(true_labels, atac_labels)
    print(f"RNA ARI: {ari_rna:.3f}, ATAC ARI: {ari_atac:.3f}")


def plot_umap(mu_rna, mu_atac, title):
    # Combine latent vectors
    combined = np.concatenate([mu_rna, mu_atac], axis=0)
    labels = ["RNA"] * n_cells + ["ATAC"] * n_cells
    
    # UMAP projection
    reducer = umap.UMAP()
    embed = reducer.fit_transform(combined)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embed[:,0], y=embed[:,1], hue=labels, palette="tab10", s=20)
    plt.title(title)
    plt.show()


def plot_reconstruction(original, reconstructed, modality="RNA"):
    plt.figure(figsize=(6, 6))
    plt.scatter(original.flatten(), reconstructed.detach().numpy().flatten(), alpha=0.1)
    plt.xlabel(f"Original {modality}")
    plt.ylabel(f"Reconstructed {modacity}")
    plt.title(f"Reconstruction vs. Ground Truth ({modality})")
    plt.plot([-3,3], [-3,3], 'r--')  # Perfect reconstruction line
    plt.show()

# Track losses for plotting
train_losses.append(total_loss.item())

# Periodically evaluate
if epoch % 20 == 0:
    with torch.no_grad():
        # Get latent representations
        _, mu_rna_eval, _ = vae_rna(rna_tensor)
        _, mu_atac_eval, _ = vae_atac(atac_tensor)
        
        # Evaluate alignment
        evaluate_alignment(mu_rna_eval, mu_atac_eval, z)
        
        # Visualize latent space
        plot_umap(mu_rna_eval.numpy(), mu_atac_eval.numpy(), 
                 f"Latent Space (Epoch {epoch})")

# Final evaluation
with torch.no_grad():
    # Get final latent representations
    _, mu_rna_final, _ = vae_rna(rna_tensor)
    _, mu_atac_final, _ = vae_atac(atac_tensor)
    
    # Plot reconstructions
    plot_reconstruction(rna_data, recon_rna, "RNA")
    plot_reconstruction(atac_data, recon_atac, "ATAC")
    
    # Compare latent spaces before/after training
    plot_umap(z, z, "True Shared Latent Space (Simulated)")
    plot_umap(mu_rna_final.numpy(), mu_atac_final.numpy(), "Aligned Latent Space")
    
    # Plot training curve
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Training Progress")
    plt.show()


