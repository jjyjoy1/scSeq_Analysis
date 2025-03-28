import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr
from typing import List, Tuple, Dict, Optional

class EncoderModule(nn.Module):
    """Encoder module for a specific omics modality."""
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        
        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(nn.LeakyReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Mean and log variance layers for the VAE
        self.mean_encoder = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_encoder = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of (hidden representation, mean, logvar)
        """
        hidden = self.encoder(x)
        mean = self.mean_encoder(hidden)
        logvar = self.logvar_encoder(hidden)
        return hidden, mean, logvar


class DecoderModule(nn.Module):
    """Decoder module for a specific omics modality."""
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        
        # Build decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        
        for dim in hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(nn.LeakyReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
            
        decoder_layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            z: Latent space representation of shape [batch_size, latent_dim]
            
        Returns:
            Reconstructed input of shape [batch_size, output_dim]
        """
        return self.decoder(z)


class CycleAttentionModule(nn.Module):
    """Cross-modal cycle attention mechanism."""
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        
        # Multi-head attention layers
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        
        # Layer norm for stability
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # MLP for processing after attention
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying attention mechanism.
        
        Args:
            query: Query tensor from the first modality [seq_len, batch, hidden_dim]
            key: Key tensor from the second modality [seq_len, batch, hidden_dim]
            value: Value tensor from the second modality [seq_len, batch, hidden_dim]
            
        Returns:
            Attended features
        """
        # Apply multi-head attention
        attended_features, _ = self.attention(query, key, value)
        
        # Apply residual connection and normalization
        normalized = self.norm1(query + attended_features)
        
        # Apply MLP
        output = self.mlp(normalized)
        
        # Apply second residual connection and normalization
        return self.norm2(normalized + output)


class DCCA(nn.Module):
    """Deep Cross-omics Cycle Attention (DCCA) model."""
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dims: Dict[str, List[int]],
        latent_dim: int,
        cycle_iterations: int = 3,
        num_heads: int = 4,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.modality_names = list(modality_dims.keys())
        self.latent_dim = latent_dim
        self.cycle_iterations = cycle_iterations
        
        # Create encoder for each modality
        self.encoders = nn.ModuleDict({
            mod_name: EncoderModule(
                input_dim=modality_dims[mod_name],
                hidden_dims=hidden_dims[mod_name],
                latent_dim=latent_dim,
                dropout_rate=dropout_rate
            ) for mod_name in self.modality_names
        })
        
        # Create decoder for each modality
        self.decoders = nn.ModuleDict({
            mod_name: DecoderModule(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims[mod_name][::-1],  # Reverse the hidden dims
                output_dim=modality_dims[mod_name],
                dropout_rate=dropout_rate
            ) for mod_name in self.modality_names
        })
        
        # Create cycle attention modules between each pair of modalities
        self.attention_modules = nn.ModuleDict()
        for i, mod_i in enumerate(self.modality_names):
            for j, mod_j in enumerate(self.modality_names):
                if i != j:
                    key = f"{mod_i}_to_{mod_j}"
                    self.attention_modules[key] = CycleAttentionModule(
                        hidden_dim=hidden_dims[mod_i][-1],
                        num_heads=num_heads
                    )
        
        # Projectors to align hidden representations across modalities
        self.projectors = nn.ModuleDict({
            mod_name: nn.Linear(hidden_dims[mod_name][-1], hidden_dims[mod_name][-1])
            for mod_name in self.modality_names
        })
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mean: Mean vector
            logvar: Log variance vector
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode_modality(self, x: torch.Tensor, modality: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a single modality.
        
        Args:
            x: Input tensor for the modality
            modality: Modality name
            
        Returns:
            Tuple of (hidden representation, mean, logvar)
        """
        return self.encoders[modality](x)
    
    def cycle_attention(self, hidden_states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply cyclical attention between modalities.
        
        Args:
            hidden_states: Dictionary of hidden states for each modality
            
        Returns:
            Updated hidden states after attention
        """
        updated_states = {mod: state.clone() for mod, state in hidden_states.items()}
        
        # Apply cycle attention for the specified number of iterations
        for _ in range(self.cycle_iterations):
            temp_states = {mod: state.clone() for mod, state in updated_states.items()}
            
            for i, mod_i in enumerate(self.modality_names):
                attended_features = []
                
                # Apply attention from each other modality
                for j, mod_j in enumerate(self.modality_names):
                    if i != j:
                        key = f"{mod_i}_to_{mod_j}"
                        
                        # Project hidden states for compatibility
                        query = temp_states[mod_i].unsqueeze(0)  # [1, batch, hidden_dim]
                        key_value = temp_states[mod_j].unsqueeze(0)  # [1, batch, hidden_dim]
                        
                        # Apply attention
                        attended = self.attention_modules[key](query, key_value, key_value)
                        attended_features.append(attended.squeeze(0))  # [batch, hidden_dim]
                
                # Average the attended features
                if attended_features:
                    avg_attended = torch.stack(attended_features).mean(dim=0)
                    
                    # Apply residual connection
                    updated_states[mod_i] = self.projectors[mod_i](temp_states[mod_i] + avg_attended)
        
        return updated_states
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the DCCA model.
        
        Args:
            inputs: Dictionary mapping modality names to input tensors
            
        Returns:
            Dictionary containing model outputs including reconstructions,
            latent representations, KL divergence losses, etc.
        """
        # Encode each modality
        hidden_states = {}
        means = {}
        logvars = {}
        
        for mod_name in self.modality_names:
            if mod_name in inputs:
                hidden, mean, logvar = self.encode_modality(inputs[mod_name], mod_name)
                hidden_states[mod_name] = hidden
                means[mod_name] = mean
                logvars[mod_name] = logvar
        
        # Apply cyclical attention between modalities
        attended_states = self.cycle_attention(hidden_states)
        
        # Get updated means and logvars after attention
        updated_means = {}
        updated_logvars = {}
        
        for mod_name in self.modality_names:
            if mod_name in inputs:
                # Re-encode to get updated mean and logvar
                updated_means[mod_name] = self.encoders[mod_name].mean_encoder(attended_states[mod_name])
                updated_logvars[mod_name] = self.encoders[mod_name].logvar_encoder(attended_states[mod_name])
        
        # Sample from latent distributions
        latents = {}
        for mod_name in self.modality_names:
            if mod_name in inputs:
                latents[mod_name] = self.reparameterize(updated_means[mod_name], updated_logvars[mod_name])
        
        # Calculate integrated latent representation (average of modality-specific latents)
        available_modalities = [mod for mod in self.modality_names if mod in inputs]
        if available_modalities:
            integrated_latent = torch.stack([latents[mod] for mod in available_modalities]).mean(dim=0)
        else:
            raise ValueError("No valid modalities provided in input")
        
        # Decode each modality from both its specific latent and the integrated latent
        reconstructions = {}
        for mod_name in self.modality_names:
            if mod_name in inputs:
                reconstructions[f"{mod_name}_specific"] = self.decoders[mod_name](latents[mod_name])
                reconstructions[f"{mod_name}_integrated"] = self.decoders[mod_name](integrated_latent)
        
        # Calculate KL divergences
        kl_losses = {}
        for mod_name in self.modality_names:
            if mod_name in inputs:
                # KL between modality-specific latent distribution and prior
                kl_losses[f"{mod_name}_kl"] = torch.mean(
                    -0.5 * torch.sum(1 + updated_logvars[mod_name] - updated_means[mod_name].pow(2) - updated_logvars[mod_name].exp(), dim=1)
                )
        
        # Calculate cross-modal alignment losses
        alignment_losses = {}
        if len(available_modalities) > 1:
            for i, mod_i in enumerate(available_modalities):
                for j, mod_j in enumerate(available_modalities):
                    if i < j:  # Calculate only once for each pair
                        alignment_losses[f"{mod_i}_{mod_j}_align"] = F.mse_loss(latents[mod_i], latents[mod_j])
        
        return {
            "reconstructions": reconstructions,
            "latents": latents,
            "integrated_latent": integrated_latent,
            "kl_losses": kl_losses,
            "alignment_losses": alignment_losses,
            "means": updated_means,
            "logvars": updated_logvars
        }
    
    def calculate_total_loss(
        self, 
        inputs: Dict[str, torch.Tensor], 
        outputs: Dict[str, torch.Tensor],
        beta: float = 1.0,
        gamma: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate total loss for the model.
        
        Args:
            inputs: Dictionary of input tensors by modality
            outputs: Model outputs from forward pass
            beta: Weight for KL divergence losses
            gamma: Weight for alignment losses
            
        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        
        # Reconstruction losses
        recon_losses = {}
        for mod_name in self.modality_names:
            if mod_name in inputs:
                # MSE for continuous data (can be customized for count data, e.g., NB loss)
                recon_losses[f"{mod_name}_specific"] = F.mse_loss(
                    outputs["reconstructions"][f"{mod_name}_specific"], 
                    inputs[mod_name]
                )
                recon_losses[f"{mod_name}_integrated"] = F.mse_loss(
                    outputs["reconstructions"][f"{mod_name}_integrated"], 
                    inputs[mod_name]
                )
        
        # Sum up all reconstruction losses
        losses["reconstruction"] = sum(recon_losses.values())
        
        # Sum up all KL divergence losses
        losses["kl"] = beta * sum(outputs["kl_losses"].values())
        
        # Sum up all alignment losses
        if outputs["alignment_losses"]:
            losses["alignment"] = gamma * sum(outputs["alignment_losses"].values())
        else:
            losses["alignment"] = torch.tensor(0.0, device=inputs[list(inputs.keys())[0]].device)
        
        # Total loss
        losses["total"] = losses["reconstruction"] + losses["kl"] + losses["alignment"]
        
        return losses

    def get_latent_representation(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get integrated latent representation for new data.
        
        Args:
            inputs: Dictionary of input tensors by modality
            
        Returns:
            Integrated latent representation
        """
        with torch.no_grad():
            outputs = self.forward(inputs)
            return outputs["integrated_latent"]


# Example training function
def train_dcca(
    model: DCCA,
    train_data: Dict[str, torch.Tensor],
    val_data: Optional[Dict[str, torch.Tensor]] = None,
    num_epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    beta: float = 1.0,
    gamma: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the DCCA model.
    
    Args:
        model: DCCA model
        train_data: Dictionary of training data tensors by modality
        val_data: Optional dictionary of validation data tensors
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        beta: Weight for KL divergence losses
        gamma: Weight for alignment losses
        device: Device for training
    
    Returns:
        Training history
    """
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create data loaders
    n_samples = len(next(iter(train_data.values())))
    indices = torch.randperm(n_samples)
    
    history = {
        "train_loss": [],
        "val_loss": [] if val_data else None
    }
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Train by batch
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # Prepare batch
            batch_data = {mod: data[batch_indices].to(device) for mod, data in train_data.items()}
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            # Calculate loss
            losses = model.calculate_total_loss(batch_data, outputs, beta=beta, gamma=gamma)
            total_loss = losses["total"]
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item() * len(batch_indices)
        
        # Record average loss
        avg_train_loss = epoch_loss / n_samples
        history["train_loss"].append(avg_train_loss)
        
        # Validation
        if val_data:
            model.eval()
            with torch.no_grad():
                val_data_device = {mod: data.to(device) for mod, data in val_data.items()}
                val_outputs = model(val_data_device)
                val_losses = model.calculate_total_loss(val_data_device, val_outputs, beta=beta, gamma=gamma)
                history["val_loss"].append(val_losses["total"].item())
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}", end="")
            if val_data:
                print(f", Val Loss: {history['val_loss'][-1]:.4f}")
            else:
                print()
    
    return history


# Example function to analyze results with a single-cell multi-omics dataset
def analyze_scmultiomics_integration(model, data_dict, cell_types, n_neighbors=15):
    """
    Analyze integration of single-cell multi-omics data.
    
    Args:
        model: Trained DCCA model
        data_dict: Dictionary of data tensors by modality
        cell_types: Array of cell type labels
        n_neighbors: Number of neighbors for UMAP
        
    Returns:
        AnnData object with integrated representation
    """
    # Get integrated latent representation
    model.eval()
    with torch.no_grad():
        inputs = {mod: tensor.to(next(model.parameters()).device) for mod, tensor in data_dict.items()}
        outputs = model(inputs)
        latent_rep = outputs["integrated_latent"].cpu().numpy()
    
    # Create AnnData object with integrated representation
    adata = ad.AnnData(X=latent_rep)
    
    # Add cell type information
    adata.obs['cell_type'] = cell_types
    
    # Calculate neighborhood graph and UMAP embedding
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.umap(adata)
    
    # Compute silhouette score to quantify clustering quality
    sil_score = silhouette_score(latent_rep, cell_types)
    print(f"Silhouette score: {sil_score:.4f}")
    
    return adata


# Example usage with a simulated dataset
def simulate_multiomics_data(
    n_cells: int = 1000, 
    n_genes: int = 2000, 
    n_peaks: int = 5000, 
    n_proteins: int = 500, 
    n_cell_types: int = 5
):
    """Simulate a simplified multi-omics dataset."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate cell type labels
    cell_types = np.random.choice(n_cell_types, size=n_cells)
    cell_type_names = [f"CellType_{i}" for i in range(n_cell_types)]
    cell_type_labels = np.array([cell_type_names[i] for i in cell_types])
    
    # Generate latent factors that influence all modalities
    n_factors = 20
    latent_factors = np.random.normal(0, 1, size=(n_cells, n_factors))
    
    # Add cell type-specific patterns to latent factors
    for i in range(n_cell_types):
        mask = cell_types == i
        latent_factors[mask] += np.random.normal(i, 0.5, size=(mask.sum(), n_factors))
    
    # Generate RNA-seq data (log-normalized counts)
    gene_loadings = np.random.normal(0, 1, size=(n_factors, n_genes))
    rna_data = np.dot(latent_factors, gene_loadings)
    rna_data = np.exp(rna_data + np.random.normal(0, 0.5, size=rna_data.shape))
    
    # Generate ATAC-seq data (binary accessibility)
    peak_loadings = np.random.normal(0, 1, size=(n_factors, n_peaks))
    atac_data = np.dot(latent_factors, peak_loadings)
    atac_data = 1 / (1 + np.exp(-atac_data))  # Sigmoid
    atac_data = (atac_data > 0.5).astype(float)
    
    # Generate protein data
    protein_loadings = np.random.normal(0, 1, size=(n_factors, n_proteins))
    protein_data = np.dot(latent_factors, protein_loadings)
    protein_data = np.exp(protein_data + np.random.normal(0, 0.3, size=protein_data.shape))
    
    # Convert to PyTorch tensors
    data_dict = {
        "rna": torch.tensor(rna_data, dtype=torch.float32),
        "atac": torch.tensor(atac_data, dtype=torch.float32),
        "protein": torch.tensor(protein_data, dtype=torch.float32)
    }
    
    return data_dict, cell_type_labels


# Main function to demonstrate the DCCA model
def run_dcca_demo():
    # Simulate multi-omics data
    print("Simulating multi-omics data...")
    data_dict, cell_types = simulate_multiomics_data()
    
    # Split into train and validation sets
    n_samples = len(cell_types)
    n_train = int(0.8 * n_samples)
    
    train_indices = torch.randperm(n_samples)[:n_train]
    val_indices = torch.randperm(n_samples)[n_train:]
    
    train_data = {mod: data[train_indices] for mod, data in data_dict.items()}
    val_data = {mod: data[val_indices] for mod, data in data_dict.items()}
    
    # Define model parameters
    modality_dims = {
        "rna": train_data["rna"].shape[1],
        "atac": train_data["atac"].shape[1],
        "protein": train_data["protein"].shape[1]
    }
    
    hidden_dims = {
        "rna": [512, 256, 128],
        "atac": [512, 256, 128],
        "protein": [256, 128, 128]
    }
    
    latent_dim = 32
    
    # Create and train the model
    print("Creating DCCA model...")
    model = DCCA(
        modality_dims=modality_dims,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        cycle_iterations=2
    )
    
    print("Training DCCA model...")
    history = train_dcca(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=50,
        batch_size=64,
        beta=0.5,  # Weight for KL divergence
        gamma=1.0  # Weight for alignment loss
    )
    
    # Analyze integration results
    print("Analyzing integration results...")
    adata = analyze_scmultiomics_integration(
        model=model,
        data_dict=data_dict,
        cell_types=cell_types
    )
    
    print("DCCA demonstration completed!")
    return model, adata, history


if __name__ == "__main__":
    run_dcca_demo()




