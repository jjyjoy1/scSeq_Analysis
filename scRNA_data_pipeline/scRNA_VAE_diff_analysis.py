#Focus on batch integration, normalization, dimensionality reduction, clustering, and Bayesian differential expression analysis using variational autoencoders (VAEs).

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from scipy import sparse
from typing import Dict, List, Optional, Tuple, Union

class VAEEncoder(nn.Module):
    """
    Encoder network for scRNA-seq VAE.
    Maps count data to latent space distribution parameters.
    """
    def __init__(
        self, 
        input_dim: int,
        latent_dim: int = 20,
        hidden_dims: List[int] = [128, 128],
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False
    ):
        super().__init__()
        
        # Build encoder layers
        modules = []
        
        # Input layer
        modules.append(nn.Linear(input_dim, hidden_dims[0]))
        if use_batch_norm:
            modules.append(nn.BatchNorm1d(hidden_dims[0]))
        if use_layer_norm:
            modules.append(nn.LayerNorm(hidden_dims[0]))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if use_batch_norm:
                modules.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            if use_layer_norm:
                modules.append(nn.LayerNorm(hidden_dims[i + 1]))
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(dropout_rate))
        
        self.encoder = nn.Sequential(*modules)
        
        # Mean and variance layers
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Batch correction layers
        self.batch_encoder = None
        
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode input
        h = self.encoder(x)
        
        # Get latent distribution parameters
        mu = self.mu(h)
        log_var = self.log_var(h)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) while allowing backprop.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEDecoder(nn.Module):
    """
    Decoder network for scRNA-seq VAE.
    Reconstructs gene expression from latent space.
    """
    def __init__(
        self, 
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128],
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        dispersion: str = "gene"
    ):
        super().__init__()
        
        # Build decoder layers
        modules = []
        
        # Input layer from latent space
        modules.append(nn.Linear(latent_dim, hidden_dims[0]))
        if use_batch_norm:
            modules.append(nn.BatchNorm1d(hidden_dims[0]))
        if use_layer_norm:
            modules.append(nn.LayerNorm(hidden_dims[0]))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if use_batch_norm:
                modules.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            if use_layer_norm:
                modules.append(nn.LayerNorm(hidden_dims[i + 1]))
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(dropout_rate))
        
        self.decoder = nn.Sequential(*modules)
        
        # Output layers for negative binomial parameters
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Softmax(dim=-1)  # Ensure values sum to 1 across genes
        )
        
        # Gene-specific dispersion parameters (phi)
        if dispersion == "gene":
            self.px_r = nn.Parameter(torch.randn(output_dim))
        else:  # "gene-cell" or "gene-batch"
            self.px_r = nn.Linear(hidden_dims[-1], output_dim)
            
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Maps latent space to distribution parameters for each gene.
        """
        # Decode latent representation
        h = self.decoder(z)
        
        # Get negative binomial parameters
        px_scale = self.px_scale_decoder(h)
        
        # Gene-specific dispersion
        if isinstance(self.px_r, nn.Parameter):
            px_r = F.softplus(self.px_r).expand(z.size(0), -1)
        else:
            px_r = F.softplus(self.px_r(h))
            
        return {
            "px_scale": px_scale,  # Mean parameter
            "px_r": px_r           # Dispersion parameter
        }


class scRNAVAE(nn.Module):
    """
    Full VAE model for scRNA-seq data with batch correction capability.
    """
    def __init__(
        self,
        n_genes: int,
        n_batches: int = 0,
        latent_dim: int = 20,
        hidden_dims: List[int] = [128, 128],
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_genes = n_genes
        self.n_batches = n_batches
        
        # Set up encoder and decoder
        self.encoder = VAEEncoder(
            input_dim=n_genes, 
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm
        )
        
        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            output_dim=n_genes,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dispersion=dispersion
        )
        
        # Batch correction encoder if needed
        if n_batches > 0:
            self.batch_encoder = nn.Sequential(
                nn.Embedding(n_batches, 10),
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, latent_dim)
            )
        else:
            self.batch_encoder = None
            
    def forward(self, x: torch.Tensor, batch_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VAE.
        """
        # Encoder pass
        mu, log_var = self.encoder(x)
        
        # Adjust latent space based on batch
        if batch_indices is not None and self.batch_encoder is not None:
            batch_effect = self.batch_encoder(batch_indices)
            mu = mu + batch_effect
        
        # Reparameterization
        z = self.encoder.reparameterize(mu, log_var)
        
        # Decoder pass
        decoded = self.decoder(z)
        
        return {
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "px_scale": decoded["px_scale"],
            "px_r": decoded["px_r"]
        }
    
    def loss_function(
        self, 
        x: torch.Tensor, 
        outputs: Dict[str, torch.Tensor],
        kl_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the ELBO loss: reconstruction loss + KL divergence.
        Uses negative binomial loss for count data.
        """
        mu, log_var = outputs["mu"], outputs["log_var"]
        px_scale, px_r = outputs["px_scale"], outputs["px_r"]
        
        # Library size normalization factor (sum of all counts per cell)
        library_size = torch.sum(x, dim=1, keepdim=True)
        
        # Scale the means by library size
        scale_factor = px_scale * library_size
        
        # Negative binomial log likelihood (reconstruction loss)
        reconst_loss = -torch.sum(
            torch.lgamma(x + px_r) - torch.lgamma(px_r) - torch.lgamma(x + 1) +
            px_r * torch.log(px_r / (px_r + scale_factor)) +
            x * torch.log(scale_factor / (px_r + scale_factor)),
            dim=1
        )
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        
        # Total loss (ELBO)
        loss = reconst_loss + kl_weight * kl_div
        
        return {
            "loss": torch.mean(loss),
            "reconst_loss": torch.mean(reconst_loss),
            "kl_div": torch.mean(kl_div)
        }
        
    def get_latent(self, x: torch.Tensor, batch_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get latent representation (mean of the latent distribution).
        """
        mu, _ = self.encoder(x)
        
        # Adjust for batch if needed
        if batch_indices is not None and self.batch_encoder is not None:
            batch_effect = self.batch_encoder(batch_indices)
            mu = mu + batch_effect
            
        return mu
    
    def sample_latent(
        self, 
        x: torch.Tensor, 
        batch_indices: Optional[torch.Tensor] = None,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        Sample from the latent distribution multiple times.
        Used for uncertainty estimation.
        """
        mu, log_var = self.encoder(x)
        
        # Adjust for batch if needed
        if batch_indices is not None and self.batch_encoder is not None:
            batch_effect = self.batch_encoder(batch_indices)
            mu = mu + batch_effect
        
        # Extract dimensions
        batch_size = mu.size(0)
        
        # Expand parameters for multiple samples
        mu_expanded = mu.unsqueeze(1).expand(-1, n_samples, -1).reshape(batch_size * n_samples, -1)
        log_var_expanded = log_var.unsqueeze(1).expand(-1, n_samples, -1).reshape(batch_size * n_samples, -1)
        
        # Sample from expanded parameters
        std = torch.exp(0.5 * log_var_expanded)
        eps = torch.randn_like(std)
        z = mu_expanded + eps * std
        
        return z.reshape(batch_size, n_samples, -1)
    

def train_vae(
    model: scRNAVAE,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    kl_weight: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, List[float]]:
    """
    Train the VAE model.
    
    Returns:
        Dict containing training and validation metrics
    """
    print(f"Training on {device}")
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )
    
    # Store metrics
    metrics = {
        "train_loss": [],
        "train_reconst_loss": [],
        "train_kl_div": [],
        "val_loss": [],
        "val_reconst_loss": [],
        "val_kl_div": []
    }
    
    # Training loop
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_reconst_loss = 0.0
        train_kl_div = 0.0
        
        for batch in train_dataloader:
            x = batch[0].to(device)
            batch_indices = batch[1].to(device) if len(batch) > 1 else None
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x, batch_indices)
            
            # Compute loss
            loss_dict = model.loss_function(x, outputs, kl_weight=kl_weight)
            loss = loss_dict["loss"]
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_reconst_loss += loss_dict["reconst_loss"].item()
            train_kl_div += loss_dict["kl_div"].item()
        
        # Average metrics over batches
        train_loss /= len(train_dataloader)
        train_reconst_loss /= len(train_dataloader)
        train_kl_div /= len(train_dataloader)
        
        # Store metrics
        metrics["train_loss"].append(train_loss)
        metrics["train_reconst_loss"].append(train_reconst_loss)
        metrics["train_kl_div"].append(train_kl_div)
        
        # Validation
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            val_reconst_loss = 0.0
            val_kl_div = 0.0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    x = batch[0].to(device)
                    batch_indices = batch[1].to(device) if len(batch) > 1 else None
                    
                    # Forward pass
                    outputs = model(x, batch_indices)
                    
                    # Compute loss
                    loss_dict = model.loss_function(x, outputs, kl_weight=kl_weight)
                    
                    # Update metrics
                    val_loss += loss_dict["loss"].item()
                    val_reconst_loss += loss_dict["reconst_loss"].item()
                    val_kl_div += loss_dict["kl_div"].item()
            
            # Average metrics over batches
            val_loss /= len(val_dataloader)
            val_reconst_loss /= len(val_dataloader)
            val_kl_div /= len(val_dataloader)
            
            # Store metrics
            metrics["val_loss"].append(val_loss)
            metrics["val_reconst_loss"].append(val_reconst_loss)
            metrics["val_kl_div"].append(val_kl_div)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}")
    
    return metrics


def prepare_data_from_anndata(
    adata: ad.AnnData,
    batch_key: Optional[str] = None,
    train_size: float = 0.9,
    batch_size: int = 128,
    use_highly_variable: bool = True
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Prepare PyTorch DataLoaders from AnnData object.
    
    Returns:
        Tuple containing (train_dataloader, val_dataloader, data_info)
    """
    # Use highly variable genes if available
    if use_highly_variable and "highly_variable" in adata.var:
        adata_use = adata[:, adata.var.highly_variable].copy()
    else:
        adata_use = adata.copy()
    
    # Convert to dense format if sparse
    X = adata_use.X
    if isinstance(X, sparse.spmatrix):
        X = X.toarray()
    
    # Create tensor
    X_tensor = torch.FloatTensor(X)
    
    # Prepare batch indices if available
    if batch_key is not None and batch_key in adata_use.obs:
        # Map batch categories to integers
        batch_categories = adata_use.obs[batch_key].cat.categories
        batch_map = {cat: i for i, cat in enumerate(batch_categories)}
        batch_indices = adata_use.obs[batch_key].map(batch_map).values
        batch_indices_tensor = torch.LongTensor(batch_indices)
        dataset = TensorDataset(X_tensor, batch_indices_tensor)
    else:
        dataset = TensorDataset(X_tensor)
    
    # Split into train and validation
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=1-train_size,
        random_state=42
    )
    
    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
    )
    
    val_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
    )
    
    # Store additional info
    data_info = {
        "n_genes": adata_use.shape[1],
        "gene_names": adata_use.var_names.tolist(),
        "n_batches": len(batch_categories) if batch_key is not None else 0,
        "batch_categories": batch_categories.tolist() if batch_key is not None else None,
    }
    
    return train_dataloader, val_dataloader, data_info


def bayesian_differential_expression(
    model: scRNAVAE,
    adata: ad.AnnData,
    groupby: str,
    group1: str,
    group2: str,
    batch_key: Optional[str] = None,
    n_samples: int = 50,
    batch_size: int = 128,
    min_cells: int = 10,
    fdr_threshold: float = 0.05,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> pd.DataFrame:
    """
    Perform Bayesian differential expression analysis using the VAE.
    
    Args:
        model: Trained VAE model
        adata: AnnData object with cell group annotations
        groupby: Column in adata.obs for cell grouping
        group1, group2: Groups to compare
        batch_key: Column for batch information
        n_samples: Number of samples from latent distribution for uncertainty estimation
        batch_size: Batch size for inference
        min_cells: Minimum number of cells required in each group
        fdr_threshold: FDR threshold for significance
        device: Device for computation
        
    Returns:
        DataFrame with DE results
    """
    model = model.to(device)
    model.eval()
    
    # Select cells from the two groups
    cells1 = adata[adata.obs[groupby] == group1]
    cells2 = adata[adata.obs[groupby] == group2]
    
    # Check if enough cells in each group
    if (len(cells1) < min_cells) or (len(cells2) < min_cells):
        raise ValueError(f"Not enough cells in groups: {len(cells1)} vs {len(cells2)}. Minimum required: {min_cells}")
    
    # Get counts
    X1 = cells1.X
    X2 = cells2.X
    
    # Convert to dense if sparse
    if isinstance(X1, sparse.spmatrix):
        X1 = X1.toarray()
    if isinstance(X2, sparse.spmatrix):
        X2 = X2.toarray()
    
    # Prepare batch indices if available
    batch_indices1 = None
    batch_indices2 = None
    
    if batch_key is not None and batch_key in adata.obs:
        # Map batch categories to integers
        batch_categories = adata.obs[batch_key].cat.categories
        batch_map = {cat: i for i, cat in enumerate(batch_categories)}
        
        batch_indices1 = cells1.obs[batch_key].map(batch_map).values
        batch_indices2 = cells2.obs[batch_key].map(batch_map).values
    
    # Set up results storage
    gene_names = adata.var_names.tolist()
    n_genes = len(gene_names)
    
    # Generate posterior samples
    group1_means = []
    group2_means = []
    
    with torch.no_grad():
        # Process group 1
        for i in range(0, len(X1), batch_size):
            batch_x = torch.FloatTensor(X1[i:i+batch_size]).to(device)
            
            if batch_indices1 is not None:
                batch_b = torch.LongTensor(batch_indices1[i:i+batch_size]).to(device)
            else:
                batch_b = None
            
            # Get latent samples
            z_samples = model.sample_latent(batch_x, batch_b, n_samples=n_samples)
            
            # Shape: [batch_size, n_samples, latent_dim]
            batch_size_actual = z_samples.size(0)
            
            # Process each sample through decoder
            for s in range(n_samples):
                # Get one latent sample for each cell
                z_sample = z_samples[:, s, :]
                
                # Decode to get gene expression
                decoder_output = model.decoder(z_sample)
                px_scale = decoder_output["px_scale"]
                
                # Compute mean expression
                if s == 0:
                    batch_means = px_scale.unsqueeze(1)
                else:
                    batch_means = torch.cat([batch_means, px_scale.unsqueeze(1)], dim=1)
            
            # Average over all samples for each cell
            group1_means.append(batch_means.mean(dim=1))
        
        # Process group 2
        for i in range(0, len(X2), batch_size):
            batch_x = torch.FloatTensor(X2[i:i+batch_size]).to(device)
            
            if batch_indices2 is not None:
                batch_b = torch.LongTensor(batch_indices2[i:i+batch_size]).to(device)
            else:
                batch_b = None
            
            # Get latent samples
            z_samples = model.sample_latent(batch_x, batch_b, n_samples=n_samples)
            
            # Shape: [batch_size, n_samples, latent_dim]
            batch_size_actual = z_samples.size(0)
            
            # Process each sample through decoder
            for s in range(n_samples):
                # Get one latent sample for each cell
                z_sample = z_samples[:, s, :]
                
                # Decode to get gene expression
                decoder_output = model.decoder(z_sample)
                px_scale = decoder_output["px_scale"]
                
                # Compute mean expression
                if s == 0:
                    batch_means = px_scale.unsqueeze(1)
                else:
                    batch_means = torch.cat([batch_means, px_scale.unsqueeze(1)], dim=1)
            
            # Average over all samples for each cell
            group2_means.append(batch_means.mean(dim=1))
    
    # Concatenate batches
    group1_means = torch.cat(group1_means, dim=0).cpu().numpy()
    group2_means = torch.cat(group2_means, dim=0).cpu().numpy()
    
    # Compute statistics for each gene
    results = []
    
    for g in range(n_genes):
        # Get mean expression for this gene
        expr1 = group1_means[:, g]
        expr2 = group2_means[:, g]
        
        # Compute statistics
        mean1 = np.mean(expr1)
        mean2 = np.mean(expr2)
        
        # Log2 fold change
        epsilon = 1e-8  # To avoid log(0)
        log2fc = np.log2((mean2 + epsilon) / (mean1 + epsilon))
        
        # Compute p-value using t-test
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(expr1, expr2, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(expr1, ddof=1) + np.var(expr2, ddof=1)) / 2)
        effect_size = (mean2 - mean1) / (pooled_std + epsilon)
        
        # Store results
        results.append({
            'gene': gene_names[g],
            'mean1': mean1,
            'mean2': mean2,
            'log2fc': log2fc,
            'p_value': p_val,
            'effect_size': effect_size
        })
    
    # Create dataframe
    results_df = pd.DataFrame(results)
    
    # Calculate FDR (Benjamini-Hochberg correction)
    from statsmodels.stats.multitest import multipletests
    _, adj_pvals, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
    results_df['adj_p_value'] = adj_pvals
    
    # Flag significant genes
    results_df['significant'] = results_df['adj_p_value'] < fdr_threshold
    
    # Sort by absolute log2FC
    results_df = results_df.sort_values('log2fc', key=abs, ascending=False)
    
    return results_df


def plot_de_results(
    de_results: pd.DataFrame,
    n_top_genes: int = 20,
    log2fc_threshold: float = 0.5,
    fdr_threshold: float = 0.05,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2"
) -> plt.Figure:
    """
    Create a volcano plot of differential expression results.
    
    Args:
        de_results: DataFrame with DE results from bayesian_differential_expression
        n_top_genes: Number of top genes to label in the plot
        log2fc_threshold: Log2 fold change threshold for significance
        fdr_threshold: FDR threshold for significance
        group1_name, group2_name: Names for the groups compared
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Transform p-values to -log10 scale
    de_results['neg_log10_pval'] = -np.log10(de_results['adj_p_value'] + 1e-10)
    
    # Define colors based on significance and fold change
    colors = []
    for _, row in de_results.iterrows():
        if row['adj_p_value'] < fdr_threshold:
            if row['log2fc'] > log2fc_threshold:
                colors.append('red')  # Up-regulated
            elif row['log2fc'] < -log2fc_threshold:
                colors.append('blue')  # Down-regulated
            else:
                colors.append('gray')  # Significant but small effect
        else:
            colors.append('gray')  # Not significant
    
    # Create scatter plot
    ax.scatter(
        de_results['log2fc'],
        de_results['neg_log10_pval'],
        c=colors,
        alpha=0.6,
        s=30
    )
    
    # Add threshold lines
    ax.axhline(-np.log10(fdr_threshold), ls='--', color='gray')
    ax.axvline(log2fc_threshold, ls='--', color='gray')
    ax.axvline(-log2fc_threshold, ls='--', color='gray')
    
    # Label top genes
    significant = de_results['adj_p_value'] < fdr_threshold
    fold_change = abs(de_results['log2fc']) > log2fc_threshold
    top_genes = de_results[significant & fold_change].head(n_top_genes)
    
    for _, row in top_genes.iterrows():
        ax.text(
            row['log2fc'] + 0.1,
            row['neg_log10_pval'] + 0.1,
            row['gene'],
            fontsize=9
        )
    
    # Set labels and title
    ax.set_xlabel('Log2 Fold Change', fontsize=12)
    ax.set_ylabel('-log10(Adjusted p-value)', fontsize=12)
    ax.set_title(f'Differential Expression: {group2_name} vs {group1_name}', fontsize=14)
    
    # Add legend for colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label=f'Up in {group2_name}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=f'Up in {group1_name}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    # Display summary statistics
    n_sig_up = sum((de_results['adj_p_value'] < fdr_threshold) & (de_results['log2fc'] > log2fc_threshold))
    n_sig_down = sum((de_results['adj_p_value'] < fdr_threshold) & (de_results['log2fc'] < -log2fc_threshold))
    
    text = (
        f'Total genes: {len(de_results)}\n'
        f'Up-regulated: {n_sig_up}\n'
        f'Down-regulated: {n_sig_down}\n'
        f'FDR threshold: {fdr_threshold}\n'
        f'Log2FC threshold: {log2fc_threshold}'
    )
    
    # Place text box in upper left
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    return fig


# Example usage:
def scvi_de_analysis_example(adata, batch_key='batch', condition_key='condition'):
    """
    Example workflow for Bayesian differential expression using VAE.
    """
    # Prepare data for VAE
    train_dl, val_dl, data_info = prepare_data_from_anndata(
        adata,
        batch_key=batch_key,
        use_highly_variable=True,
        batch_size=128
    )
    
    # Create model
    model = scRNAVAE(
        n_genes=data_info['n_genes'],
        n_batches=data_info['n_batches'],
        latent_dim=20,
        hidden_dims=[128, 128],
        dropout_rate=0.1
    )
    
    # Train model
    metrics = train_vae(
        model,
        train_dl,
        val_dl,
        n_epochs=100,
        lr=1e-3,
        kl_weight=1.0
    )
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_kl_div'], label='Train KL')
    plt.plot(metrics['val_kl_div'], label='Val KL')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    
    # Perform differential expression between conditions
    conditions = adata.obs[condition_key].cat.categories.tolist()
    if len(conditions) >= 2:
        de_results = bayesian_differential_expression(
            model,
            adata,
            groupby=condition_key,
            group1=conditions[0],
            group2=conditions[1],
            batch_key=batch_key,
            n_samples=50
        )
        
        # Plot results
        fig = plot_de_results(
            de_results,
            n_top_genes=20,
            group1_name=conditions[0],
            group2_name=conditions[1]
        )
        fig.savefig('de_volcano_plot.png')
        
        # Visualize uncertainty for top genes
        top_genes = de_results.head(5)['gene'].tolist()
        fig = visualize_uncertainty(
            model,
            adata,
            gene_names=top_genes,
            groupby=condition_key,
            n_samples=20
        )
        fig.savefig('gene_uncertainty.png')
    
    return model, de_results


def perform_integrated_analysis(
    adata_list: List[ad.AnnData],
    batch_names: List[str],
    conditions: Optional[Dict[str, List[str]]] = None,
    output_dir: str = "./integrated_results"
):
    """
    Perform an integrated analysis of multiple scRNA-seq datasets with VAE-based integration.
    
    Args:
        adata_list: List of AnnData objects to integrate
        batch_names: List of batch names corresponding to each AnnData
        conditions: Dict mapping each batch to list of condition labels
        output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize standard pipeline
    pipeline = ScRNASeqPipeline(output_dir=output_dir)
    
    # Convert to dictionary for loading
    data_dict = {name: adata for name, adata in zip(batch_names, adata_list)}
    
    # Add condition information if provided
    if conditions:
        for batch_name, condition_list in conditions.items():
            for i, adata in enumerate(adata_list):
                if batch_names[i] == batch_name:
                    adata.obs['condition'] = condition_list
    
    # Run standard pipeline through cluster identification
    pipeline.load_data(data_dict, batch_key="batch") \
        .qc_and_filtering() \
        .normalize_and_scale() \
        .run_pca() \
        .prepare_scvi(batch_key="batch") \
        .run_scvi(n_latent=30, n_epochs=200) \
        .run_umap(use_scvi=True) \
        .run_clustering() \
        .find_markers() \
        .save_results()
    
    # If condition information is available, run Bayesian DE
    if conditions and 'condition' in pipeline.adata.obs:
        pipeline.run_bayesian_de(groupby='condition')
    
    # Get latent representation from scVI for custom analysis
    adata_integrated = pipeline.adata.copy()
    
    # Return both the pipeline and the integrated AnnData
    return pipeline, adata_integrated


def visualize_uncertainty(
    model: scRNAVAE,
    adata: ad.AnnData,
    gene_names: List[str],
    groupby: str,
    n_samples: int = 20,
    batch_key: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> plt.Figure:
    """
    Visualize expression uncertainty for selected genes across groups.
    
    Args:
        model: Trained VAE model
        adata: AnnData object with cell group annotations
        gene_names: List of genes to visualize
        groupby: Column in adata.obs for cell grouping
        n_samples: Number of samples from latent distribution
        batch_key: Column for batch information
        device: Device for computation
        
    Returns:
        Matplotlib figure object
    """
    model = model.to(device)
    model.eval()
    
    # Get gene indices
    gene_indices = [list(adata.var_names).index(gene) for gene in gene_names if gene in adata.var_names]
    if not gene_indices:
        raise ValueError("None of the specified genes found in the dataset")
    
    # Get groups
    groups = adata.obs[groupby].cat.categories.tolist()
    
    # Set up figure
    n_genes = len(gene_indices)
    fig, axes = plt.subplots(n_genes, len(groups), figsize=(3*len(groups), 3*n_genes), sharey='row')
    
    if n_genes == 1:
        axes = axes.reshape(1, -1)
    
    # Process each group
    for g_idx, group in enumerate(groups):
        cells = adata[adata.obs[groupby] == group]
        
        # Convert to dense if sparse
        X = cells.X
        if isinstance(X, sparse.spmatrix):
            X = X.toarray()
        
        # Prepare batch indices if needed
        batch_indices = None
        if batch_key is not None and batch_key in cells.obs:
            batch_cats = adata.obs[batch_key].cat.categories
            batch_map = {cat: i for i, cat in enumerate(batch_cats)}
            batch_indices = cells.obs[batch_key].map(batch_map).values
        
        # Generate posterior samples
        all_samples = []
        
        with torch.no_grad():
            batch_x = torch.FloatTensor(X).to(device)
            if batch_indices is not None:
                batch_b = torch.LongTensor(batch_indices).to(device)
            else:
                batch_b = None
            
            # Get latent samples
            z_samples = model.sample_latent(batch_x, batch_b, n_samples=n_samples)
            
            # Process each sample
            for s in range(n_samples):
                z_sample = z_samples[:, s, :]
                decoder_output = model.decoder(z_sample)
                px_scale = decoder_output["px_scale"]
                all_samples.append(px_scale.cpu().numpy())
        
        # Stack samples
        all_samples = np.stack(all_samples)  # [n_samples, n_cells, n_genes]
        
        # For each gene, create violin plot of expression distribution
        for gene_idx, gene_name in enumerate(gene_names):
            if gene_name not in adata.var_names:
                continue
                
            gene_pos = gene_indices[gene_idx]
            
            # Extract expression for this gene across all cells and samples
            gene_expr = all_samples[:, :, gene_pos]  # [n_samples, n_cells]
            
            # Plot as violin
            ax = axes[gene_idx, g_idx]
            
            # Reshape to have one distribution per cell
            reshaped_expr = gene_expr.T  # [n_cells, n_samples]
            
            # Plot violin for each cell (one distribution per cell)
            parts = ax.violinplot(
                reshaped_expr,
                positions=range(len(reshaped_expr)),
                showmeans=False,
                showmedians=True,
                showextrema=False
            )
            
            # Set color based on group
            for pc in parts['bodies']:
                pc.set_facecolor(f'C{g_idx}')
                pc.set_alpha(0.7)
            
            # Add empirical expression values as points
            ax.scatter(
                range(len(reshaped_expr)),
                X[:, gene_pos],
                color='black',
                alpha=0.5,
                s=10
            )
            
            # Add labels
            if g_idx == 0:
                ax.set_ylabel(f'{gene_name}\nExpression', fontsize=12)
            
            if gene_idx == 0:
                ax.set_title(f'{group}', fontsize=12)
            
            # Remove x-axis labels
            ax.set_xticks([])
            ax.set_xlim(-1, len(reshaped_expr))
            
    plt.tight_layout()
    return fig


