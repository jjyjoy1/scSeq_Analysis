"""
Deep Count Autoencoder (DCA) for scRNA-seq data denoising and batch correction

This implementation adapts the DCA approach to handle both denoising and batch effect correction
for single-cell RNA sequencing data, using a Zero-Inflated Negative Binomial (ZINB) loss function
to model the count distribution of scRNA-seq data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from scipy import sparse
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, regularizers, callbacks
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import warnings

# Set TensorFlow warnings to error only
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_pbmc_dataset():
    """
    Load the PBMC dataset from scvi-tools.
    
    Returns:
        AnnData: Combined PBMC dataset with batch information
    """
    print("Loading PBMC dataset...")
    
    try:
        import scvi
        pbmc_data = scvi.data.pbmc_dataset()
        print(f"PBMC dataset: {pbmc_data.shape[0]} cells, {pbmc_data.shape[1]} genes")
        
        # Convert batch to categorical
        pbmc_data.obs['batch'] = pbmc_data.obs['batch'].astype('category')
        
        # Save raw data for future use
        os.makedirs("dca_results", exist_ok=True)
        pbmc_data.write("dca_results/pbmc_raw.h5ad")
        
        return pbmc_data
    
    except ImportError:
        print("scvi-tools not found. Please install it with: pip install scvi-tools")
        return None


def preprocess_data(adata, n_top_genes=2000, min_genes=200, min_cells=3):
    """
    Preprocess data for DCA - filter cells and genes, normalize, and select highly variable genes.
    
    Parameters:
        adata (AnnData): The annotated data matrix
        n_top_genes (int): Number of highly variable genes to select
        min_genes (int): Minimum number of genes per cell
        min_cells (int): Minimum number of cells per gene
        
    Returns:
        AnnData: Preprocessed data
    """
    print("Preprocessing data...")
    
    # Create a copy to avoid modifying original
    adata = adata.copy()
    
    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # Store raw counts
    adata.layers["counts"] = adata.X.copy()
    
    # Convert to dense if sparse
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    
    # Check for and replace infinite values
    if np.any(~np.isfinite(adata.X)):
        print("Warning: Infinite values found in data matrix. Replacing with zeros.")
        adata.X[~np.isfinite(adata.X)] = 0
    
    # Select highly variable genes
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key='batch')
        adata_subset = adata[:, adata.var.highly_variable].copy()
    except Exception as e:
        print(f"Error finding highly variable genes with batch correction: {e}")
        try:
            print("Falling back to standard highly variable genes without batch correction")
            # Try with flavor='seurat_v3' which might be more robust
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3')
            adata_subset = adata[:, adata.var.highly_variable].copy()
        except Exception as e2:
            print(f"Error with seurat_v3 method: {e2}")
            print("Using manual approach for selecting variable genes")
            
            # Calculate gene stats manually
            gene_means = np.mean(adata.X, axis=0)
            gene_vars = np.var(adata.X, axis=0)
            
            # Calculate coefficient of variation (CV)
            cv = gene_vars / (gene_means + 1e-8)  # Add small value to avoid division by zero
            
            # Select top genes by CV
            top_genes_idx = np.argsort(cv)[-n_top_genes:]
            
            # Update highly_variable in var
            adata.var['highly_variable'] = False
            adata.var.iloc[top_genes_idx, adata.var.columns.get_loc('highly_variable')] = True
            
            adata_subset = adata[:, adata.var.highly_variable].copy()
    
    print(f"After preprocessing: {adata_subset.shape[0]} cells, {adata_subset.shape[1]} genes")
    return adata_subset

def make_count_matrix(adata):
    """
    Prepare count matrix from AnnData object.
    
    Parameters:
        adata (AnnData): AnnData object
        
    Returns:
        numpy.ndarray: Count matrix
    """
    if "counts" in adata.layers:
        if sparse.issparse(adata.layers["counts"]):
            counts = adata.layers["counts"].toarray()
        else:
            counts = adata.layers["counts"]
    else:
        if sparse.issparse(adata.X):
            counts = adata.X.toarray()
        else:
            counts = adata.X

    # Ensure non-negative values
    counts[counts < 0] = 0
    
    return counts


def zinb_loss(y_true, y_pred, theta=None, pi=None, eps=1e-10):
    """
    Zero-inflated negative binomial loss function.
    
    Parameters:
        y_true: True counts
        y_pred: Predicted means
        theta: Dispersion parameter
        pi: Zero-inflation parameter
        eps: Small value to avoid numerical issues
    
    Returns:
        loss: ZINB loss
    """
    # If theta or pi not provided, use the other outputs from the model
    if theta is None:
        # Assuming y_pred is [means, theta, pi]
        y_pred, theta, pi = tf.split(y_pred, 3, axis=-1)
    
    # Clip values for numerical stability
    y_true = tf.clip_by_value(y_true, 0, 1e6)
    y_pred = tf.clip_by_value(y_pred, eps, 1e6)
    theta = tf.clip_by_value(theta, eps, 1e6)
    pi = tf.clip_by_value(pi, eps, 1.0 - eps)
    
    # Negative binomial part (for non-zero counts)
    nb_case = tf.math.lgamma(y_true + theta) - tf.math.lgamma(theta) - tf.math.lgamma(y_true + 1.0) + \
              theta * tf.math.log(theta) + \
              y_true * tf.math.log(y_pred) - \
              (y_true + theta) * tf.math.log(theta + y_pred)
    
    # Zero-inflation part
    zero_nb = theta * (tf.math.log(theta) - tf.math.log(theta + y_pred))
    zero_case = tf.math.log(pi + (1.0 - pi) * tf.exp(zero_nb))
    
    # Non-zero case
    nonzero_case = tf.math.log(1.0 - pi) + nb_case
    
    # Combine zero and non-zero cases
    result = tf.where(
        tf.less(y_true, 1e-8),
        zero_case,
        nonzero_case
    )
    
    return -result


class MeanAct(layers.Layer):
    """
    Mean activation function.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        return tf.math.exp(inputs)


class DispAct(layers.Layer):
    """
    Dispersion activation function.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        return tf.math.softplus(inputs) + 1e-4


class PiAct(layers.Layer):
    """
    Pi activation function for zero-inflation probabilities.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        return tf.math.sigmoid(inputs)


def build_dca_autoencoder(input_dim, hidden_dim=128, latent_dim=32, 
                       l1_reg=0.0, l2_reg=0.0, activation='relu', 
                       batchnorm=True, dropout_rate=0.1, use_zinb=True):
    """
    Build DCA autoencoder model.
    
    Parameters:
        input_dim (int): Input dimension (number of genes)
        hidden_dim (int): Size of hidden layer
        latent_dim (int): Size of latent layer (bottleneck)
        l1_reg (float): L1 regularization strength
        l2_reg (float): L2 regularization strength
        activation (str): Activation function
        batchnorm (bool): Whether to use batch normalization
        dropout_rate (float): Dropout rate
        use_zinb (bool): Whether to use zero-inflated negative binomial distribution
    
    Returns:
        model: Keras model
    """
    # Create regularizer
    regularizer = None
    if l1_reg > 0 or l2_reg > 0:
        regularizer = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    
    # Input layer
    input_layer = layers.Input(shape=(input_dim,), name='count_input')
    
    # Encoder layers
    x = layers.Dense(hidden_dim, activation=None, kernel_regularizer=regularizer)(input_layer)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Additional encoder layer
    x = layers.Dense(hidden_dim // 2, activation=None, kernel_regularizer=regularizer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Bottleneck layer
    latent = layers.Dense(latent_dim, activation=None, kernel_regularizer=regularizer, name='latent_layer')(x)
    if batchnorm:
        latent = layers.BatchNormalization()(latent)
    latent = layers.Activation(activation)(latent)
    
    # Decoder layers
    x = layers.Dense(hidden_dim // 2, activation=None, kernel_regularizer=regularizer)(latent)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(hidden_dim, activation=None, kernel_regularizer=regularizer)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layers for ZINB distribution
    # 1. Mean
    mean = layers.Dense(input_dim, activation=None, kernel_regularizer=regularizer, name='decoder_mean')(x)
    mean_act = MeanAct(name='mean_activation')(mean)
    
    # 2. Dispersion
    dispersion = layers.Dense(input_dim, activation=None, kernel_regularizer=regularizer, name='decoder_dispersion')(x)
    disp_act = DispAct(name='dispersion_activation')(dispersion)
    
    if use_zinb:
        # 3. Zero-inflation
        pi = layers.Dense(input_dim, activation=None, kernel_regularizer=regularizer, name='decoder_pi')(x)
        pi_act = PiAct(name='pi_activation')(pi)
        
        # Create model with separate outputs
        model = Model(inputs=input_layer, outputs=[mean_act, disp_act, pi_act], name='DCA')
    else:
        # Just use negative binomial (NB) without zero-inflation
        model = Model(inputs=input_layer, outputs=[mean_act, disp_act], name='DCA')
    
    return model



def custom_loss_wrapper(use_zinb):
    """
    Wrapper for creating custom loss function with or without zero-inflation.
    
    Parameters:
        use_zinb (bool): Whether to use Zero-Inflated Negative Binomial loss
        
    Returns:
        loss_fn: Loss function
    """
    def nb_loss(y_true, y_pred):
        """Negative Binomial loss function (without zero-inflation)"""
        # Split predictions into mean and dispersion
        y_pred_mean, y_pred_disp = y_pred
        return zinb_loss(y_true, y_pred_mean, theta=y_pred_disp, pi=None)
    
    def zinb_loss_fn(y_true, y_pred):
        """Zero-Inflated Negative Binomial loss function"""
        # Split predictions into mean, dispersion, and pi
        y_pred_mean, y_pred_disp, y_pred_pi = y_pred
        return zinb_loss(y_true, y_pred_mean, theta=y_pred_disp, pi=y_pred_pi)
    
    return zinb_loss_fn if use_zinb else nb_loss


def zinb_loss_separate(y_true, mean, disp, pi=None, eps=1e-10):
    """
    Zero-inflated negative binomial loss function with separate tensor inputs.
    
    Parameters:
        y_true: True counts
        mean: Predicted means
        disp: Dispersion parameter
        pi: Zero-inflation parameter (optional)
        eps: Small value to avoid numerical issues
    
    Returns:
        loss: ZINB loss
    """
    # Clip values for numerical stability
    y_true = tf.clip_by_value(y_true, 0, 1e6)
    mean = tf.clip_by_value(mean, eps, 1e6)
    disp = tf.clip_by_value(disp, eps, 1e6)
    
    # Negative binomial part (for non-zero counts)
    nb_case = tf.math.lgamma(y_true + disp) - tf.math.lgamma(disp) - tf.math.lgamma(y_true + 1.0) + \
              disp * tf.math.log(disp) + \
              y_true * tf.math.log(mean) - \
              (y_true + disp) * tf.math.log(disp + mean)
    
    if pi is not None:
        # Clip pi for numerical stability
        pi = tf.clip_by_value(pi, eps, 1.0 - eps)
        
        # Zero-inflation part
        zero_nb = disp * (tf.math.log(disp) - tf.math.log(disp + mean))
        zero_case = tf.math.log(pi + (1.0 - pi) * tf.exp(zero_nb))
        
        # Non-zero case
        nonzero_case = tf.math.log(1.0 - pi) + nb_case
        
        # Combine zero and non-zero cases
        result = tf.where(
            tf.less(y_true, 1e-8),
            zero_case,
            nonzero_case
        )
    else:
        # Simple negative binomial (no zero-inflation)
        zero_nb = disp * (tf.math.log(disp) - tf.math.log(disp + mean))
        result = tf.where(
            tf.less(y_true, 1e-8),
            zero_nb,
            nb_case
        )
    
    return -tf.reduce_mean(result)


def load_dca_model(model_path):
    """
    Load a previously saved DCA model.
    
    Parameters:
        model_path (str): Path to saved model directory
        
    Returns:
        model: Loaded Keras model
    """
    import json
    
    # Load model information
    with open(f"{model_path}/model_info.json", "r") as json_file:
        model_info = json.load(json_file)
    
    # Extract model parameters
    input_dim = model_info['input_dim']
    hidden_dim = model_info['hidden_dim']
    latent_dim = model_info['latent_dim']
    use_zinb = model_info['use_zinb']
    
    # Rebuild the model
    model = build_dca_autoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        use_zinb=use_zinb
    )
    
    # Create a dummy input to build the model
    dummy_input = np.zeros((1, input_dim))
    _ = model(dummy_input)
    
    # Load weights
    model.load_weights(f"{model_path}/weights")
    
    print(f"Model loaded from {model_path}")
    return model


import optuna
from optuna.trial import Trial
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import time


def optimize_dca_hyperparameters(
    adata, 
    n_trials=20, 
    timeout=None, 
    batch_key='batch', 
    study_name="dca_optimization",
    storage=None,  # Optional database URL for study persistence
    save_best_params=True,
    save_path="dca_results/best_params.json"
):
    """
    Optimize DCA hyperparameters using Optuna.
    
    Parameters:
        adata (AnnData): Annotated data matrix
        n_trials (int): Number of optimization trials
        timeout (int): Timeout in seconds for the optimization (optional)
        batch_key (str): Key for batch annotation in adata.obs
        study_name (str): Name of the Optuna study
        storage (str): Database URL for Optuna study persistence (optional)
        save_best_params (bool): Whether to save best parameters to file
        save_path (str): Path to save best parameters
        
    Returns:
        dict: Best hyperparameters
    """
    print(f"Starting hyperparameter optimization with Optuna (n_trials={n_trials})")
    
    # Prepare data once to avoid redundant preprocessing
    count_matrix = make_count_matrix(adata)
    count_train, count_val = train_test_split(count_matrix, test_size=0.2, random_state=42)
    
    # Get input dimension
    input_dim = count_matrix.shape[1]
    
    def objective(trial: Trial):
        """Optuna objective function to minimize validation loss"""
        
        # Sample hyperparameters
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        latent_dim = trial.suggest_int("latent_dim", 16, 64)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        l1_reg = trial.suggest_float("l1_reg", 0.0, 0.01)
        l2_reg = trial.suggest_float("l2_reg", 0.0, 0.01)
        use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])
        
        # Always use ZINB since it's generally better for scRNA-seq
        use_zinb = True
        
        # Build model with trial params
        model = build_dca_autoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            activation='relu',
            batchnorm=use_batchnorm,
            dropout_rate=dropout_rate,
            use_zinb=use_zinb
        )
        
        # Define custom TF functions for loss calculation
        @tf.function
        def compute_loss(y_true, mean, disp, pi):
            return zinb_loss_separate(y_true, mean, disp, pi)
        
        # Custom training step
        @tf.function
        def train_step(x, y, optimizer):
            with tf.GradientTape() as tape:
                mean, disp, pi = model(x, training=True)
                loss = compute_loss(y, mean, disp, pi)
            
            trainable_vars = model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))
            return loss
        
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Mini training loop with early stopping
        patience = 10
        best_val_loss = float('inf')
        patience_counter = 0
        max_epochs = 100  # Limit for trial duration
        
        # For storing losses
        train_losses = []
        val_losses = []
        
        for epoch in range(max_epochs):
            # Shuffle training data
            indices = np.random.permutation(len(count_train))
            
            # Train in batches
            epoch_loss = 0
            num_batches = int(np.ceil(len(count_train) / batch_size))
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, len(count_train))
                batch_indices = indices[start_idx:end_idx]
                
                x_batch = count_train[batch_indices]
                batch_loss = train_step(x_batch, x_batch, optimizer)
                epoch_loss += batch_loss
            
            epoch_loss /= num_batches
            train_losses.append(float(epoch_loss))
            
            # Validate
            val_loss = 0
            num_val_batches = int(np.ceil(len(count_val) / batch_size))
            
            for batch in range(num_val_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, len(count_val))
                
                x_batch = count_val[start_idx:end_idx]
                
                # Compute validation loss
                mean, disp, pi = model(x_batch, training=False)
                batch_val_loss = compute_loss(x_batch, mean, disp, pi)
                val_loss += batch_val_loss
            
            val_loss /= num_val_batches
            val_losses.append(float(val_loss))
            
            # Report intermediate value to Optuna
            trial.report(val_loss, epoch)
            
            # Print progress for every 10 epochs
            if epoch % 10 == 0:
                print(f"Trial {trial.number}, Epoch {epoch}/{max_epochs}: val_loss={val_loss:.4f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Trial {trial.number} stopped early at epoch {epoch}")
                    break
            
            # Stop trial if Optuna suggests it
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # Return best validation loss
        return best_val_loss
    
    # Create or load study
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner()
        )
    except:
        # If storage is not available or study cannot be loaded
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner()
        )
    
    # Run optimization
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    end_time = time.time()
    
    # Print optimization results
    print("\n" + "="*50)
    print("Hyperparameter Optimization Results")
    print("="*50)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print(f"Optimization time: {(end_time - start_time)/60:.2f} minutes")
    print("\nBest hyperparameters:")
    
    # Get best parameters
    best_params = study.best_params
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Save best parameters if requested
    if save_best_params:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        import json
        with open(save_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\nBest parameters saved to {save_path}")
    
    # Create plots of optimization history
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        # Make directory for plots
        os.makedirs("dca_results/optuna_plots", exist_ok=True)
        
        # Plot optimization history
        fig = plot_optimization_history(study)
        fig.write_image("dca_results/optuna_plots/optimization_history.png")
        
        # Plot parameter importances
        fig = plot_param_importances(study)
        fig.write_image("dca_results/optuna_plots/param_importances.png")
        
        print("Optimization plots saved to dca_results/optuna_plots/")
    except ImportError:
        print("Plotting requires plotly and kaleido packages. Skipping plot generation.")
    
    return best_params


def train_dca_with_best_params(adata, best_params, batch_key='batch', use_zinb=True,
                              save_path=None, save_model_path=None):
    """
    Train DCA model with optimized hyperparameters.
    
    Parameters:
        adata (AnnData): Annotated data matrix
        best_params (dict): Dictionary of best hyperparameters from Optuna
        batch_key (str): Key for batch annotation in adata.obs
        use_zinb (bool): Whether to use Zero-Inflated Negative Binomial loss
        save_path (str): Path to save denoised data
        save_model_path (str): Path to save the model
        
    Returns:
        tuple: (model, denoised_adata)
    """
    # Extract parameters, filling in defaults for any missing ones
    hidden_dim = best_params.get("hidden_dim", 128)
    latent_dim = best_params.get("latent_dim", 32)
    learning_rate = best_params.get("learning_rate", 1e-3)
    batch_size = best_params.get("batch_size", 128)
    dropout_rate = best_params.get("dropout_rate", 0.1)
    l1_reg = best_params.get("l1_reg", 0.0)
    l2_reg = best_params.get("l2_reg", 0.0)
    use_batchnorm = best_params.get("use_batchnorm", True)
    
    print(f"Training DCA with optimized parameters:")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  latent_dim: {latent_dim}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  batch_size: {batch_size}")
    print(f"  dropout_rate: {dropout_rate}")
    print(f"  l1_reg: {l1_reg}")
    print(f"  l2_reg: {l2_reg}")
    print(f"  use_batchnorm: {use_batchnorm}")
    print(f"  use_zinb: {use_zinb}")
    
    # Train model with optimized hyperparameters
    model, adata_denoised = train_dca(
        adata,
        batch_key=batch_key,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        use_batchnorm=use_batchnorm,
        use_zinb=use_zinb,
        save_path=save_path,
        save_model_path=save_model_path,
        epochs=300  # Keep longer epochs for final training
    )
    
    return model, adata_denoised


def train_dca(adata, batch_key='batch', latent_dim=32, hidden_dim=128, 
           learning_rate=1e-3, epochs=300, batch_size=128, use_zinb=True,
           save_path=None, save_model_path=None, dropout_rate=0.1,
           l1_reg=0.0, l2_reg=0.0, use_batchnorm=True):
    """
    Train DCA model on scRNA-seq data.
    
    Parameters:
        adata (AnnData): Annotated data matrix
        batch_key (str): Key for batch annotation in adata.obs
        latent_dim (int): Dimension of latent space
        hidden_dim (int): Dimension of hidden layers
        learning_rate (float): Learning rate for optimizer
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        use_zinb (bool): Whether to use Zero-Inflated Negative Binomial loss
        save_path (str): Path to save denoised data
        save_model_path (str): Path to save the model
        dropout_rate (float): Dropout rate for network layers
        l1_reg (float): L1 regularization strength
        l2_reg (float): L2 regularization strength
        use_batchnorm (bool): Whether to use batch normalization
        
    Returns:
        tuple: (model, denoised_adata)
    """
    print(f"Training DCA with parameters:")
    print(f"  latent_dim={latent_dim}")
    print(f"  hidden_dim={hidden_dim}")
    print(f"  learning_rate={learning_rate}")
    print(f"  batch_size={batch_size}")
    print(f"  dropout_rate={dropout_rate}")
    print(f"  l1_reg={l1_reg}")
    print(f"  l2_reg={l2_reg}")
    print(f"  use_batchnorm={use_batchnorm}")
    print(f"  use_zinb={use_zinb}")
    
    # Get count matrix
    count_matrix = make_count_matrix(adata)
    
    # Build model
    input_dim = count_matrix.shape[1]
    model = build_dca_autoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        activation='relu',
        batchnorm=use_batchnorm,
        dropout_rate=dropout_rate,
        use_zinb=use_zinb
    )
    
    # Define custom TF functions for loss calculation
    @tf.function
    def compute_loss(y_true, mean, disp, pi=None):
        return zinb_loss_separate(y_true, mean, disp, pi)
    
    # Custom training step
    @tf.function
    def train_step(x, y, optimizer):
        with tf.GradientTape() as tape:
            if use_zinb:
                mean, disp, pi = model(x, training=True)
                loss = compute_loss(y, mean, disp, pi)
            else:
                mean, disp = model(x, training=True)
                loss = compute_loss(y, mean, disp)
        
        trainable_vars = model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Split data for training and validation
    count_train, count_val = train_test_split(count_matrix, test_size=0.1, random_state=42)
    
    # Implement custom training loop
    print("Starting training loop...")
    history = {"loss": [], "val_loss": []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    best_weights = None
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(count_train))
        
        # Train in batches
        epoch_loss = 0
        num_batches = int(np.ceil(len(count_train) / batch_size))
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(count_train))
            batch_indices = indices[start_idx:end_idx]
            
            x_batch = count_train[batch_indices]
            batch_loss = train_step(x_batch, x_batch, optimizer)
            epoch_loss += batch_loss
        
        epoch_loss /= num_batches
        
        # Validate
        val_loss = 0
        num_val_batches = int(np.ceil(len(count_val) / batch_size))
        
        for batch in range(num_val_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(count_val))
            
            x_batch = count_val[start_idx:end_idx]
            
            # Compute validation loss
            if use_zinb:
                mean, disp, pi = model(x_batch, training=False)
                batch_val_loss = compute_loss(x_batch, mean, disp, pi)
            else:
                mean, disp = model(x_batch, training=False)
                batch_val_loss = compute_loss(x_batch, mean, disp)
            
            val_loss += batch_val_loss
        
        val_loss /= num_val_batches
        
        # Record history
        history["loss"].append(float(epoch_loss))
        history["val_loss"].append(float(val_loss))
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best weights
            best_weights = [var.numpy() for var in model.weights]
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best weights
    if best_weights is not None:
        for i, weight in enumerate(best_weights):
            model.weights[i].assign(weight)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    
    # Save figure
    os.makedirs("dca_results/figures", exist_ok=True)
    plt.savefig("dca_results/figures/training_history.png", dpi=300)
    
    # Get denoised data (mean parameter from the model)
    if use_zinb:
        mean, dispersion, pi = model.predict(count_matrix)
    else:
        mean, dispersion = model.predict(count_matrix)
        pi = None
    
    # Get latent representation
    encoder = Model(inputs=model.input, outputs=model.get_layer('latent_layer').output)
    latent_representation = encoder.predict(count_matrix)
    
    # Update AnnData object with denoised data and latent representation
    adata_denoised = adata.copy()
    adata_denoised.X = mean
    adata_denoised.obsm['X_dca'] = latent_representation
    adata_denoised.obsm['dca_mean'] = mean
    adata_denoised.obsm['dca_dispersion'] = dispersion
    if use_zinb and pi is not None:
        adata_denoised.obsm['dca_dropout'] = pi
    
    # Calculate UMAP on latent representation
    sc.pp.neighbors(adata_denoised, use_rep='X_dca')
    sc.tl.umap(adata_denoised)
    
    # Save denoised data if path provided
    if save_path:
        print(f"Saving denoised data to {save_path}")
        adata_denoised.write(save_path)
    
    # Save model if path provided
    if save_model_path:
        print(f"Saving model weights to {save_model_path}")
        os.makedirs(save_model_path, exist_ok=True)
        
        # Save weights instead of the full model
        model.save_weights(f"{save_model_path}/weights")
        
        # Save model architecture as JSON
        model_json = model.to_json()
        with open(f"{save_model_path}/model_architecture.json", "w") as json_file:
            json_file.write(model_json)
        
        # Save additional model information
        model_info = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
            'use_zinb': use_zinb,
            'dropout_rate': dropout_rate,
            'l1_reg': l1_reg,
            'l2_reg': l2_reg,
            'use_batchnorm': use_batchnorm
        }
        
        import json
        with open(f"{save_model_path}/model_info.json", "w") as json_file:
            json.dump(model_info, json_file)
        
        print(f"Model information and weights saved to {save_model_path}")
    
    return model, adata_denoised



def evaluate_dca_correction(adata, batch_key='batch', cell_type_key=None):
    """
    Evaluate DCA denoising and batch correction.
    
    Parameters:
        adata (AnnData): Annotated data matrix with denoised data
        batch_key (str): Key for batch annotation
        cell_type_key (str): Key for cell type annotation
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    results = {}
    
    # Calculate silhouette scores on original data
    if 'X_pca' not in adata.obsm:
        sc.pp.pca(adata)
    
    batch_sil_pca = silhouette_score(
        adata.obsm['X_pca'],
        adata.obs[batch_key].cat.codes,
        metric='euclidean'
    )
    results['batch_silhouette_pca'] = batch_sil_pca
    
    # Calculate silhouette scores on DCA latent space
    batch_sil_dca = silhouette_score(
        adata.obsm['X_dca'],
        adata.obs[batch_key].cat.codes,
        metric='euclidean'
    )
    results['batch_silhouette_dca'] = batch_sil_dca
    
    # Calculate cell type silhouette scores if available
    if cell_type_key and cell_type_key in adata.obs.columns:
        cell_sil_pca = silhouette_score(
            adata.obsm['X_pca'],
            adata.obs[cell_type_key].cat.codes,
            metric='euclidean'
        )
        results['celltype_silhouette_pca'] = cell_sil_pca
        
        cell_sil_dca = silhouette_score(
            adata.obsm['X_dca'],
            adata.obs[cell_type_key].cat.codes,
            metric='euclidean'
        )
        results['celltype_silhouette_dca'] = cell_sil_dca
    
    # Print results
    print("\nBatch correction evaluation:")
    print(f"Batch silhouette score (PCA): {batch_sil_pca:.4f}")
    print(f"Batch silhouette score (DCA): {batch_sil_dca:.4f}")
    
    if cell_type_key and cell_type_key in adata.obs.columns:
        print(f"Cell type silhouette score (PCA): {cell_sil_pca:.4f}")
        print(f"Cell type silhouette score (DCA): {cell_sil_dca:.4f}")
    
    # Check if batch effect is reduced
    batch_effect_reduction = batch_sil_pca - batch_sil_dca
    results['batch_effect_reduction'] = batch_effect_reduction
    
    if batch_sil_dca < batch_sil_pca:
        print("\n✓ Batch effect reduced (lower batch silhouette score)")
    else:
        print("\n✗ Batch effect may not be sufficiently reduced")
    
    if cell_type_key and cell_type_key in adata.obs.columns:
        if results['celltype_silhouette_dca'] >= results['celltype_silhouette_pca']:
            print("✓ Cell type separation preserved or improved")
        else:
            print("✗ Cell type separation may be compromised")
    
    return results


def visualize_dca_results(adata, batch_key='batch', cell_type_key=None, save_dir='dca_results/figures'):
    """
    Visualize DCA results.
    
    Parameters:
        adata (AnnData): Annotated data matrix with denoised data
        batch_key (str): Key for batch annotation
        cell_type_key (str): Key for cell type annotation
        save_dir (str): Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Before vs. after correction plot with batches
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before (PCA-based visualization)
    if 'X_pca' not in adata.obsm:
        sc.pp.pca(adata)
    if 'X_umap' not in adata.obsm:
        sc.pp.neighbors(adata, use_rep='X_pca')
        sc.tl.umap(adata)
    
    sc.pl.umap(adata, color=batch_key, ax=ax1, show=False, title='Before correction (PCA-based)')
    
    # After (DCA-based visualization)
    # Use the UMAP that was computed on the DCA latent space
    sc.pl.umap(adata, color=batch_key, ax=ax2, show=False, title='After correction (DCA)')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/batch_correction_comparison.png", dpi=300)
    
    # Cell type visualization if available
    if cell_type_key and cell_type_key in adata.obs.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        sc.pl.umap(adata, color=cell_type_key, ax=ax1, show=False, title='Before correction (PCA-based)')
        sc.pl.umap(adata, color=cell_type_key, ax=ax2, show=False, title='After correction (DCA)')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/celltype_visualization.png", dpi=300)
    
    # Create a scatter plot of the latent space
    if 'X_dca' in adata.obsm:
        latent_dim = adata.obsm['X_dca'].shape[1]
        
        if latent_dim >= 2:
            # Plot first 2 dimensions of latent space
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Color by batch
            for batch_id, batch_name in enumerate(adata.obs[batch_key].cat.categories):
                mask = adata.obs[batch_key] == batch_name
                ax1.scatter(
                    adata.obsm['X_dca'][mask, 0],
                    adata.obsm['X_dca'][mask, 1],
                    label=batch_name,
                    alpha=0.7,
                    s=10
                )
            
            ax1.set_title('DCA Latent Space (colored by batch)')
            ax1.set_xlabel('DCA 1')
            ax1.set_ylabel('DCA 2')
            ax1.legend()
            
            # Color by cell type if available
            if cell_type_key and cell_type_key in adata.obs.columns:
                for cell_id, cell_name in enumerate(adata.obs[cell_type_key].cat.categories):
                    mask = adata.obs[cell_type_key] == cell_name
                    ax2.scatter(
                        adata.obsm['X_dca'][mask, 0],
                        adata.obsm['X_dca'][mask, 1],
                        label=cell_name,
                        alpha=0.7,
                        s=10
                    )
                
                ax2.set_title('DCA Latent Space (colored by cell type)')
                ax2.set_xlabel('DCA 1')
                ax2.set_ylabel('DCA 2')
                ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/latent_space_visualization.png", dpi=300)
    
    # Visualize distribution of denoised vs. original counts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original data
    if sparse.issparse(adata.layers['counts']):
        original_counts = adata.layers['counts'].toarray()
    else:
        original_counts = adata.layers['counts']
    
    # Flatten and remove zeros for visualization
    original_flat = original_counts.flatten()
    original_flat = original_flat[original_flat > 0]
    
    # Denoised data
    denoised_flat = adata.X.flatten()
    denoised_flat = denoised_flat[denoised_flat > 0]
    
    # Plot histograms
    ax1.hist(np.log1p(original_flat), bins=50, alpha=0.7, label='Original')
    ax1.set_title('Distribution of Original Counts (log1p)')
    ax1.set_xlabel('Log1p(Counts)')
    ax1.set_ylabel('Frequency')
    
    ax2.hist(np.log1p(denoised_flat), bins=50, alpha=0.7, label='Denoised')
    ax2.set_title('Distribution of Denoised Counts (log1p)')
    ax2.set_xlabel('Log1p(Counts)')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/count_distribution_comparison.png", dpi=300)
    
    # Visualize the dropout probabilities if available
    if 'dca_dropout' in adata.obsm:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        dropout_probs = adata.obsm['dca_dropout'].flatten()
        ax.hist(dropout_probs, bins=50, alpha=0.7)
        ax.set_title('Distribution of Dropout Probabilities')
        ax.set_xlabel('Dropout Probability')
        ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/dropout_probabilities.png", dpi=300)


def main(use_optuna=True, n_trials=20, timeout=None):
    """
    Main function to demonstrate DCA on PBMC data.
    
    Parameters:
        use_optuna (bool): Whether to use Optuna for hyperparameter optimization
        n_trials (int): Number of optimization trials if using Optuna
        timeout (int): Timeout in seconds for optimization if using Optuna
    """
    print("Starting DCA analysis...")
    
    # Set GPU memory growth to avoid OOM errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    
    # Create results directory
    os.makedirs("dca_results", exist_ok=True)
    
    # Load PBMC dataset
    pbmc_data = load_pbmc_dataset()
    if pbmc_data is None:
        return
    
    # Perform initial check for infinite values
    if sparse.issparse(pbmc_data.X):
        has_inf = np.any(~np.isfinite(pbmc_data.X.data))
    else:
        has_inf = np.any(~np.isfinite(pbmc_data.X))
    
    if has_inf:
        print("Warning: Original data contains infinite values. They will be handled during preprocessing.")
    
    # Preprocess data
    pbmc_preprocessed = preprocess_data(
        pbmc_data,
        n_top_genes=2000,
        min_genes=200,
        min_cells=3
    )
    
    # Save preprocessed data
    pbmc_preprocessed.write("dca_results/pbmc_preprocessed.h5ad")
    
    if use_optuna:
        print("Using Optuna for hyperparameter optimization...")
        
        # Run Optuna optimization
        best_params = optimize_dca_hyperparameters(
            pbmc_preprocessed,
            n_trials=n_trials,
            timeout=timeout,
            batch_key='batch',
            study_name="dca_pbmc_optimization",
            save_best_params=True,
            save_path="dca_results/best_params.json"
        )
        
        # Train with optimized parameters
        model, pbmc_denoised = train_dca_with_best_params(
            pbmc_preprocessed,
            best_params,
            batch_key='batch',
            use_zinb=True,
            save_path="dca_results/pbmc_denoised_optimized.h5ad",
            save_model_path="dca_results/dca_model_optimized"
        )
    else:
        # Train with default parameters
        model, pbmc_denoised = train_dca(
            pbmc_preprocessed,
            batch_key='batch',
            latent_dim=32,
            hidden_dim=128,
            learning_rate=1e-3,
            epochs=300,
            batch_size=128,
            use_zinb=True,
            save_path="dca_results/pbmc_denoised.h5ad",
            save_model_path="dca_results/dca_model"
        )
    
    # Get label key
    label_key = 'str_labels' if 'str_labels' in pbmc_denoised.obs.columns else 'labels'
    
    # Evaluate batch correction
    evaluation_results = evaluate_dca_correction(
        pbmc_denoised,
        batch_key='batch',
        cell_type_key=label_key
    )
    
    # Save evaluation results
    pd.DataFrame([evaluation_results]).to_csv("dca_results/evaluation_results.csv", index=False)
    
    # Visualize results
    visualize_dca_results(
        pbmc_denoised,
        batch_key='batch',
        cell_type_key=label_key,
        save_dir='dca_results/figures'
    )
    
    print("\nDCA analysis complete! Results saved to 'dca_results' directory.")
    print("You can view the results using the Streamlit app by pointing it to the saved h5ad files.")
    
    if use_optuna:
        print("\nOptimization results:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print("\nCheck dca_results/optuna_plots/ for visualization of optimization results")
    
    return pbmc_denoised


if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='DCA for scRNA-seq data')
    parser.add_argument('--no-optuna', action='store_true', help='Disable Optuna optimization')
    parser.add_argument('--trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout for Optuna in seconds')
    parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension (if not using Optuna)')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension (if not using Optuna)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate (if not using Optuna)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (if not using Optuna)')
    parser.add_argument('--no-zinb', action='store_true', help='Disable Zero-Inflated Negative Binomial loss')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run main function
    main(
        use_optuna=not args.no_optuna,
        n_trials=args.trials,
        timeout=args.timeout
    )

