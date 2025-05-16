
```
# The Weighted Composite Score Explained

## Score Calculation
```python
score = (
    w["var_exp"] * s_var
    + w["sil_pca"] * s_sil_pca
    + w["lisi"] * s_lisi
    + w["sil_dca"] * s_sil_dca
    + w["dca_loss"] * s_dca_loss
)
```

## Score Components

### 1. Variance Explained (`s_var` - 20% weight)
- **Measures**: Amount of data variance captured by principal components  
- **Importance**: Higher values indicate better preservation of original data information  
- **Scaling**: 0-1 scale (1 = 100% variance explained)  
- **Direction**: Higher is better  
- **Weight**: 20% - Important but not dominant to avoid capturing technical variation  

### 2. PCA Silhouette Score (`s_sil_pca` - 20% weight)
- **Measures**: Batch separation in PCA space  
- **Importance**: Lower values indicate better batch mixing  
- **Scaling**: Transformed from [-1,1] to [0,1] (0 = better mixing)  
- **Direction**: Lower is better (scale flipped for scoring)  
- **Weight**: 20% - Baseline comparison for DCA model  

### 3. LISI Score (`s_lisi` - 10% weight)
- **Measures**: Local batch mixing via Inverse Simpson's Index  
- **Importance**: Higher raw values = better mixing  
- **Scaling**: Transformed to 1/(1+LISI) (lower = better)  
- **Direction**: Lower transformed value is better  
- **Weight**: 10% - Focuses on local neighborhood structures  

### 4. DCA Silhouette Score (`s_sil_dca` - 25% weight)
- **Measures**: Batch separation in DCA latent space  
- **Importance**: Lower values indicate better mixing  
- **Scaling**: Transformed from [-1,1] to [0,1] (0 = better mixing)  
- **Direction**: Lower is better (scale flipped)  
- **Weight**: 25% - Directly measures DCA latent space quality  

### 5. DCA Loss Value (`s_dca_loss` - 25% weight)
- **Measures**: Data reconstruction quality (ZINB loss)  
- **Importance**: Lower loss = better modeling  
- **Scaling**: Sigmoid transform 1/(1+exp(loss/100)) → [0,1]  
- **Direction**: Higher transformed value is better  
- **Weight**: 25% - Directly measures model performance  

---

## The Balancing Act
The weighted sum balances competing objectives:
- **Data representation quality** (variance explained, DCA loss)  
- **Batch effect removal** (silhouette scores, LISI)  
- **Biological signal preservation** (implicit in all metrics)  

Default weights prioritize reconstruction accuracy and batch mixing equally, with supporting metrics.

---

## Weight Modification Strategies

### 1. Adjusting Based on Research Priorities
**Default Weights**:
```python
DEFAULT_WEIGHTS = {
    "var_exp": 0.2,
    "sil_pca": 0.2,
    "lisi": 0.1,
    "sil_dca": 0.25,
    "dca_loss": 0.25,
}
```

**Scenario: Prioritize Batch Correction**
```python
DEFAULT_WEIGHTS = {
    "var_exp": 0.1,     # Reduced
    "sil_pca": 0.15,    # Reduced
    "lisi": 0.25,       # Increased
    "sil_dca": 0.35,    # Increased
    "dca_loss": 0.15,   # Reduced
}
```

**Scenario: Preserve Biological Signal**
```python
DEFAULT_WEIGHTS = {
    "var_exp": 0.3,     # Increased
    "sil_pca": 0.1,     # Reduced
    "lisi": 0.05,       # Reduced
    "sil_dca": 0.15,    # Reduced
    "dca_loss": 0.4,    # Increased
}
```

**Scenario: Highly Imbalanced Batches**
```python
DEFAULT_WEIGHTS = {
    "var_exp": 0.15,
    "sil_pca": 0.1,     # Reduced
    "lisi": 0.35,       # Significantly increased
    "sil_dca": 0.2,
    "dca_loss": 0.2,
}
```

### 2. Dynamic or Dataset-Specific Weights
**Command-line Example**:
```python
ap.add_argument("--var_exp_weight", type=float, default=0.2)
ap.add_argument("--sil_pca_weight", type=float, default=0.2)
# etc.
```

### 3. Automatic Dataset-Dependent Weighting
```python
def calculate_dataset_weights(adata):
    """Determine optimal weights based on dataset characteristics."""
    weights = DEFAULT_WEIGHTS.copy()
    
    # Heterogeneous cell types → emphasize biological signal
    if len(adata.obs["cell_type"].unique()) > 10:
        weights["var_exp"] += 0.1
        weights["dca_loss"] += 0.1
        weights["sil_dca"] -= 0.1
        weights["lisi"] -= 0.1
    
    # Many batches → emphasize batch correction
    if len(adata.obs["batch"].unique()) > 3:
        weights["lisi"] += 0.1
        weights["sil_dca"] += 0.1
        weights["var_exp"] -= 0.1
        weights["dca_loss"] -= 0.1
        
    # Normalize weights to sum to 1
    total = sum(weights.values())
    for k in weights:
        weights[k] /= total
        
    return weights
```

```

