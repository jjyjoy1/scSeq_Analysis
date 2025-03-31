Single-cell CRISPR analysis pipelines share many steps with routine single-cell RNA-seq (scRNA-seq) pipelines, but there are several key differences specifically related to handling CRISPR perturbations.

Here’s how a **single-cell CRISPR pipeline differs from a routine scRNA-seq pipeline**, step-by-step:

---

### Steps Common to Both Pipelines:

- **Demultiplexing and Read Alignment**
- **Cell Barcode Identification**
- **Gene Expression Quantification**
- **Quality Control (QC)**
- **Normalization**
- **Batch Correction & Integration**
- **Dimensionality Reduction (e.g., PCA, UMAP)**
- **Clustering and Cell-State Identification**
- **Differential Expression (DE) Analysis**
- **Visualization of Results**

---

### Steps Unique to Single-Cell CRISPR Pipelines:

#### 1. **sgRNA Read Processing & Assignment**  
- CRISPR pipelines require specific handling of reads containing the guide RNA sequences (sgRNAs), which identify the perturbations applied to each cell.
- sgRNAs are assigned to individual cells based on a separate read or library (feature barcode library in platforms like 10x Genomics).
- Routine scRNA-seq experiments don't involve sgRNA sequences, so this step is completely absent.

#### 2. **Perturbation Effect Estimation**  
- Routine scRNA-seq identifies cell types and gene expression differences across biological conditions or cell states, but doesn’t directly quantify "perturbation effects."
- Single-cell CRISPR explicitly estimates how perturbations affect cell states or gene expression, often using regression models or sophisticated inference methods (e.g., **SCEPTRE**, **scMAGeCK**, **crisprQTL**).

#### 3. **Perturbation-Guided Differential Expression Analysis**  
- Standard scRNA-seq DE analysis typically compares defined biological groups or clusters.
- Single-cell CRISPR DE analysis explicitly compares cells carrying a particular perturbation (e.g., cells with gene X knocked out) versus control (non-targeted or cells without perturbation).

#### 4. **Inference of Gene Regulatory Networks (GRNs)**  
- Single-cell CRISPR data enables systematic perturbation-based inference of gene regulatory networks, allowing for the assignment of regulatory relationships.
- Routine scRNA-seq typically relies on correlation-based methods rather than causal perturbation-driven inferences.

---

### Summary of Major Differences:

| Pipeline Step                   | Routine scRNA-seq                     | Single-cell CRISPR                      |
|---------------------------------|---------------------------------------|-----------------------------------------|
| **sgRNA Assignment**            | Not applicable                        | Essential; dedicated sgRNA assignment  |
| **Perturbation effect**         | Not applicable                        | Explicitly modeled using perturbations |
| **Differential Expression**     | Between groups/states                 | Between cells with vs. without sgRNA perturbations |
| **Gene Regulatory Network**     | Correlation or co-expression inference | Causal inference using perturbation data|

---

### Practical Implications:
- Single-cell CRISPR analysis involves a dual-layered complexity:
  - **Transcriptomic analysis** similar to standard scRNA-seq.
  - **Genotype (perturbation) analysis** to establish causal relationships between perturbed genes and observed phenotypes.

---

These differences highlight how single-cell CRISPR pipelines expand upon standard scRNA-seq analysis, adding steps that explicitly leverage perturbation data for deeper biological interpretation.

