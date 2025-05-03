# Visium Spatial Transcriptomics Analysis Pipeline

A comprehensive Snakemake pipeline for analyzing 10x Genomics Visium spatial transcriptomics data, with support for other spatial platforms including Slide-seq and Stereo-seq.

## Overview

This pipeline provides end-to-end analysis of spatial transcriptomics data, from raw FASTQ files to final visualizations and reports. It includes quality control, preprocessing, clustering, cell type deconvolution, pathway analysis, and more.

### Key Features

- **Comprehensive Analysis**: Complete workflow from raw data to publication-ready figures
- **Dual Analysis Approaches**: Both R/Seurat and Python/Scanpy pipelines
- **Cell Type Deconvolution**: SPOTlight-based estimation of cellular composition
- **Pathway Analysis**: GO, KEGG, and Reactome enrichment analysis
- **Interactive Reports**: HTML reports with integrated visualizations
- **Multi-Sample Support**: Handles single or multiple samples with automatic integration
- **Platform Flexibility**: Adaptable for Visium, Slide-seq, and Stereo-seq data

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Pipeline Overview](#pipeline-overview)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Output Structure](#output-structure)
7. [Platform Support](#platform-support)
8. [Troubleshooting](#troubleshooting)
9. [Citation](#citation)

## Installation

### Prerequisites

1. **Snakemake** (>= 7.0.0)

2. **Space Ranger** (for Visium data)
```bash
# Download from 10x Genomics website
curl -o spaceranger-2.1.0.tar.gz "https://cf.10xgenomics.com/..."
tar -xzf spaceranger-2.1.0.tar.gz
```

3. **R packages** (>= 4.1.0)
```r
# Required packages for R analysis
install.packages(c("Seurat", "tidyverse", "patchwork", "viridis", "pheatmap"))
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install(c("clusterProfiler", "pathview", "org.Hs.eg.db", "enrichplot", "ReactomePA"))
```

4. **Python packages** (>= 3.8)
```bash
# Install packages
pip install scanpy squidpy numpy pandas matplotlib seaborn plotly
```

## Quick Start

1. **Prepare your data structure**:
```
project/
├── data/
│   ├── sample1/
│   │   ├── fastq/
│   │   ├── tissue_image.tif
│   │   └── tissue_positions_list.csv
│   └── sample2/
│       ├── fastq/
│       ├── tissue_image.tif
│       └── tissue_positions_list.csv
├── reference/
│   ├── refdata-gex-GRCh38-2024-A/
│   └── slide_file.gpr
└── config.yaml
```

2. **Configure the pipeline**:
```bash
# Copy and edit the configuration template
cp config_template.yaml config.yaml
vim config.yaml  # Edit paths and parameters
```

3. **Run the pipeline**:
```bash
# Dry run to check configuration
snakemake -n

# Run with 16 cores
snakemake --cores 16

# Run with Slurm cluster
snakemake --profile slurm
```

## Pipeline Overview

The pipeline consists of the following major steps:

1. **Quality Control**: FastQC and MultiQC for raw FASTQ files
2. **Space Ranger Processing**: Read alignment and feature counting
3. **Data Integration**: Combining multiple samples (if applicable)
4. **Clustering Analysis**: Identifying spatial domains
5. **Differential Expression**: Finding marker genes for each cluster
6. **Spatial Variable Gene Analysis**: Detecting spatially patterned expression
7. **Cell Type Deconvolution**: Estimating cellular composition of each spot
8. **Pathway Analysis**: Functional enrichment of spatial domains
9. **Report Generation**: Creating comprehensive HTML reports

### Pipeline Flowchart

```
FASTQ files → FastQC → Space Ranger → Feature Matrix
                                            ↓
                          Seurat Analysis ⟷ Scanpy Analysis
                                    ↓           ↓
                         Spatial Clustering   UMAP/PCA
                                    ↓           ↓
                            Marker Genes → Pathway Analysis
                                    ↓
                        Cell Type Deconvolution → Final Report
```


### Analysis Parameters
```yaml
analysis:
  qc:
    min_genes: 200
    max_genes: 6000
    max_mt_percent: 20
  clustering:
    resolutions: [0.2, 0.4, 0.6, 0.8, 1.0]
  pathway:
    organism: "human"
```

## Usage

### Running Specific Steps

```bash
# Run only QC
snakemake -R fastqc multiqc

# Run only Space Ranger
snakemake spaceranger_count

# Generate only reports
snakemake final_report
```

### Running with Different Platforms

```bash
# For Slide-seq data
snakemake --config platform=slideseq

# For Stereo-seq data
snakemake --config platform=stereo
```

### Advanced Options

```bash
# Run with specific resources
snakemake --resources mem_mb=100000

# Run with job grouping
snakemake --group-by=rule

# Run with prioritization
snakemake --prioritize process_spatial_data
```

## Output Structure

```
results/
├── qc/
│   ├── sample1_fastqc.html
│   └── multiqc_report.html
├── spaceranger/
│   └── sample1/
│       └── outs/
├── seurat/
│   ├── seurat_analysis.rds
│   └── spatial_clusters.pdf
├── scanpy/
│   ├── scanpy_analysis.h5ad
│   └── spatial_markers.csv
└── final_results/
    ├── spatial_clusters.pdf
    ├── deconvolution_results.csv
    ├── pathway_analysis.csv
    └── final_report.html
```

## Platform Support

The pipeline supports multiple spatial transcriptomics platforms:

### 10x Genomics Visium (Default)
- Uses Space Ranger for preprocessing
- Standard 55μm spot size
- Optimized for tissue sections

### Slide-seq/Slide-seq V2
- Uses custom preprocessing scripts
- Higher resolution (10μm beads)
- Adjusted neighborhood parameters

### BGI Stereo-seq
- Uses spateo/stereopy tools
- Ultra-high resolution (~1μm)
- Includes binning for multi-scale analysis

To specify a platform:
```bash
snakemake --config platform=slideseq
```

## Troubleshooting

### Common Issues

1. **Space Ranger Memory Errors**
   ```bash
   # Increase memory allocation
   snakemake --resources mem_mb=64000
   ```

2. **R Package Conflicts**
   ```bash
   # Create isolated R environment
   conda create -n r-env r-base=4.1.0
   ```

3. **JAVA Memory Issues**
   ```bash
   export _JAVA_OPTIONS="-Xmx8g"
   ```


And the relevant tools used in the pipeline:
- Seurat: Butler et al. (2018) Nature Biotechnology
- Scanpy: Wolf et al. (2018) Genome Biology
- SPOTlight: Elosua-Bayes et al. (2021) Nucleic Acids Research
- Space Ranger: 10x Genomics


## Contributors

Jiyang Jiang,  jiyang.jiang@gmail.com

