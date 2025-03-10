# Single-cell RNA sequencing analysis pipeline using Seurat and Harmony
# This R script provides a comprehensive workflow for scRNA-seq analysis
# including batch integration, normalization, clustering, and differential expression

# Install required packages if needed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

required_packages <- c("Seurat", "harmony", "SeuratWrappers", "ggplot2", "dplyr", 
                       "patchwork", "glmGamPoi", "DESeq2", "scater", "sctransform",
                       "MAST", "pheatmap", "viridis", "RColorBrewer")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    if (pkg %in% c("DESeq2", "MAST", "scater", "glmGamPoi")) {
      BiocManager::install(pkg)
    } else {
      install.packages(pkg)
    }
  }
}

# Load libraries
library(Seurat)
library(harmony)
library(SeuratWrappers)
library(ggplot2)
library(dplyr)
library(patchwork)
library(MAST)
library(RColorBrewer)
library(viridis)
library(pheatmap)
library(glmGamPoi)

#' Create a Seurat object from raw counts
#'
#' @param counts_matrix A matrix of raw UMI counts (genes x cells)
#' @param meta_data A data frame with cell metadata (optional)
#' @param min_cells Minimum number of cells expressing a gene
#' @param min_features Minimum number of genes detected in a cell
#' @param project_name Name for the Seurat project
#'
#' @return A Seurat object
create_seurat_object <- function(counts_matrix, meta_data = NULL, 
                                min_cells = 3, min_features = 200,
                                project_name = "scRNA_project") {
  
  seurat_obj <- CreateSeuratObject(counts = counts_matrix,
                                   meta.data = meta_data,
                                   project = project_name,
                                   min.cells = min_cells, 
                                   min.features = min_features)
  
  return(seurat_obj)
}

#' Load and merge multiple scRNA-seq datasets
#'
#' @param file_paths List of file paths to count matrices
#' @param file_format Format of input files (e.g., "10X_h5", "10X_mtx", "csv")
#' @param batch_ids Vector of batch identifiers corresponding to each file
#' @param meta_data_paths List of file paths to metadata files (optional)
#'
#' @return A merged Seurat object with batch information
load_and_merge_datasets <- function(file_paths, file_format = "10X_h5",
                                   batch_ids = NULL, meta_data_paths = NULL) {
  
  if (is.null(batch_ids)) {
    batch_ids <- paste0("batch_", 1:length(file_paths))
  }
  
  # Initialize list to store Seurat objects
  seurat_objects <- list()
  
  # Load each dataset
  for (i in seq_along(file_paths)) {
    file_path <- file_paths[i]
    batch_id <- batch_ids[i]
    
    message(paste0("Loading dataset: ", batch_id))
    
    # Load counts based on format
    if (file_format == "10X_h5") {
      counts <- Read10X_h5(file_path)
    } else if (file_format == "10X_mtx") {
      counts <- Read10X(file_path)
    } else if (file_format == "csv") {
      counts <- read.csv(file_path, row.names = 1)
    } else {
      stop("Unsupported file format. Use '10X_h5', '10X_mtx', or 'csv'.")
    }
    
    # Load metadata if provided
    meta_data <- NULL
    if (!is.null(meta_data_paths) && i <= length(meta_data_paths)) {
      if (!is.na(meta_data_paths[i])) {
        meta_data <- read.csv(meta_data_paths[i], row.names = 1)
      }
    }
    
    # Create Seurat object
    seurat_obj <- create_seurat_object(counts, meta_data, project_name = batch_id)
    
    # Add batch information
    seurat_obj$batch <- batch_id
    
    # Store in list
    seurat_objects[[i]] <- seurat_obj
  }
  
  # Merge all datasets
  if (length(seurat_objects) > 1) {
    merged_obj <- merge(seurat_objects[[1]], y = seurat_objects[2:length(seurat_objects)], 
                         add.cell.ids = batch_ids)
  } else {
    merged_obj <- seurat_objects[[1]]
  }
  
  return(merged_obj)
}

#' Calculate QC metrics and filter low quality cells
#'
#' @param seurat_obj Seurat object
#' @param min_features Minimum number of genes per cell
#' @param max_features Maximum number of genes per cell (for doublet filtering)
#' @param percent_mt_cutoff Maximum percentage of mitochondrial reads
#' @param output_dir Directory to save plots
#'
#' @return Filtered Seurat object
perform_qc_and_filtering <- function(seurat_obj, min_features = 200, 
                                   max_features = 6000, percent_mt_cutoff = 20,
                                   output_dir = "./results") {
  
  # Create output directory if it doesn't exist
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Calculate mitochondrial percentage
  seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^MT-")
  
  # Generate QC plots before filtering
  p1 <- VlnPlot(seurat_obj, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), 
                ncol = 3, pt.size = 0.1) + 
        theme(legend.position = "none")
  
  p2 <- FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "nFeature_RNA") +
        geom_smooth(method = "lm") +
        theme(legend.position = "none")
  
  p3 <- FeatureScatter(seurat_obj, feature1 = "nFeature_RNA", feature2 = "percent.mt") +
        theme(legend.position = "none")
  
  qc_plot <- p1 / (p2 | p3)
  
  # Save QC plot
  ggsave(paste0(output_dir, "/qc_before_filtering.png"), qc_plot, width = 12, height = 10, dpi = 100)
  
  # Cell counts before filtering
  message(paste0("Cells before filtering: ", ncol(seurat_obj)))
  
  # Filter cells based on QC metrics
  seurat_obj <- subset(seurat_obj, 
                       subset = nFeature_RNA > min_features & 
                       nFeature_RNA < max_features & 
                       percent.mt < percent_mt_cutoff)
  
  # Cell counts after filtering
  message(paste0("Cells after filtering: ", ncol(seurat_obj)))
  
  # Generate QC plots after filtering
  p1 <- VlnPlot(seurat_obj, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), 
                ncol = 3, pt.size = 0.1) + 
        theme(legend.position = "none")
  
  p2 <- FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "nFeature_RNA") +
        geom_smooth(method = "lm") +
        theme(legend.position = "none")
  
  p3 <- FeatureScatter(seurat_obj, feature1 = "nFeature_RNA", feature2 = "percent.mt") +
        theme(legend.position = "none")
  
  qc_plot <- p1 / (p2 | p3)
  
  # Save QC plot after filtering
  ggsave(paste0(output_dir, "/qc_after_filtering.png"), qc_plot, width = 12, height = 10, dpi = 100)
  
  return(seurat_obj)
}

#' Normalize and identify variable features
#'
#' @param seurat_obj Seurat object
#' @param normalization_method Method for normalization (LogNormalize or SCT)
#' @param n_variable_features Number of variable features to select
#' @param scale_factor Scale factor for normalization
#' @param output_dir Directory to save plots
#'
#' @return Normalized Seurat object with variable features identified
normalize_and_scale <- function(seurat_obj, normalization_method = "LogNormalize",
                              n_variable_features = 2000, scale_factor = 10000,
                              output_dir = "./results") {
  
  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Normalize data
  if (normalization_method == "SCT") {
    # SCTransform normalization
    seurat_obj <- SCTransform(seurat_obj, verbose = TRUE, 
                             variable.features.n = n_variable_features,
                             conserve.memory = TRUE,
                             method = "glmGamPoi")
  } else {
    # Standard log-normalization
    seurat_obj <- NormalizeData(seurat_obj, normalization.method = normalization_method, 
                               scale.factor = scale_factor)
    
    # Find variable features
    seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", 
                                     nfeatures = n_variable_features)
    
    # Scale data
    all_genes <- rownames(seurat_obj)
    seurat_obj <- ScaleData(seurat_obj, features = all_genes)
  }
  
  # Plot variable features
  var_features_plot <- VariableFeaturePlot(seurat_obj)
  
  # Label top 10 variable features
  top10 <- head(VariableFeatures(seurat_obj), 10)
  var_features_plot <- LabelPoints(plot = var_features_plot, points = top10, repel = TRUE)
  
  # Save variable features plot
  ggsave(paste0(output_dir, "/variable_features.png"), var_features_plot, width = 10, height = 8, dpi = 100)
  
  return(seurat_obj)
}

#' Run dimensionality reduction using PCA
#'
#' @param seurat_obj Seurat object
#' @param n_pcs Number of principal components to compute
#' @param output_dir Directory to save plots
#'
#' @return Seurat object with PCA computation
run_pca <- function(seurat_obj, n_pcs = 50, output_dir = "./results") {
  
  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Run PCA
  seurat_obj <- RunPCA(seurat_obj, npcs = n_pcs, verbose = FALSE)
  
  # Plot PCA results
  pca_plot <- DimPlot(seurat_obj, reduction = "pca", group.by = "batch")
  ggsave(paste0(output_dir, "/pca.png"), pca_plot, width = 10, height = 8, dpi = 100)
  
  # Elbow plot to determine number of PCs to use
  elbow_plot <- ElbowPlot(seurat_obj, ndims = n_pcs)
  ggsave(paste0(output_dir, "/elbow_plot.png"), elbow_plot, width = 8, height = 6, dpi = 100)
  
  # Heatmap of top PCs
  pdf(paste0(output_dir, "/pca_heatmap.pdf"), width = 12, height = 20)
  DimHeatmap(seurat_obj, dims = 1:15, cells = 500, balanced = TRUE)
  dev.off()
  
  return(seurat_obj)
}

#' Integrate datasets using Harmony for batch correction
#'
#' @param seurat_obj Seurat object with multiple batches
#' @param batch_var Variable name in metadata containing batch information
#' @param n_pcs Number of PCs to use for integration
#' @param output_dir Directory to save plots
#'
#' @return Seurat object with integrated embeddings
run_harmony_integration <- function(seurat_obj, batch_var = "batch", 
                                  n_pcs = 30, output_dir = "./results") {
  
  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Check if PCA has been run
  if (!"pca" %in% names(seurat_obj@reductions)) {
    seurat_obj <- run_pca(seurat_obj, n_pcs = n_pcs, output_dir = output_dir)
  }
  
  # Run Harmony for batch integration
  seurat_obj <- RunHarmony(seurat_obj, group.by.vars = batch_var, 
                         assay.use = ifelse("SCT" %in% names(seurat_obj@assays), "SCT", "RNA"),
                         reduction = "pca", dims.use = 1:n_pcs, 
                         plot_convergence = TRUE)
  
  # Plot UMAP based on harmony embeddings
  seurat_obj <- RunUMAP(seurat_obj, reduction = "harmony", dims = 1:n_pcs)
  
  # Compare batch effects before and after correction
  p1 <- DimPlot(seurat_obj, reduction = "umap", group.by = batch_var, pt.size = 0.5) +
        ggtitle("After Harmony Integration") +
        theme(legend.position = "right")
  
  # Run UMAP on original PCA for comparison
  seurat_obj <- RunUMAP(seurat_obj, reduction = "pca", dims = 1:n_pcs, reduction.name = "umap.pca")
  
  p2 <- DimPlot(seurat_obj, reduction = "umap.pca", group.by = batch_var, pt.size = 0.5) +
        ggtitle("Before Harmony Integration") +
        theme(legend.position = "right")
  
  # Combine plots and save
  comparison_plot <- p1 | p2
  ggsave(paste0(output_dir, "/harmony_integration.png"), comparison_plot, width = 14, height = 7, dpi = 100)
  
  # Return to harmony-based UMAP
  seurat_obj <- RunUMAP(seurat_obj, reduction = "harmony", dims = 1:n_pcs, reduction.name = "umap")
  
  return(seurat_obj)
}

#' Run clustering analysis
#'
#' @param seurat_obj Seurat object
#' @param reduction Dimensionality reduction to use (pca or harmony)
#' @param resolution Resolution parameter for clustering
#' @param n_dims Number of dimensions to use
#' @param output_dir Directory to save plots
#'
#' @return Seurat object with clusters identified
run_clustering <- function(seurat_obj, reduction = "harmony", resolution = 0.8, 
                         n_dims = 30, output_dir = "./results") {
  
  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Ensure UMAP has been run
  if (!"umap" %in% names(seurat_obj@reductions)) {
    seurat_obj <- RunUMAP(seurat_obj, reduction = reduction, dims = 1:n_dims)
  }
  
  # Find neighbors
  seurat_obj <- FindNeighbors(seurat_obj, reduction = reduction, dims = 1:n_dims)
  
  # Find clusters at specified resolution
  seurat_obj <- FindClusters(seurat_obj, resolution = resolution)
  
  # Plot clusters
  umap_clusters <- DimPlot(seurat_obj, reduction = "umap", label = TRUE, label.size = 5, pt.size = 0.5) +
                  ggtitle(paste0("Clustering at Resolution ", resolution))
  
  ggsave(paste0(output_dir, "/clusters_resolution_", resolution, ".png"), 
         umap_clusters, width = 10, height = 8, dpi = 100)
  
  # Plot batches vs clusters
  p1 <- DimPlot(seurat_obj, reduction = "umap", group.by = "batch", pt.size = 0.5) +
        ggtitle("Batches") +
        theme(legend.position = "right")
  
  p2 <- DimPlot(seurat_obj, reduction = "umap", group.by = "seurat_clusters", 
                label = TRUE, label.size = 5, pt.size = 0.5) +
        ggtitle("Clusters") +
        theme(legend.position = "none")
  
  batches_vs_clusters <- p1 | p2
  ggsave(paste0(output_dir, "/batches_vs_clusters.png"), 
         batches_vs_clusters, width = 14, height = 7, dpi = 100)
  
  return(seurat_obj)
}

#' Find marker genes for each cluster
#'
#' @param seurat_obj Seurat object with clusters identified
#' @param min_pct Minimum percentage of cells in either population expressing the gene
#' @param logfc_threshold Minimum log fold-change threshold
#' @param test_use Statistical test to use (e.g., "wilcox", "MAST", "DESeq2")
#' @param output_dir Directory to save results
#'
#' @return Seurat object with marker genes identified and a list of markers
find_markers <- function(seurat_obj, min_pct = 0.25, logfc_threshold = 0.25,
                       test_use = "wilcox", output_dir = "./results") {
  
  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(paste0(output_dir, "/markers"), showWarnings = FALSE, recursive = TRUE)
  
  # Initialize list to store markers
  all_markers <- list()
  
  # Get cluster IDs
  clusters <- unique(seurat_obj$seurat_clusters)
  
  # Find markers for each cluster compared to all other clusters
  all_markers_df <- FindAllMarkers(seurat_obj, 
                                 only.pos = TRUE,
                                 min.pct = min_pct,
                                 logfc.threshold = logfc_threshold,
                                 test.use = test_use)
  
  # Save all markers to CSV
  write.csv(all_markers_df, file = paste0(output_dir, "/all_markers.csv"), row.names = FALSE)
  
  # Top markers for each cluster
  top_markers <- all_markers_df %>%
                group_by(cluster) %>%
                top_n(n = 20, wt = avg_log2FC)
  
  # Generate heatmap of top markers
  top_genes_per_cluster <- top_markers %>%
                          group_by(cluster) %>%
                          top_n(n = 5, wt = avg_log2FC) %>%
                          pull(gene)
  
  # Create heatmap using scaled data
  seurat_obj <- ScaleData(seurat_obj, features = unique(top_genes_per_cluster))
  
  # Generate and save heatmap
  pdf(paste0(output_dir, "/top_markers_heatmap.pdf"), width = 14, height = 12)
  DoHeatmap(seurat_obj, features = unique(top_genes_per_cluster), group.by = "seurat_clusters") +
    scale_fill_gradientn(colors = colorRampPalette(c("blue", "white", "red"))(100))
  dev.off()
  
  # Generate dot plot of top markers
  dotplot <- DotPlot(seurat_obj, features = unique(top_genes_per_cluster), 
                   group.by = "seurat_clusters") + 
            RotatedAxis() +
            coord_flip() +
            scale_color_viridis()
  
  ggsave(paste0(output_dir, "/top_markers_dotplot.png"), dotplot, width = 12, height = 10, dpi = 100)
  
  # Generate feature plots for top 5 markers per cluster
  for (cluster in unique(top_markers$cluster)) {
    # Get top 5 markers for this cluster
    top5 <- top_markers %>%
            filter(cluster == !!cluster) %>%
            top_n(n = 5, wt = avg_log2FC) %>%
            pull(gene)
    
    # Generate feature plot
    feature_plot <- FeaturePlot(seurat_obj, features = top5, 
                             min.cutoff = "q9", ncol = 3)
    
    # Save plot
    ggsave(paste0(output_dir, "/markers/cluster", cluster, "_top5_markers.png"), 
           feature_plot, width = 15, height = 10, dpi = 100)
    
    # Store in list
    all_markers[[paste0("cluster_", cluster)]] <- top_markers %>% 
                                               filter(cluster == !!cluster)
  }
  
  # Return markers and updated Seurat object
  return(list(seurat_obj = seurat_obj, markers = all_markers, all_markers_df = all_markers_df))
}

#' Run Bayesian differential expression using MAST
#'
#' @param seurat_obj Seurat object
#' @param ident.1 First identity class for comparison
#' @param ident.2 Second identity class for comparison (optional)
#' @param group_by Variable to group cells by (default: seurat_clusters)
#' @param min_pct Minimum percentage of cells expressing gene
#' @param output_dir Directory to save results
#'
#' @return Differential expression results
bayesian_differential_expression <- function(seurat_obj, ident.1, ident.2 = NULL,
                                           group_by = "seurat_clusters",
                                           min_pct = 0.1, output_dir = "./results") {
  
  # Create output directory
  dir.create(paste0(output_dir, "/differential_expression"), showWarnings = FALSE, recursive = TRUE)
  
  # Set identity class
  Idents(seurat_obj) <- group_by
  
  # Run differential expression using MAST
  if (is.null(ident.2)) {
    # One vs all comparison
    de_results <- FindMarkers(seurat_obj,
                            ident.1 = ident.1,
                            min.pct = min_pct,
                            test.use = "MAST")
    
    comp_name <- paste0(group_by, "_", ident.1, "_vs_all")
  } else {
    # Direct comparison between two groups
    de_results <- FindMarkers(seurat_obj,
                            ident.1 = ident.1,
                            ident.2 = ident.2,
                            min.pct = min_pct,
                            test.use = "MAST")
    
    comp_name <- paste0(group_by, "_", ident.1, "_vs_", ident.2)
  }
  
  # Save results
  write.csv(de_results, file = paste0(output_dir, "/differential_expression/", comp_name, ".csv"))
  
  # Create volcano plot
  de_results$gene <- rownames(de_results)
  
  # Add significance and fold-change thresholds
  de_results$significance <- ifelse(de_results$p_val_adj < 0.05, 
                                  ifelse(de_results$avg_log2FC > 0.5, "Up",
                                         ifelse(de_results$avg_log2FC < -0.5, "Down", "NS")),
                                  "NS")
  
  # Set color scheme
  de_results$significance <- factor(de_results$significance, levels = c("Up", "Down", "NS"))
  color_values <- c("Up" = "red", "Down" = "blue", "NS" = "grey")
  
  # Create volcano plot
  volcano_plot <- ggplot(de_results, aes(x = avg_log2FC, y = -log10(p_val_adj), 
                                     color = significance)) +
                geom_point(alpha = 0.7, size = 1.5) +
                scale_color_manual(values = color_values) +
                theme_minimal() +
                labs(title = paste0("Differential Expression: ", comp_name),
                     x = "Log2 Fold Change",
                     y = "-log10(adjusted p-value)") +
                geom_hline(yintercept = -log10(0.05), linetype = "dashed") +
                geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed") +
                theme(legend.title = element_blank(),
                      plot.title = element_text(hjust = 0.5, size = 14))
  
  # Label top genes
  top_up <- de_results %>%
           filter(significance == "Up") %>%
           top_n(n = 10, wt = -p_val_adj)
  
  top_down <- de_results %>%
             filter(significance == "Down") %>%
             top_n(n = 10, wt = -p_val_adj)
  
  top_genes <- rbind(top_up, top_down)
  
  volcano_plot <- volcano_plot +
                geom_text_repel(data = top_genes,
                               aes(label = gene),
                               size = 3,
                               max.overlaps = 20,
                               box.padding = 0.5)
  
  # Save volcano plot
  ggsave(paste0(output_dir, "/differential_expression/", comp_name, "_volcano.png"), 
         volcano_plot, width = 10, height = 8, dpi = 100)
  
  # Return DE results
  return(de_results)
}

#' Annotate cell clusters based on marker genes
#'
#' @param seurat_obj Seurat object with clusters identified
#' @param markers_dict Named list of cell type marker genes
#' @param output_dir Directory to save results
#'
#' @return Seurat object with cell type annotations
annotate_clusters <- function(seurat_obj, markers_dict, output_dir = "./results") {
  
  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Check if marker genes are present in the dataset
  all_genes <- rownames(seurat_obj)
  
  # Initialize score matrix
  cell_types <- names(markers_dict)
  clusters <- unique(seurat_obj$seurat_clusters)
  
  score_matrix <- matrix(0, nrow = length(clusters), ncol = length(cell_types))
  rownames(score_matrix) <- clusters
  colnames(score_matrix) <- cell_types
  
  # Calculate enrichment score for each cell type in each cluster
  for (ct_idx in seq_along(cell_types)) {
    cell_type <- cell_types[ct_idx]
    markers <- markers_dict[[cell_type]]
    
    # Find markers present in dataset
    markers_present <- markers[markers %in% all_genes]
    
    if (length(markers_present) == 0) {
      warning(paste("No markers found for", cell_type))
      next
    }
    
    # Calculate average expression of marker genes in each cluster
    for (cluster in clusters) {
      # Get cells in this cluster
      cells_in_cluster <- WhichCells(seurat_obj, idents = cluster)
      
      # Get average expression of marker genes in this cluster
      avg_expr <- Matrix::rowMeans(seurat_obj[markers_present, cells_in_cluster]@assays$RNA@data)
      
      # Calculate score as sum of average expression values
      score_matrix[cluster, cell_type] <- sum(avg_expr)
    }
  }
  
  # Normalize scores by row
  score_matrix_norm <- t(apply(score_matrix, 1, function(x) x / max(x, na.rm = TRUE)))
  
  # Create heatmap of scores
  pdf(paste0(output_dir, "/cluster_annotation_heatmap.pdf"), width = 10, height = 8)
  pheatmap(score_matrix_norm, color = colorRampPalette(c("navy", "white", "firebrick3"))(100),
          main = "Cell Type Enrichment Scores By Cluster")
  dev.off()
  
  # Assign cell type to each cluster based on highest score
  cluster_annotations <- apply(score_matrix_norm, 1, function(x) {
    cell_type <- names(which.max(x))
    score <- max(x, na.rm = TRUE)
    
    # Only assign if score is above threshold
    if (score >= 0.5) {
      return(cell_type)
    } else {
      return("Unknown")
    }
  })
  
  # Add annotations to Seurat object
  seurat_obj$cell_type <- plyr::mapvalues(seurat_obj$seurat_clusters, 
                                        from = names(cluster_annotations),
                                        to = cluster_annotations)
  
  # Create DimPlot with cell type annotations
  dimplot <- DimPlot(seurat_obj, reduction = "umap", group.by = "cell_type", 
                   label = TRUE, label.size = 5, repel = TRUE) +
            ggtitle("Cell Types") +
            theme(legend.position = "right")
  
  ggsave(paste0(output_dir, "/cell_type_annotations.png"), dimplot, width = 12, height = 10, dpi = 100)
  
  # Return annotated Seurat object
  return(seurat_obj)
}

#' Run the entire scRNA-seq analysis pipeline
#'
#' @param file_paths List of paths to count matrices
#' @param batch_ids Vector of batch identifiers
#' @param file_format Format of input files
#' @param meta_data_paths List of paths to metadata files (optional)
#' @param output_dir Directory to save results
#' @param markers_dict Named list of cell type marker genes (optional)
#' @param normalization_method Method for normalization
#' @param n_variable_features Number of variable features to select
#' @param n_pcs Number of principal components to use
#' @param resolution Resolution parameter for clustering
#'
#' @return List containing the Seurat object and analysis results
run_scrna_seq_pipeline <- function(file_paths, 
                                 batch_ids = NULL,
                                 file_format = "10X_h5",
                                 meta_data_paths = NULL,
                                 output_dir = "./scrna_analysis",
                                 markers_dict = NULL,
                                 normalization_method = "LogNormalize",
                                 n_variable_features = 2000,
                                 n_pcs = 30,
                                 resolution = 0.8) {
  
  # Create main output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # 1. Load and merge datasets
  message("Loading and merging datasets...")
  seurat_obj <- load_and_merge_datasets(file_paths = file_paths,
                                      file_format = file_format,
                                      batch_ids = batch_ids,
                                      meta_data_paths = meta_data_paths)
  
  # 2. QC and filtering
  message("Performing QC and filtering...")
  seurat_obj <- perform_qc_and_filtering(seurat_obj, 
                                       output_dir = output_dir)
  
  # 3. Normalization and variable feature selection
  message("Normalizing data and finding variable features...")
  seurat_obj <- normalize_and_scale(seurat_obj,
                                  normalization_method = normalization_method,
                                  n_variable_features = n_variable_features,
                                  output_dir = output_dir)
  
  # 4. Run PCA
  message("Running PCA...")
  seurat_obj <- run_pca(seurat_obj, 
                      n_pcs = n_pcs, 
                      output_dir = output_dir)
  
  # 5. Batch integration with Harmony
  if (length(unique(seurat_obj$batch)) > 1) {
    message("Running Harmony batch integration...")
    seurat_obj <- run_harmony_integration(seurat_obj, 
                                        batch_var = "batch",
                                        n_pcs = n_pcs,
                                        output_dir = output_dir)
  } else {
    message("Skipping batch integration as only one batch is present...")
    seurat_obj <- RunUMAP(seurat_obj, reduction = "pca", dims = 1:n_pcs)
  }
  
  # 6. Clustering
  message("Running clustering analysis...")
  seurat_obj <- run_clustering(seurat_obj,
                             reduction = ifelse("harmony" %in% names(seurat_obj@reductions), 
                                              "harmony", "pca"),
                             resolution = resolution,
                             n_dims = n_pcs,
                             output_dir = output_dir)
  
  # 7. Find markers
  message("Finding marker genes...")
  markers_results <- find_markers(seurat_obj, output_dir = output_dir)
  seurat_obj <- markers_results$seurat_obj
  
  # 8. Annotate clusters if marker dictionary provided
  if (!is.null(markers_dict)) {
    message("Annotating clusters with cell types...")
    seurat_obj <- annotate_clusters(seurat_obj, 
                                  markers_dict = markers_dict,
                                  output_dir = output_dir)
  }
  
  # 9. Save Seurat object
  saveRDS(seurat_obj, file = paste0(output_dir, "/seurat_object.rds"))
  
  # Return results
  return(list(
    seurat_obj = seurat_obj,
    markers = markers_results$all_markers_df
  ))
}

#' Example usage of the scRNA-seq analysis pipeline
#'
#' This function demonstrates how to use the pipeline with a sample dataset
example_scrna_analysis <- function() {
  # Define cell type markers for annotation
  immune_markers <- list(
    "T_cells" = c("CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4"),
    "B_cells" = c("CD19", "MS4A1", "CD79A", "CD79B"),
    "NK_cells" = c("NCAM1", "NKG7", "KLRD1", "KLRF1"),
    "Monocytes" = c("CD14", "LYZ", "FCGR3A", "MS4A7"),
    "Dendritic_cells" = c("FCER1A", "CLEC10A", "ITGAX", "ITGAM"),
    "Neutrophils" = c("S100A8", "S100A9", "FCGR3B", "CSF3R")
  )
  
  # Use PBMC data as an example
  pbmc_data <- Read10X_h5(system.file("extdata", "pbmc_10k_v3.h5", package = "SeuratData"))
  
  # Create a temporary directory to save the matrix
  tmp_dir <- tempdir()
  sample1_dir <- file.path(tmp_dir, "sample1")
  sample2_dir <- file.path(tmp_dir, "sample2")
  dir.create(sample1_dir, recursive = TRUE)
  dir.create(sample2_dir, recursive = TRUE)
  
  # Simulate two batches by splitting the data
  cells_batch1 <- colnames(pbmc_data)[1:(ncol(pbmc_data)/2)]
  cells_batch2 <- colnames(pbmc_data)[(ncol(pbmc_data)/2 + 1):ncol(pbmc_data)]
  
  batch1_data <- pbmc_data[, cells_batch1]
  batch2_data <- pbmc_data[, cells_batch2]
  
  # Save as MTX files
  DropletUtils::write10xCounts(sample1_dir, batch1_data)
  DropletUtils::write10xCounts(sample2_dir, batch2_data)
  
  # Set up file paths
  file_paths <- c(sample1_dir, sample2_dir)
  batch_ids <- c("batch1", "batch2")
  
  # Run the pipeline
  results <- run_scrna_seq_pipeline(
    file_paths = file_paths,
    batch_ids = batch_ids,
    file_format = "10X_mtx",
    output_dir = "./example_analysis",
    markers_dict = immune_markers,
    normalization_method = "SCT",
    resolution = 0.6
  )
  
  return(results)
}

#' Differential expression analysis between conditions
#'
#' @param seurat_obj Seurat object
#' @param condition_col Column name in metadata that contains condition information
#' @param condition1 First condition for comparison
#' @param condition2 Second condition for comparison
#' @param output_dir Directory to save results
#'
#' @return Differential expression results
condition_differential_expression <- function(seurat_obj, condition_col,
                                            condition1, condition2,
                                            output_dir = "./results") {
  
  # Create output directory
  dir.create(paste0(output_dir, "/condition_de"), showWarnings = FALSE, recursive = TRUE)
  
  # Set identity to condition
  Idents(seurat_obj) <- condition_col
  
  # Run DE analysis using MAST
  de_results <- FindMarkers(seurat_obj,
                          ident.1 = condition1,
                          ident.2 = condition2,
                          test.use = "MAST",
                          min.pct = 0.1)
  
  # Save results
  write.csv(de_results, file = paste0(output_dir, "/condition_de/", 
                                   condition1, "_vs_", condition2, ".csv"))
  
  # Create volcano plot
  de_results$gene <- rownames(de_results)
  
  # Add significance and fold-change thresholds
  de_results$significance <- ifelse(de_results$p_val_adj < 0.05, 
                                  ifelse(de_results$avg_log2FC > 0.5, "Up",
                                         ifelse(de_results$avg_log2FC < -0.5, "Down", "NS")),
                                  "NS")
  
  # Set color scheme
  de_results$significance <- factor(de_results$significance, levels = c("Up", "Down", "NS"))
  color_values <- c("Up" = "red", "Down" = "blue", "NS" = "grey")
  
  # Create volcano plot
  volcano_plot <- ggplot(de_results, aes(x = avg_log2FC, y = -log10(p_val_adj), 
                                     color = significance)) +
                geom_point(alpha = 0.7, size = 1.5) +
                scale_color_manual(values = color_values) +
                theme_minimal() +
                labs(title = paste0("Differential Expression: ", condition1, " vs ", condition2),
                     x = "Log2 Fold Change",
                     y = "-log10(adjusted p-value)") +
                geom_hline(yintercept = -log10(0.05), linetype = "dashed") +
                geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed") +
                theme(legend.title = element_blank(),
                      plot.title = element_text(hjust = 0.5, size = 14))
  
  # Label top genes
  top_up <- de_results %>%
           filter(significance == "Up") %>%
           top_n(n = 10, wt = -p_val_adj)
  
  top_down <- de_results %>%
             filter(significance == "Down") %>%
             top_n(n = 10, wt = -p_val_adj)
  
  top_genes <- rbind(top_up, top_down)
  
  volcano_plot <- volcano_plot +
                geom_text_repel(data = top_genes,
                               aes(label = gene),
                               size = 3,
                               max.overlaps = 20,
                               box.padding = 0.5)
  
  # Save volcano plot
  ggsave(paste0(output_dir, "/condition_de/", condition1, "_vs_", condition2, "_volcano.png"), 
         volcano_plot, width = 10, height = 8, dpi = 100)
  
  # Create heatmap of top DE genes
  top_genes_all <- rbind(
    top_n(de_results, n = 25, wt = avg_log2FC),
    top_n(de_results, n = 25, wt = -avg_log2FC)
  )
  
  # Scale data for heatmap
  seurat_obj <- ScaleData(seurat_obj, features = top_genes_all$gene)
  
  # Make heatmap
  heatmap <- DoHeatmap(seurat_obj, 
                     features = top_genes_all$gene, 
                     group.by = condition_col,
                     slot = "scale.data") +
           scale_fill_gradientn(colors = colorRampPalette(c("blue", "white", "red"))(100))
  
  # Save heatmap
  ggsave(paste0(output_dir, "/condition_de/", condition1, "_vs_", condition2, "_heatmap.png"), 
         heatmap, width = 12, height = 10, dpi = 100)
  
  return(de_results)
}

# Complete R workflow for scRNA-seq analysis including batch integration, normalization, clustering and DE analysis


