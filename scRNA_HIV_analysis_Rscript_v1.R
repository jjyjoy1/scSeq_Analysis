# Load required packages
library(Seurat)
library(dplyr)
library(ggplot2)
library(patchwork)
library(sctransform)
library(monocle3)
library(clusterProfiler)
library(DESeq2)
library(biomaRt)
library(pheatmap)
library(CellChat)
library(infercnv)
library(DropletUtils)
library(SingleR)
library(celldex)
library(DoubletFinder)
library(tidyverse)

# Step 1: Quality Control of Raw Reads
# Note: This usually happens outside R with FastQC/MultiQC
#Define data_dir
data_dir = 'path_fastq_dir'

# Step 2: Read in 10X data
read_and_qc <- function(data_dir, project_name, min.cells = 3, min.features = 200, max.features = 5000, percent.mt = 20) {
  # Read 10X data
  data <- Read10X(data.dir = data_dir)
  
  # Create Seurat object
  seurat_obj <- CreateSeuratObject(counts = data, project = project_name, min.cells = min.cells, min.features = min.features)
  
  # Calculate mitochondrial content
  seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^MT-")
  
  # Plot QC metrics
  p1 <- VlnPlot(seurat_obj, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)
  p2 <- FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
  p3 <- FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "percent.mt")
  print(p1 / (p2 | p3))
  
  # Filter cells
  seurat_obj <- subset(seurat_obj, subset = nFeature_RNA > min.features & nFeature_RNA < max.features & percent.mt < percent.mt)
  
  return(seurat_obj)
}

# Step 3: Normalization and Feature Selection
normalize_and_reduce <- function(seurat_obj, n.pcs = 30, resolution = 0.5, n.neighbors = 10) {
  # SCTransform normalization (preferred over standard normalization for most cases)
  seurat_obj <- SCTransform(seurat_obj, verbose = FALSE)
  
  # Run PCA
  seurat_obj <- RunPCA(seurat_obj, npcs = n.pcs, verbose = FALSE)
  
  # Visualize PCA results
  print(ElbowPlot(seurat_obj))
  
  # Find neighbors and clusters
  seurat_obj <- FindNeighbors(seurat_obj, dims = 1:n.pcs)
  seurat_obj <- FindClusters(seurat_obj, resolution = resolution)
  
  # Run UMAP and t-SNE
  seurat_obj <- RunUMAP(seurat_obj, dims = 1:n.pcs, n.neighbors = n.neighbors)
  seurat_obj <- RunTSNE(seurat_obj, dims = 1:n.pcs)
  
  # Visualize clusters
  p1 <- DimPlot(seurat_obj, reduction = "umap", group.by = "seurat_clusters", label = TRUE)
  p2 <- DimPlot(seurat_obj, reduction = "tsne", group.by = "seurat_clusters", label = TRUE)
  print(p1 | p2)
  
  return(seurat_obj)
}

# Step 4: Find Marker Genes
find_markers <- function(seurat_obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25) {
  # Find markers for every cluster
  all.markers <- FindAllMarkers(seurat_obj, only.pos = only.pos, min.pct = min.pct, logfc.threshold = logfc.threshold)
  
  # Get top markers per cluster
  top_markers <- all.markers %>%
    group_by(cluster) %>%
    top_n(n = 10, wt = avg_log2FC)
  
  # Create heatmap of top markers
  DoHeatmap(seurat_obj, features = top_markers$gene, angle = 90) + NoLegend()
  
  # Return marker genes
  return(all.markers)
}

# Step 5: Pathway and Enrichment Analysis
run_enrichment <- function(markers, organism = "hsa") {
  # For each cluster
  clusters <- unique(markers$cluster)
  enrichment_results <- list()
  
  for(cluster in clusters) {
    # Get top genes for this cluster
    cluster_markers <- markers %>%
      filter(cluster == !!cluster & p_val_adj < 0.05) %>%
      arrange(desc(avg_log2FC))
    
    # Get gene list
    gene_list <- cluster_markers$gene
    
    # Run GO enrichment
    ego <- enrichGO(gene = gene_list, 
                    OrgDb = org.Hs.eg.db, 
                    keyType = 'SYMBOL',
                    ont = "BP", 
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05, 
                    qvalueCutoff = 0.2)
    
    # Run KEGG pathway analysis
    # First convert gene symbols to Entrez IDs
    gene_ids <- bitr(gene_list, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
    
    kk <- enrichKEGG(gene = gene_ids$ENTREZID,
                     organism = organism,
                     pvalueCutoff = 0.05)
    
    # Store results
    enrichment_results[[paste0("cluster_", cluster)]] <- list(go = ego, kegg = kk)
    
    # Plot results
    if(nrow(as.data.frame(ego)) > 0) {
      p1 <- barplot(ego, showCategory = 15)
      print(p1)
    }
    
    if(nrow(as.data.frame(kk)) > 0) {
      p2 <- barplot(kk, showCategory = 15)
      print(p2)
    }
  }
  
  return(enrichment_results)
}

# Step 6: Cell Cycle Analysis
analyze_cell_cycle <- function(seurat_obj) {
  # Cell cycle gene sets
  s.genes <- cc.genes$s.genes
  g2m.genes <- cc.genes$g2m.genes
  
  # Score cells based on cell cycle markers
  seurat_obj <- CellCycleScoring(seurat_obj, s.features = s.genes, g2m.features = g2m.genes)
  
  # Visualize cell cycle phases on UMAP
  p <- DimPlot(seurat_obj, reduction = "umap", group.by = "Phase", cols = c("red", "green", "blue"))
  print(p)
  
  return(seurat_obj)
}

# Step 7: Trajectory Analysis with Monocle3
trajectory_analysis <- function(seurat_obj) {
  # Convert Seurat to cell_data_set for Monocle3
  cds <- as.cell_data_set(seurat_obj)
  
  # Extract UMAP coordinates from Seurat
  cds@reducedDims$UMAP <- seurat_obj@reductions$umap@cell.embeddings
  
  # Cluster cells
  cds <- cluster_cells(cds)
  
  # Learn the trajectory
  cds <- learn_graph(cds)
  
  # Plot the trajectory
  p1 <- plot_cells(cds, color_cells_by = "cluster", label_groups_by_cluster = TRUE,
                   label_leaves = FALSE, label_branch_points = FALSE)
  
  p2 <- plot_cells(cds, color_cells_by = "cluster", label_groups_by_cluster = FALSE,
                   label_leaves = TRUE, label_branch_points = TRUE,
                   graph_label_size = 1.5)
  
  print(p1)
  print(p2)
  
  return(cds)
}

# Step 8: Ligand-Receptor Analysis with CellChat
ligand_receptor_analysis <- function(seurat_obj) {
  # Create CellChat object
  cellchat <- createCellChat(object = seurat_obj, group.by = "seurat_clusters")
  
  # Set the database
  cellchat@DB <- CellChatDB.human
  
  # Pre-process the expression data
  cellchat <- subsetData(cellchat)
  cellchat <- identifyOverExpressedGenes(cellchat)
  cellchat <- identifyOverExpressedInteractions(cellchat)
  
  # Compute communication probability and infer cellular communication network
  cellchat <- computeCommunProb(cellchat)
  cellchat <- computeCommunProbPathway(cellchat)
  cellchat <- aggregateNet(cellchat)
  
  # Visualize the inferred cell-cell communication network
  groupSize <- as.numeric(table(cellchat@idents))
  netVisual_circle(cellchat@net$count, vertex.weight = groupSize, weight.scale = T, label.edge= F, title.name = "Number of interactions")
  
  # Visualize communication network for specific pathways
  pathways.show <- c("WNT", "TGFb", "VEGF")
  for (pathway in pathways.show) {
    if (pathway %in% cellchat@netP$pathways) {
      netVisual_aggregate(cellchat, signaling = pathway, layout = "circle")
    }
  }
  
  return(cellchat)
}

# Step 9: CNV Analysis with inferCNV
cnv_analysis <- function(seurat_obj, reference_cells = NULL) {
  # Create inferCNV object
  # If reference_cells is NULL, we'll use a random subset of cells as reference
  if (is.null(reference_cells)) {
    # Randomly select 10% of cells as reference
    cell_names <- colnames(seurat_obj)
    reference_cells <- sample(cell_names, size = round(length(cell_names) * 0.1))
  }
  
  # Prepare annotations file
  annotations <- data.frame(
    cell_id = colnames(seurat_obj),
    type = ifelse(colnames(seurat_obj) %in% reference_cells, "reference", "sample")
  )
  
  # Get gene positions (normally you would use a genome annotation file)
  # This is a placeholder for getting gene positions
  genes <- rownames(seurat_obj)
  gene_pos <- data.frame(
    gene = genes,
    chr = sapply(strsplit(genes, "[.]"), function(x) paste0("chr", x[1])),
    start = 1:length(genes),
    end = 1:length(genes) + 1000
  )
  
  # Create inferCNV object
  infercnv_obj <- CreateInfercnvObject(
    raw_counts_matrix = as.matrix(GetAssayData(seurat_obj, slot = "counts")),
    annotations_file = annotations,
    gene_order_file = gene_pos,
    ref_group_names = "reference"
  )
  
  # Run inferCNV analysis
  infercnv_obj <- infercnv::run(infercnv_obj,
                              cutoff = 1,
                              out_dir = "infercnv_output",
                              cluster_by_groups = TRUE,
                              denoise = TRUE,
                              HMM = TRUE)
  
  return(infercnv_obj)
}

# Step 10: Integration of Multiple Samples
integrate_samples <- function(seurat_list, n.pcs = 30, n.neighbors = 10, resolution = 0.5) {
  # Find integration features
  features <- SelectIntegrationFeatures(object.list = seurat_list)
  
  # Find integration anchors
  seurat.anchors <- FindIntegrationAnchors(object.list = seurat_list, anchor.features = features)
  
  # Integrate data
  seurat.integrated <- IntegrateData(anchorset = seurat.anchors)
  
  # Switch to integrated assay for downstream analysis
  DefaultAssay(seurat.integrated) <- "integrated"
  
  # Standard workflow for visualization and clustering
  seurat.integrated <- ScaleData(seurat.integrated, verbose = FALSE)
  seurat.integrated <- RunPCA(seurat.integrated, npcs = n.pcs, verbose = FALSE)
  seurat.integrated <- RunUMAP(seurat.integrated, reduction = "pca", dims = 1:n.pcs, n.neighbors = n.neighbors)
  seurat.integrated <- FindNeighbors(seurat.integrated, reduction = "pca", dims = 1:n.pcs)
  seurat.integrated <- FindClusters(seurat.integrated, resolution = resolution)
  
  # Visualization
  p1 <- DimPlot(seurat.integrated, reduction = "umap", group.by = "orig.ident")
  p2 <- DimPlot(seurat.integrated, reduction = "umap", group.by = "seurat_clusters", label = TRUE)
  print(p1 | p2)
  
  return(seurat.integrated)
}

# Step 11: Differential Expression Between Conditions
diff_expression_analysis <- function(seurat_obj, group_by = "condition") {
  # Ensure the grouping variable exists
  if (!group_by %in% colnames(seurat_obj@meta.data)) {
    stop(paste("Column", group_by, "not found in Seurat object metadata"))
  }
  
  # Get the unique groups
  groups <- unique(seurat_obj@meta.data[[group_by]])
  
  if (length(groups) != 2) {
    stop("Need exactly two groups for comparison")
  }
  
  # Find markers between the two conditions
  markers <- FindMarkers(seurat_obj, 
                       ident.1 = groups[1], 
                       ident.2 = groups[2], 
                       group.by = group_by,
                       min.pct = 0.25)
  
  # Add gene names as a column
  markers$gene <- rownames(markers)
  
  # Volcano plot
  EnhancedVolcano(markers,
                lab = markers$gene,
                x = 'avg_log2FC',
                y = 'p_val_adj',
                pCutoff = 0.05,
                FCcutoff = 0.5,
                title = paste('Differential Expression:', groups[1], 'vs', groups[2]))
  
  # Return the results
  return(markers)
}

# Step 12: Cell Type Annotation with SingleR
annotate_cell_types <- function(seurat_obj, ref_dataset = "HumanPrimaryCellAtlasData") {
  # Get reference dataset
  if(ref_dataset == "HumanPrimaryCellAtlasData") {
    ref <- HumanPrimaryCellAtlasData()
  } else if(ref_dataset == "BlueprintEncodeData") {
    ref <- BlueprintEncodeData()
  } else {
    stop("Unsupported reference dataset")
  }
  
  # Run SingleR
  predictions <- SingleR(test = GetAssayData(seurat_obj, slot = "data"),
                         ref = ref,
                         labels = ref$label.main)
  
  # Add cell type annotations to Seurat object
  seurat_obj$SingleR.labels <- predictions$labels
  
  # Visualize annotations
  p <- DimPlot(seurat_obj, reduction = "umap", group.by = "SingleR.labels", label = TRUE, repel = TRUE)
  print(p)
  
  return(seurat_obj)
}

# Step 13: Main Workflow
main <- function() {
  # Set directories
  data_dirs <- c(
    healthy = "/path/to/healthy_donor/filtered_feature_bc_matrix/",
    hiv = "/path/to/hiv_patient/filtered_feature_bc_matrix/"
  )
  
  # Process each sample separately
  seurat_list <- list()
  for(i in seq_along(data_dirs)) {
    sample_name <- names(data_dirs)[i]
    sample_dir <- data_dirs[i]
    
    # Read and QC
    seurat_obj <- read_and_qc(sample_dir, sample_name)
    
    # Add condition information
    seurat_obj$condition <- ifelse(grepl("healthy", sample_name), "healthy", "hiv")
    
    # Normalize and reduce dimensions
    seurat_obj <- normalize_and_reduce(seurat_obj)
    
    # Find markers
    markers <- find_markers(seurat_obj)
    
    # Enrichment analysis
    enrichment <- run_enrichment(markers)
    
    # Cell cycle analysis
    seurat_obj <- analyze_cell_cycle(seurat_obj)
    
    # Save processed object
    saveRDS(seurat_obj, file = paste0(sample_name, "_processed.rds"))
    
    # Add to list for integration
    seurat_list[[sample_name]] <- seurat_obj
  }
  
  # Integrate samples
  integrated <- integrate_samples(seurat_list)
  
  # Annotate cell types
  integrated <- annotate_cell_types(integrated)
  
  # Find markers for integrated data
  integrated_markers <- find_markers(integrated)
  
  # Differential expression between conditions
  de_results <- diff_expression_analysis(integrated, group_by = "condition")
  
  # Trajectory analysis
  cds <- trajectory_analysis(integrated)
  
  # Ligand-receptor analysis
  cellchat <- ligand_receptor_analysis(integrated)
  
  # Save final objects
  saveRDS(integrated, file = "integrated_seurat.rds")
  saveRDS(cds, file = "trajectory_monocle.rds")
  saveRDS(cellchat, file = "ligand_receptor_cellchat.rds")
  
  # Return final objects
  return(list(seurat = integrated, monocle = cds, cellchat = cellchat))
}

# Run the analysis
results <- main()
