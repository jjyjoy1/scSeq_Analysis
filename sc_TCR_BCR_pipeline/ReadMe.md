In this repo I prepared two single cell sequences pipelines, one for scTCR-seq data analysis; anther for scRNA-DEA analysis
For more complex experiment, I will prepare later

A. A global road diagram for scTCR-seq data analysis

                Raw scTCR-seq Data
          │  Extraction & QC of             │
          │   TCR Sequences                 │
          
          │ V(D)J Annotation &              │
          │ CDR3 Identification             │
          │ Clonotype Assignment &          │
          │ Repertoire Compilation          │
          │ Feature Analysis:               │
          │ V(D)J Usage, CDR3               │
          │ Length, Motif Analysis,         │
          │ Clonal Expansion                │
          │ Diversity Metrics &             │
          │ Repertoire Overlap              │
          │ Comparative Statistical         │
          │ Analysis (Healthy vs. condition)    │
          │ Integration with Gene           │
          │ Expression (Optional)           │
          │ Visualization &                 │
          │ Downstream Analyses             │


B. A globe roadmap for scRNA-seq DEA

          Raw Sequencing Data
        │  Individual QC &        │
        │   Preprocessing         │
        │Normalization &          │
        │ Feature Selection       │
        │    Data Integration     │
        │   (Batch Correction)    │
        │Dimensionality Reduction | 
        │   (PCA → UMAP/t-SNE)    │
        │     Clustering          │
        │ Cell Type Annotation    │
        │ Differential            │
        │ Expression Analysis     │
        │ (Within Cell Types)     │
        │  Downstream             │
        │   Analyses (Pathway,    |
        │   Trajectory, etc.)     │
 

C. For more complex experiment design and data integration, which involves multiple data modalities—such as scRNA-seq, surface protein (e.g., CITE-seq/ADT), and AIRR (adaptive immune receptor) data—and possibly multiple samples or even experiments, need to plan for integration at several levels. 

                 │  Raw Data (RNA, ADT, AIRR Sequencing)         │
                 │  Modality-Specific Preprocessing & QC         │
                 │  (RNA: normalization, HVGs; ADT: CLR, etc.)   │
                 │  (AIRR: extraction, V(D)J annotation)         │
                 │   Barcode Matching &                          │
                 │   Cell-Level Linking                          │
                 │   Within-Sample (Experiment) Integration      │
                 │  • RNA + ADT Integration (e.g., WNN, totalVI) │
                 │  • AIRR data merged via cell barcodes         │
                 │   Integration Across Samples/Experiments      │
                 │   (Batch correction across donors/exp.)       │
                 │  Downstream Analyses:                         │
                 │  • Clustering & Cell Type Annotation          │
                 │  • Differential Expression/Protein Analysis   │
                 │  • AIRR Comparative Analysis (diversity,      │
                 │    clonality, V/J usage)                      │
                 │  • Linking clonotypes with cell states        │
                 │   Visualization & Biological Insights         │



