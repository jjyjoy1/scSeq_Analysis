'''
The script will complete pair-end fastq.gz files, single cell RNA sequences, HIV related analysis.  
The code contains multiple functions contain the following tasks: 
Data quality control chart
Cell dimensionality reduction clustering diagram
Cell type definition
Gene expression display for each cell cluster
Gene function enrichment analysis of cell clusters
Cell cluster state evolution
Ligand-receptor interaction
Cell cycle analysis
Transcription factor regulation analysis
Single-cell SNP, Indel, CNV analysis
'''

#Step 1: Quality Control and Preprocessing

import os
import subprocess
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths to your data
fastq_dir = "/path/to/fastq/files/"
output_dir = "/path/to/output/"


# Quality control with FastQC
def run_fastqc(fastq_dir, output_dir):
    os.makedirs(output_dir + "/fastqc", exist_ok=True)
    fastq_files = [f for f in os.listdir(fastq_dir) if f.endswith('.fastq.gz')]
    
    for file in fastq_files:
        cmd = f"fastqc {fastq_dir}/{file} -o {output_dir}/fastqc"
        subprocess.run(cmd, shell=True)
    
    # Summarize with MultiQC
    cmd = f"multiqc {output_dir}/fastqc -o {output_dir}/fastqc"
    subprocess.run(cmd, shell=True)
    
    
#Step 2: Alignment and Feature Counting with Cell Ranger
def run_cellranger(fastq_dir, output_dir, sample_name, reference):
    cmd = f"cellranger count --id={sample_name} \
           --fastqs={fastq_dir} \
           --sample={sample_name} \
           --transcriptome={reference} \
           --localcores=16 \
           --localmem=64"
    subprocess.run(cmd, shell=True)
    

#Step 3: Data Loading and Initial QC with Scanpy

def load_and_qc(h5_file, min_genes=200, min_cells=3, max_genes=5000, max_mt=20):
    # Load the data
    adata = sc.read_10x_h5(h5_file) 
    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    # Filter cells based on QC metrics
    adata = adata[adata.obs.n_genes_by_counts < max_genes, :]
    adata = adata[adata.obs.pct_counts_mt < max_mt, :]
    return adata


# Create QC plots
def plot_qc(adata, output_dir):
    sc.settings.figdir = output_dir + "/figures"
    os.makedirs(sc.settings.figdir, exist_ok=True)
    # QC violin plots
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                 jitter=0.4, multi_panel=True, save='_qc_metrics.pdf')
    # Gene-count vs MT content scatter
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', save='_count_vs_mt.pdf')
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', save='_count_vs_genes.pdf')


#Step 4: Normalization, Feature Selection, and Dimensionality Reduction
def normalize_and_reduce(adata):
    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)    
    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)    
    # Keep only highly variable genes for dimensionality reduction
    adata_hvg = adata[:, adata.var.highly_variable]    
    # Scale data
    sc.pp.scale(adata_hvg, max_value=10)    
    # PCA
    sc.tl.pca(adata_hvg, svd_solver='arpack')    
    # UMAP and t-SNE
    sc.pp.neighbors(adata_hvg, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata_hvg)
    sc.tl.tsne(adata_hvg)    
    # Return both objects
    return adata, adata_hvg


#Step 5: Clustering and Cell Type Annotation
def cluster_and_annotate(adata_hvg):
    # Find clusters
    sc.tl.leiden(adata_hvg, resolution=0.5)
    
    # Find marker genes
    sc.tl.rank_genes_groups(adata_hvg, 'leiden', method='wilcoxon')
    
    # Plot results
    sc.pl.umap(adata_hvg, color='leiden', save='_clusters.pdf')
    sc.pl.rank_genes_groups(adata_hvg, n_genes=25, sharey=False, save='_marker_genes.pdf')
    
    # Get top marker genes for each cluster
    marker_genes = pd.DataFrame()
    for i in range(len(adata_hvg.uns['rank_genes_groups']['names'][0])):
        cluster_markers = pd.DataFrame({
            'gene': adata_hvg.uns['rank_genes_groups']['names'][i],
            'score': adata_hvg.uns['rank_genes_groups']['scores'][i],
            'logfoldchanges': adata_hvg.uns['rank_genes_groups']['logfoldchanges'][i],
            'pvals': adata_hvg.uns['rank_genes_groups']['pvals'][i],
            'pvals_adj': adata_hvg.uns['rank_genes_groups']['pvals_adj'][i],
            'cluster': i
        })
        marker_genes = pd.concat([marker_genes, cluster_markers])
    
    return adata_hvg, marker_genes
    
    

#Step 6: Differential Expression Between Conditions
def differential_expression(adata, condition_col='condition'):
    # Ensure condition column exists
    if condition_col not in adata.obs.columns:
        raise ValueError(f"Column {condition_col} not found in adata.obs")
    
    # Split by condition
    conditions = adata.obs[condition_col].unique()
    if len(conditions) != 2:
        raise ValueError("Need exactly two conditions for comparison")
    
    # Create a mask for each condition
    mask1 = adata.obs[condition_col] == conditions[0]
    mask2 = adata.obs[condition_col] == conditions[1]
    
    # Run differential expression
    sc.tl.rank_genes_groups(adata, condition_col, groups=[conditions[0]], reference=conditions[1], method='wilcoxon')
    
    # Get results
    de_genes = sc.get.rank_genes_groups_df(adata, group=conditions[0])
    
    return de_genes
    

#Step 7: Cell Cycle Analysis
def cell_cycle_analysis(adata):
    # Cell cycle genes (you may need to get the appropriate gene list for your species)
    s_genes = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 'CDCA7', 'DTL']
    g2m_genes = ['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67']
    
    # Score cell cycle
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    
    # Plot
    sc.pl.umap(adata, color=['phase', 'S_score', 'G2M_score'], save='_cell_cycle.pdf')
    
    return adata
    

#Step 8: Pathway and Gene Set Enrichment Analysis
def pathway_analysis(marker_genes, output_dir):
    try:
        import gseapy as gp
        
        # Create directory for results
        os.makedirs(output_dir + "/enrichment", exist_ok=True)
        
        # For each cluster
        clusters = marker_genes['cluster'].unique()
        enrichment_results = {}
        
        for cluster in clusters:
            # Get top genes for this cluster
            top_genes = marker_genes[marker_genes['cluster'] == cluster].sort_values('pvals_adj').head(100)['gene'].tolist()
            
            # Run enrichment
            enr = gp.enrichr(gene_list=top_genes,
                           gene_sets=['GO_Biological_Process_2021', 'KEGG_2021_Human'],
                           outdir=output_dir + f"/enrichment/cluster_{cluster}",
                           cutoff=0.05)
            
            enrichment_results[cluster] = enr.results
            
        return enrichment_results
    
    except ImportError:
        print("gseapy not installed. Please install with: pip install gseapy")
        return None

 
#Step 9: Trajectory Analysis with PAGA
def trajectory_analysis(adata):
    # Run PAGA for trajectory inference
    sc.tl.paga(adata, groups='leiden')
    sc.pl.paga(adata, save='_paga.pdf')
    
    # Compute a single-cell embedding using PAGA-initialized embedding
    sc.tl.draw_graph(adata, init_pos='paga')
    sc.pl.draw_graph(adata, color='leiden', legend_loc='on data', save='_paga_graph.pdf')
    
    return adata
    

#Step 10: Ligand-Receptor Analysis
def ligand_receptor_analysis(adata, output_dir):
    # Export data for CellPhoneDB
    cellphonedb_dir = output_dir + "/cellphonedb"
    os.makedirs(cellphonedb_dir, exist_ok=True)
    
    # Export counts matrix
    counts = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
    counts.to_csv(f"{cellphonedb_dir}/counts.txt", sep='\t')
    
    # Export cell type annotations
    meta = pd.DataFrame({"Cell": adata.obs_names, "cell_type": adata.obs['leiden']})
    meta.set_index("Cell", inplace=True)
    meta.to_csv(f"{cellphonedb_dir}/meta.txt", sep='\t')
    
    # Run CellPhoneDB
    cmd = f"cellphonedb method statistical_analysis {cellphonedb_dir}/meta.txt " \
          f"{cellphonedb_dir}/counts.txt --output-path {cellphonedb_dir} " \
          f"--counts-data gene_name"
    subprocess.run(cmd, shell=True)
    
    # Generate plots
    cmd = f"cellphonedb plot dot_plot --means-path {cellphonedb_dir}/means.txt " \
          f"--pvalues-path {cellphonedb_dir}/pvalues.txt " \
          f"--output-path {cellphonedb_dir} " \
          f"--output-name dot_plot.pdf"
    subprocess.run(cmd, shell=True)
    
    # Read results
    means = pd.read_csv(f"{cellphonedb_dir}/means.txt", sep='\t')
    pvals = pd.read_csv(f"{cellphonedb_dir}/pvalues.txt", sep='\t')
    
    return means, pvals
    

#Step 11: SNP/Indel/CNV Analysis
def snp_indel_cnv_analysis(bam_files, output_dir, reference_genome):
    # SNP and Indel analysis with GATK
    for bam_file in bam_files:
        sample_name = os.path.basename(bam_file).split('.')[0]
        
        # Add read groups
        cmd = f"java -jar picard.jar AddOrReplaceReadGroups I={bam_file} " \
              f"O={output_dir}/{sample_name}_rg.bam RGID=1 RGLB=lib1 RGPL=illumina " \
              f"RGPU=unit1 RGSM={sample_name}"
        subprocess.run(cmd, shell=True)
        
        # Mark duplicates
        cmd = f"java -jar picard.jar MarkDuplicates I={output_dir}/{sample_name}_rg.bam " \
              f"O={output_dir}/{sample_name}_rg_md.bam CREATE_INDEX=true " \
              f"VALIDATION_STRINGENCY=SILENT M={output_dir}/{sample_name}_marked_dup_metrics.txt"
        subprocess.run(cmd, shell=True)
        
        # Base recalibration
        cmd = f"gatk BaseRecalibrator -I {output_dir}/{sample_name}_rg_md.bam " \
              f"-R {reference_genome} --known-sites dbsnp.vcf " \
              f"-O {output_dir}/{sample_name}_recal_data.table"
        subprocess.run(cmd, shell=True)
        
        # Apply BQSR
        cmd = f"gatk ApplyBQSR -I {output_dir}/{sample_name}_rg_md.bam " \
              f"-R {reference_genome} --bqsr-recal-file {output_dir}/{sample_name}_recal_data.table " \
              f"-O {output_dir}/{sample_name}_recal.bam"
        subprocess.run(cmd, shell=True)
        
        # Call variants
        cmd = f"gatk HaplotypeCaller -I {output_dir}/{sample_name}_recal.bam " \
              f"-R {reference_genome} -O {output_dir}/{sample_name}.vcf.gz"
        subprocess.run(cmd, shell=True)
    
    # CNV analysis with inferCNV
    # This would require additional code to format data appropriately for inferCNV


#Step 12: Integration of Multiple Samples
def integrate_samples(adata_list, batch_key='sample'):
    import scvi
    
    # Concatenate AnnData objects
    adata_concat = adata_list[0].concatenate(adata_list[1:], batch_key=batch_key)
    
    # Set up scVI model
    scvi.data.setup_anndata(adata_concat, batch_key=batch_key)
    
    # Train model
    model = scvi.model.SCVI(adata_concat)
    model.train()
    
    # Get latent representation
    adata_concat.obsm["X_scVI"] = model.get_latent_representation()
    
    # Run UMAP on integrated data
    sc.pp.neighbors(adata_concat, use_rep="X_scVI")
    sc.tl.umap(adata_concat)
    
    # Plot by batch to check integration
    sc.pl.umap(adata_concat, color=batch_key, save='_batch_integration.pdf')
    
    return adata_concat, model
    

#Step 13: Main Workflow
def main():
    # Paths
    fastq_dir = "/path/to/fastq/files/"
    output_dir = "/path/to/output/"
    reference = "/path/to/reference/"
    
    # Step 1: Quality control
    run_fastqc(fastq_dir, output_dir)
    
    # Step 2: Cell Ranger (example for two samples)
    samples = ["healthy_donor", "hiv_patient"]
    for sample in samples:
        run_cellranger(fastq_dir, output_dir, sample, reference)
    
    # Step 3-12: Process each sample
    adata_list = []
    for sample in samples:
        h5_file = f"{output_dir}/{sample}/outs/filtered_feature_bc_matrix.h5"
        
        # Load and QC
        adata = load_and_qc(h5_file)
        plot_qc(adata, output_dir)
        
        # Add sample info
        adata.obs['sample'] = sample
        adata.obs['condition'] = 'healthy' if 'healthy' in sample else 'hiv'
        
        # Normalize and reduce dimensions
        adata, adata_hvg = normalize_and_reduce(adata)
        
        # Cluster and find markers
        adata_hvg, marker_genes = cluster_and_annotate(adata_hvg)
        
        # Cell cycle analysis
        adata_hvg = cell_cycle_analysis(adata_hvg)
        
        # Pathway analysis
        enrichment_results = pathway_analysis(marker_genes, output_dir)
        
        # Save results
        adata_hvg.write(f"{output_dir}/{sample}_processed.h5ad")
        
        # Add to list for integration
        adata_list.append(adata_hvg)
    
    # Integrate samples
    adata_integrated, model = integrate_samples(adata_list)
    
    # Run additional analyses on integrated data
    adata_integrated, integrated_markers = cluster_and_annotate(adata_integrated)
    
    # Compare conditions
    de_genes = differential_expression(adata_integrated, condition_col='condition')
    
    # Trajectory analysis
    adata_integrated = trajectory_analysis(adata_integrated)
    
    # Save final object
    adata_integrated.write(f"{output_dir}/integrated_analysis.h5ad")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()

























