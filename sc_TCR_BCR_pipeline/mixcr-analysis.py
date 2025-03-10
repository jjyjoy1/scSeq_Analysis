# Single Cell TCR/BCR Analysis Pipeline using MiXCR
# This script processes fastq files with MiXCR and performs comprehensive repertoire analysis

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import glob
import subprocess
import re
from scipy import stats
import math
from Bio import SeqIO, Seq, motifs
from Bio.SeqUtils import GC
import networkx as nx
import argparse
import json
from adjustText import adjust_text
import itertools
import multiprocessing
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy


class MiXCRAnalyzer:
    def __init__(self, data_dir, output_dir, receptor_type='tcr', species='hsa', threads=None, sample_info=None):
        """
        Initialize the MiXCR analyzer with input and output directories
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the fastq files
        output_dir : str
            Directory for saving analysis results
        receptor_type : str
            Type of receptor to analyze: 'tcr' or 'bcr'
        species : str
            Species code (hsa for human, mmu for mouse)
        threads : int
            Number of threads to use for MiXCR analysis
        sample_info : pd.DataFrame, optional
            Information about samples (condition, subject, etc.)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.receptor_type = receptor_type.lower()
        self.species = species
        self.threads = threads or max(1, multiprocessing.cpu_count() - 1)
        self.sample_info = sample_info
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize dataframes to store results
        self.clonotypes_df = None
        self.repertoire_df = None
        self.vdj_usage = None
        self.diversity_metrics = None
        self.isotype_distribution = None  # Only for BCR
        self.shm_data = None  # Only for BCR
        
        # Define chains based on receptor type
        if self.receptor_type == 'tcr':
            self.heavy_chains = ['TRA', 'TRB']  # Alpha and Beta chains
            self.light_chains = []
        elif self.receptor_type == 'bcr':
            self.heavy_chains = ['IGH']
            self.light_chains = ['IGK', 'IGL']
        else:
            raise ValueError("receptor_type must be either 'tcr' or 'bcr'")
    
    def run_mixcr_analysis(self, preset=None):
        """
        Run MiXCR analysis on all FASTQ files
        
        Parameters:
        -----------
        preset : str, optional
            MiXCR preset to use (e.g., 'hs-tcrb' or 'hs-bcr'). If None, will be determined from receptor type.
        """
        print(f"Starting MiXCR analysis for {self.receptor_type.upper()} data...")
        
        # Create directory for MiXCR results
        mixcr_dir = os.path.join(self.output_dir, "mixcr_output")
        os.makedirs(mixcr_dir, exist_ok=True)
        
        # Get all FASTQ files
        fastq_files = glob.glob(os.path.join(self.data_dir, "*.fastq")) + \
                      glob.glob(os.path.join(self.data_dir, "*.fastq.gz")) + \
                      glob.glob(os.path.join(self.data_dir, "*_R1_*.fastq.gz"))
        
        # Determine preset if not provided
        if preset is None:
            if self.receptor_type == 'tcr':
                if self.species == 'hsa':
                    preset = 'hs-tcrb'  # Human TCR beta
                else:
                    preset = 'mm-tcrb'  # Mouse TCR beta
            else:  # BCR
                if self.species == 'hsa':
                    preset = 'hs-bcr'   # Human BCR
                else:
                    preset = 'mm-bcr'   # Mouse BCR
        
        # Group files by sample name
        sample_files = defaultdict(list)
        
        for fastq_file in fastq_files:
            sample_name = os.path.basename(fastq_file).split('_')[0]
            sample_files[sample_name].append(fastq_file)
        
        # Process each sample
        for sample_name, files in sample_files.items():
            print(f"Processing sample {sample_name} with MiXCR...")
            
            # If paired-end data, find read pairs
            paired_files = []
            if len(files) >= 2:
                # Look for read pairs (R1 and R2)
                r1_files = [f for f in files if "_R1_" in f or "_R1." in f]
                r2_files = [f for f in files if "_R2_" in f or "_R2." in f]
                
                # Match R1 and R2 files
                for r1 in r1_files:
                    r2 = r1.replace("_R1_", "_R2_").replace("_R1.", "_R2.")
                    if r2 in r2_files:
                        paired_files.append((r1, r2))
            
            # Use paired files if available, otherwise use all files individually
            input_files = paired_files if paired_files else [(f,) for f in files]
            
            # Run MiXCR for each file or file pair
            for file_tuple in input_files:
                # Determine output name
                if len(file_tuple) == 2:
                    # Paired-end
                    file_base = os.path.basename(file_tuple[0]).replace("_R1_", "_").replace("_R1.", ".")
                    file_prefix = os.path.splitext(file_base)[0]
                else:
                    # Single-end
                    file_base = os.path.basename(file_tuple[0])
                    file_prefix = os.path.splitext(file_base)[0]
                
                output_prefix = os.path.join(mixcr_dir, f"{sample_name}_{file_prefix}")
                
                # For single-cell data with barcodes, we need to export alignments with barcodes
                # and adjust the assemblePartial and assemblePairs commands
                mixcr_cmd = [
                    "mixcr", "analyze", preset,
                    "--threads", str(self.threads),
                    "--starting-material", "rna",
                    "--tag-pattern", "UMI:NNNNNNNNNN", # Adjust UMI pattern if needed
                    "--tag-pattern", "BC:NNNNNNNNNNNNNN", # Adjust barcode pattern if needed
                    "-s", self.species,
                    "--report", f"{output_prefix}_report.txt"
                ]
                
                # Add input files
                mixcr_cmd.extend(file_tuple)
                
                # Add output file
                mixcr_cmd.append(f"{output_prefix}")
                
                print(f"Running MiXCR command: {' '.join(mixcr_cmd)}")
                
                try:
                    # Run MiXCR command
                    subprocess.run(mixcr_cmd, check=True)
                    
                    # Export clonotypes to TSV
                    self._export_clonotypes(f"{output_prefix}.clns", f"{output_prefix}_clonotypes.tsv")
                    
                except subprocess.CalledProcessError as e:
                    print(f"Error running MiXCR for sample {sample_name}: {str(e)}")
        
        # Compile all clonotype files
        self._compile_clonotypes(mixcr_dir)
        
        print("MiXCR analysis completed!")
        return self.clonotypes_df
    
    def _export_clonotypes(self, clns_file, output_file):
        """
        Export clonotypes from MiXCR binary format to TSV
        
        Parameters:
        -----------
        clns_file : str
            MiXCR binary clonotypes file
        output_file : str
            Output TSV file
        """
        if not os.path.exists(clns_file):
            print(f"Warning: Clonotype file {clns_file} not found.")
            return
        
        export_fields = [
            "cloneId", "targetSequences", "targetQualities", 
            "allVHitsWithScore", "allDHitsWithScore", "allJHitsWithScore", "allCHitsWithScore",
            "nSeqCDR3", "aaSeqCDR3", "vdjdist", "cloneCount", "cloneFraction",
            "readIds"
        ]
        
        if self.receptor_type == 'bcr':
            # Add BCR-specific fields
            export_fields.extend([
                "somatic", "hypermutations", "allGGene"
            ])
        
        # Export clonotypes as TSV
        cmd = [
            "mixcr", "exportClones",
            "--chains-sticks", "-o",  # Include chain information as "sticks" (e.g., TRA|TRB)
            *export_fields,
            clns_file, output_file
        ]
        
        subprocess.run(cmd, check=True)
    
    def _compile_clonotypes(self, mixcr_dir):
        """
        Compile clonotype information from all MiXCR output files
        """
        print("Compiling clonotype data...")
        
        # Find all clonotype files
        clonotype_files = glob.glob(os.path.join(mixcr_dir, "*_clonotypes.tsv"))
        
        if not clonotype_files:
            print("Error: No clonotype files found.")
            return None
        
        all_clonotypes = []
        
        for clonotype_file in clonotype_files:
            # Extract sample name from filename
            file_base = os.path.basename(clonotype_file)
            sample_name = file_base.split('_')[0]
            
            try:
                # Read clonotype file
                df = pd.read_csv(clonotype_file, sep='\t')
                
                if df.empty:
                    print(f"Warning: Empty clonotype file {clonotype_file}")
                    continue
                
                # Add sample name
                df['sample'] = sample_name
                
                # Extract chain information
                chain_columns = self._extract_chain_info(df)
                
                # Add chain-specific columns to dataframe
                for col, values in chain_columns.items():
                    df[col] = values
                
                # Rename columns for consistency
                df = df.rename(columns={
                    'cloneId': 'clone_id',
                    'cloneCount': 'clone_count',
                    'cloneFraction': 'clone_fraction',
                    'aaSeqCDR3': 'cdr3_aa',
                    'nSeqCDR3': 'cdr3_nt'
                })
                
                all_clonotypes.append(df)
                
            except Exception as e:
                print(f"Error processing clonotype file {clonotype_file}: {str(e)}")
        
        if not all_clonotypes:
            print("Error: No valid clonotype data found.")
            return None
        
        # Combine all clonotype data
        combined_df = pd.concat(all_clonotypes, ignore_index=True)
        
        # Save compiled clonotype data
        output_file = os.path.join(self.output_dir, "all_clonotypes.csv")
        combined_df.to_csv(output_file, index=False)
        
        self.clonotypes_df = combined_df
        
        # Compile repertoire summary
        self._compile_repertoire()
        
        print(f"Compiled {len(combined_df)} clonotypes from {len(clonotype_files)} files")
        return combined_df
    
    def _extract_chain_info(self, df):
        """
        Extract chain-specific information from MiXCR output
        
        Parameters:
        -----------
        df : pd.DataFrame
            MiXCR output dataframe
        
        Returns:
        --------
        dict:
            Dictionary with chain-specific columns
        """
        # Initialize chain columns
        chain_columns = {}
        
        for chain in self.heavy_chains + self.light_chains:
            chain_columns[f"{chain.lower()}_v_gene"] = [""] * len(df)
            chain_columns[f"{chain.lower()}_d_gene"] = [""] * len(df)
            chain_columns[f"{chain.lower()}_j_gene"] = [""] * len(df)
            chain_columns[f"{chain.lower()}_c_gene"] = [""] * len(df)
            chain_columns[f"{chain.lower()}_cdr3"] = [""] * len(df)
            
            if self.receptor_type == 'bcr' and chain == 'IGH':
                # Add isotype column for BCR heavy chain
                chain_columns['isotype'] = [""] * len(df)
        
        # Process each row
        for idx, row in df.iterrows():
            # Extract V gene hits
            v_hits = str(row['allVHitsWithScore']).split(",")
            d_hits = str(row['allDHitsWithScore']).split(",")
            j_hits = str(row['allJHitsWithScore']).split(",")
            c_hits = str(row['allCHitsWithScore']).split(",") if 'allCHitsWithScore' in row else []
            
            # Determine chains present in this clonotype
            chains = []
            for hit in v_hits:
                if not hit or hit == 'None' or hit == 'nan':
                    continue
                # Extract chain from V gene name (e.g., TRBV7-2*01 -> TRB)
                chain_match = re.match(r'([A-Z]{2,3})[VDJC]', hit)
                if chain_match:
                    chains.append(chain_match.group(1))
            
            # Only keep unique chains
            chains = list(set(chains))
            
            # Process each chain
            for chain in chains:
                if chain in self.heavy_chains + self.light_chains:
                    chain_lower = chain.lower()
                    
                    # Find genes for this chain
                    v_gene = next((g for g in v_hits if g.startswith(f"{chain}V")), "")
                    d_gene = next((g for g in d_hits if g.startswith(f"{chain}D")), "")
                    j_gene = next((g for g in j_hits if g.startswith(f"{chain}J")), "")
                    c_gene = next((g for g in c_hits if g.startswith(f"{chain}C")), "")
                    
                    # Clean up gene names (remove scores and alleles)
                    v_gene = re.sub(r'\(.*?\)', '', v_gene).split('*')[0] if v_gene else ""
                    d_gene = re.sub(r'\(.*?\)', '', d_gene).split('*')[0] if d_gene else ""
                    j_gene = re.sub(r'\(.*?\)', '', j_gene).split('*')[0] if j_gene else ""
                    c_gene = re.sub(r'\(.*?\)', '', c_gene).split('*')[0] if c_gene else ""
                    
                    # Store in columns
                    chain_columns[f"{chain_lower}_v_gene"][idx] = v_gene
                    chain_columns[f"{chain_lower}_d_gene"][idx] = d_gene
                    chain_columns[f"{chain_lower}_j_gene"][idx] = j_gene
                    chain_columns[f"{chain_lower}_c_gene"][idx] = c_gene
                    
                    # For BCR, extract isotype from C gene
                    if self.receptor_type == 'bcr' and chain == 'IGH' and c_gene:
                        isotype_match = re.search(r'IGHG[1-4]|IGHA[1-2]|IGHM|IGHD|IGHE', c_gene)
                        if isotype_match:
                            chain_columns['isotype'][idx] = isotype_match.group(0)
                    
                    # Extract CDR3 sequence for this chain (if available)
                    if 'aaSeqCDR3' in row and row['aaSeqCDR3']:
                        chain_columns[f"{chain_lower}_cdr3"][idx] = row['aaSeqCDR3']
        
        return chain_columns
    
    def _compile_repertoire(self):
        """
        Compile repertoire information from clonotype data
        """
        if self.clonotypes_df is None:
            print("Error: No clonotype data available. Run run_mixcr_analysis first.")
            return None
            
        # Group by sample and compile repertoire info
        repertoire_data = []
        
        for sample, sample_df in self.clonotypes_df.groupby('sample'):
            # Get condition if available
            condition = "Unknown"
            if self.sample_info is not None and 'condition' in self.sample_info.columns:
                sample_condition = self.sample_info[self.sample_info['sample'] == sample]
                if not sample_condition.empty:
                    condition = sample_condition['condition'].iloc[0]
            
            # Process TCR/BCR repertoire differently
            if self.receptor_type == 'tcr':
                # For TCR, process Alpha and Beta chains
                alpha_df = sample_df[sample_df['tra_v_gene'] != ""]
                beta_df = sample_df[sample_df['trb_v_gene'] != ""]
                
                # Calculate unique clonotypes count
                unique_clonotypes = len(sample_df)
                
                # Calculate unique V and J gene usage
                unique_v_genes = set(
                    [v for v in alpha_df['tra_v_gene'] if v] + 
                    [v for v in beta_df['trb_v_gene'] if v]
                )
                unique_j_genes = set(
                    [j for j in alpha_df['tra_j_gene'] if j] + 
                    [j for j in beta_df['trb_j_gene'] if j]
                )
                
                # Calculate diversity metrics
                diversity_metrics = self._calculate_diversity_metrics(sample_df)
                
                # Store repertoire info
                repertoire_data.append({
                    'sample': sample,
                    'condition': condition,
                    'unique_clonotypes': unique_clonotypes,
                    'alpha_chains': len(alpha_df),
                    'beta_chains': len(beta_df),
                    'unique_v_genes': len(unique_v_genes),
                    'unique_j_genes': len(unique_j_genes),
                    'shannon_entropy': diversity_metrics['shannon_entropy'],
                    'clonality': diversity_metrics['clonality'],
                    'simpson_index': diversity_metrics['simpson_index']
                })
                
            else:  # BCR
                # For BCR, process Heavy and Light chains
                heavy_df = sample_df[sample_df['igh_v_gene'] != ""]
                kappa_df = sample_df[sample_df['igk_v_gene'] != ""]
                lambda_df = sample_df[sample_df['igl_v_gene'] != ""]
                
                # Calculate unique clonotypes count
                unique_clonotypes = len(sample_df)
                
                # Calculate unique V and J gene usage
                unique_v_genes = set(
                    [v for v in heavy_df['igh_v_gene'] if v] + 
                    [v for v in kappa_df['igk_v_gene'] if v] +
                    [v for v in lambda_df['igl_v_gene'] if v]
                )
                unique_j_genes = set(
                    [j for j in heavy_df['igh_j_gene'] if j] + 
                    [j for j in kappa_df['igk_j_gene'] if j] +
                    [j for j in lambda_df['igl_j_gene'] if j]
                )
                
                # Calculate diversity metrics
                diversity_metrics = self._calculate_diversity_metrics(sample_df)
                
                # Calculate isotype distribution
                isotype_counts = Counter(heavy_df['isotype'])
                isotype_percent = {
                    isotype: (count / len(heavy_df) * 100) if len(heavy_df) > 0 else 0
                    for isotype, count in isotype_counts.items()
                }
                
                # Store repertoire info
                repertoire_data.append({
                    'sample': sample,
                    'condition': condition,
                    'unique_clonotypes': unique_clonotypes,
                    'heavy_chains': len(heavy_df),
                    'kappa_chains': len(kappa_df),
                    'lambda_chains': len(lambda_df),
                    'unique_v_genes': len(unique_v_genes),
                    'unique_j_genes': len(unique_j_genes),
                    'shannon_entropy': diversity_metrics['shannon_entropy'],
                    'clonality': diversity_metrics['clonality'],
                    'simpson_index': diversity_metrics['simpson_index'],
                    'isotype_distribution': isotype_percent
                })
        
        # Convert to DataFrame
        self.repertoire_df = pd.DataFrame(repertoire_data)
        
        # Save repertoire data
        output_file = os.path.join(self.output_dir, f"{self.receptor_type}_repertoire_summary.csv")
        self.repertoire_df.to_csv(output_file, index=False)
        
        return self.repertoire_df
    
    def _calculate_diversity_metrics(self, df):
        """
        Calculate diversity metrics for a sample
        
        Parameters:
        -----------
        df : pd.DataFrame
            Clonotype dataframe for a sample
        
        Returns:
        --------
        dict:
            Dictionary with diversity metrics
        """
        # Extract clone frequencies
        frequencies = df['clone_fraction'].values
        
        # Calculate Shannon entropy
        shannon_entropy = 0
        for freq in frequencies:
            if freq > 0:
                shannon_entropy -= freq * np.log2(freq)
        
        # Normalized Shannon entropy
        max_entropy = np.log2(len(frequencies)) if len(frequencies) > 0 else 0
        norm_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
        
        # Clonality (1 - normalized entropy)
        clonality = 1 - norm_entropy
        
        # Simpson index (probability that two randomly chosen clones are identical)
        simpson_index = sum(freq**2 for freq in frequencies)
        
        # Inverse Simpson (1/D) - number of equally dominant clones
        inverse_simpson = 1 / simpson_index if simpson_index > 0 else 0
        
        return {
            'shannon_entropy': shannon_entropy,
            'normalized_entropy': norm_entropy,
            'clonality': clonality,
            'simpson_index': simpson_index,
            'inverse_simpson': inverse_simpson
        }
    
    def analyze_vdj_usage(self):
        """
        Analyze V(D)J gene usage patterns
        """
        print("Analyzing V(D)J gene usage...")
        
        if self.clonotypes_df is None:
            print("Error: No clonotype data available. Run run_mixcr_analysis first.")
            return None
        
        # Create a directory for VDJ usage analysis
        vdj_dir = os.path.join(self.output_dir, "vdj_usage")
        os.makedirs(vdj_dir, exist_ok=True)
        
        # Initialize dictionaries to store V(D)J usage data
        v_usage = defaultdict(lambda: defaultdict(int))
        j_usage = defaultdict(lambda: defaultdict(int))
        d_usage = defaultdict(lambda: defaultdict(int))
        vj_pairing = defaultdict(lambda: defaultdict(int))
        
        # Calculate gene usage (weighted by clone count)
        for _, clone in self.clonotypes_df.iterrows():
            sample = clone['sample']
            clone_count = clone['clone_count']
            
            # Process each chain type
            for chain in self.heavy_chains + self.light_chains:
                chain_lower = chain.lower()
                v_gene = clone.get(f"{chain_lower}_v_gene", "")
                d_gene = clone.get(f"{chain_lower}_d_gene", "")
                j_gene = clone.get(f"{chain_lower}_j_gene", "")
                
                if v_gene:
                    v_usage[sample][v_gene] += clone_count
                if d_gene:
                    d_usage[sample][d_gene] += clone_count
                if j_gene:
                    j_usage[sample][j_gene] += clone_count
                if v_gene and j_gene:
                    vj_pairing[sample][f"{v_gene}:{j_gene}"] += clone_count
        
        # Convert to DataFrames
        v_usage_df = pd.DataFrame(v_usage).fillna(0)
        j_usage_df = pd.DataFrame(j_usage).fillna(0)
        d_usage_df = pd.DataFrame(d_usage).fillna(0)
        vj_pairing_df = pd.DataFrame(vj_pairing).fillna(0)
        
        # Save usage data
        v_usage_df.to_csv(os.path.join(vdj_dir, "v_gene_usage.csv"))
        j_usage_df.to_csv(os.path.join(vdj_dir, "j_gene_usage.csv"))
        d_usage_df.to_csv(os.path.join(vdj_dir, "d_gene_usage.csv"))
        vj_pairing_df.to_csv(os.path.join(vdj_dir, "vj_pairing.csv"))
        
        # Store results
        self.vdj_usage = {
            'v_usage': v_usage_df,
            'j_usage': j_usage_df,
            'd_usage': d_usage_df,
            'vj_pairing': vj_pairing_df
        }
        
        # Create plots
        self._plot_gene_usage(v_usage_df, "V gene", vdj_dir)
        self._plot_gene_usage(j_usage_df, "J gene", vdj_dir)
        if not d_usage_df.empty:
            self._plot_gene_usage(d_usage_df, "D gene", vdj_dir)
        
        # Plot V-J gene pairing
        self._plot_vj_pairing(vj_pairing_df, vdj_dir)
        
        # Add gene usage by chain
        self._analyze_gene_usage_by_chain(vdj_dir)
        
        print("V(D)J usage analysis completed!")
        return self.vdj_usage
    
    def _plot_gene_usage(self, usage_df, gene_type, output_dir):
        """
        Plot gene usage data as heatmap and bar plots
        
        Parameters:
        -----------
        usage_df : pd.DataFrame
            Gene usage dataframe
        gene_type : str
            Type of gene (V, D, or J)
        output_dir : str
            Output directory
        """
        if usage_df.empty:
            return
            
        # Normalize by total usage within each sample
        normalized_df = usage_df.copy()
        for col in normalized_df.columns:
            total = normalized_df[col].sum()
            if total > 0:
                normalized_df[col] = normalized_df[col] / total * 100
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(normalized_df, cmap="viridis", annot=False)
        plt.title(f"{gene_type} Usage (% of repertoire)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{gene_type.lower().replace(' ', '_')}_usage_heatmap.pdf"))
        plt.close()
        
        # Plot top 10 genes as bar plots for each sample
        for sample in usage_df.columns:
            plt.figure(figsize=(10, 6))
            
            # Get top 10 genes
            top_genes = normalized_df[sample].nlargest(10)
            
            # Plot
            sns.barplot(x=top_genes.index, y=top_genes.values)
            plt.title(f"Top 10 {gene_type} Usage in {sample}")
            plt.ylabel("% of repertoire")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{sample}_{gene_type.lower().replace(' ', '_')}_top10.pdf"))
            plt.close()
    
    def _plot_vj_pairing(self, vj_pairing_df, output_dir):
        """
        Plot V-J gene pairing as chord diagram or heatmap
        
        Parameters:
        -----------
        vj_pairing_df : pd.DataFrame
            V-J pairing dataframe
        output_dir : str
            Output directory
        """
        if vj_pairing_df.empty:
            return
        
        # For each sample, create a separate V-J pairing plot
        for sample in vj_pairing_df.columns:
            # Extract V-J pairs for this sample
            sample_pairs = vj_pairing_df[sample].copy()
            sample_pairs = sample_pairs[sample_pairs > 0]
            
            if sample_pairs.empty:
                continue
            
            # Get top 20 V-J pairs
            top_pairs = sample_pairs.nlargest(20)
            
            # Extract V and J genes
            v_genes = set()
            j_genes = set()
            for pair in top_pairs.index:
                v, j = pair.split(':')
                v_genes.add(v)
                j_genes.add(j)
            
            # Create a matrix for the heatmap
            v_genes = sorted(v_genes)
            j_genes = sorted(j_genes)
            
            # Initialize matrix
            matrix = np.zeros((len(v_genes), len(j_genes)))
            
            # Fill matrix
            for i, v in enumerate(v_genes):
                for j, j_gene in enumerate(j_genes):
                    pair = f"{v}:{j_gene}"
                    if pair in top_pairs:
                        matrix[i, j] = top_pairs[pair]
            
            # Normalize
            matrix_norm = matrix.copy()
            if matrix.sum() > 0:
                matrix_norm = matrix_norm / matrix.sum() * 100
            
            # Plot heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(matrix_norm, cmap="viridis", annot=True, fmt=".1f",
                       xticklabels=j_genes, yticklabels=v_genes)
            plt.title(f"Top V-J Pairing in {sample} (% of repertoire)")
            plt.xlabel("J Gene")
            plt.ylabel("V Gene")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{sample}_vj_pairing_heatmap.pdf"))
            plt.close()
    
    def _analyze_gene_usage_by_chain(self, output_dir):
        """
        Analyze gene usage patterns separately for each chain type
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        if self.clonotypes_df is None:
            return
            
        # Separate V gene usage by chain type
        chain_v_usage = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for _, row in self.clonotypes_df.iterrows():
            sample = row['sample']
            clone_count = row['clone_count']
            
            # Process each chain type
            for chain in self.heavy_chains + self.light_chains:
                chain_lower = chain.lower()
                v_gene = row.get(f"{chain_lower}_v_gene", "")
                
                if v_gene:
                    chain_v_usage[chain][sample][v_gene] += clone_count
        
        # Convert to DataFrames and plot
        for chain in chain_v_usage:
            v_usage_chain = pd.DataFrame(chain_v_usage[chain]).fillna(0)
            
            # Save
            v_usage_chain.to_csv(os.path.join(output_dir, f"{chain}_v_gene_usage.csv"))
            
            # Normalize and plot
            normalized_df = v_usage_chain.copy()
            for col in normalized_df.columns:
                total = normalized_df[col].sum()
                if total > 0:
                    normalized_df[col] = normalized_df[col] / total * 100
            
            # Plot heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(normalized_df, cmap="viridis", annot=False)
            plt.title(f"{chain} V Gene Usage (% of repertoire)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{chain}_v_gene_usage_heatmap.pdf"))
            plt.close()
            
            # Plot top 10 genes combined across samples
            plt.figure(figsize=(10, 6))
            top_genes = normalized_df.sum(axis=1).nlargest(10).index
            normalized_df.loc[top_genes].mean(axis=1).plot(kind='bar')
            plt.title(f"Top 10 {chain} V Gene Usage (Average Across Samples)")
            plt.ylabel("% of repertoire")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{chain}_v_gene_usage_top10.pdf"))
            plt.close()
    
    def analyze_cdr3_features(self):
        """
        Analyze CDR3 sequence features (length distribution, amino acid composition, motifs)
        """
        print("Analyzing CDR3 features...")
        
        if self.clonotypes_df is None:
            print("Error: No clonotype data available. Run run_mixcr_analysis first.")
            return None
        
        # Create a directory for CDR3 analysis
        cdr3_dir = os.path.join(self.output_dir, "cdr3_analysis")
        os.makedirs(cdr3_dir, exist_ok=True)
        
        # Analyze CDR3 length distribution
        self._analyze_cdr3_length(cdr3_dir)
        
        # Analyze CDR3 amino acid composition
        self._analyze_cdr3_aa_composition(cdr3_dir)
        
        # Analyze CDR3 motifs
        self._analyze_cdr3_motifs(cdr3_dir)
        
        # Analyze physicochemical properties
        self._analyze_cdr3_properties(cdr3_dir)
        
        print("CDR3 feature analysis completed!")
    
    def _analyze_cdr3_length(self, output_dir):
        """
        Analyze CDR3 length distribution
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        # Prepare data for each chain type
        chain_lengths = []
        
        for _, clone in self.clonotypes_df.iterrows():
            sample = clone['sample']
            clone_count = clone['clone_count']
            
            # Get condition if available
            condition = "Unknown"
            if self.sample_info is not None:
                sample_condition = self.sample_info[self.sample_info['sample'] == sample]
                if not sample_condition.empty:
                    condition = sample_condition['condition'].iloc[0]
            
            # Process each chain type
            for chain in self.heavy_chains + self.light_chains:
                chain_lower = chain.lower()
                cdr3 = clone.get(f"{chain_lower}_cdr3", "")
                
                if cdr3 and cdr3 != "":
                    isotype = clone.get('isotype', None) if self.receptor_type == 'bcr' and chain == 'IGH' else None
                    
                    for _ in range(int(clone_count)):
                        chain_lengths.append({
                            'sample': sample,
                            'condition': condition,
                            'chain': chain,
                            'isotype': isotype,
                            'length': len(cdr3)
                        })
        
        # Convert to DataFrame
        lengths_df = pd.DataFrame(chain_lengths)
        
        # Save data
        lengths_df.to_csv(os.path.join(output_dir, "cdr3_lengths.csv"), index=False)
        
        # Plot length distribution by chain and condition
        if not lengths_df.empty:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=lengths_df, x='chain', y='length', hue='condition')
            plt.title("CDR3 Length Distribution by Chain and Condition")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cdr3_length_boxplot.pdf"))
            plt.close()
            
            # Plot length distribution histograms
            plt.figure(figsize=(12, 6))
            sns.histplot(data=lengths_df, x='length', hue='chain', multiple='dodge', bins=range(5, 31))
            plt.title("CDR3 Length Distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cdr3_length_histogram.pdf"))
            plt.close()
            
            # For BCR, plot length distribution by isotype for heavy chains
            if self.receptor_type == 'bcr':
                heavy_df = lengths_df[lengths_df['chain'] == 'IGH']
                if not heavy_df.empty and 'isotype' in heavy_df.columns:
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(data=heavy_df, x='isotype', y='length')
                    plt.title("Heavy Chain CDR3 Length by Isotype")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "igh_cdr3_length_by_isotype.pdf"))
                    plt.close()
            
            # If conditions are available, plot by condition
            if len(lengths_df['condition'].unique()) > 1:
                g = sns.FacetGrid(lengths_df, col='chain', row='condition', height=4, aspect=1.5)
                g.map(sns.histplot, 'length', bins=range(5, 31))
                g.add_legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "cdr3_length_by_condition.pdf"))
                plt.close()
    
    def _analyze_cdr3_aa_composition(self, output_dir):
        """
        Analyze CDR3 amino acid composition
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        aa_counts = {aa: defaultdict(int) for aa in amino_acids}
        
        # Count amino acids in CDR3 sequences (weighted by clone count)
        for _, clone in self.clonotypes_df.iterrows():
            sample = clone['sample']
            clone_count = clone['clone_count']
            
            # Get condition if available
            condition = "Unknown"
            if self.sample_info is not None:
                sample_condition = self.sample_info[self.sample_info['sample'] == sample]
                if not sample_condition.empty:
                    condition = sample_condition['condition'].iloc[0]
            
            # Process each chain type
            for chain in self.heavy_chains + self.light_chains:
                chain_lower = chain.lower()
                cdr3 = clone.get(f"{chain_lower}_cdr3", "")
                
                if cdr3 and cdr3 != "":
                    for aa in cdr3:
                        if aa in amino_acids:
                            aa_counts[aa][(sample, condition, chain)] += clone_count
        
        # Convert to DataFrame
        aa_composition = []
        
        for aa in amino_acids:
            for (sample, condition, chain), count in aa_counts[aa].items():
                aa_composition.append({
                    'sample': sample,
                    'condition': condition,
                    'chain': chain,
                    'amino_acid': aa,
                    'count': count
                })
        
        aa_composition_df = pd.DataFrame(aa_composition)
        
        # Save data
        aa_composition_df.to_csv(os.path.join(output_dir, "cdr3_aa_composition.csv"), index=False)
        
        # Normalize by total count within each sample and chain
        if not aa_composition_df.empty:
            total_counts = aa_composition_df.groupby(['sample', 'chain'])['count'].sum().reset_index()
            total_counts.rename(columns={'count': 'total'}, inplace=True)
            
            aa_composition_df = aa_composition_df.merge(total_counts, on=['sample', 'chain'])
            aa_composition_df['frequency'] = aa_composition_df['count'] / aa_composition_df['total'] * 100
            
            # Plot amino acid composition
            plt.figure(figsize=(15, 10))
            g = sns.FacetGrid(data=aa_composition_df, col='chain', row='condition', height=4, aspect=1.5)
            g.map(sns.barplot, 'amino_acid', 'frequency')
            g.add_legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cdr3_aa_composition.pdf"))
            plt.close()
            
            # Plot heatmap of amino acid composition by chain
            for chain in aa_composition_df['chain'].unique():
                chain_data = aa_composition_df[aa_composition_df['chain'] == chain]
                if not chain_data.empty:
                    pivot_data = chain_data.pivot_table(index='amino_acid', columns='sample', values='frequency', fill_value=0)
                    
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(pivot_data, cmap='viridis', annot=True, fmt='.1f')
                    plt.title(f"{chain} CDR3 Amino Acid Composition")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{chain}_cdr3_aa_heatmap.pdf"))
                    plt.close()
    
    def _analyze_cdr3_motifs(self, output_dir):
        """
        Analyze CDR3 sequence motifs
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        # Get all unique CDR3 sequences with their counts for each chain
        chain_seqs = defaultdict(lambda: defaultdict(int))
        
        for _, clone in self.clonotypes_df.iterrows():
            clone_count = clone['clone_count']
            
            # Process each chain type
            for chain in self.heavy_chains + self.light_chains:
                chain_lower = chain.lower()
                cdr3 = clone.get(f"{chain_lower}_cdr3", "")
                
                if cdr3 and cdr3 != "":
                    chain_seqs[chain][cdr3] += clone_count
        
        # Find common motifs in CDR3 sequences for each chain
        for chain, seqs in chain_seqs.items():
            if seqs:
                motifs = self._find_motifs(seqs, chain, output_dir)
                
                # Save motif data
                if motifs:
                    with open(os.path.join(output_dir, f"{chain}_motifs.txt"), "w") as f:
                        for motif, count in motifs.items():
                            f.write(f"{motif}\t{count}\n")
    
    def _find_motifs(self, seq_dict, chain, output_dir, min_count=5, motif_length=3):
        """
        Find common motifs in CDR3 sequences
        
        Parameters:
        -----------
        seq_dict : dict
            Dictionary of CDR3 sequences and their counts
        chain : str
            Chain type
        output_dir : str
            Output directory
        min_count : int
            Minimum count for a motif to be considered
        motif_length : int
            Length of motifs to identify
        
        Returns:
        --------
        dict:
            Dictionary of motifs and their counts
        """
        if not seq_dict:
            return {}
            
        # Find all motifs of specified length
        motif_counts = defaultdict(int)
        
        for seq, count in seq_dict.items():
            if len(seq) >= motif_length:
                for i in range(len(seq) - motif_length + 1):
                    motif = seq[i:i+motif_length]
                    motif_counts[motif] += count
        
        # Filter by minimum count
        filtered_motifs = {motif: count for motif, count in motif_counts.items() if count >= min_count}
        
        # Plot top motifs
        if filtered_motifs:
            top_motifs = dict(sorted(filtered_motifs.items(), key=lambda x: x[1], reverse=True)[:20])
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x=list(top_motifs.keys()), y=list(top_motifs.values()))
            plt.title(f"Top {chain} CDR3 Motifs (length {motif_length})")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{chain.lower()}_top_motifs.pdf"))
            plt.close()
        
        return filtered_motifs
    
    def _analyze_cdr3_properties(self, output_dir):
        """
        Analyze physiochemical properties of CDR3 sequences
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        # Define property scales
        hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        charge = {
            'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0, 
            'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0, 
            'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1, 
            'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
        }
        
        # Calculate properties for each CDR3
        properties = []
        
        for _, clone in self.clonotypes_df.iterrows():
            sample = clone['sample']
            clone_count = clone['clone_count']
            
            # Get condition if available
            condition = "Unknown"
            if self.sample_info is not None:
                sample_condition = self.sample_info[self.sample_info['sample'] == sample]
                if not sample_condition.empty:
                    condition = sample_condition['condition'].iloc[0]
            
            # Process each chain type
            for chain in self.heavy_chains + self.light_chains:
                chain_lower = chain.lower()
                cdr3 = clone.get(f"{chain_lower}_cdr3", "")
                
                if cdr3 and cdr3 != "":
                    isotype = clone.get('isotype', None) if self.receptor_type == 'bcr' and chain == 'IGH' else None
                    
                    # Calculate properties
                    hydrophobicity_score = sum(hydrophobicity.get(aa, 0) for aa in cdr3) / len(cdr3)
                    charge_score = sum(charge.get(aa, 0) for aa in cdr3)
                    
                    properties.append({
                        'sample': sample,
                        'condition': condition,
                        'chain': chain,
                        'isotype': isotype,
                        'cdr3_length': len(cdr3),
                        'hydrophobicity': hydrophobicity_score,
                        'charge': charge_score,
                        'clone_count': clone_count
                    })
        
        # Convert to DataFrame
        properties_df = pd.DataFrame(properties)
        
        # Save data
        properties_df.to_csv(os.path.join(output_dir, "cdr3_properties.csv"), index=False)
        
        # Plot property distributions
        if not properties_df.empty:
            # Hydrophobicity by chain
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=properties_df, x='chain', y='hydrophobicity', hue='condition')
            plt.title("CDR3 Hydrophobicity by Chain")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cdr3_hydrophobicity_by_chain.pdf"))
            plt.close()
            
            # Charge by chain
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=properties_df, x='chain', y='charge', hue='condition')
            plt.title("CDR3 Net Charge by Chain")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cdr3_charge_by_chain.pdf"))
            plt.close()
            
            # Hydrophobicity vs Length
            plt.figure(figsize=(10, 6))
            g = sns.scatterplot(data=properties_df, x='cdr3_length', y='hydrophobicity', 
                              hue='chain', size='clone_count', sizes=(20, 200), alpha=0.7)
            plt.title("CDR3 Hydrophobicity vs Length")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cdr3_hydrophobicity_vs_length.pdf"))
            plt.close()
            
            # For BCR, if isotype information is available, plot by isotype
            if self.receptor_type == 'bcr':
                heavy_df = properties_df[properties_df['chain'] == 'IGH']
                if not heavy_df.empty and 'isotype' in heavy_df.columns:
                    heavy_df = heavy_df[~heavy_df['isotype'].isna()]
                    
                    if not heavy_df.empty:
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(data=heavy_df, x='isotype', y='hydrophobicity')
                        plt.title("Heavy Chain CDR3 Hydrophobicity by Isotype")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, "igh_cdr3_hydrophobicity_by_isotype.pdf"))
                        plt.close()
                        
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(data=heavy_df, x='isotype', y='charge')
                        plt.title("Heavy Chain CDR3 Net Charge by Isotype")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, "igh_cdr3_charge_by_isotype.pdf"))
                        plt.close()
    
    def analyze_isotype_distribution(self):
        """
        Analyze isotype distribution (BCR only)
        """
        if self.receptor_type != 'bcr':
            print("Isotype analysis is only applicable for BCR data.")
            return None
        
        print("Analyzing isotype distribution...")
        
        if self.clonotypes_df is None:
            print("Error: No clonotype data available. Run run_mixcr_analysis first.")
            return None
        
        # Create directory for isotype analysis
        isotype_dir = os.path.join(self.output_dir, "isotype_analysis")
        os.makedirs(isotype_dir, exist_ok=True)
        
        # Calculate isotype distribution
        isotype_counts = []
        
        for sample, sample_df in self.clonotypes_df.groupby('sample'):
            # Get condition if available
            condition = "Unknown"
            if self.sample_info is not None:
                sample_condition = self.sample_info[self.sample_info['sample'] == sample]
                if not sample_condition.empty:
                    condition = sample_condition['condition'].iloc[0]
            
            # Count clones by isotype (weighted by clone count)
            isotype_count = defaultdict(int)
            total_clones = 0
            
            for _, clone in sample_df.iterrows():
                isotype = clone.get('isotype', 'Unknown')
                clone_count = clone['clone_count']
                
                isotype_count[isotype] += clone_count
                total_clones += clone_count
            
            # Calculate percentages
            for isotype, count in isotype_count.items():
                isotype_counts.append({
                    'sample': sample,
                    'condition': condition,
                    'isotype': isotype,
                    'count': count,
                    'percentage': (count / total_clones * 100) if total_clones > 0 else 0
                })
        
        # Convert to DataFrame
        isotype_df = pd.DataFrame(isotype_counts)
        
        # Save isotype distribution data
        isotype_df.to_csv(os.path.join(isotype_dir, "isotype_distribution.csv"), index=False)
        
        # Plot isotype distribution
        if not isotype_df.empty:
            plt.figure(figsize=(12, 8))
            sns.barplot(data=isotype_df, x='sample', y='percentage', hue='isotype')
            plt.title("Isotype Distribution by Sample")
            plt.xlabel("Sample")
            plt.ylabel("Percentage (%)")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(isotype_dir, "isotype_distribution_by_sample.pdf"))
            plt.close()
            
            # If conditions are available, plot isotype by condition
            if len(isotype_df['condition'].unique()) > 1:
                # Calculate mean isotype percentages by condition
                condition_isotype = isotype_df.groupby(['condition', 'isotype'])['percentage'].mean().reset_index()
                
                plt.figure(figsize=(10, 6))
                sns.barplot(data=condition_isotype, x='condition', y='percentage', hue='isotype')
                plt.title("Isotype Distribution by Condition")
                plt.xlabel("Condition")
                plt.ylabel("Percentage (%)")
                plt.tight_layout()
                plt.savefig(os.path.join(isotype_dir, "isotype_distribution_by_condition.pdf"))
                plt.close()
                
                # Statistical tests for isotype distribution differences
                if len(isotype_df['condition'].unique()) == 2:
                    # t-test for two conditions
                    test_results = []
                    
                    for isotype in isotype_df['isotype'].unique():
                        if isotype == 'Unknown':
                            continue
                            
                        isotype_data = isotype_df[isotype_df['isotype'] == isotype]
                        conditions = isotype_df['condition'].unique()
                        
                        group1 = isotype_data[isotype_data['condition'] == conditions[0]]['percentage'].values
                        group2 = isotype_data[isotype_data['condition'] == conditions[1]]['percentage'].values
                        
                        if len(group1) > 0 and len(group2) > 0:
                            stat, pval = stats.ttest_ind(group1, group2, equal_var=False)
                            
                            test_results.append({
                                'isotype': isotype,
                                'mean_percentage_condition1': np.mean(group1),
                                'mean_percentage_condition2': np.mean(group2),
                                'difference': np.mean(group1) - np.mean(group2),
                                'p_value': pval,
                                'significant': pval < 0.05
                            })
                    
                    if test_results:
                        test_df = pd.DataFrame(test_results)
                        test_df.to_csv(os.path.join(isotype_dir, "isotype_statistical_tests.csv"), index=False)
        
        # Store isotype distribution data
        self.isotype_distribution = isotype_df
        
        print("Isotype distribution analysis completed!")
        return isotype_df
    
    def analyze_clonal_expansion(self):
        """
        Analyze clonal expansion and identify expanded clonotypes
        """
        print("Analyzing clonal expansion...")
        
        if self.clonotypes_df is None:
            print("Error: No clonotype data available. Run run_mixcr_analysis first.")
            return None
        
        # Create a directory for clonal expansion analysis
        expansion_dir = os.path.join(self.output_dir, "clonal_expansion")
        os.makedirs(expansion_dir, exist_ok=True)
        
        # Calculate clonal expansion metrics
        expansion_metrics = []
        
        for sample, sample_df in self.clonotypes_df.groupby('sample'):
            # Get condition if available
            condition = "Unknown"
            if self.sample_info is not None:
                sample_condition = self.sample_info[self.sample_info['sample'] == sample]
                if not sample_condition.empty:
                    condition = sample_condition['condition'].iloc[0]
            
            # Get clone fractions
            clone_fractions = sample_df['clone_fraction'].values
            
            # Calculate expansion metrics
            total_clonotypes = len(sample_df)
            
            # Gini index for inequality in clone size distribution
            gini_index = self._calculate_gini(sample_df['clone_count'])
            
            # Clonality (1 - normalized Shannon entropy)
            clonality = 1 - self._calculate_diversity_metrics(sample_df)['normalized_entropy']
            
            # Top clone frequency
            top_clone_freq = sample_df['clone_fraction'].max() if not sample_df.empty else 0
            
            # Percent of repertoire occupied by top 10 clones
            top10_percent = sample_df.nlargest(10, 'clone_fraction')['clone_fraction'].sum() if total_clonotypes >= 10 else 1.0
            
            # Expansion threshold (clones > 0.1% of repertoire)
            expanded_clones = sample_df[sample_df['clone_fraction'] > 0.001]
            expanded_count = len(expanded_clones)
            expanded_percent = expanded_clones['clone_fraction'].sum() if not expanded_clones.empty else 0
            
            # Store metrics
            expansion_metrics.append({
                'sample': sample,
                'condition': condition,
                'total_clonotypes': total_clonotypes,
                'gini_index': gini_index,
                'clonality': clonality,
                'top_clone_freq': top_clone_freq,
                'top10_percent': top10_percent,
                'expanded_clone_count': expanded_count,
                'expanded_clone_percent': expanded_percent
            })
        
        # Convert to DataFrame
        expansion_df = pd.DataFrame(expansion_metrics)
        
        # Save data
        expansion_df.to_csv(os.path.join(expansion_dir, "clonal_expansion_metrics.csv"), index=False)
        
        # Plot clonal expansion metrics
        self._plot_expansion_metrics(expansion_df, expansion_dir)
        
        # Identify and save expanded clonotypes
        self._identify_expanded_clonotypes(expansion_dir)
        
        print("Clonal expansion analysis completed!")
        return expansion_df
    
    def _calculate_gini(self, values):
        """
        Calculate Gini index (measure of inequality)
        
        Parameters:
        -----------
        values : array-like
            Array of values (clone counts)
        
        Returns:
        --------
        float:
            Gini index
        """
        if len(values) <= 1 or np.sum(values) == 0:
            return 0
        
        # Sort values
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        # Calculate Gini index
        cum_sum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cum_sum) / cum_sum[-1]) / n
    
    def _plot_expansion_metrics(self, expansion_df, output_dir):
        """
        Plot clonal expansion metrics
        
        Parameters:
        -----------
        expansion_df : pd.DataFrame
            Expansion metrics dataframe
        output_dir : str
            Output directory
        """
        # Prepare data for plotting
        if 'condition' in expansion_df.columns and len(expansion_df['condition'].unique()) > 1:
            # Create plots comparing conditions
            metrics = ['gini_index', 'clonality', 'top_clone_freq', 'top10_percent', 
                     'expanded_clone_count', 'expanded_clone_percent']
            
            for metric in metrics:
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=expansion_df, x='condition', y=metric)
                sns.stripplot(data=expansion_df, x='condition', y=metric, color='black', dodge=True)
                plt.title(f"{metric.replace('_', ' ').title()} by Condition")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{metric}_by_condition.pdf"))
                plt.close()
        
        # Plot frequency distribution of clones for each sample
        self._plot_clone_distribution(output_dir)
    
    def _plot_clone_distribution(self, output_dir):
        """
        Plot clone size distribution for each sample
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        for sample, sample_df in self.clonotypes_df.groupby('sample'):
            # Get condition if available
            condition = "Unknown"
            if self.sample_info is not None:
                sample_condition = self.sample_info[self.sample_info['sample'] == sample]
                if not sample_condition.empty:
                    condition = sample_condition['condition'].iloc[0]
            
            # Sort clones by frequency
            sorted_clones = sample_df.sort_values('clone_fraction', ascending=False).reset_index(drop=True)
            
            # Create rank vs frequency plot
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(sorted_clones) + 1), sorted_clones['clone_fraction'] * 100)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Clone Rank')
            plt.ylabel('Frequency (%)')
            plt.title(f"Clone Size Distribution - {sample} ({condition})")
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{sample}_clone_distribution.pdf"))
            plt.close()
    
    def _identify_expanded_clonotypes(self, output_dir):
        """
        Identify and save expanded clonotypes
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        # Define expansion threshold (clones > 0.1% of repertoire)
        threshold = 0.001
        
        # Identify expanded clones for each sample
        expanded_clones = []
        
        for sample, sample_df in self.clonotypes_df.groupby('sample'):
            # Get condition if available
            condition = "Unknown"
            if self.sample_info is not None:
                sample_condition = self.sample_info[self.sample_info['sample'] == sample]
                if not sample_condition.empty:
                    condition = sample_condition['condition'].iloc[0]
            
            # Find expanded clones
            sample_expanded = sample_df[sample_df['clone_fraction'] > threshold].copy()
            sample_expanded['condition'] = condition
            
            expanded_clones.append(sample_expanded)
        
        # Combine all expanded clones
        if expanded_clones:
            all_expanded = pd.concat(expanded_clones, ignore_index=True)
            all_expanded.to_csv(os.path.join(output_dir, "expanded_clonotypes.csv"), index=False)
            
            # Plot expanded clone characteristics
            self._plot_expanded_characteristics(all_expanded, output_dir)
    
    def _plot_expanded_characteristics(self, expanded_df, output_dir):
        """
        Plot characteristics of expanded clones
        
        Parameters:
        -----------
        expanded_df : pd.DataFrame
            Expanded clones dataframe
        output_dir : str
            Output directory
        """
        if expanded_df.empty:
            return
            
        # Plot CDR3 length distribution of expanded clones
        # Group by chain type
        chain_lengths = []
        
        for _, clone in expanded_df.iterrows():
            # Process each chain type
            for chain in self.heavy_chains + self.light_chains:
                chain_lower = chain.lower()
                cdr3 = clone.get(f"{chain_lower}_cdr3", "")
                
                if cdr3 and cdr3 != "":
                    chain_lengths.append({
                        'chain': chain,
                        'length': len(cdr3)
                    })
        
        if chain_lengths:
            lengths_df = pd.DataFrame(chain_lengths)
            
            plt.figure(figsize=(10, 6))
            sns.histplot(data=lengths_df, x='length', hue='chain', bins=range(5, 31), multiple='dodge')
            plt.title("CDR3 Length Distribution of Expanded Clones")
            plt.xlabel("CDR3 Length")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "expanded_cdr3_length.pdf"))
            plt.close()
        
        # For BCR, plot isotype distribution of expanded clones
        if self.receptor_type == 'bcr' and 'isotype' in expanded_df.columns:
            isotype_counts = expanded_df['isotype'].value_counts()
            
            plt.figure(figsize=(10, 6))
            isotype_counts.plot(kind='bar')
            plt.title("Isotype Distribution of Expanded Clones")
            plt.xlabel("Isotype")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "expanded_isotype_distribution.pdf"))
            plt.close()
    
    def calculate_diversity_metrics(self):
        """
        Calculate repertoire diversity metrics
        """
        print("Calculating diversity metrics...")
        
        if self.clonotypes_df is None:
            print("Error: No clonotype data available. Run run_mixcr_analysis first.")
            return None
        
        # Create directory for diversity analysis
        diversity_dir = os.path.join(self.output_dir, "diversity")
        os.makedirs(diversity_dir, exist_ok=True)
        
        # Calculate diversity metrics for each sample
        diversity_metrics = []
        
        for sample, sample_df in self.clonotypes_df.groupby('sample'):
            # Get condition if available
            condition = "Unknown"
            if self.sample_info is not None:
                sample_condition = self.sample_info[self.sample_info['sample'] == sample]
                if not sample_condition.empty:
                    condition = sample_condition['condition'].iloc[0]
            
            # Calculate diversity metrics
            metrics = self._calculate_diversity_metrics(sample_df)
            
            # Store metrics
            metrics_dict = {
                'sample': sample,
                'condition': condition,
                'richness': len(sample_df),
                'shannon_entropy': metrics['shannon_entropy'],
                'normalized_entropy': metrics['normalized_entropy'],
                'clonality': 1 - metrics['normalized_entropy'],
                'simpson_index': metrics['simpson_index'],
                'inverse_simpson': metrics['inverse_simpson']
            }
            
            diversity_metrics.append(metrics_dict)
        
        # Convert to DataFrame
        diversity_df = pd.DataFrame(diversity_metrics)
        
        # Save diversity metrics
        diversity_df.to_csv(os.path.join(diversity_dir, "diversity_metrics.csv"), index=False)
        
        # Plot diversity metrics
        self._plot_diversity_metrics(diversity_df, diversity_dir)
        
        # Store results
        self.diversity_metrics = diversity_df
        
        print("Diversity metrics calculation completed!")
        return diversity_df
    
    def _plot_diversity_metrics(self, diversity_df, output_dir):
        """
        Plot diversity metrics
        
        Parameters:
        -----------
        diversity_df : pd.DataFrame
            Diversity metrics dataframe
        output_dir : str
            Output directory
        """
        # Check if we have multiple conditions to compare
        if 'condition' in diversity_df.columns and len(diversity_df['condition'].unique()) > 1:
            # Metrics to plot
            metrics = ['richness', 'shannon_entropy', 'normalized_entropy', 'clonality', 
                     'simpson_index', 'inverse_simpson']
            
            # Plot each metric
            for metric in metrics:
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=diversity_df, x='condition', y=metric)
                sns.stripplot(data=diversity_df, x='condition', y=metric, color='black', dodge=True)
                plt.title(f"{metric.replace('_', ' ').title()} by Condition")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{metric}_by_condition.pdf"))
                plt.close()
            
            # Add statistical tests for comparison between conditions
            if len(diversity_df['condition'].unique()) == 2:
                # t-test for two conditions
                stats_results = []
                
                for metric in metrics:
                    group1 = diversity_df[diversity_df['condition'] == diversity_df['condition'].unique()[0]][metric].values
                    group2 = diversity_df[diversity_df['condition'] == diversity_df['condition'].unique()[1]][metric].values
                    
                    if len(group1) > 0 and len(group2) > 0:
                        stat, pval = stats.ttest_ind(group1, group2, equal_var=False)
                        
                        stats_results.append({
                            'metric': metric,
                            'test': 't-test',
                            'statistic': stat,
                            'p_value': pval,
                            'significant': pval < 0.05
                        })
                
                # Save statistical test results
                if stats_results:
                    stats_df = pd.DataFrame(stats_results)
                    stats_df.to_csv(os.path.join(output_dir, "diversity_statistics.csv"), index=False)
        
        # Create radar chart for diversity metrics
        metrics = ['richness', 'shannon_entropy', 'clonality', 'inverse_simpson']
        
        # Normalize metrics to 0-1 scale for radar chart
        radar_df = diversity_df[['sample'] + metrics].copy()
        
        for metric in metrics:
            min_val = radar_df[metric].min()
            max_val = radar_df[metric].max()
            
            if max_val > min_val:
                radar_df[metric] = (radar_df[metric] - min_val) / (max_val - min_val)
        
        # Create radar chart
        samples = radar_df['sample'].values
        
        # Number of variables
        N = len(metrics)
        
        # Create angle for each variable
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create radar plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Draw one line per sample and fill area
        for i, sample in enumerate(samples):
            values = radar_df[radar_df['sample'] == sample][metrics].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=sample)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels and title
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        plt.title("Diversity Metrics Comparison", size=15)
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "diversity_radar_chart.pdf"))
        plt.close()
    
    def calculate_repertoire_overlap(self):
        """
        Calculate repertoire overlap between samples
        """
        print("Calculating repertoire overlap...")
        
        if self.clonotypes_df is None:
            print("Error: No clonotype data available. Run run_mixcr_analysis first.")
            return None
        
        # Create directory for overlap analysis
        overlap_dir = os.path.join(self.output_dir, "overlap")
        os.makedirs(overlap_dir, exist_ok=True)
        
        # Get all samples
        samples = self.clonotypes_df['sample'].unique()
        n_samples = len(samples)
        
        if n_samples <= 1:
            print("Warning: Need at least 2 samples to calculate overlap.")
            return None
        
        # Initialize overlap matrices
        jaccard_matrix = np.zeros((n_samples, n_samples))
        morisita_matrix = np.zeros((n_samples, n_samples))
        overlap_count_matrix = np.zeros((n_samples, n_samples))
        
        # For each pair of samples, calculate overlap metrics
        for i, sample1 in enumerate(samples):
            # Get CDR3 sequences and their frequencies for sample1
            data1 = self.clonotypes_df[self.clonotypes_df['sample'] == sample1]
            
            # Extract unique CDR3 sequences and their frequencies
            seq1_freqs = {}
            for chain in self.heavy_chains + self.light_chains:
                chain_lower = chain.lower()
                
                for _, clone in data1.iterrows():
                    cdr3 = clone.get(f"{chain_lower}_cdr3", "")
                    if cdr3 and cdr3 != "":
                        seq1_freqs[cdr3] = clone['clone_fraction']
            
            for j, sample2 in enumerate(samples):
                if i == j:
                    # Self-comparison
                    jaccard_matrix[i, j] = 1.0
                    morisita_matrix[i, j] = 1.0
                    overlap_count_matrix[i, j] = len(seq1_freqs)
                    continue
                
                # Get CDR3 sequences and their frequencies for sample2
                data2 = self.clonotypes_df[self.clonotypes_df['sample'] == sample2]
                
                # Extract unique CDR3 sequences and their frequencies
                seq2_freqs = {}
                for chain in self.heavy_chains + self.light_chains:
                    chain_lower = chain.lower()
                    
                    for _, clone in data2.iterrows():
                        cdr3 = clone.get(f"{chain_lower}_cdr3", "")
                        if cdr3 and cdr3 != "":
                            seq2_freqs[cdr3] = clone['clone_fraction']
                
                # Calculate overlap metrics
                
                # Jaccard index
                overlap_seqs = set(seq1_freqs.keys()) & set(seq2_freqs.keys())
                all_seqs = set(seq1_freqs.keys()) | set(seq2_freqs.keys())
                
                jaccard = len(overlap_seqs) / len(all_seqs) if all_seqs else 0
                jaccard_matrix[i, j] = jaccard
                
                # Morisita-Horn index (weighted by frequency)
                if overlap_seqs:
                    sum_freq1_squared = sum(f**2 for f in seq1_freqs.values())
                    sum_freq2_squared = sum(f**2 for f in seq2_freqs.values())
                    
                    sum_freq_products = 0
                    for seq in overlap_seqs:
                        sum_freq_products += seq1_freqs[seq] * seq2_freqs[seq]
                    
                    if sum_freq1_squared > 0 and sum_freq2_squared > 0:
                        morisita = (2 * sum_freq_products) / (sum_freq1_squared + sum_freq2_squared)
                        morisita_matrix[i, j] = morisita
                
                # Count of overlapping sequences
                overlap_count_matrix[i, j] = len(overlap_seqs)
        
        # Create DataFrames
        jaccard_df = pd.DataFrame(jaccard_matrix, index=samples, columns=samples)
        morisita_df = pd.DataFrame(morisita_matrix, index=samples, columns=samples)
        overlap_count_df = pd.DataFrame(overlap_count_matrix, index=samples, columns=samples)
        
        # Save overlap metrics
        jaccard_df.to_csv(os.path.join(overlap_dir, "jaccard_overlap.csv"))
        morisita_df.to_csv(os.path.join(overlap_dir, "morisita_overlap.csv"))
        overlap_count_df.to_csv(os.path.join(overlap_dir, "overlap_counts.csv"))
        
        # Plot heatmaps
        plt.figure(figsize=(10, 8))
        sns.heatmap(jaccard_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        plt.title("Jaccard Overlap Index")
        plt.tight_layout()
        plt.savefig(os.path.join(overlap_dir, "jaccard_heatmap.pdf"))
        plt.close()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(morisita_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        plt.title("Morisita-Horn Overlap Index")
        plt.tight_layout()
        plt.savefig(os.path.join(overlap_dir, "morisita_heatmap.pdf"))
        plt.close()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(overlap_count_df, annot=True, cmap="YlGnBu")
        plt.title(f"Number of Overlapping {self.receptor_type.upper()} Sequences")
        plt.tight_layout()
        plt.savefig(os.path.join(overlap_dir, "overlap_count_heatmap.pdf"))
        plt.close()
        
        print("Repertoire overlap calculation completed!")
        return {
            'jaccard': jaccard_df,
            'morisita': morisita_df,
            'counts': overlap_count_df
        }
    
    def comparative_analysis(self):
        """
        Perform comparative statistical analysis between conditions
        """
        print("Performing comparative analysis between conditions...")
        
        if self.clonotypes_df is None:
            print("Error: No clonotype data available. Run run_mixcr_analysis first.")
            return None
        
        # Check if condition information is available
        if self.sample_info is None:
            print("Warning: No sample information with conditions available. Skipping comparative analysis.")
            return None
        
        # Create directory for comparative analysis
        comparative_dir = os.path.join(self.output_dir, "comparative_analysis")
        os.makedirs(comparative_dir, exist_ok=True)
        
        # Get all samples and conditions
        samples = self.clonotypes_df['sample'].unique()
        sample_conditions = {}
        
        for sample in samples:
            condition = "Unknown"
            sample_condition = self.sample_info[self.sample_info['sample'] == sample]
            if not sample_condition.empty:
                condition = sample_condition['condition'].iloc[0]
            sample_conditions[sample] = condition
        
        # Add condition to clonotypes
        self.clonotypes_df['condition'] = self.clonotypes_df['sample'].map(sample_conditions)
        
        # Get unique conditions
        conditions = list(set(sample_conditions.values()))
        
        if len(conditions) < 2:
            print("Warning: Need at least 2 conditions to perform comparative analysis.")
            return None
        
        # Perform comparative analyses
        self._analyze_vdj_usage_by_condition(comparative_dir)
        self._analyze_cdr3_features_by_condition(comparative_dir)
        self._analyze_clonal_expansion_by_condition(comparative_dir)
        
        # Add BCR-specific analyses
        if self.receptor_type == 'bcr':
            self._analyze_isotype_distribution_by_condition(comparative_dir)
        
        print("Comparative analysis completed!")
    
    def _analyze_vdj_usage_by_condition(self, output_dir):
        """
        Analyze V(D)J gene usage differences between conditions
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        # Group data by condition
        condition_data = self.clonotypes_df.groupby('condition')
        
        # Initialize dictionaries for V gene usage by condition
        v_usage_by_condition = defaultdict(lambda: defaultdict(int))
        
        # Calculate V gene usage by condition (weighted by clone count)
        for condition, group in condition_data:
            for _, clone in group.iterrows():
                clone_count = clone['clone_count']
                
                # Process each chain type
                for chain in self.heavy_chains + self.light_chains:
                    chain_lower = chain.lower()
                    v_gene = clone.get(f"{chain_lower}_v_gene", "")
                    
                    if v_gene:
                        v_usage_by_condition[condition][v_gene] += clone_count
        
        # Convert to DataFrame
        v_usage_df = pd.DataFrame(v_usage_by_condition).fillna(0)
        
        # Normalize by total usage within each condition
        v_usage_norm = v_usage_df.copy()
        for col in v_usage_norm.columns:
            total = v_usage_norm[col].sum()
            if total > 0:
                v_usage_norm[col] = v_usage_norm[col] / total * 100
        
        # Save V gene usage by condition
        v_usage_norm.to_csv(os.path.join(output_dir, "v_gene_usage_by_condition.csv"))
        
        # Plot top 20 V genes by condition
        plt.figure(figsize=(14, 10))
        
        # Get top 20 V genes across all conditions
        top_v_genes = v_usage_norm.sum(axis=1).nlargest(20).index
        v_usage_top = v_usage_norm.loc[top_v_genes]
        
        # Plot heatmap
        sns.heatmap(v_usage_top, cmap="YlGnBu", annot=True, fmt=".1f")
        plt.title("Top 20 V Gene Usage by Condition")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "v_gene_usage_by_condition_heatmap.pdf"))
        plt.close()
        
        # Prepare data for grouped bar plot
        plot_data = []
        for v_gene in top_v_genes:
            for condition in v_usage_norm.columns:
                plot_data.append({
                    'V_gene': v_gene,
                    'Condition': condition,
                    'Frequency': v_usage_norm.loc[v_gene, condition]
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Plot grouped bar chart
        plt.figure(figsize=(16, 8))
        sns.barplot(data=plot_df, x='V_gene', y='Frequency', hue='Condition')
        plt.title("Top 20 V Gene Usage by Condition")
        plt.xlabel("V Gene")
        plt.ylabel("Frequency (%)")
        plt.xticks(rotation=90)
        plt.legend(title="Condition")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "v_gene_usage_by_condition_barplot.pdf"))
        plt.close()
        
        # Perform statistical tests for each V gene (if there are exactly 2 conditions)
        if len(v_usage_norm.columns) == 2:
            # t-test for two conditions
            test_results = []
            
            for v_gene in v_usage_norm.index:
                val1 = v_usage_norm.loc[v_gene, v_usage_norm.columns[0]]
                val2 = v_usage_norm.loc[v_gene, v_usage_norm.columns[1]]
                
                # Skip rare genes with very low usage
                if val1 + val2 < 0.5:
                    continue
                
                # We don't have enough samples for a proper t-test,
                # so we'll just calculate the fold change
                fold_change = val1 / val2 if val2 > 0 else float('inf')
                
                test_results.append({
                    'V_gene': v_gene,
                    f'{v_usage_norm.columns[0]}_freq': val1,
                    f'{v_usage_norm.columns[1]}_freq': val2,
                    'fold_change': fold_change,
                    'difference': val1 - val2
                })
            
            if test_results:
                test_df = pd.DataFrame(test_results)
                test_df = test_df.sort_values('difference', ascending=False)
                test_df.to_csv(os.path.join(output_dir, "v_gene_comparative_analysis.csv"), index=False)
                
                # Plot top differential V genes
                top_diff = test_df.nlargest(10, 'difference')
                bottom_diff = test_df.nsmallest(10, 'difference')
                top_diff_genes = pd.concat([top_diff, bottom_diff])
                
                plt.figure(figsize=(12, 8))
                sns.barplot(data=top_diff_genes, x='V_gene', y='difference')
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.title(f"Top Differentially Used V Genes ({v_usage_norm.columns[0]} vs {v_usage_norm.columns[1]})")
                plt.xlabel("V Gene")
                plt.ylabel(f"Difference in Usage (% {v_usage_norm.columns[0]} - % {v_usage_norm.columns[1]})")
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "differential_v_gene_usage.pdf"))
                plt.close()
    
    def _analyze_cdr3_features_by_condition(self, output_dir):
        """
        Analyze CDR3 feature differences between conditions
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        # Group data by condition
        condition_data = self.clonotypes_df.groupby('condition')
        
        # CDR3 lengths by condition
        cdr3_lengths = []
        
        for condition, group in condition_data:
            for _, clone in group.iterrows():
                clone_count = clone['clone_count']
                
                # Process each chain type
                for chain in self.heavy_chains + self.light_chains:
                    chain_lower = chain.lower()
                    cdr3 = clone.get(f"{chain_lower}_cdr3", "")
                    
                    if cdr3 and cdr3 != "":
                        cdr3_lengths.append({
                            'condition': condition,
                            'chain': chain,
                            'length': len(cdr3),
                            'count': clone_count
                        })
        
        # Convert to DataFrame
        lengths_df = pd.DataFrame(cdr3_lengths)
        
        if not lengths_df.empty:
            # Save data
            lengths_df.to_csv(os.path.join(output_dir, "cdr3_lengths_by_condition.csv"), index=False)
            
            # Plot CDR3 length distribution by condition
            plt.figure(figsize=(12, 6))
            for condition in lengths_df['condition'].unique():
                condition_data = lengths_df[lengths_df['condition'] == condition]
                
                # Calculate length distribution (weighted by count)
                length_dist = {}
                for _, row in condition_data.iterrows():
                    length = row['length']
                    count = row['count']
                    length_dist[length] = length_dist.get(length, 0) + count
                
                # Convert to array for plotting
                lengths = sorted(length_dist.keys())
                counts = [length_dist[l] for l in lengths]
                
                # Normalize
                counts = np.array(counts) / sum(counts) * 100
                
                plt.plot(lengths, counts, label=condition)
            
            plt.xlabel('CDR3 Length')
            plt.ylabel('Percentage (%)')
            plt.title('CDR3 Length Distribution by Condition')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cdr3_length_by_condition.pdf"))
            plt.close()
            
            # Plot by chain and condition
            g = sns.FacetGrid(lengths_df, col='chain', row='condition', height=4, aspect=1.5)
            g.map(sns.histplot, 'length', bins=range(5, 31), weights='count')
            g.add_legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cdr3_length_by_chain_condition.pdf"))
            plt.close()
            
            # Statistical comparison of lengths (if exactly 2 conditions)
            if len(lengths_df['condition'].unique()) == 2:
                stats_df = []
                
                for chain in lengths_df['chain'].unique():
                    chain_data = lengths_df[lengths_df['chain'] == chain]
                    
                    # Group by condition and calculate weighted average length
                    condition_lengths = {}
                    condition_counts = {}
                    
                    for condition in chain_data['condition'].unique():
                        cond_data = chain_data[chain_data['condition'] == condition]
                        
                        # Weighted average length
                        weighted_length = np.average(cond_data['length'], weights=cond_data['count'])
                        condition_lengths[condition] = weighted_length
                        condition_counts[condition] = cond_data['count'].sum()
                    
                    # Calculate difference
                    conditions = list(condition_lengths.keys())
                    
                    stats_df.append({
                        'chain': chain,
                        f'{conditions[0]}_avg_length': condition_lengths[conditions[0]],
                        f'{conditions[1]}_avg_length': condition_lengths[conditions[1]],
                        'difference': condition_lengths[conditions[0]] - condition_lengths[conditions[1]]
                    })
                
                if stats_df:
                    stats_df = pd.DataFrame(stats_df)
                    stats_df.to_csv(os.path.join(output_dir, "cdr3_length_comparison.csv"), index=False)
                    
                    # Plot comparison
                    plt.figure(figsize=(8, 6))
                    sns.barplot(data=stats_df, x='chain', y='difference')
                    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    plt.title(f"Difference in Average CDR3 Length ({conditions[0]} - {conditions[1]})")
                    plt.ylabel('Difference (amino acids)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "cdr3_length_comparison.pdf"))
                    plt.close()
            
            # Analyze physicochemical properties by condition
            props_by_condition = self._analyze_aa_properties_by_condition(output_dir)
    
    def _analyze_aa_properties_by_condition(self, output_dir):
        """
        Analyze amino acid properties of CDR3 sequences by condition
        
        Parameters:
        -----------
        output_dir : str
            Output directory
            
        Returns:
        --------
        pd.DataFrame:
            Properties by condition
        """
        # Define property scales
        hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        charge = {
            'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0, 
            'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0, 
            'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1, 
            'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
        }
        
        # Calculate properties by condition
        properties = []
        
        for condition, condition_df in self.clonotypes_df.groupby('condition'):
            for _, clone in condition_df.iterrows():
                clone_count = clone['clone_count']
                
                # Process each chain type
                for chain in self.heavy_chains + self.light_chains:
                    chain_lower = chain.lower()
                    cdr3 = clone.get(f"{chain_lower}_cdr3", "")
                    
                    if cdr3 and cdr3 != "":
                        # Calculate properties
                        hydro_score = sum(hydrophobicity.get(aa, 0) for aa in cdr3) / len(cdr3)
                        charge_score = sum(charge.get(aa, 0) for aa in cdr3)
                        
                        properties.append({
                            'condition': condition,
                            'chain': chain,
                            'cdr3_length': len(cdr3),
                            'hydrophobicity': hydro_score,
                            'charge': charge_score,
                            'count': clone_count
                        })
        
        # Convert to DataFrame
        props_df = pd.DataFrame(properties)
        
        if not props_df.empty:
            # Save data
            props_df.to_csv(os.path.join(output_dir, "cdr3_properties_by_condition.csv"), index=False)
            
            # Plot hydrophobicity by condition
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=props_df, x='chain', y='hydrophobicity', hue='condition')
            plt.title("CDR3 Hydrophobicity by Chain and Condition")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cdr3_hydrophobicity_by_condition.pdf"))
            plt.close()
            
            # Plot charge by condition
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=props_df, x='chain', y='charge', hue='condition')
            plt.title("CDR3 Net Charge by Chain and Condition")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cdr3_charge_by_condition.pdf"))
            plt.close()
            
            # Statistical comparison (if exactly 2 conditions)
            if len(props_df['condition'].unique()) == 2:
                stats_df = []
                
                for chain in props_df['chain'].unique():
                    chain_data = props_df[props_df['chain'] == chain]
                    
                    # Group by condition and calculate weighted average properties
                    condition_hydro = {}
                    condition_charge = {}
                    
                    for condition in chain_data['condition'].unique():
                        cond_data = chain_data[chain_data['condition'] == condition]
                        
                        # Weighted average properties
                        weighted_hydro = np.average(cond_data['hydrophobicity'], weights=cond_data['count'])
                        weighted_charge = np.average(cond_data['charge'], weights=cond_data['count'])
                        
                        condition_hydro[condition] = weighted_hydro
                        condition_charge[condition] = weighted_charge
                    
                    # Calculate differences
                    conditions = list(condition_hydro.keys())
                    
                    stats_df.append({
                        'chain': chain,
                        'property': 'hydrophobicity',
                        f'{conditions[0]}_avg': condition_hydro[conditions[0]],
                        f'{conditions[1]}_avg': condition_hydro[conditions[1]],
                        'difference': condition_hydro[conditions[0]] - condition_hydro[conditions[1]]
                    })
                    
                    stats_df.append({
                        'chain': chain,
                        'property': 'charge',
                        f'{conditions[0]}_avg': condition_charge[conditions[0]],
                        f'{conditions[1]}_avg': condition_charge[conditions[1]],
                        'difference': condition_charge[conditions[0]] - condition_charge[conditions[1]]
                    })
                
                if stats_df:
                    stats_df = pd.DataFrame(stats_df)
                    stats_df.to_csv(os.path.join(output_dir, "cdr3_properties_comparison.csv"), index=False)
                    
                    # Plot property comparison
                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=stats_df, x='chain', y='difference', hue='property')
                    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    plt.title(f"Difference in CDR3 Properties ({conditions[0]} - {conditions[1]})")
                    plt.ylabel('Difference')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "cdr3_properties_comparison.pdf"))
                    plt.close()
        
        return props_df
    
    def _analyze_clonal_expansion_by_condition(self, output_dir):
        """
        Analyze clonal expansion differences between conditions
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        # Group data by condition and sample
        condition_samples = {}
        for sample in self.clonotypes_df['sample'].unique():
            condition = self.clonotypes_df[self.clonotypes_df['sample'] == sample]['condition'].iloc[0]
            if condition not in condition_samples:
                condition_samples[condition] = []
            condition_samples[condition].append(sample)
        
        # Calculate expansion metrics for each sample
        expansion_metrics = []
        
        for condition, samples in condition_samples.items():
            for sample in samples:
                sample_df = self.clonotypes_df[self.clonotypes_df['sample'] == sample]
                
                # Calculate metrics
                clone_counts = sample_df['clone_count'].values
                clone_fractions = sample_df['clone_fraction'].values
                
                # Gini index
                gini_index = self._calculate_gini(clone_counts)
                
                # Top clone frequency
                top_clone_freq = max(clone_fractions) if clone_fractions.size > 0 else 0
                
                # Clonality (1 - normalized entropy)
                metrics = self._calculate_diversity_metrics(sample_df)
                clonality = 1 - metrics['normalized_entropy']
                
                # Percent of repertoire occupied by top 10 clones
                top10_percent = sum(sorted(clone_fractions, reverse=True)[:10]) if len(clone_fractions) >= 10 else sum(clone_fractions)
                
                # Expanded clones (> 0.1% of repertoire)
                expanded_count = sum(1 for f in clone_fractions if f > 0.001)
                expanded_percent = sum(f for f in clone_fractions if f > 0.001)
                
                expansion_metrics.append({
                    'sample': sample,
                    'condition': condition,
                    'total_clonotypes': len(sample_df),
                    'gini_index': gini_index,
                    'clonality': clonality,
                    'top_clone_freq': top_clone_freq,
                    'top10_percent': top10_percent,
                    'expanded_clone_count': expanded_count,
                    'expanded_clone_percent': expanded_percent
                })
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(expansion_metrics)
        
        if not metrics_df.empty:
            # Save data
            metrics_df.to_csv(os.path.join(output_dir, "expansion_metrics_by_condition.csv"), index=False)
            
            # Plot metrics by condition
            metrics_to_plot = ['gini_index', 'clonality', 'top_clone_freq', 'top10_percent',
                            'expanded_clone_count', 'expanded_clone_percent']
            
            for metric in metrics_to_plot:
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=metrics_df, x='condition', y=metric)
                sns.stripplot(data=metrics_df, x='condition', y=metric, color='black', dodge=True)
                plt.title(f"{metric.replace('_', ' ').title()} by Condition")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{metric}_by_condition.pdf"))
                plt.close()
            
            # Statistical comparison (if exactly 2 conditions)
            if len(metrics_df['condition'].unique()) == 2:
                stats_df = []
                
                conditions = list(metrics_df['condition'].unique())
                
                for metric in metrics_to_plot:
                    group1 = metrics_df[metrics_df['condition'] == conditions[0]][metric].values
                    group2 = metrics_df[metrics_df['condition'] == conditions[1]][metric].values
                    
                    if len(group1) > 0 and len(group2) > 0:
                        # If enough samples, perform t-test
                        if len(group1) >= 3 and len(group2) >= 3:
                            stat, pval = stats.ttest_ind(group1, group2, equal_var=False)
                            test_type = 't-test'
                        else:
                            # Otherwise just calculate the difference
                            stat = np.mean(group1) - np.mean(group2)
                            pval = np.nan
                            test_type = 'mean difference'
                        
                        stats_df.append({
                            'metric': metric,
                            'test': test_type,
                            f'{conditions[0]}_mean': np.mean(group1),
                            f'{conditions[1]}_mean': np.mean(group2),
                            'difference': np.mean(group1) - np.mean(group2),
                            'statistic': stat,
                            'p_value': pval,
                            'significant': pval < 0.05 if not np.isnan(pval) else np.nan
                        })
                
                if stats_df:
                    stats_df = pd.DataFrame(stats_df)
                    stats_df.to_csv(os.path.join(output_dir, "expansion_metrics_comparison.csv"), index=False)
                    
                    # Plot metrics comparison
                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=stats_df, x='metric', y='difference')
                    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    plt.title(f"Difference in Expansion Metrics ({conditions[0]} - {conditions[1]})")
                    plt.ylabel('Difference')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "expansion_metrics_comparison.pdf"))
                    plt.close()
    
    def _analyze_isotype_distribution_by_condition(self, output_dir):
        """
        Analyze isotype distribution differences between conditions (BCR only)
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        if self.receptor_type != 'bcr':
            return
        
        # Check if we have isotype data
        if 'isotype' not in self.clonotypes_df.columns:
            return
        
        # Group data by condition
        condition_isotypes = defaultdict(lambda: defaultdict(int))
        
        for condition, condition_df in self.clonotypes_df.groupby('condition'):
            for _, clone in condition_df.iterrows():
                isotype = clone.get('isotype', 'Unknown')
                clone_count = clone['clone_count']
                
                condition_isotypes[condition][isotype] += clone_count
        
        # Convert to DataFrame
        isotype_data = []
        
        for condition, isotypes in condition_isotypes.items():
            total = sum(isotypes.values())
            
            for isotype, count in isotypes.items():
                isotype_data.append({
                    'condition': condition,
                    'isotype': isotype,
                    'count': count,
                    'percentage': (count / total * 100) if total > 0 else 0
                })
        
        isotype_df = pd.DataFrame(isotype_data)
        
        if not isotype_df.empty:
            # Save data
            isotype_df.to_csv(os.path.join(output_dir, "isotype_distribution_by_condition.csv"), index=False)
            
            # Plot isotype distribution by condition
            plt.figure(figsize=(12, 8))
            sns.barplot(data=isotype_df, x='isotype', y='percentage', hue='condition')
            plt.title("Isotype Distribution by Condition")
            plt.xlabel("Isotype")
            plt.ylabel("Percentage (%)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "isotype_distribution_by_condition.pdf"))
            plt.close()
            
            # Statistical comparison (if exactly 2 conditions)
            if len(isotype_df['condition'].unique()) == 2:
                stats_df = []
                
                conditions = list(isotype_df['condition'].unique())
                
                for isotype in isotype_df['isotype'].unique():
                    if isotype == 'Unknown':
                        continue
                        
                    val1 = isotype_df[(isotype_df['condition'] == conditions[0]) & 
                                    (isotype_df['isotype'] == isotype)]['percentage'].values
                    
                    val2 = isotype_df[(isotype_df['condition'] == conditions[1]) & 
                                    (isotype_df['isotype'] == isotype)]['percentage'].values
                    
                    if len(val1) > 0 and len(val2) > 0:
                        stats_df.append({
                            'isotype': isotype,
                            f'{conditions[0]}_percent': val1[0],
                            f'{conditions[1]}_percent': val2[0],
                            'difference': val1[0] - val2[0]
                        })
                
                if stats_df:
                    stats_df = pd.DataFrame(stats_df)
                    stats_df = stats_df.sort_values('difference', ascending=False)
                    stats_df.to_csv(os.path.join(output_dir, "isotype_distribution_comparison.csv"), index=False)
                    
                    # Plot isotype comparison
                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=stats_df, x='isotype', y='difference')
                    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    plt.title(f"Difference in Isotype Distribution ({conditions[0]} - {conditions[1]})")
                    plt.ylabel('Difference in Percentage (%)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "isotype_distribution_comparison.pdf"))
                    plt.close()
    
    def visualize_results(self):
        """
        Create comprehensive visualizations and summary report
        """
        print("Creating comprehensive visualizations...")
        
        # Create directory for final visualizations
        viz_dir = os.path.join(self.output_dir, "final_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create summary report
        self._create_summary_report(viz_dir)
        
        # Create summary plots
        self._create_summary_plots(viz_dir)
        
        print("Visualization completed!")
    
    def _create_summary_plots(self, output_dir):
        """
        Create summary plots for the analysis
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        # Plot sample overview
        if self.repertoire_df is not None:
            # Plot number of clonotypes per sample
            plt.figure(figsize=(10, 6))
            sns.barplot(data=self.repertoire_df, x='sample', y='unique_clonotypes')
            plt.title("Number of Unique Clonotypes per Sample")
            plt.xlabel("Sample")
            plt.ylabel("Unique Clonotypes")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "clonotypes_per_sample.pdf"))
            plt.close()
        
        # Plot V gene usage
        if self.vdj_usage is not None and 'v_usage' in self.vdj_usage:
            v_usage = self.vdj_usage['v_usage']
            
            # Get top V genes across all samples
            v_usage_sum = v_usage.sum(axis=1)
            top_v_genes = v_usage_sum.nlargest(20).index
            
            # Plot top V genes
            plt.figure(figsize=(12, 8))
            v_usage_top = v_usage.loc[top_v_genes]
            
            # Normalize by sample
            v_usage_norm = v_usage_top.copy()
            for col in v_usage_norm.columns:
                total = v_usage_norm[col].sum()
                if total > 0:
                    v_usage_norm[col] = v_usage_norm[col] / total * 100
            
            # Plot heatmap
            sns.heatmap(v_usage_norm, cmap="YlGnBu", annot=True, fmt=".1f")
            plt.title("Top V Gene Usage Across Samples")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "top_v_genes_heatmap.pdf"))
            plt.close()
        
        # For BCR, plot isotype distribution
        if self.receptor_type == 'bcr' and self.clonotypes_df is not None and 'isotype' in self.clonotypes_df.columns:
            # Calculate overall isotype distribution
            isotype_counts = Counter()
            isotype_dict = {}
            
            for sample, sample_df in self.clonotypes_df.groupby('sample'):
                sample_counts = Counter()
                for _, clone in sample_df.iterrows():
                    isotype = clone.get('isotype', 'Unknown')
                    clone_count = clone['clone_count']
                    sample_counts[isotype] += clone_count
                    isotype_counts[isotype] += clone_count
                
                # Calculate percentages
                total = sum(sample_counts.values())
                isotype_dict[sample] = {isotype: (count / total * 100) if total > 0 else 0
                                     for isotype, count in sample_counts.items()}
            
            # Convert to DataFrame for plotting
            isotype_data = []
            for sample, isotypes in isotype_dict.items():
                for isotype, percentage in isotypes.items():
                    isotype_data.append({
                        'sample': sample,
                        'isotype': isotype,
                        'percentage': percentage
                    })
            
            isotype_df = pd.DataFrame(isotype_data)
            
            if not isotype_df.empty:
                # Plot isotype distribution
                plt.figure(figsize=(12, 8))
                sns.barplot(data=isotype_df, x='sample', y='percentage', hue='isotype')
                plt.title("Isotype Distribution Across Samples")
                plt.xlabel("Sample")
                plt.ylabel("Percentage (%)")
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "isotype_distribution.pdf"))
                plt.close()
        
        # Plot diversity metrics comparison
        if self.diversity_metrics is not None:
            # Create radar chart
            metrics = ['richness', 'shannon_entropy', 'clonality', 'inverse_simpson']
            
            # Normalize metrics to 0-1 scale for radar chart
            radar_df = self.diversity_metrics[['sample'] + metrics].copy()
            
            for metric in metrics:
                min_val = radar_df[metric].min()
                max_val = radar_df[metric].max()
                
                if max_val > min_val:
                    radar_df[metric] = (radar_df[metric] - min_val) / (max_val - min_val)
            
            # Create radar chart
            samples = radar_df['sample'].values
            
            # Number of variables
            N = len(metrics)
            
            # Create angle for each variable
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Create radar plot
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            # Draw one line per sample and fill area
            for i, sample in enumerate(samples):
                values = radar_df[radar_df['sample'] == sample][metrics].values.flatten().tolist()
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, label=sample)
                ax.fill(angles, values, alpha=0.1)
            
            # Set labels and title
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            plt.title("Diversity Metrics Comparison", size=15)
            plt.legend(loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "diversity_radar_chart.pdf"))
            plt.close()
    
    def _create_summary_report(self, output_dir):
        """
        Create summary report in HTML format
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        # Start HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.receptor_type.upper()} Repertoire Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .section {{ margin-bottom: 30px; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .plot img {{ max-width: 800px; }}
            </style>
        </head>
        <body>
            <h1>{self.receptor_type.upper()} Repertoire Analysis Report</h1>
            
            <div class="section summary">
                <h2>Analysis Summary</h2>
        """
        
        # Add general info
        if self.repertoire_df is not None:
            num_samples = len(self.repertoire_df)
            
            # Get conditions
            conditions = "None"
            if self.sample_info is not None and 'condition' in self.sample_info.columns:
                conditions = ", ".join(self.sample_info['condition'].unique())
            
            # Calculate total metrics
            total_clonotypes = sum(self.repertoire_df['unique_clonotypes'])
            
            html_content += f"""
                <p>Number of samples analyzed: {num_samples}</p>
                <p>Total unique clonotypes: {total_clonotypes}</p>
                <p>Conditions: {conditions}</p>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Repertoire Metrics</h2>
                
                <table>
                    <tr>
                        <th>Sample</th>
                        <th>Condition</th>
                        <th>Unique Clonotypes</th>
                        <th>Shannon Entropy</th>
                        <th>Clonality</th>
        """
        
        # Add receptor-specific columns
        if self.receptor_type == 'tcr':
            html_content += """
                        <th>Alpha Chains</th>
                        <th>Beta Chains</th>
            """
        else:  # BCR
            html_content += """
                        <th>Heavy Chains</th>
                        <th>Light Chains</th>
            """
        
        html_content += """
                    </tr>
        """
        
        # Add sample data
        if self.repertoire_df is not None:
            for _, row in self.repertoire_df.iterrows():
                html_content += f"""
                    <tr>
                        <td>{row['sample']}</td>
                        <td>{row['condition']}</td>
                        <td>{row['unique_clonotypes']}</td>
                        <td>{row['shannon_entropy']:.2f}</td>
                        <td>{row['clonality']:.2f}</td>
                """
                
                # Add receptor-specific data
                if self.receptor_type == 'tcr':
                    html_content += f"""
                        <td>{row['alpha_chains']}</td>
                        <td>{row['beta_chains']}</td>
                    """
                else:  # BCR
                    html_content += f"""
                        <td>{row['heavy_chains']}</td>
                        <td>{row['kappa_chains'] + row['lambda_chains']}</td>
                    """
                
                html_content += """
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Key Visualizations</h2>
                
                <div class="plot">
                    <h3>Clonotypes per Sample</h3>
                    <img src="clonotypes_per_sample.pdf" alt="Clonotypes per Sample">
                </div>
                
                <div class="plot">
                    <h3>V Gene Usage</h3>
                    <img src="top_v_genes_heatmap.pdf" alt="V Gene Usage">
                </div>
                
                <div class="plot">
                    <h3>Diversity Metrics</h3>
                    <img src="diversity_radar_chart.pdf" alt="Diversity Metrics">
                </div>
        """
        
        # Add BCR-specific plots
        if self.receptor_type == 'bcr':
            html_content += """
                <div class="plot">
                    <h3>Isotype Distribution</h3>
                    <img src="isotype_distribution.pdf" alt="Isotype Distribution">
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Analysis Details</h2>
                <p>This report was generated using MiXCR for repertoire analysis. The analysis included the following steps:</p>
                <ul>
                    <li>Extraction of {self.receptor_type.upper()} sequences from raw data</li>
                    <li>V(D)J gene annotation and CDR3 identification</li>
                    <li>Clonotype assignment and frequency calculation</li>
                    <li>V(D)J gene usage analysis</li>
                    <li>CDR3 sequence feature analysis</li>
                    <li>Clonal expansion analysis</li>
                    <li>Diversity metrics calculation</li>
                    <li>Repertoire overlap assessment</li>
        """
        
        # Add BCR-specific steps
        if self.receptor_type == 'bcr':
            html_content += """
                    <li>Isotype distribution analysis</li>
            """
        
        html_content += """
                </ul>
                <p>For full details, please refer to the detailed analysis folders in the output directory.</p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(os.path.join(output_dir, f"{self.receptor_type}_analysis_report.html"), "w") as f:
            f.write(html_content)


# Main function to run the pipeline
def run_mixcr_analysis(data_dir, output_dir, receptor_type='tcr', species='hsa', sample_info_file=None, threads=None):
    """
    Run the complete MiXCR-based repertoire analysis pipeline
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the FASTQ files
    output_dir : str
        Directory for saving analysis results
    receptor_type : str
        Type of receptor to analyze: 'tcr' or 'bcr'
    species : str
        Species code (hsa for human, mmu for mouse)
    sample_info_file : str, optional
        Path to sample information file (CSV with sample, condition columns)
    threads : int, optional
        Number of threads to use for MiXCR analysis
    
    Returns:
    --------
    MiXCRAnalyzer:
        Analyzer object with analysis results
    """
    print(f"Starting {receptor_type.upper()} analysis pipeline using MiXCR...")
    
    # Load sample information if provided
    sample_info = None
    if sample_info_file and os.path.exists(sample_info_file):
        sample_info = pd.read_csv(sample_info_file)
        print(f"Loaded sample information for {len(sample_info)} samples")
    
    # Initialize analyzer
    analyzer = MiXCRAnalyzer(data_dir, output_dir, receptor_type, species, threads, sample_info)
    
    # Run MiXCR analysis
    analyzer.run_mixcr_analysis()
    
    # Run analysis steps
    analyzer.analyze_vdj_usage()
    analyzer.analyze_cdr3_features()
    analyzer.analyze_clonal_expansion()
    analyzer.calculate_diversity_metrics()
    analyzer.calculate_repertoire_overlap()
    
    # Run BCR-specific analyses
    if receptor_type == 'bcr':
        analyzer.analyze_isotype_distribution()
    
    # Run comparative analysis if sample information with conditions is available
    if sample_info is not None and 'condition' in sample_info.columns:
        analyzer.comparative_analysis()
    
    # Create visualizations and summary report
    analyzer.visualize_results()
    
    print(f"{receptor_type.upper()} analysis pipeline completed!")
    return analyzer


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description=f'MiXCR-based TCR/BCR repertoire analysis pipeline')
    
    parser.add_argument('--data_dir', required=True, help='Directory containing the FASTQ files')
    parser.add_argument('--output_dir', required=True, help='Directory for saving analysis results')
    parser.add_argument('--receptor_type', choices=['tcr', 'bcr'], default='tcr', help='Type of receptor to analyze (tcr or bcr)')
    parser.add_argument('--species', choices=['hsa', 'mmu'], default='hsa', help='Species code (hsa for human, mmu for mouse)')
    parser.add_argument('--sample_info', help='Path to sample information file (CSV with sample,condition columns)')
    parser.add_argument('--threads', type=int, help='Number of threads to use for MiXCR analysis')
    
    args = parser.parse_args()
    
    run_mixcr_analysis(
        args.data_dir,
        args.output_dir,
        args.receptor_type,
        args.species,
        args.sample_info,
        args.threads
    )
