# Single Cell TCR Analysis Pipeline for 10X Genomics Data
# This script processes fastq.gz files from 10X Genomics scTCR-seq and performs comprehensive TCR repertoire analysis

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
import scanpy as sc
import anndata
import json
import pickle


class scTCRAnalyzer:
    def __init__(self, data_dir, output_dir, sample_info=None):
        """
        Initialize the scTCR analyzer with input and output directories
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the raw fastq.gz files
        output_dir : str
            Directory for saving analysis results
        sample_info : pd.DataFrame, optional
            Information about samples (condition, subject, etc.)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.sample_info = sample_info
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize dataframes to store results
        self.clonotype_df = None
        self.repertoire_df = None
        self.vdj_usage = None
        self.diversity_metrics = None
        
    def extract_and_qc_tcr(self, cellranger_path="cellranger"):
        """
        Extract TCR sequences from fastq.gz files using Cell Ranger
        
        Parameters:
        -----------
        cellranger_path : str
            Path to cellranger executable
        """
        print("Starting TCR sequence extraction and QC...")
        
        # Find all sample directories
        sample_dirs = glob.glob(os.path.join(self.data_dir, "*"))
        
        for sample_dir in sample_dirs:
            sample_name = os.path.basename(sample_dir)
            output_path = os.path.join(self.output_dir, sample_name)
            
            # Run Cell Ranger vdj
            cmd = [
                cellranger_path, "vdj",
                "--id=" + sample_name,
                "--fastqs=" + sample_dir,
                "--reference=refdata-cellranger-vdj-GRCh38-alts-ensembl-5.0.0",
                "--sample=" + sample_name,
                "--localcores=8",
                "--localmem=64"
            ]
            
            print(f"Processing sample {sample_name} with command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
        print("TCR extraction and VDJ assembly completed!")
        
    def compile_clonotypes(self):
        """
        Compile clonotype information from Cell Ranger vdj output
        """
        print("Compiling clonotype data...")
        
        all_clonotypes = []
        
        # Find all analysis directories
        analysis_dirs = glob.glob(os.path.join(self.output_dir, "*/outs"))
        
        for analysis_dir in analysis_dirs:
            sample_name = os.path.basename(os.path.dirname(analysis_dir))
            
            # Load clonotype data from all_contig_annotations.json
            clonotype_file = os.path.join(analysis_dir, "all_contig_annotations.json")
            
            if os.path.exists(clonotype_file):
                with open(clonotype_file, 'r') as f:
                    data = json.load(f)
                
                for cell in data:
                    # Extract relevant TCR information
                    cell_id = cell.get('barcode')
                    
                    # Process each chain
                    for contig in cell.get('contig_annotations', []):
                        if contig.get('productive', False) and contig.get('is_cell', False):
                            chain_type = contig.get('chain')
                            if chain_type in ['TRA', 'TRB']:
                                clonotype_info = {
                                    'sample': sample_name,
                                    'cell_id': cell_id,
                                    'clonotype_id': cell.get('clonotype_id', 'None'),
                                    'chain': chain_type,
                                    'v_gene': contig.get('v_gene', ''),
                                    'd_gene': contig.get('d_gene', '') if chain_type == 'TRB' else None,
                                    'j_gene': contig.get('j_gene', ''),
                                    'c_gene': contig.get('c_gene', ''),
                                    'cdr3': contig.get('cdr3', ''),
                                    'cdr3_nt': contig.get('cdr3_nt', ''),
                                    'reads': contig.get('reads', 0),
                                    'umis': contig.get('umis', 0)
                                }
                                all_clonotypes.append(clonotype_info)
            else:
                print(f"Warning: Could not find annotations file for sample {sample_name}")
        
        # Convert to DataFrame
        self.clonotype_df = pd.DataFrame(all_clonotypes)
        
        # Save compiled clonotype data
        self.clonotype_df.to_csv(os.path.join(self.output_dir, "all_clonotypes.csv"), index=False)
        
        # Create a summarized repertoire dataframe
        self._compile_repertoire()
        
        print(f"Compiled {len(self.clonotype_df)} TCR sequences from {len(analysis_dirs)} samples")
        return self.clonotype_df
    
    def _compile_repertoire(self):
        """
        Compile repertoire information from clonotype data
        """
        if self.clonotype_df is None:
            print("Error: No clonotype data available. Run compile_clonotypes first.")
            return None
            
        # Group by sample and clonotype_id to count cells per clonotype
        clonotype_counts = self.clonotype_df.groupby(['sample', 'clonotype_id']).agg({
            'cell_id': 'nunique',
        }).rename(columns={'cell_id': 'clone_size'}).reset_index()
        
        # Get the V, J, CDR3 information for each clonotype (TRA and TRB)
        clonotype_info = {}
        
        for sample in self.clonotype_df['sample'].unique():
            sample_data = self.clonotype_df[self.clonotype_df['sample'] == sample]
            
            for clonotype_id in sample_data['clonotype_id'].unique():
                if clonotype_id == 'None':
                    continue
                    
                clonotype_data = sample_data[sample_data['clonotype_id'] == clonotype_id]
                
                tra_data = clonotype_data[clonotype_data['chain'] == 'TRA'].iloc[0] if not clonotype_data[clonotype_data['chain'] == 'TRA'].empty else None
                trb_data = clonotype_data[clonotype_data['chain'] == 'TRB'].iloc[0] if not clonotype_data[clonotype_data['chain'] == 'TRB'].empty else None
                
                clonotype_info[(sample, clonotype_id)] = {
                    'tra_v': tra_data['v_gene'] if tra_data is not None else None,
                    'tra_j': tra_data['j_gene'] if tra_data is not None else None,
                    'tra_cdr3': tra_data['cdr3'] if tra_data is not None else None,
                    'tra_cdr3_nt': tra_data['cdr3_nt'] if tra_data is not None else None,
                    'trb_v': trb_data['v_gene'] if trb_data is not None else None,
                    'trb_d': trb_data['d_gene'] if trb_data is not None else None,
                    'trb_j': trb_data['j_gene'] if trb_data is not None else None,
                    'trb_cdr3': trb_data['cdr3'] if trb_data is not None else None,
                    'trb_cdr3_nt': trb_data['cdr3_nt'] if trb_data is not None else None,
                }
        
        # Add clonotype information to the counts
        repertoire_data = []
        
        for (sample, clonotype_id), count_data in clonotype_counts.iterrows():
            if clonotype_id == 'None':
                continue
                
            clone_size = count_data['clone_size']
            
            if (sample, clonotype_id) in clonotype_info:
                info = clonotype_info[(sample, clonotype_id)]
                
                repertoire_data.append({
                    'sample': sample,
                    'clonotype_id': clonotype_id,
                    'clone_size': clone_size,
                    'tra_v': info['tra_v'],
                    'tra_j': info['tra_j'],
                    'tra_cdr3': info['tra_cdr3'],
                    'tra_cdr3_nt': info['tra_cdr3_nt'],
                    'trb_v': info['trb_v'],
                    'trb_d': info['trb_d'],
                    'trb_j': info['trb_j'],
                    'trb_cdr3': info['trb_cdr3'],
                    'trb_cdr3_nt': info['trb_cdr3_nt']
                })
        
        self.repertoire_df = pd.DataFrame(repertoire_data)
        
        # Calculate frequency and add condition information if available
        if self.repertoire_df is not None:
            # Calculate total cells per sample
            total_cells = self.repertoire_df.groupby('sample')['clone_size'].sum().to_dict()
            self.repertoire_df['frequency'] = self.repertoire_df.apply(
                lambda x: x['clone_size'] / total_cells[x['sample']], axis=1
            )
            
            # Add condition information if sample_info is available
            if self.sample_info is not None:
                sample_to_condition = self.sample_info.set_index('sample')['condition'].to_dict()
                self.repertoire_df['condition'] = self.repertoire_df['sample'].map(sample_to_condition)
        
        # Save repertoire data
        self.repertoire_df.to_csv(os.path.join(self.output_dir, "tcr_repertoire.csv"), index=False)
        
        return self.repertoire_df
    
    def analyze_vdj_usage(self):
        """
        Analyze V(D)J gene usage patterns
        """
        print("Analyzing V(D)J gene usage...")
        
        if self.repertoire_df is None:
            print("Error: No repertoire data available. Run compile_clonotypes first.")
            return None
        
        # Create a directory for VDJ usage analysis
        vdj_dir = os.path.join(self.output_dir, "vdj_usage")
        os.makedirs(vdj_dir, exist_ok=True)
        
        # Initialize dictionaries to store V(D)J usage data
        v_usage = defaultdict(lambda: defaultdict(int))
        j_usage = defaultdict(lambda: defaultdict(int))
        d_usage = defaultdict(lambda: defaultdict(int))
        vj_pairing = defaultdict(lambda: defaultdict(int))
        
        # Calculate gene usage (weighted by clone size)
        for _, clone in self.repertoire_df.iterrows():
            sample = clone['sample']
            
            # TRA genes
            if not pd.isna(clone['tra_v']):
                v_usage[sample][clone['tra_v']] += clone['clone_size']
            if not pd.isna(clone['tra_j']):
                j_usage[sample][clone['tra_j']] += clone['clone_size']
            if not pd.isna(clone['tra_v']) and not pd.isna(clone['tra_j']):
                vj_pairing[sample][f"{clone['tra_v']}:{clone['tra_j']}"] += clone['clone_size']
            
            # TRB genes
            if not pd.isna(clone['trb_v']):
                v_usage[sample][clone['trb_v']] += clone['clone_size']
            if not pd.isna(clone['trb_j']):
                j_usage[sample][clone['trb_j']] += clone['clone_size']
            if not pd.isna(clone['trb_d']):
                d_usage[sample][clone['trb_d']] += clone['clone_size']
            if not pd.isna(clone['trb_v']) and not pd.isna(clone['trb_j']):
                vj_pairing[sample][f"{clone['trb_v']}:{clone['trb_j']}"] += clone['clone_size']
        
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
        
        # Plotting
        self._plot_gene_usage(v_usage_df, "V gene", vdj_dir)
        self._plot_gene_usage(j_usage_df, "J gene", vdj_dir)
        if not d_usage_df.empty:
            self._plot_gene_usage(d_usage_df, "D gene", vdj_dir)
        
        # Store results
        self.vdj_usage = {
            'v_usage': v_usage_df,
            'j_usage': j_usage_df,
            'd_usage': d_usage_df,
            'vj_pairing': vj_pairing_df
        }
        
        print("VDJ usage analysis completed!")
        return self.vdj_usage
    
    def _plot_gene_usage(self, usage_df, gene_type, output_dir):
        """
        Plot gene usage data as heatmap and bar plots
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
    
    def analyze_cdr3_features(self):
        """
        Analyze CDR3 sequence features (length distribution, amino acid composition, motifs)
        """
        print("Analyzing CDR3 features...")
        
        if self.repertoire_df is None:
            print("Error: No repertoire data available. Run compile_clonotypes first.")
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
        
        print("CDR3 feature analysis completed!")
    
    def _analyze_cdr3_length(self, output_dir):
        """
        Analyze CDR3 length distribution
        """
        # Prepare data
        tra_lengths = []
        trb_lengths = []
        
        for _, clone in self.repertoire_df.iterrows():
            sample = clone['sample']
            condition = clone.get('condition', 'Unknown')
            
            # Get condition if available
            if 'condition' not in clone and self.sample_info is not None:
                condition = self.sample_info.loc[self.sample_info['sample'] == sample, 'condition'].iloc[0]
            
            # TRA CDR3 length
            if not pd.isna(clone['tra_cdr3']) and clone['tra_cdr3']:
                for _ in range(int(clone['clone_size'])):
                    tra_lengths.append({
                        'sample': sample,
                        'condition': condition,
                        'chain': 'TRA',
                        'length': len(clone['tra_cdr3'])
                    })
            
            # TRB CDR3 length
            if not pd.isna(clone['trb_cdr3']) and clone['trb_cdr3']:
                for _ in range(int(clone['clone_size'])):
                    trb_lengths.append({
                        'sample': sample,
                        'condition': condition,
                        'chain': 'TRB',
                        'length': len(clone['trb_cdr3'])
                    })
        
        # Convert to DataFrames
        tra_lengths_df = pd.DataFrame(tra_lengths)
        trb_lengths_df = pd.DataFrame(trb_lengths)
        
        # Combine
        all_lengths_df = pd.concat([tra_lengths_df, trb_lengths_df])
        
        # Save data
        all_lengths_df.to_csv(os.path.join(output_dir, "cdr3_lengths.csv"), index=False)
        
        # Plot length distribution by chain and condition
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=all_lengths_df, x='chain', y='length', hue='condition')
        plt.title("CDR3 Length Distribution by Chain and Condition")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cdr3_length_boxplot.pdf"))
        plt.close()
        
        # Plot length distribution histograms
        plt.figure(figsize=(12, 6))
        sns.histplot(data=all_lengths_df, x='length', hue='chain', multiple='dodge', bins=range(5, 31))
        plt.title("CDR3 Length Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cdr3_length_histogram.pdf"))
        plt.close()
        
        # If conditions are available, plot by condition
        if 'condition' in all_lengths_df.columns and len(all_lengths_df['condition'].unique()) > 1:
            g = sns.FacetGrid(all_lengths_df, col='chain', row='condition', height=4, aspect=1.5)
            g.map(sns.histplot, 'length', bins=range(5, 31))
            g.add_legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cdr3_length_by_condition.pdf"))
            plt.close()
    
    def _analyze_cdr3_aa_composition(self, output_dir):
        """
        Analyze CDR3 amino acid composition
        """
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        aa_counts = {aa: defaultdict(int) for aa in amino_acids}
        
        # Count amino acids in CDR3 sequences (weighted by clone size)
        for _, clone in self.repertoire_df.iterrows():
            sample = clone['sample']
            condition = clone.get('condition', 'Unknown')
            
            # Get condition if available
            if 'condition' not in clone and self.sample_info is not None:
                condition = self.sample_info.loc[self.sample_info['sample'] == sample, 'condition'].iloc[0]
            
            # Process TRA CDR3
            if not pd.isna(clone['tra_cdr3']) and clone['tra_cdr3']:
                for aa in clone['tra_cdr3']:
                    if aa in amino_acids:
                        aa_counts[aa][(sample, condition, 'TRA')] += clone['clone_size']
            
            # Process TRB CDR3
            if not pd.isna(clone['trb_cdr3']) and clone['trb_cdr3']:
                for aa in clone['trb_cdr3']:
                    if aa in amino_acids:
                        aa_counts[aa][(sample, condition, 'TRB')] += clone['clone_size']
        
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
        
        # Normalize by total count within each sample and chain
        total_counts = aa_composition_df.groupby(['sample', 'chain'])['count'].sum().reset_index()
        total_counts.rename(columns={'count': 'total'}, inplace=True)
        
        aa_composition_df = aa_composition_df.merge(total_counts, on=['sample', 'chain'])
        aa_composition_df['frequency'] = aa_composition_df['count'] / aa_composition_df['total'] * 100
        
        # Save data
        aa_composition_df.to_csv(os.path.join(output_dir, "cdr3_aa_composition.csv"), index=False)
        
        # Plot amino acid composition
        plt.figure(figsize=(15, 10))
        g = sns.FacetGrid(data=aa_composition_df, col='chain', row='condition', height=4, aspect=1.5)
        g.map(sns.barplot, 'amino_acid', 'frequency')
        g.add_legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cdr3_aa_composition.pdf"))
        plt.close()
    
    def _analyze_cdr3_motifs(self, output_dir):
        """
        Analyze CDR3 sequence motifs
        """
        # Get all unique CDR3 sequences with their counts
        tra_seqs = defaultdict(int)
        trb_seqs = defaultdict(int)
        
        for _, clone in self.repertoire_df.iterrows():
            if not pd.isna(clone['tra_cdr3']) and clone['tra_cdr3']:
                tra_seqs[clone['tra_cdr3']] += clone['clone_size']
            if not pd.isna(clone['trb_cdr3']) and clone['trb_cdr3']:
                trb_seqs[clone['trb_cdr3']] += clone['clone_size']
        
        # Find common motifs in CDR3 sequences
        tra_motifs = self._find_motifs(tra_seqs, 'TRA', output_dir)
        trb_motifs = self._find_motifs(trb_seqs, 'TRB', output_dir)
        
        # Save motif data
        if tra_motifs:
            with open(os.path.join(output_dir, "tra_motifs.txt"), "w") as f:
                for motif, count in tra_motifs.items():
                    f.write(f"{motif}\t{count}\n")
        
        if trb_motifs:
            with open(os.path.join(output_dir, "trb_motifs.txt"), "w") as f:
                for motif, count in trb_motifs.items():
                    f.write(f"{motif}\t{count}\n")
    
    def _find_motifs(self, seq_dict, chain, output_dir, min_count=5, motif_length=3):
        """
        Find common motifs in CDR3 sequences
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

    def analyze_clonal_expansion(self):
        """
        Analyze clonal expansion and identify expanded clonotypes
        """
        print("Analyzing clonal expansion...")
        
        if self.repertoire_df is None:
            print("Error: No repertoire data available. Run compile_clonotypes first.")
            return None
        
        # Create a directory for clonal expansion analysis
        expansion_dir = os.path.join(self.output_dir, "clonal_expansion")
        os.makedirs(expansion_dir, exist_ok=True)
        
        # Calculate clonal expansion metrics
        expansion_metrics = []
        
        for sample in self.repertoire_df['sample'].unique():
            sample_data = self.repertoire_df[self.repertoire_df['sample'] == sample]
            
            # Get condition if available
            condition = "Unknown"
            if 'condition' in self.repertoire_df.columns:
                condition = sample_data['condition'].iloc[0]
            elif self.sample_info is not None:
                condition = self.sample_info.loc[self.sample_info['sample'] == sample, 'condition'].iloc[0]
            
            # Calculate expansion metrics
            total_clonotypes = len(sample_data)
            total_cells = sample_data['clone_size'].sum()
            
            # Gini index for inequality in clone size distribution
            gini_index = self._calculate_gini(sample_data['clone_size'])
            
            # Clonality (1 - normalized Shannon entropy)
            clonality = self._calculate_clonality(sample_data['frequency'])
            
            # Top clone frequency
            top_clone_freq = sample_data['frequency'].max() if not sample_data.empty else 0
            
            # Percent of repertoire occupied by top 10 clones
            top10_percent = sample_data.nlargest(10, 'frequency')['frequency'].sum() if total_clonotypes >= 10 else 1.0
            
            # Expansion threshold (clones > 0.1% of repertoire)
            expanded_clones = sample_data[sample_data['frequency'] > 0.001]
            expanded_count = len(expanded_clones)
            expanded_percent = expanded_clones['frequency'].sum() if not expanded_clones.empty else 0
            
            # Store metrics
            expansion_metrics.append({
                'sample': sample,
                'condition': condition,
                'total_clonotypes': total_clonotypes,
                'total_cells': total_cells,
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
        """
        if len(values) <= 1 or values.sum() == 0:
            return 0
        
        # Sort values
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        # Calculate Gini index
        cum_sum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cum_sum) / cum_sum[-1]) / n
    
    def _calculate_clonality(self, frequencies):
        """
        Calculate clonality (1 - normalized Shannon entropy)
        """
        if len(frequencies) <= 1:
            return 0
            
        # Calculate Shannon entropy
        entropy = 0
        for freq in frequencies:
            if freq > 0:
                entropy -= freq * np.log2(freq)
        
        # Normalize entropy and calculate clonality
        max_entropy = np.log2(len(frequencies))
        if max_entropy == 0:
            return 0
            
        normalized_entropy = entropy / max_entropy
        clonality = 1 - normalized_entropy
        
        return clonality
    
    def _plot_expansion_metrics(self, expansion_df, output_dir):
        """
        Plot clonal expansion metrics
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
        """
        for sample in self.repertoire_df['sample'].unique():
            sample_data = self.repertoire_df[self.repertoire_df['sample'] == sample]
            
            # Get condition if available
            condition = "Unknown"
            if 'condition' in self.repertoire_df.columns:
                condition = sample_data['condition'].iloc[0]
            elif self.sample_info is not None:
                condition = self.sample_info.loc[self.sample_info['sample'] == sample, 'condition'].iloc[0]
            
            # Sort clones by frequency
            sorted_clones = sample_data.sort_values('frequency', ascending=False).reset_index(drop=True)
            
            # Create rank vs frequency plot
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(sorted_clones) + 1), sorted_clones['frequency'] * 100)
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
        """
        # Define expansion threshold (clones > 0.1% of repertoire)
        threshold = 0.001
        
        # Identify expanded clones for each sample
        expanded_clones = []
        
        for sample in self.repertoire_df['sample'].unique():
            sample_data = self.repertoire_df[self.repertoire_df['sample'] == sample]
            
            # Get condition if available
            condition = "Unknown"
            if 'condition' in self.repertoire_df.columns:
                condition = sample_data['condition'].iloc[0]
            elif self.sample_info is not None:
                condition = self.sample_info.loc[self.sample_info['sample'] == sample, 'condition'].iloc[0]
            
            # Find expanded clones
            sample_expanded = sample_data[sample_data['frequency'] > threshold].copy()
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
        """
        if expanded_df.empty:
            return
            
        # Plot CDR3 length distribution of expanded clones
        plt.figure(figsize=(10, 6))
        
        # TRA CDR3 lengths
        tra_lengths = []
        for _, clone in expanded_df.iterrows():
            if not pd.isna(clone['tra_cdr3']) and clone['tra_cdr3']:
                tra_lengths.append(len(clone['tra_cdr3']))
        
        # TRB CDR3 lengths
        trb_lengths = []
        for _, clone in expanded_df.iterrows():
            if not pd.isna(clone['trb_cdr3']) and clone['trb_cdr3']:
                trb_lengths.append(len(clone['trb_cdr3']))
        
        if tra_lengths and trb_lengths:
            plt.hist([tra_lengths, trb_lengths], bins=range(5, 31), 
                    label=['TRA', 'TRB'], alpha=0.7, edgecolor='black')
            plt.legend()
        elif tra_lengths:
            plt.hist(tra_lengths, bins=range(5, 31), label='TRA', alpha=0.7, edgecolor='black')
            plt.legend()
        elif trb_lengths:
            plt.hist(trb_lengths, bins=range(5, 31), label='TRB', alpha=0.7, edgecolor='black')
            plt.legend()
        
        plt.title("CDR3 Length Distribution of Expanded Clones")
        plt.xlabel("CDR3 Length")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "expanded_cdr3_length.pdf"))
        plt.close()
    
    def calculate_diversity_metrics(self):
        """
        Calculate repertoire diversity metrics
        """
        print("Calculating diversity metrics...")
        
        if self.repertoire_df is None:
            print("Error: No repertoire data available. Run compile_clonotypes first.")
            return None
        
        # Create directory for diversity analysis
        diversity_dir = os.path.join(self.output_dir, "diversity")
        os.makedirs(diversity_dir, exist_ok=True)
        
        # Calculate diversity metrics for each sample
        diversity_metrics = []
        
        for sample in self.repertoire_df['sample'].unique():
            sample_data = self.repertoire_df[self.repertoire_df['sample'] == sample]
            
            # Get condition if available
            condition = "Unknown"
            if 'condition' in self.repertoire_df.columns:
                condition = sample_data['condition'].iloc[0]
            elif self.sample_info is not None:
                condition = self.sample_info.loc[self.sample_info['sample'] == sample, 'condition'].iloc[0]
            
            # Extract frequencies
            frequencies = sample_data['frequency'].values
            
            # Calculate diversity metrics
            richness = len(frequencies)  # Number of unique clonotypes
            
            # Shannon entropy and normalized entropy
            shannon_entropy = -np.sum(np.where(frequencies > 0, frequencies * np.log2(frequencies), 0))
            max_entropy = np.log2(richness) if richness > 0 else 0
            normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
            
            # Inverse Simpson index (1/D)
            inverse_simpson = 1 / np.sum(frequencies**2) if np.sum(frequencies**2) > 0 else 0
            
            # Hill numbers (effective number of species)
            hill_1 = np.exp(shannon_entropy)  # exponential of Shannon entropy
            hill_2 = inverse_simpson  # inverse Simpson index
            
            # Clonality
            clonality = 1 - normalized_entropy
            
            # Gini index
            gini_index = self._calculate_gini(sample_data['clone_size'])
            
            # Store metrics
            diversity_metrics.append({
                'sample': sample,
                'condition': condition,
                'richness': richness,
                'shannon_entropy': shannon_entropy,
                'normalized_entropy': normalized_entropy,
                'inverse_simpson': inverse_simpson,
                'hill_1': hill_1,
                'hill_2': hill_2,
                'clonality': clonality,
                'gini_index': gini_index
            })
        
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
        """
        # Check if we have multiple conditions to compare
        if 'condition' in diversity_df.columns and len(diversity_df['condition'].unique()) > 1:
            # Metrics to plot
            metrics = ['richness', 'shannon_entropy', 'normalized_entropy', 'inverse_simpson', 
                      'hill_1', 'hill_2', 'clonality', 'gini_index']
            
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
            stats_results = []
            
            for metric in metrics:
                groups = []
                conditions = diversity_df['condition'].unique()
                
                for condition in conditions:
                    groups.append(diversity_df[diversity_df['condition'] == condition][metric].values)
                
                # Perform statistical test if we have at least two groups
                if len(groups) >= 2:
                    if len(groups) == 2:
                        # t-test for two groups
                        stat, pval = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                        test_name = "t-test"
                    else:
                        # ANOVA for more than two groups
                        stat, pval = stats.f_oneway(*groups)
                        test_name = "ANOVA"
                    
                    stats_results.append({
                        'metric': metric,
                        'test': test_name,
                        'statistic': stat,
                        'p_value': pval,
                        'significant': pval < 0.05
                    })
            
            # Save statistical test results
            if stats_results:
                stats_df = pd.DataFrame(stats_results)
                stats_df.to_csv(os.path.join(output_dir, "diversity_statistics.csv"), index=False)
    
    def calculate_repertoire_overlap(self):
        """
        Calculate repertoire overlap between samples
        """
        print("Calculating repertoire overlap...")
        
        if self.repertoire_df is None:
            print("Error: No repertoire data available. Run compile_clonotypes first.")
            return None
        
        # Create directory for overlap analysis
        overlap_dir = os.path.join(self.output_dir, "overlap")
        os.makedirs(overlap_dir, exist_ok=True)
        
        # Get all samples
        samples = self.repertoire_df['sample'].unique()
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
            data1 = self.repertoire_df[self.repertoire_df['sample'] == sample1]
            
            # Extract unique CDR3 sequences
            set1_tra = set()
            set1_trb = set()
            freq1 = defaultdict(float)
            
            for _, row in data1.iterrows():
                if not pd.isna(row['tra_cdr3']) and row['tra_cdr3']:
                    set1_tra.add(row['tra_cdr3'])
                    freq1[row['tra_cdr3']] += row['frequency']
                if not pd.isna(row['trb_cdr3']) and row['trb_cdr3']:
                    set1_trb.add(row['trb_cdr3'])
                    freq1[row['trb_cdr3']] += row['frequency']
            
            for j, sample2 in enumerate(samples):
                if i == j:
                    # Self-comparison
                    jaccard_matrix[i, j] = 1.0
                    morisita_matrix[i, j] = 1.0
                    overlap_count_matrix[i, j] = len(set1_tra) + len(set1_trb)
                    continue
                
                data2 = self.repertoire_df[self.repertoire_df['sample'] == sample2]
                
                # Extract unique CDR3 sequences
                set2_tra = set()
                set2_trb = set()
                freq2 = defaultdict(float)
                
                for _, row in data2.iterrows():
                    if not pd.isna(row['tra_cdr3']) and row['tra_cdr3']:
                        set2_tra.add(row['tra_cdr3'])
                        freq2[row['tra_cdr3']] += row['frequency']
                    if not pd.isna(row['trb_cdr3']) and row['trb_cdr3']:
                        set2_trb.add(row['trb_cdr3'])
                        freq2[row['trb_cdr3']] += row['frequency']
                
                # Calculate overlap
                overlap_tra = set1_tra.intersection(set2_tra)
                overlap_trb = set1_trb.intersection(set2_trb)
                
                # Jaccard index
                union_tra = set1_tra.union(set2_tra)
                union_trb = set1_trb.union(set2_trb)
                
                jaccard_tra = len(overlap_tra) / len(union_tra) if union_tra else 0
                jaccard_trb = len(overlap_trb) / len(union_trb) if union_trb else 0
                
                # Combine TRA and TRB Jaccard
                jaccard_matrix[i, j] = (jaccard_tra + jaccard_trb) / 2
                
                # Morisita-Horn index (weighted by frequency)
                overlap_seqs = set(freq1.keys()).intersection(set(freq2.keys()))
                
                if overlap_seqs:
                    sum_freq1_squared = sum([freq1[seq]**2 for seq in freq1])
                    sum_freq2_squared = sum([freq2[seq]**2 for seq in freq2])
                    
                    sum_freq_products = sum([freq1[seq] * freq2[seq] for seq in overlap_seqs])
                    
                    if sum_freq1_squared > 0 and sum_freq2_squared > 0:
                        morisita_matrix[i, j] = (2 * sum_freq_products) / (sum_freq1_squared + sum_freq2_squared)
                
                # Count of overlapping sequences
                overlap_count_matrix[i, j] = len(overlap_tra) + len(overlap_trb)
        
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
        plt.title("Number of Overlapping TCR Sequences")
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
        
        if self.repertoire_df is None:
            print("Error: No repertoire data available. Run compile_clonotypes first.")
            return None
        
        # Check if condition information is available
        if 'condition' not in self.repertoire_df.columns and self.sample_info is None:
            print("Warning: No condition information available. Skipping comparative analysis.")
            return None
        
        # Add condition information if not already present
        if 'condition' not in self.repertoire_df.columns and self.sample_info is not None:
            sample_to_condition = self.sample_info.set_index('sample')['condition'].to_dict()
            self.repertoire_df['condition'] = self.repertoire_df['sample'].map(sample_to_condition)
        
        # Create directory for comparative analysis
        comparative_dir = os.path.join(self.output_dir, "comparative_analysis")
        os.makedirs(comparative_dir, exist_ok=True)
        
        # Get unique conditions
        conditions = self.repertoire_df['condition'].unique()
        
        if len(conditions) < 2:
            print("Warning: Need at least 2 conditions to perform comparative analysis.")
            return None
        
        # Perform comparative analyses
        self._analyze_vdj_usage_by_condition(comparative_dir)
        self._analyze_cdr3_features_by_condition(comparative_dir)
        self._analyze_clonal_expansion_by_condition(comparative_dir)
        
        print("Comparative analysis completed!")
    
    def _analyze_vdj_usage_by_condition(self, output_dir):
        """
        Analyze V(D)J gene usage differences between conditions
        """
        # Create V gene usage table by condition
        v_usage_by_condition = defaultdict(lambda: defaultdict(int))
        
        for _, clone in self.repertoire_df.iterrows():
            condition = clone.get('condition', 'Unknown')
            
            # TRA V gene
            if not pd.isna(clone['tra_v']):
                v_usage_by_condition[condition][clone['tra_v']] += clone['clone_size']
            
            # TRB V gene
            if not pd.isna(clone['trb_v']):
                v_usage_by_condition[condition][clone['trb_v']] += clone['clone_size']
        
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
        
        # Plot
        sns.barplot(data=plot_df, x='V_gene', y='Frequency', hue='Condition')
        plt.title("Top 20 V Gene Usage by Condition")
        plt.xlabel("V Gene")
        plt.ylabel("Frequency (%)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "v_gene_usage_by_condition.pdf"))
        plt.close()
        
        # Perform statistical tests for each V gene
        v_genes = list(v_usage_norm.index)
        conditions = list(v_usage_norm.columns)
        
        if len(conditions) == 2:
            # t-test for two conditions
            test_results = []
            
            for v_gene in v_genes:
                # Get frequencies for each sample in each condition
                samples_by_condition = defaultdict(list)
                
                for _, clone in self.repertoire_df.iterrows():
                    condition = clone.get('condition', 'Unknown')
                    sample = clone['sample']
                    
                    if not pd.isna(clone['tra_v']) and clone['tra_v'] == v_gene:
                        samples_by_condition[condition].append(sample)
                    
                    if not pd.isna(clone['trb_v']) and clone['trb_v'] == v_gene:
                        samples_by_condition[condition].append(sample)
                
                # If we have samples in both conditions, perform t-test
                if len(samples_by_condition[conditions[0]]) > 0 and len(samples_by_condition[conditions[1]]) > 0:
                    group1 = [v_usage_norm.loc[v_gene, conditions[0]]]
                    group2 = [v_usage_norm.loc[v_gene, conditions[1]]]
                    
                    stat, pval = stats.ttest_ind(group1, group2, equal_var=False)
                    
                    test_results.append({
                        'V_gene': v_gene,
                        'Condition1_freq': v_usage_norm.loc[v_gene, conditions[0]],
                        'Condition2_freq': v_usage_norm.loc[v_gene, conditions[1]],
                        'Difference': v_usage_norm.loc[v_gene, conditions[0]] - v_usage_norm.loc[v_gene, conditions[1]],
                        'p_value': pval,
                        'significant': pval < 0.05
                    })
            
            if test_results:
                test_df = pd.DataFrame(test_results)
                test_df = test_df.sort_values('p_value')
                test_df.to_csv(os.path.join(output_dir, "v_gene_statistical_tests.csv"), index=False)
    
    def _analyze_cdr3_features_by_condition(self, output_dir):
        """
        Analyze CDR3 feature differences between conditions
        """
        # Analyze CDR3 length distribution by condition
        tra_lengths = []
        trb_lengths = []
        
        for _, clone in self.repertoire_df.iterrows():
            condition = clone.get('condition', 'Unknown')
            
            # TRA CDR3 length
            if not pd.isna(clone['tra_cdr3']) and clone['tra_cdr3']:
                tra_lengths.append({
                    'condition': condition,
                    'chain': 'TRA',
                    'length': len(clone['tra_cdr3']),
                    'frequency': clone['frequency']
                })
            
            # TRB CDR3 length
            if not pd.isna(clone['trb_cdr3']) and clone['trb_cdr3']:
                trb_lengths.append({
                    'condition': condition,
                    'chain': 'TRB',
                    'length': len(clone['trb_cdr3']),
                    'frequency': clone['frequency']
                })
        
        # Convert to DataFrames
        tra_df = pd.DataFrame(tra_lengths)
        trb_df = pd.DataFrame(trb_lengths)
        
        # Combine
        cdr3_df = pd.concat([tra_df, trb_df])
        
        # Save data
        cdr3_df.to_csv(os.path.join(output_dir, "cdr3_lengths_by_condition.csv"), index=False)
        
        # Plot CDR3 length distribution by condition and chain
        g = sns.FacetGrid(data=cdr3_df, col='chain', row='condition', height=4, aspect=1.5)
        g.map(sns.histplot, 'length', weights='frequency', bins=range(5, 31))
        g.add_legend()
        plt.savefig(os.path.join(output_dir, "cdr3_length_by_condition.pdf"))
        plt.close()
        
        # Statistical test for CDR3 length differences
        conditions = cdr3_df['condition'].unique()
        chains = ['TRA', 'TRB']
        
        if len(conditions) == 2:
            # t-test for two conditions
            test_results = []
            
            for chain in chains:
                chain_data = cdr3_df[cdr3_df['chain'] == chain]
                
                if not chain_data.empty:
                    group1 = chain_data[chain_data['condition'] == conditions[0]]['length'].values
                    group2 = chain_data[chain_data['condition'] == conditions[1]]['length'].values
                    
                    if len(group1) > 0 and len(group2) > 0:
                        stat, pval = stats.ttest_ind(group1, group2, equal_var=False)
                        
                        test_results.append({
                            'chain': chain,
                            'mean_length_condition1': np.mean(group1),
                            'mean_length_condition2': np.mean(group2),
                            'difference': np.mean(group1) - np.mean(group2),
                            'p_value': pval,
                            'significant': pval < 0.05
                        })
            
            if test_results:
                test_df = pd.DataFrame(test_results)
                test_df.to_csv(os.path.join(output_dir, "cdr3_length_statistical_tests.csv"), index=False)
    
    def _analyze_clonal_expansion_by_condition(self, output_dir):
        """
        Analyze clonal expansion differences between conditions
        """
        # Calculate expansion metrics by sample
        expansion_metrics = []
        
        for sample in self.repertoire_df['sample'].unique():
            sample_data = self.repertoire_df[self.repertoire_df['sample'] == sample]
            
            condition = sample_data['condition'].iloc[0] if 'condition' in sample_data.columns else "Unknown"
            
            # Calculate expansion metrics
            total_clonotypes = len(sample_data)
            
            # Gini index for inequality in clone size distribution
            gini_index = self._calculate_gini(sample_data['clone_size'])
            
            # Clonality (1 - normalized Shannon entropy)
            clonality = self._calculate_clonality(sample_data['frequency'])
            
            # Top clone frequency
            top_clone_freq = sample_data['frequency'].max() if not sample_data.empty else 0
            
            # Percent of repertoire occupied by top 10 clones
            top10_percent = sample_data.nlargest(10, 'frequency')['frequency'].sum() if total_clonotypes >= 10 else 1.0
            
            # Expansion threshold (clones > 0.1% of repertoire)
            expanded_clones = sample_data[sample_data['frequency'] > 0.001]
            expanded_count = len(expanded_clones)
            expanded_percent = expanded_clones['frequency'].sum() if not expanded_clones.empty else 0
            
            expansion_metrics.append({
                'sample': sample,
                'condition': condition,
                'gini_index': gini_index,
                'clonality': clonality,
                'top_clone_freq': top_clone_freq,
                'top10_percent': top10_percent,
                'expanded_clone_count': expanded_count,
                'expanded_clone_percent': expanded_percent
            })
        
        expansion_df = pd.DataFrame(expansion_metrics)
        
        # Save data
        expansion_df.to_csv(os.path.join(output_dir, "expansion_metrics_by_sample.csv"), index=False)
        
        # Plot expansion metrics by condition
        metrics = ['gini_index', 'clonality', 'top_clone_freq', 'top10_percent', 
                  'expanded_clone_count', 'expanded_clone_percent']
        
        conditions = expansion_df['condition'].unique()
        
        for metric in metrics:
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=expansion_df, x='condition', y=metric)
            sns.stripplot(data=expansion_df, x='condition', y=metric, color='black', dodge=True)
            plt.title(f"{metric.replace('_', ' ').title()} by Condition")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{metric}_by_condition.pdf"))
            plt.close()
        
        # Statistical tests
        if len(conditions) == 2:
            # t-test for two conditions
            test_results = []
            
            for metric in metrics:
                group1 = expansion_df[expansion_df['condition'] == conditions[0]][metric].values
                group2 = expansion_df[expansion_df['condition'] == conditions[1]][metric].values
                
                if len(group1) > 0 and len(group2) > 0:
                    stat, pval = stats.ttest_ind(group1, group2, equal_var=False)
                    
                    test_results.append({
                        'metric': metric,
                        'mean_condition1': np.mean(group1),
                        'mean_condition2': np.mean(group2),
                        'difference': np.mean(group1) - np.mean(group2),
                        'p_value': pval,
                        'significant': pval < 0.05
                    })
            
            if test_results:
                test_df = pd.DataFrame(test_results)
                test_df.to_csv(os.path.join(output_dir, "expansion_statistical_tests.csv"), index=False)
    
    def integrate_with_expression(self, gene_expression_file):
        """
        Integrate TCR data with gene expression (optional)
        
        Parameters:
        -----------
        gene_expression_file : str
            Path to gene expression file (10X Cell Ranger format)
        """
        print("Integrating TCR data with gene expression...")
        
        if self.clonotype_df is None:
            print("Error: No clonotype data available. Run compile_clonotypes first.")
            return None
        
        # Create directory for integrated analysis
        integration_dir = os.path.join(self.output_dir, "expression_integration")
        os.makedirs(integration_dir, exist_ok=True)
        
        # Load gene expression data
        if not os.path.exists(gene_expression_file):
            print(f"Error: Gene expression file {gene_expression_file} not found.")
            return None
        
        try:
            # Load gene expression using scanpy
            adata = sc.read_10x_h5(gene_expression_file)
            print(f"Loaded gene expression data with {adata.shape[0]} cells and {adata.shape[1]} genes")
            
            # Process gene expression data
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            
            # Normalize and log-transform
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            
            # Find variable genes
            sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            
            # Keep only variable genes
            adata = adata[:, adata.var.highly_variable]
            
            # Scale data
            sc.pp.scale(adata, max_value=10)
            
            # Run PCA
            sc.tl.pca(adata, svd_solver='arpack')
            
            # Compute neighborhood graph
            sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
            
            # Run UMAP
            sc.tl.umap(adata)
            
            # Run clustering
            sc.tl.leiden(adata, resolution=0.5)
            
            # Match barcodes with TCR data
            tcr_barcodes = self.clonotype_df['cell_id'].unique()
            common_barcodes = set(adata.obs.index).intersection(set(tcr_barcodes))
            
            print(f"Found {len(common_barcodes)} cells with both gene expression and TCR data")
            
            if not common_barcodes:
                print("Error: No common barcodes between gene expression and TCR data.")
                return None
            
            # Create TCR metadata for gene expression
            tcr_meta = {}
            
            for _, row in self.clonotype_df.iterrows():
                cell_id = row['cell_id']
                if cell_id in common_barcodes:
                    if cell_id not in tcr_meta:
                        tcr_meta[cell_id] = {
                            'has_tcr': True,
                            'clonotype_id': row['clonotype_id'],
                            'sample': row['sample']
                        }
                    
                    if row['chain'] == 'TRA':
                        tcr_meta[cell_id]['tra_v'] = row['v_gene']
                        tcr_meta[cell_id]['tra_j'] = row['j_gene']
                        tcr_meta[cell_id]['tra_cdr3'] = row['cdr3']
                    elif row['chain'] == 'TRB':
                        tcr_meta[cell_id]['trb_v'] = row['v_gene']
                        tcr_meta[cell_id]['trb_d'] = row['d_gene']
                        tcr_meta[cell_id]['trb_j'] = row['j_gene']
                        tcr_meta[cell_id]['trb_cdr3'] = row['cdr3']
            
            # Add TCR metadata to anndata object
            tcr_df = pd.DataFrame.from_dict(tcr_meta, orient='index')
            
            # Add empty rows for cells without TCR data
            for cell in adata.obs.index:
                if cell not in tcr_df.index:
                    tcr_df.loc[cell] = pd.Series({
                        'has_tcr': False,
                        'clonotype_id': None,
                        'sample': None,
                        'tra_v': None,
                        'tra_j': None,
                        'tra_cdr3': None,
                        'trb_v': None,
                        'trb_d': None,
                        'trb_j': None,
                        'trb_cdr3': None
                    })
            
            # Sort to match adata order
            tcr_df = tcr_df.loc[adata.obs.index]
            
            # Add to adata.obs
            for col in tcr_df.columns:
                adata.obs[col] = tcr_df[col]
            
            # Save integrated object
            adata.write(os.path.join(integration_dir, "integrated_tcr_gene_expression.h5ad"))
            
            # Create visualization plots
            self._visualize_integrated_data(adata, integration_dir)
            
            print("Integration with gene expression completed!")
            return adata
            
        except Exception as e:
            print(f"Error integrating with gene expression: {str(e)}")
            return None
    
    def _visualize_integrated_data(self, adata, output_dir):
        """
        Create visualization plots for integrated data
        """
        # UMAP colored by TCR presence
        sc.pl.umap(adata, color=['has_tcr'], save=os.path.join(output_dir, "umap_has_tcr.pdf"))
        
        # UMAP colored by clonotype (for top 10 clonotypes)
        if 'clonotype_id' in adata.obs.columns:
            # Get top 10 clonotypes
            top_clonotypes = adata.obs['clonotype_id'].value_counts().head(10).index
            
            # Create a new column for top clonotypes
            adata.obs['top_clonotype'] = adata.obs['clonotype_id']
            adata.obs.loc[~adata.obs['top_clonotype'].isin(top_clonotypes), 'top_clonotype'] = 'Other'
            
            # Plot
            sc.pl.umap(adata, color=['top_clonotype'], save=os.path.join(output_dir, "umap_top_clonotypes.pdf"))
        
        # Differential gene expression between expanded and non-expanded clones
        if 'has_tcr' in adata.obs.columns and adata.obs['has_tcr'].sum() > 0:
            sc.tl.rank_genes_groups(adata, 'has_tcr', method='wilcoxon')
            sc.pl.rank_genes_groups(adata, n_genes=20, save=os.path.join(output_dir, "diff_genes_tcr.pdf"))
            
            # Save differentially expressed genes
            diff_genes = sc.get.rank_genes_groups_df(adata, group='True')
            diff_genes.to_csv(os.path.join(output_dir, "diff_genes_tcr.csv"))
        
        # If we have cluster information, check TCR distribution across clusters
        if 'leiden' in adata.obs.columns:
            tcr_cluster = pd.crosstab(adata.obs['leiden'], adata.obs['has_tcr'])
            tcr_cluster.to_csv(os.path.join(output_dir, "tcr_distribution_by_cluster.csv"))
            
            # Plot
            plt.figure(figsize=(12, 8))
            tcr_cluster_percent = tcr_cluster.div(tcr_cluster.sum(axis=1), axis=0) * 100
            tcr_cluster_percent.plot(kind='bar', stacked=True)
            plt.title("TCR Distribution by Cluster")
            plt.xlabel("Cluster")
            plt.ylabel("Percentage")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "tcr_distribution_by_cluster.pdf"))
            plt.close()
    
    def visualize_results(self):
        """
        Create comprehensive visualizations of analysis results
        """
        print("Creating comprehensive visualizations...")
        
        # Create directory for final visualizations
        viz_dir = os.path.join(self.output_dir, "final_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create summary plots
        self._create_summary_plots(viz_dir)
        
        # Create HTML report
        self._create_html_report(viz_dir)
        
        print("Visualization completed!")
    
    def _create_summary_plots(self, output_dir):
        """
        Create summary plots for the analysis
        """
        # Sample overview
        if self.repertoire_df is not None:
            # Number of clonotypes per sample
            clonotypes_per_sample = self.repertoire_df.groupby('sample').size().reset_index()
            clonotypes_per_sample.columns = ['sample', 'num_clonotypes']
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=clonotypes_per_sample, x='sample', y='num_clonotypes')
            plt.title("Number of Clonotypes per Sample")
            plt.xlabel("Sample")
            plt.ylabel("Number of Clonotypes")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "clonotypes_per_sample.pdf"))
            plt.close()
            
            # Clone size distribution
            plt.figure(figsize=(12, 8))
            for sample in self.repertoire_df['sample'].unique():
                sample_data = self.repertoire_df[self.repertoire_df['sample'] == sample]
                sorted_freq = np.sort(sample_data['frequency'].values)[::-1]
                plt.plot(range(1, len(sorted_freq) + 1), sorted_freq, label=sample)
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Clone Rank")
            plt.ylabel("Clone Frequency")
            plt.title("Clone Size Distribution by Sample")
            plt.legend()
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "clone_size_distribution.pdf"))
            plt.close()
        
        # V gene usage heatmap
        if self.vdj_usage and 'v_usage' in self.vdj_usage:
            v_usage = self.vdj_usage['v_usage']
            
            # Get top 20 V genes
            top_v_genes = v_usage.sum(axis=1).nlargest(20).index
            v_usage_top = v_usage.loc[top_v_genes]
            
            # Normalize
            v_usage_norm = v_usage_top.copy()
            for col in v_usage_norm.columns:
                total = v_usage_norm[col].sum()
                if total > 0:
                    v_usage_norm[col] = v_usage_norm[col] / total * 100
            
            # Plot heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(v_usage_norm, cmap="YlGnBu", annot=True, fmt=".1f")
            plt.title("Top 20 V Gene Usage (% of repertoire)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "top_v_gene_usage_heatmap.pdf"))
            plt.close()
        
        # Diversity metrics comparison
        if self.diversity_metrics is not None:
            # Prepare for radar chart
            metrics = ['richness', 'shannon_entropy', 'inverse_simpson', 'clonality']
            
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
    
    def _create_html_report(self, output_dir):
        """
        Create HTML report summarizing analysis results
        """
        # Create HTML report
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>scTCR-seq Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #2c3e50; }
                .section { margin-bottom: 30px; }
                .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .plot { text-align: center; margin: 20px 0; }
                .plot img { max-width: 800px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            </style>
        </head>
        <body>
            <h1>Single Cell TCR-seq Analysis Report</h1>
            
            <div class="section summary">
                <h2>Analysis Summary</h2>
        """
        
        # Add general info
        if self.repertoire_df is not None:
            num_samples = len(self.repertoire_df['sample'].unique())
            num_clonotypes = len(self.repertoire_df)
            
            html_content += f"""
                <p>Number of samples analyzed: {num_samples}</p>
                <p>Total number of unique clonotypes: {num_clonotypes}</p>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Sample Information</h2>
                <table>
                    <tr>
                        <th>Sample</th>
                        <th>Condition</th>
                        <th>Clonotypes</th>
                        <th>Top Clone Frequency</th>
                        <th>Clonality</th>
                    </tr>
        """
        
        if self.repertoire_df is not None:
            for sample in self.repertoire_df['sample'].unique():
                sample_data = self.repertoire_df[self.repertoire_df['sample'] == sample]
                
                condition = "Unknown"
                if 'condition' in sample_data.columns:
                    condition = sample_data['condition'].iloc[0]
                
                num_clonotypes = len(sample_data)
                top_clone_freq = sample_data['frequency'].max() if not sample_data.empty else 0
                top_clone_freq_pct = f"{top_clone_freq * 100:.2f}%"
                
                # Calculate clonality
                clonality = self._calculate_clonality(sample_data['frequency'])
                
                html_content += f"""
                    <tr>
                        <td>{sample}</td>
                        <td>{condition}</td>
                        <td>{num_clonotypes}</td>
                        <td>{top_clone_freq_pct}</td>
                        <td>{clonality:.4f}</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Key Plots</h2>
                
                <div class="plot">
                    <h3>Clone Size Distribution</h3>
                    <img src="clone_size_distribution.pdf" alt="Clone Size Distribution">
                </div>
                
                <div class="plot">
                    <h3>V Gene Usage</h3>
                    <img src="top_v_gene_usage_heatmap.pdf" alt="V Gene Usage">
                </div>
                
                <div class="plot">
                    <h3>Diversity Metrics</h3>
                    <img src="diversity_radar_chart.pdf" alt="Diversity Metrics">
                </div>
            </div>
            
            <div class="section">
                <h2>Analysis Details</h2>
                <p>This report summarizes the results of a single-cell TCR repertoire analysis. The analysis included the following steps:</p>
                <ul>
                    <li>Extraction and QC of TCR sequences from 10X Genomics data</li>
                    <li>V(D)J annotation and CDR3 identification</li>
                    <li>Clonotype assignment and repertoire compilation</li>
                    <li>Analysis of V(D)J usage, CDR3 length, motifs, and clonal expansion</li>
                    <li>Calculation of diversity metrics and repertoire overlap</li>
                    <li>Comparative statistical analysis between conditions</li>
                </ul>
                <p>For full details, please refer to the output files in the analysis directory.</p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(os.path.join(output_dir, "analysis_report.html"), "w") as f:
            f.write(html_content)


# Main execution function
def run_sctcr_analysis(data_dir, output_dir, sample_info_file=None, gene_expression_file=None):
    """
    Run the complete scTCR analysis pipeline
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the raw fastq.gz files
    output_dir : str
        Directory for saving analysis results
    sample_info_file : str, optional
        Path to sample information file (CSV with sample, condition columns)
    gene_expression_file : str, optional
        Path to gene expression file for integration
    """
    print("Starting scTCR analysis pipeline...")
    
    # Load sample information if provided
    sample_info = None
    if sample_info_file and os.path.exists(sample_info_file):
        sample_info = pd.read_csv(sample_info_file)
        print(f"Loaded sample information for {len(sample_info)} samples")
    
    # Initialize analyzer
    analyzer = scTCRAnalyzer(data_dir, output_dir, sample_info)
    
    # Run analysis steps
    analyzer.extract_and_qc_tcr()
    analyzer.compile_clonotypes()
    analyzer.analyze_vdj_usage()
    analyzer.analyze_cdr3_features()
    analyzer.analyze_clonal_expansion()
    analyzer.calculate_diversity_metrics()
    analyzer.calculate_repertoire_overlap()
    analyzer.comparative_analysis()
    
    # Optional: Integrate with gene expression
    if gene_expression_file:
        analyzer.integrate_with_expression(gene_expression_file)
    
    # Create final visualizations
    analyzer.visualize_results()
    
    print("scTCR analysis pipeline completed!")
    return analyzer


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Single-cell TCR analysis pipeline for 10X Genomics data')
    parser.add_argument('--data_dir', required=True, help='Directory containing the raw fastq.gz files')
    parser.add_argument('--output_dir', required=True, help='Directory for saving analysis results')
    parser.add_argument('--sample_info', help='Path to sample information file (CSV with sample,condition columns)')
    parser.add_argument('--gene_expression', help='Path to gene expression file for integration')
    
    args = parser.parse_args()
    
    # Run the analysis
    run_sctcr_analysis(args.data_dir, args.output_dir, args.sample_info, args.gene_expression)
        """
        
