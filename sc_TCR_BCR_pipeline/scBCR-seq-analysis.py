# Single Cell BCR Analysis Pipeline for 10X Genomics Data
# This script processes fastq.gz files from 10X Genomics scBCR-seq and performs comprehensive BCR repertoire analysis

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
from Bio import SeqIO, Seq, motifs, pairwise2, Align
from Bio.SeqUtils import GC
import scanpy as sc
import anndata
import json
import pickle
import itertools
from adjustText import adjust_text
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy

class scBCRAnalyzer:
    def __init__(self, data_dir, output_dir, sample_info=None):
        """
        Initialize the scBCR analyzer with input and output directories
        
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
        self.isotype_distribution = None
        self.shm_data = None
        self.lineage_trees = None
        
    def extract_and_qc_bcr(self, cellranger_path="cellranger"):
        """
        Extract BCR sequences from fastq.gz files using Cell Ranger
        
        Parameters:
        -----------
        cellranger_path : str
            Path to cellranger executable
        """
        print("Starting BCR sequence extraction and QC...")
        
        # Find all sample directories
        sample_dirs = glob.glob(os.path.join(self.data_dir, "*"))
        
        for sample_dir in sample_dirs:
            sample_name = os.path.basename(sample_dir)
            output_path = os.path.join(self.output_dir, sample_name)
            
            # Run Cell Ranger vdj (with --chain=B for B-cell receptor)
            cmd = [
                cellranger_path, "vdj",
                "--id=" + sample_name,
                "--fastqs=" + sample_dir,
                "--reference=refdata-cellranger-vdj-GRCh38-alts-ensembl-5.0.0",
                "--sample=" + sample_name,
                "--chain=B",  # Specifically for BCR
                "--localcores=8",
                "--localmem=64"
            ]
            
            print(f"Processing sample {sample_name} with command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
        print("BCR extraction and VDJ assembly completed!")
        
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
                    # Extract relevant BCR information
                    cell_id = cell.get('barcode')
                    
                    # Process each chain
                    for contig in cell.get('contig_annotations', []):
                        if contig.get('productive', False) and contig.get('is_cell', False):
                            chain_type = contig.get('chain')
                            
                            # Process only heavy and light chains
                            if chain_type in ['IGH', 'IGK', 'IGL']:
                                # Get V(D)J genes
                                v_gene = contig.get('v_gene', '')
                                d_gene = contig.get('d_gene', '') if chain_type == 'IGH' else None
                                j_gene = contig.get('j_gene', '')
                                c_gene = contig.get('c_gene', '')
                                
                                # Extract isotype from c_gene for heavy chains
                                isotype = None
                                if chain_type == 'IGH' and c_gene:
                                    isotype_match = re.search(r'IGHG[1-4]|IGHA[1-2]|IGHM|IGHD|IGHE', c_gene)
                                    if isotype_match:
                                        isotype = isotype_match.group(0)
                                
                                # Get CDR3 and full sequence
                                cdr3 = contig.get('cdr3', '')
                                cdr3_nt = contig.get('cdr3_nt', '')
                                
                                # Get full_length information and sequences
                                full_length = contig.get('full_length', False)
                                fwr1 = contig.get('fwr1', '')
                                fwr1_nt = contig.get('fwr1_nt', '')
                                cdr1 = contig.get('cdr1', '')
                                cdr1_nt = contig.get('cdr1_nt', '')
                                fwr2 = contig.get('fwr2', '')
                                fwr2_nt = contig.get('fwr2_nt', '')
                                cdr2 = contig.get('cdr2', '')
                                cdr2_nt = contig.get('cdr2_nt', '')
                                fwr3 = contig.get('fwr3', '')
                                fwr3_nt = contig.get('fwr3_nt', '')
                                fwr4 = contig.get('fwr4', '')
                                fwr4_nt = contig.get('fwr4_nt', '')
                                
                                # Get mutation information
                                v_sequence = contig.get('v_sequence', '')
                                v_germline_sequence = contig.get('v_germline_sequence', '')
                                
                                clonotype_info = {
                                    'sample': sample_name,
                                    'cell_id': cell_id,
                                    'clonotype_id': cell.get('clonotype_id', 'None'),
                                    'chain': chain_type,
                                    'v_gene': v_gene,
                                    'd_gene': d_gene,
                                    'j_gene': j_gene,
                                    'c_gene': c_gene,
                                    'isotype': isotype,
                                    'cdr3': cdr3,
                                    'cdr3_nt': cdr3_nt,
                                    'cdr1': cdr1,
                                    'cdr1_nt': cdr1_nt,
                                    'cdr2': cdr2,
                                    'cdr2_nt': cdr2_nt,
                                    'fwr1': fwr1,
                                    'fwr1_nt': fwr1_nt,
                                    'fwr2': fwr2,
                                    'fwr2_nt': fwr2_nt,
                                    'fwr3': fwr3,
                                    'fwr3_nt': fwr3_nt,
                                    'fwr4': fwr4,
                                    'fwr4_nt': fwr4_nt,
                                    'full_length': full_length,
                                    'v_sequence': v_sequence,
                                    'v_germline_sequence': v_germline_sequence,
                                    'reads': contig.get('reads', 0),
                                    'umis': contig.get('umis', 0),
                                    'high_confidence': contig.get('high_confidence', False)
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
        
        print(f"Compiled {len(self.clonotype_df)} BCR sequences from {len(analysis_dirs)} samples")
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
        
        # Get the V, J, CDR3, isotype information for each clonotype (heavy and light chains)
        clonotype_info = {}
        
        for sample in self.clonotype_df['sample'].unique():
            sample_data = self.clonotype_df[self.clonotype_df['sample'] == sample]
            
            for clonotype_id in sample_data['clonotype_id'].unique():
                if clonotype_id == 'None':
                    continue
                    
                clonotype_data = sample_data[sample_data['clonotype_id'] == clonotype_id]
                
                # Get heavy chain data
                igh_data = clonotype_data[clonotype_data['chain'] == 'IGH'].iloc[0] if not clonotype_data[clonotype_data['chain'] == 'IGH'].empty else None
                
                # Get light chain data (prioritize kappa over lambda if both present)
                igk_data = clonotype_data[clonotype_data['chain'] == 'IGK'].iloc[0] if not clonotype_data[clonotype_data['chain'] == 'IGK'].empty else None
                igl_data = clonotype_data[clonotype_data['chain'] == 'IGL'].iloc[0] if not clonotype_data[clonotype_data['chain'] == 'IGL'].empty else None
                
                light_chain_data = igk_data if igk_data is not None else igl_data
                light_chain_type = 'IGK' if igk_data is not None else ('IGL' if igl_data is not None else None)
                
                # Store clonotype information
                clonotype_info[(sample, clonotype_id)] = {
                    'igh_v': igh_data['v_gene'] if igh_data is not None else None,
                    'igh_d': igh_data['d_gene'] if igh_data is not None else None,
                    'igh_j': igh_data['j_gene'] if igh_data is not None else None,
                    'igh_cdr3': igh_data['cdr3'] if igh_data is not None else None,
                    'igh_cdr3_nt': igh_data['cdr3_nt'] if igh_data is not None else None,
                    'isotype': igh_data['isotype'] if igh_data is not None else None,
                    'light_chain': light_chain_type,
                    'light_v': light_chain_data['v_gene'] if light_chain_data is not None else None,
                    'light_j': light_chain_data['j_gene'] if light_chain_data is not None else None,
                    'light_cdr3': light_chain_data['cdr3'] if light_chain_data is not None else None,
                    'light_cdr3_nt': light_chain_data['cdr3_nt'] if light_chain_data is not None else None
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
                    'igh_v': info['igh_v'],
                    'igh_d': info['igh_d'],
                    'igh_j': info['igh_j'],
                    'igh_cdr3': info['igh_cdr3'],
                    'igh_cdr3_nt': info['igh_cdr3_nt'],
                    'isotype': info['isotype'],
                    'light_chain': info['light_chain'],
                    'light_v': info['light_v'],
                    'light_j': info['light_j'],
                    'light_cdr3': info['light_cdr3'],
                    'light_cdr3_nt': info['light_cdr3_nt']
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
        self.repertoire_df.to_csv(os.path.join(self.output_dir, "bcr_repertoire.csv"), index=False)
        
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
            
            # Heavy chain genes
            if not pd.isna(clone['igh_v']):
                v_usage[sample][clone['igh_v']] += clone['clone_size']
            if not pd.isna(clone['igh_j']):
                j_usage[sample][clone['igh_j']] += clone['clone_size']
            if not pd.isna(clone['igh_d']):
                d_usage[sample][clone['igh_d']] += clone['clone_size']
            if not pd.isna(clone['igh_v']) and not pd.isna(clone['igh_j']):
                vj_pairing[sample][f"{clone['igh_v']}:{clone['igh_j']}"] += clone['clone_size']
            
            # Light chain genes
            if not pd.isna(clone['light_v']):
                v_usage[sample][clone['light_v']] += clone['clone_size']
            if not pd.isna(clone['light_j']):
                j_usage[sample][clone['light_j']] += clone['clone_size']
            if not pd.isna(clone['light_v']) and not pd.isna(clone['light_j']):
                vj_pairing[sample][f"{clone['light_v']}:{clone['light_j']}"] += clone['clone_size']
        
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
        
        # Analyze gene usage by chain type
        self._analyze_gene_usage_by_chain(vdj_dir)
        
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
    
    def _analyze_gene_usage_by_chain(self, output_dir):
        """
        Analyze gene usage patterns separately for heavy and light chains
        """
        if self.clonotype_df is None:
            return
            
        # Separate V gene usage by chain type
        chain_v_usage = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for _, row in self.clonotype_df.iterrows():
            sample = row['sample']
            chain = row['chain']
            v_gene = row['v_gene']
            
            if not pd.isna(v_gene):
                chain_v_usage[chain][sample][v_gene] += 1
        
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
        
        # Analyze physicochemical properties
        self._analyze_cdr3_properties(cdr3_dir)
        
        print("CDR3 feature analysis completed!")
    
    def _analyze_cdr3_length(self, output_dir):
        """
        Analyze CDR3 length distribution
        """
        # Prepare data
        heavy_lengths = []
        light_lengths = []
        
        for _, clone in self.repertoire_df.iterrows():
            sample = clone['sample']
            condition = clone.get('condition', 'Unknown')
            
            # Get condition if available
            if 'condition' not in clone and self.sample_info is not None:
                condition = self.sample_info.loc[self.sample_info['sample'] == sample, 'condition'].iloc[0]
            
            # Heavy chain CDR3 length
            if not pd.isna(clone['igh_cdr3']) and clone['igh_cdr3']:
                for _ in range(int(clone['clone_size'])):
                    heavy_lengths.append({
                        'sample': sample,
                        'condition': condition,
                        'chain': 'IGH',
                        'isotype': clone['isotype'] if not pd.isna(clone['isotype']) else 'Unknown',
                        'length': len(clone['igh_cdr3'])
                    })
            
            # Light chain CDR3 length
            if not pd.isna(clone['light_cdr3']) and clone['light_cdr3']:
                for _ in range(int(clone['clone_size'])):
                    light_lengths.append({
                        'sample': sample,
                        'condition': condition,
                        'chain': clone['light_chain'] if not pd.isna(clone['light_chain']) else 'Unknown',
                        'length': len(clone['light_cdr3'])
                    })
        
        # Convert to DataFrames
        heavy_lengths_df = pd.DataFrame(heavy_lengths)
        light_lengths_df = pd.DataFrame(light_lengths)
        
        # Combine
        all_lengths_df = pd.concat([heavy_lengths_df, light_lengths_df])
        
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
        
        # Plot length distribution by isotype for heavy chains
        if not heavy_lengths_df.empty and 'isotype' in heavy_lengths_df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=heavy_lengths_df, x='isotype', y='length')
            plt.title("Heavy Chain CDR3 Length by Isotype")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "igh_cdr3_length_by_isotype.pdf"))
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
            
            # Process heavy chain CDR3
            if not pd.isna(clone['igh_cdr3']) and clone['igh_cdr3']:
                for aa in clone['igh_cdr3']:
                    if aa in amino_acids:
                        aa_counts[aa][(sample, condition, 'IGH')] += clone['clone_size']
            
            # Process light chain CDR3
            if not pd.isna(clone['light_cdr3']) and clone['light_cdr3']:
                chain = clone['light_chain'] if not pd.isna(clone['light_chain']) else 'Light'
                for aa in clone['light_cdr3']:
                    if aa in amino_acids:
                        aa_counts[aa][(sample, condition, chain)] += clone['clone_size']
        
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
        
        # Plot heatmap of amino acid composition by chain
        for chain in aa_composition_df['chain'].unique():
            chain_data = aa_composition_df[aa_composition_df['chain'] == chain]
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
        """
        # Get all unique CDR3 sequences with their counts
        igh_seqs = defaultdict(int)
        light_seqs = defaultdict(int)
        
        for _, clone in self.repertoire_df.iterrows():
            if not pd.isna(clone['igh_cdr3']) and clone['igh_cdr3']:
                igh_seqs[clone['igh_cdr3']] += clone['clone_size']
            if not pd.isna(clone['light_cdr3']) and clone['light_cdr3']:
                light_seqs[clone['light_cdr3']] += clone['clone_size']
        
        # Find common motifs in CDR3 sequences
        igh_motifs = self._find_motifs(igh_seqs, 'IGH', output_dir)
        light_motifs = self._find_motifs(light_seqs, 'Light', output_dir)
        
        # Save motif data
        if igh_motifs:
            with open(os.path.join(output_dir, "igh_motifs.txt"), "w") as f:
                for motif, count in igh_motifs.items():
                    f.write(f"{motif}\t{count}\n")
        
        if light_motifs:
            with open(os.path.join(output_dir, "light_motifs.txt"), "w") as f:
                for motif, count in light_motifs.items():
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
    
    def _analyze_cdr3_properties(self, output_dir):
        """
        Analyze physiochemical properties of CDR3 sequences
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
        
        for _, clone in self.repertoire_df.iterrows():
            sample = clone['sample']
            condition = clone.get('condition', 'Unknown')
            isotype = clone['isotype'] if not pd.isna(clone['isotype']) else None
            
            # Process heavy chain CDR3
            if not pd.isna(clone['igh_cdr3']) and clone['igh_cdr3']:
                cdr3 = clone['igh_cdr3']
                
                # Calculate properties
                hydrophobicity_score = sum(hydrophobicity.get(aa, 0) for aa in cdr3) / len(cdr3)
                charge_score = sum(charge.get(aa, 0) for aa in cdr3)
                
                properties.append({
                    'sample': sample,
                    'condition': condition,
                    'chain': 'IGH',
                    'isotype': isotype,
                    'cdr3_length': len(cdr3),
                    'hydrophobicity': hydrophobicity_score,
                    'charge': charge_score,
                    'clone_size': clone['clone_size']
                })
            
            # Process light chain CDR3
            if not pd.isna(clone['light_cdr3']) and clone['light_cdr3']:
                cdr3 = clone['light_cdr3']
                chain_type = clone['light_chain'] if not pd.isna(clone['light_chain']) else 'Light'
                
                # Calculate properties
                hydrophobicity_score = sum(hydrophobicity.get(aa, 0) for aa in cdr3) / len(cdr3)
                charge_score = sum(charge.get(aa, 0) for aa in cdr3)
                
                properties.append({
                    'sample': sample,
                    'condition': condition,
                    'chain': chain_type,
                    'isotype': None,
                    'cdr3_length': len(cdr3),
                    'hydrophobicity': hydrophobicity_score,
                    'charge': charge_score,
                    'clone_size': clone['clone_size']
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
                               hue='chain', size='clone_size', sizes=(20, 200), alpha=0.7)
            plt.title("CDR3 Hydrophobicity vs Length")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cdr3_hydrophobicity_vs_length.pdf"))
            plt.close()
            
            # If isotype information is available, plot by isotype
            igh_data = properties_df[properties_df['chain'] == 'IGH']
            if not igh_data.empty and 'isotype' in igh_data.columns:
                # Filter out rows with no isotype
                igh_data = igh_data[~igh_data['isotype'].isna()]
                
                if not igh_data.empty:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=igh_data, x='isotype', y='hydrophobicity')
                    plt.title("Heavy Chain CDR3 Hydrophobicity by Isotype")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "igh_cdr3_hydrophobicity_by_isotype.pdf"))
                    plt.close()
                    
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=igh_data, x='isotype', y='charge')
                    plt.title("Heavy Chain CDR3 Net Charge by Isotype")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "igh_cdr3_charge_by_isotype.pdf"))
                    plt.close()
    
    def analyze_isotype_distribution(self):
        """
        Analyze isotype distribution across samples and conditions
        """
        print("Analyzing isotype distribution...")
        
        if self.repertoire_df is None:
            print("Error: No repertoire data available. Run compile_clonotypes first.")
            return None
        
        # Create directory for isotype analysis
        isotype_dir = os.path.join(self.output_dir, "isotype_analysis")
        os.makedirs(isotype_dir, exist_ok=True)
        
        # Calculate isotype distribution
        isotype_counts = []
        
        for sample in self.repertoire_df['sample'].unique():
            sample_data = self.repertoire_df[self.repertoire_df['sample'] == sample]
            
            # Get condition if available
            condition = "Unknown"
            if 'condition' in sample_data.columns:
                condition = sample_data['condition'].iloc[0]
            elif self.sample_info is not None:
                condition = self.sample_info.loc[self.sample_info['sample'] == sample, 'condition'].iloc[0]
            
            # Count cells by isotype (weighted by clone size)
            isotype_count = defaultdict(int)
            total_cells = 0
            
            for _, clone in sample_data.iterrows():
                if not pd.isna(clone['isotype']):
                    isotype_count[clone['isotype']] += clone['clone_size']
                    total_cells += clone['clone_size']
                else:
                    isotype_count['Unknown'] += clone['clone_size']
                    total_cells += clone['clone_size']
            
            # Calculate percentages
            for isotype, count in isotype_count.items():
                isotype_counts.append({
                    'sample': sample,
                    'condition': condition,
                    'isotype': isotype,
                    'count': count,
                    'percentage': (count / total_cells * 100) if total_cells > 0 else 0
                })
        
        # Convert to DataFrame
        isotype_df = pd.DataFrame(isotype_counts)
        
        # Save isotype distribution data
        isotype_df.to_csv(os.path.join(isotype_dir, "isotype_distribution.csv"), index=False)
        
        # Plot isotype distribution
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
        if 'condition' in isotype_df.columns and len(isotype_df['condition'].unique()) > 1:
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
            isotypes = isotype_df['isotype'].unique()
            
            if len(isotype_df['condition'].unique()) == 2:
                # t-test for two conditions
                test_results = []
                
                for isotype in isotypes:
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
    
    def analyze_somatic_hypermutation(self):
        """
        Analyze somatic hypermutation (SHM) patterns
        """
        print("Analyzing somatic hypermutation...")
        
        if self.clonotype_df is None:
            print("Error: No clonotype data available. Run compile_clonotypes first.")
            return None
        
        # Create directory for SHM analysis
        shm_dir = os.path.join(self.output_dir, "somatic_hypermutation")
        os.makedirs(shm_dir, exist_ok=True)
        
        # Calculate SHM for each BCR sequence
        shm_data = []
        
        for _, row in self.clonotype_df.iterrows():
            # Only analyze full-length sequences with germline information
            if not row['full_length'] or pd.isna(row['v_sequence']) or pd.isna(row['v_germline_sequence']):
                continue
            
            v_seq = row['v_sequence']
            v_germline = row['v_germline_sequence']
            
            # Calculate mutations
            mutations, mutation_rate = self._calculate_mutations(v_seq, v_germline)
            
            # Get sample and condition
            sample = row['sample']
            condition = "Unknown"
            if self.sample_info is not None:
                condition = self.sample_info.loc[self.sample_info['sample'] == sample, 'condition'].iloc[0]
            
            # Store SHM data
            shm_data.append({
                'sample': sample,
                'condition': condition,
                'cell_id': row['cell_id'],
                'chain': row['chain'],
                'v_gene': row['v_gene'],
                'isotype': row['isotype'] if not pd.isna(row['isotype']) else None,
                'mutations': mutations,
                'sequence_length': len(v_germline),
                'mutation_rate': mutation_rate,
                'clonotype_id': row['clonotype_id']
            })
        
        # Convert to DataFrame
        shm_df = pd.DataFrame(shm_data)
        
        # Save SHM data
        shm_df.to_csv(os.path.join(shm_dir, "somatic_hypermutation.csv"), index=False)
        
        # Analyze and plot SHM data
        if not shm_df.empty:
            # Plot mutation rate by chain
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=shm_df, x='chain', y='mutation_rate', hue='condition')
            plt.title("Somatic Hypermutation Rate by Chain")
            plt.ylabel("Mutation Rate (%)")
            plt.tight_layout()
            plt.savefig(os.path.join(shm_dir, "mutation_rate_by_chain.pdf"))
            plt.close()
            
            # For heavy chains, plot by isotype
            igh_data = shm_df[shm_df['chain'] == 'IGH']
            if not igh_data.empty and 'isotype' in igh_data.columns:
                # Remove rows with no isotype
                igh_data = igh_data[~igh_data['isotype'].isna()]
                
                if not igh_data.empty:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=igh_data, x='isotype', y='mutation_rate')
                    plt.title("Heavy Chain Mutation Rate by Isotype")
                    plt.ylabel("Mutation Rate (%)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(shm_dir, "igh_mutation_rate_by_isotype.pdf"))
                    plt.close()
            
            # Plot mutation rate distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data=shm_df, x='mutation_rate', hue='chain', bins=20, kde=True)
            plt.title("Distribution of Somatic Hypermutation Rates")
            plt.xlabel("Mutation Rate (%)")
            plt.tight_layout()
            plt.savefig(os.path.join(shm_dir, "mutation_rate_distribution.pdf"))
            plt.close()
            
            # Calculate average mutation rate by sample and chain
            avg_mutation = shm_df.groupby(['sample', 'chain'])['mutation_rate'].mean().reset_index()
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=avg_mutation, x='sample', y='mutation_rate', hue='chain')
            plt.title("Average Mutation Rate by Sample and Chain")
            plt.xlabel("Sample")
            plt.ylabel("Average Mutation Rate (%)")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(shm_dir, "avg_mutation_rate_by_sample.pdf"))
            plt.close()
            
            # If conditions are available, compare conditions
            if 'condition' in shm_df.columns and len(shm_df['condition'].unique()) > 1:
                # Test for statistical significance
                conditions = shm_df['condition'].unique()
                chains = shm_df['chain'].unique()
                
                if len(conditions) == 2:
                    # t-test for two conditions
                    test_results = []
                    
                    for chain in chains:
                        chain_data = shm_df[shm_df['chain'] == chain]
                        
                        group1 = chain_data[chain_data['condition'] == conditions[0]]['mutation_rate'].values
                        group2 = chain_data[chain_data['condition'] == conditions[1]]['mutation_rate'].values
                        
                        if len(group1) > 0 and len(group2) > 0:
                            stat, pval = stats.ttest_ind(group1, group2, equal_var=False)
                            
                            test_results.append({
                                'chain': chain,
                                'mean_mutation_rate_condition1': np.mean(group1),
                                'mean_mutation_rate_condition2': np.mean(group2),
                                'difference': np.mean(group1) - np.mean(group2),
                                'p_value': pval,
                                'significant': pval < 0.05
                            })
                    
                    if test_results:
                        test_df = pd.DataFrame(test_results)
                        test_df.to_csv(os.path.join(shm_dir, "mutation_rate_statistical_tests.csv"), index=False)
        
        # Store SHM data
        self.shm_data = shm_df
        
        print("Somatic hypermutation analysis completed!")
        return shm_df
    
    def _calculate_mutations(self, sequence, germline):
        """
        Calculate mutations between sequence and germline
        
        Returns:
        --------
        mutations : int
            Number of mutations
        mutation_rate : float
            Mutation rate as percentage
        """
        # Ensure sequences are of the same length
        min_length = min(len(sequence), len(germline))
        sequence = sequence[:min_length]
        germline = germline[:min_length]
        
        # Count mismatches
        mutations = sum(a != b for a, b in zip(sequence, germline))
        
        # Calculate mutation rate
        mutation_rate = (mutations / min_length) * 100
        
        return mutations, mutation_rate
    
    def reconstruct_lineages(self, distance_threshold=0.15):
        """
        Reconstruct B-cell lineages based on sequence similarity
        
        Parameters:
        -----------
        distance_threshold : float
            Maximum normalized edit distance to consider sequences as related
        """
        print("Reconstructing B-cell lineages...")
        
        if self.repertoire_df is None or self.clonotype_df is None:
            print("Error: No repertoire/clonotype data available. Run compile_clonotypes first.")
            return None
        
        # Create directory for lineage analysis
        lineage_dir = os.path.join(self.output_dir, "lineage_analysis")
        os.makedirs(lineage_dir, exist_ok=True)
        
        # Group BCRs by V and J gene combination (potential lineages)
        lineage_groups = defaultdict(list)
        
        for _, clone in self.repertoire_df.iterrows():
            # Skip clones without heavy chain
            if pd.isna(clone['igh_v']) or pd.isna(clone['igh_j']) or pd.isna(clone['igh_cdr3_nt']):
                continue
                
            # Create a key based on V and J genes
            key = f"{clone['igh_v']}:{clone['igh_j']}"
            
            lineage_groups[key].append(clone)
        
        # Process each potential lineage group
        lineage_data = []
        lineage_networks = {}
        
        for vj_key, group in lineage_groups.items():
            # Skip small groups
            if len(group) < 2:
                continue
                
            # Calculate pairwise distances between CDR3 sequences
            cdr3_seqs = [clone['igh_cdr3_nt'] for clone in group]
            clone_ids = [clone['clonotype_id'] for clone in group]
            
            # Create distance matrix
            n_seqs = len(cdr3_seqs)
            distance_matrix = np.zeros((n_seqs, n_seqs))
            
            for i in range(n_seqs):
                for j in range(i+1, n_seqs):
                    # Calculate normalized edit distance
                    distance = self._calculate_edit_distance(cdr3_seqs[i], cdr3_seqs[j])
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
            
            # Create network graph
            G = nx.Graph()
            
            # Add nodes
            for i, clone_id in enumerate(clone_ids):
                G.add_node(clone_id, 
                           sequence=cdr3_seqs[i], 
                           v_gene=group[i]['igh_v'], 
                           j_gene=group[i]['igh_j'],
                           isotype=group[i]['isotype'] if not pd.isna(group[i]['isotype']) else None,
                           clone_size=group[i]['clone_size'])
            
            # Add edges for related sequences
            for i in range(n_seqs):
                for j in range(i+1, n_seqs):
                    if distance_matrix[i, j] <= distance_threshold:
                        G.add_edge(clone_ids[i], clone_ids[j], distance=distance_matrix[i, j])
            
            # Find connected components (lineages)
            connected_components = list(nx.connected_components(G))
            
            # Process each lineage
            for i, component in enumerate(connected_components):
                if len(component) >= 2:  # Only consider lineages with at least 2 members
                    lineage_id = f"{vj_key.replace(':', '_')}_L{i+1}"
                    
                    # Create subgraph for this lineage
                    lineage_graph = G.subgraph(component)
                    
                    # Store lineage graph
                    lineage_networks[lineage_id] = lineage_graph
                    
                    # Add lineage data
                    for node in component:
                        node_data = G.nodes[node]
                        
                        sample = None
                        condition = None
                        
                        # Find sample and condition for this clone
                        for clone in group:
                            if clone['clonotype_id'] == node:
                                sample = clone['sample']
                                condition = clone.get('condition', 'Unknown')
                                break
                        
                        lineage_data.append({
                            'lineage_id': lineage_id,
                            'clonotype_id': node,
                            'v_gene': node_data['v_gene'],
                            'j_gene': node_data['j_gene'],
                            'isotype': node_data['isotype'],
                            'clone_size': node_data['clone_size'],
                            'sample': sample,
                            'condition': condition,
                            'cdr3_nt': node_data['sequence']
                        })
        
        # Convert to DataFrame
        lineage_df = pd.DataFrame(lineage_data)
        
        # Save lineage data
        lineage_df.to_csv(os.path.join(lineage_dir, "lineage_assignments.csv"), index=False)
        
        # Plot lineage statistics
        if not lineage_df.empty:
            # Number of lineages
            lineage_stats = lineage_df.groupby('lineage_id').agg({
                'clonotype_id': 'count',
                'clone_size': 'sum'
            }).rename(columns={'clonotype_id': 'lineage_size'}).reset_index()
            
            # Plot lineage size distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data=lineage_stats, x='lineage_size', bins=range(2, 21))
            plt.title("Lineage Size Distribution")
            plt.xlabel("Number of Clonotypes in Lineage")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(lineage_dir, "lineage_size_distribution.pdf"))
            plt.close()
            
            # Plot top 10 largest lineages
            top_lineages = lineage_stats.nlargest(10, 'lineage_size')
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=top_lineages, x='lineage_id', y='lineage_size')
            plt.title("Top 10 Largest Lineages")
            plt.xlabel("Lineage ID")
            plt.ylabel("Number of Clonotypes")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(lineage_dir, "top_lineages.pdf"))
            plt.close()
            
            # Visualize a few lineage networks
            for i, (lineage_id, G) in enumerate(list(lineage_networks.items())[:5]):
                if len(G.nodes) >= 3:  # Only visualize reasonably sized lineages
                    self._visualize_lineage(G, lineage_id, lineage_dir)
        
        # Store lineage data
        self.lineage_trees = {
            'lineage_df': lineage_df,
            'networks': lineage_networks
        }
        
        print("Lineage reconstruction completed!")
        return self.lineage_trees
    
    def _calculate_edit_distance(self, seq1, seq2):
        """
        Calculate normalized edit distance between two sequences
        
        Returns:
        --------
        float
            Normalized edit distance (0-1)
        """
        # Use Levenshtein distance
        alignment = pairwise2.align.globalms(seq1, seq2, 2, -1, -1, -0.5, one_alignment_only=True)[0]
        
        # Calculate number of mismatches
        match_count = sum(a == b for a, b in zip(alignment.seqA, alignment.seqB) if a != '-' and b != '-')
        
        # Calculate normalized distance
        max_length = max(len(seq1), len(seq2))
        match_ratio = match_count / max_length
        distance = 1.0 - match_ratio
        
        return distance
    
    def _visualize_lineage(self, G, lineage_id, output_dir):
        """
        Visualize a lineage as a network graph
        
        Parameters:
        -----------
        G : networkx.Graph
            Lineage graph
        lineage_id : str
            Lineage identifier
        output_dir : str
            Output directory
        """
        plt.figure(figsize=(10, 8))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G)
        
        # Determine node size by clone size
        node_sizes = [G.nodes[node]['clone_size'] * 20 for node in G.nodes]
        
        # Color nodes by isotype if available
        isotype_colors = {
            'IGHM': 'blue',
            'IGHD': 'skyblue',
            'IGHG1': 'red',
            'IGHG2': 'darkred',
            'IGHG3': 'salmon',
            'IGHG4': 'coral',
            'IGHA1': 'green',
            'IGHA2': 'darkgreen',
            'IGHE': 'purple',
            None: 'gray'
        }
        
        node_colors = [isotype_colors.get(G.nodes[node]['isotype'], 'gray') for node in G.nodes]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, 
                              node_size=node_sizes, 
                              node_color=node_colors, 
                              alpha=0.8)
        
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        
        # Add labels
        labels = {node: node for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        # Add legend for isotypes
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, label=isotype, markersize=8)
                          for isotype, color in isotype_colors.items() 
                          if isotype is not None and any(G.nodes[node]['isotype'] == isotype for node in G.nodes)]
        
        plt.legend(handles=legend_elements)
        
        plt.title(f"Lineage Network: {lineage_id}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"lineage_{lineage_id}.pdf"))
        plt.close()
    
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
        
        # Heavy chain CDR3 lengths
        igh_lengths = []
        for _, clone in expanded_df.iterrows():
            if not pd.isna(clone['igh_cdr3']) and clone['igh_cdr3']:
                igh_lengths.append(len(clone['igh_cdr3']))
        
        # Light chain CDR3 lengths
        light_lengths = []
        for _, clone in expanded_df.iterrows():
            if not pd.isna(clone['light_cdr3']) and clone['light_cdr3']:
                light_lengths.append(len(clone['light_cdr3']))
        
        if igh_lengths and light_lengths:
            plt.hist([igh_lengths, light_lengths], bins=range(5, 31), 
                    label=['IGH', 'Light'], alpha=0.7, edgecolor='black')
            plt.legend()
        elif igh_lengths:
            plt.hist(igh_lengths, bins=range(5, 31), label='IGH', alpha=0.7, edgecolor='black')
            plt.legend()
        elif light_lengths:
            plt.hist(light_lengths, bins=range(5, 31), label='Light', alpha=0.7, edgecolor='black')
            plt.legend()
        
        plt.title("CDR3 Length Distribution of Expanded Clones")
        plt.xlabel("CDR3 Length")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "expanded_cdr3_length.pdf"))
        plt.close()
        
        # Plot isotype distribution of expanded clones
        if 'isotype' in expanded_df.columns:
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
            set1_igh = set()
            set1_light = set()
            freq1 = defaultdict(float)
            
            for _, row in data1.iterrows():
                if not pd.isna(row['igh_cdr3']) and row['igh_cdr3']:
                    set1_igh.add(row['igh_cdr3'])
                    freq1[row['igh_cdr3']] += row['frequency']
                if not pd.isna(row['light_cdr3']) and row['light_cdr3']:
                    set1_light.add(row['light_cdr3'])
                    freq1[row['light_cdr3']] += row['frequency']
            
            for j, sample2 in enumerate(samples):
                if i == j:
                    # Self-comparison
                    jaccard_matrix[i, j] = 1.0
                    morisita_matrix[i, j] = 1.0
                    overlap_count_matrix[i, j] = len(set1_igh) + len(set1_light)
                    continue
                
                data2 = self.repertoire_df[self.repertoire_df['sample'] == sample2]
                
                # Extract unique CDR3 sequences
                set2_igh = set()
                set2_light = set()
                freq2 = defaultdict(float)
                
                for _, row in data2.iterrows():
                    if not pd.isna(row['igh_cdr3']) and row['igh_cdr3']:
                        set2_igh.add(row['igh_cdr3'])
                        freq2[row['igh_cdr3']] += row['frequency']
                    if not pd.isna(row['light_cdr3']) and row['light_cdr3']:
                        set2_light.add(row['light_cdr3'])
                        freq2[row['light_cdr3']] += row['frequency']
                
                # Calculate overlap
                overlap_igh = set1_igh.intersection(set2_igh)
                overlap_light = set1_light.intersection(set2_light)
                
                # Jaccard index
                union_igh = set1_igh.union(set2_igh)
                union_light = set1_light.union(set2_light)
                
                jaccard_igh = len(overlap_igh) / len(union_igh) if union_igh else 0
                jaccard_light = len(overlap_light) / len(union_light) if union_light else 0
                
                # Combine heavy and light chain Jaccard
                jaccard_matrix[i, j] = (jaccard_igh + jaccard_light) / 2
                
                # Morisita-Horn index (weighted by frequency)
                overlap_seqs = set(freq1.keys()).intersection(set(freq2.keys()))
                
                if overlap_seqs:
                    sum_freq1_squared = sum([freq1[seq]**2 for seq in freq1])
                    sum_freq2_squared = sum([freq2[seq]**2 for seq in freq2])
                    
                    sum_freq_products = sum([freq1[seq] * freq2[seq] for seq in overlap_seqs])
                    
                    if sum_freq1_squared > 0 and sum_freq2_squared > 0:
                        morisita_matrix[i, j] = (2 * sum_freq_products) / (sum_freq1_squared + sum_freq2_squared)
                
                # Count of overlapping sequences
                overlap_count_matrix[i, j] = len(overlap_igh) + len(overlap_light)
        
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
        plt.title("Number of Overlapping BCR Sequences")
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
        self._analyze_isotype_distribution_by_condition(comparative_dir)
        self._analyze_shm_by_condition(comparative_dir)
        
        print("Comparative analysis completed!")
    
    def _analyze_vdj_usage_by_condition(self, output_dir):
        """
        Analyze V(D)J gene usage differences between conditions
        """
        # Create V gene usage table by condition
        v_usage_by_condition = defaultdict(lambda: defaultdict(int))
        
        for _, clone in self.repertoire_df.iterrows():
            condition = clone.get('condition', 'Unknown')
            
            # Heavy chain V gene
            if not pd.isna(clone['igh_v']):
                v_usage_by_condition[condition][clone['igh_v']] += clone['clone_size']
            
            # Light chain V gene
            if not pd.isna(clone['light_v']):
                v_usage_by_condition[condition][clone['light_v']] += clone['clone_size']
        
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
                    
                    if not pd.isna(clone['igh_v']) and clone['igh_v'] == v_gene:
                        samples_by_condition[condition].append(sample)
                    
                    if not pd.isna(clone['light_v']) and clone['light_v'] == v_gene:
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
        igh_lengths = []
        light_lengths = []
        
        for _, clone in self.repertoire_df.iterrows():
            condition = clone.get('condition', 'Unknown')
            
            # Heavy chain CDR3 length
            if not pd.isna(clone['igh_cdr3']) and clone['igh_cdr3']:
                igh_lengths.append({
                    'condition': condition,
                    'chain': 'IGH',
                    'length': len(clone['igh_cdr3']),
                    'frequency': clone['frequency']
                })
            
            # Light chain CDR3 length
            if not pd.isna(clone['light_cdr3']) and clone['light_cdr3']:
                light_lengths.append({
                    'condition': condition,
                    'chain': clone['light_chain'] if not pd.isna(clone['light_chain']) else 'Light',
                    'length': len(clone['light_cdr3']),
                    'frequency': clone['frequency']
                })
        
        # Convert to DataFrames
        igh_df = pd.DataFrame(igh_lengths)
        light_df = pd.DataFrame(light_lengths)
        
        # Combine
        cdr3_df = pd.concat([igh_df, light_df])
        
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
        chains = ['IGH', 'Light']
        
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
    
    def _analyze_isotype_distribution_by_condition(self, output_dir):
        """
        Analyze isotype distribution differences between conditions
        """
        if self.isotype_distribution is None:
            print("Warning: No isotype distribution data available.")
            return
        
        # Filter to keep only known isotypes
        isotype_df = self.isotype_distribution[self.isotype_distribution['isotype'] != 'Unknown']
        
        # Calculate mean percentage by condition and isotype
        condition_isotype = isotype_df.groupby(['condition', 'isotype'])['percentage'].mean().reset_index()
        
        # Calculate standard error
        condition_isotype_sem = isotype_df.groupby(['condition', 'isotype'])['percentage'].sem().reset_index()
        condition_isotype_sem.columns = ['condition', 'isotype', 'sem']
        
        # Merge mean and sem
        condition_isotype = pd.merge(condition_isotype, condition_isotype_sem, on=['condition', 'isotype'])
        
        # Save data
        condition_isotype.to_csv(os.path.join(output_dir, "isotype_by_condition.csv"), index=False)
        
        # Plot isotype distribution by condition
        plt.figure(figsize=(12, 8))
        g = sns.barplot(data=condition_isotype, x='isotype', y='percentage', hue='condition')
        
        # Add error bars
        for i, bar in enumerate(g.patches):
            if i < len(condition_isotype):
                g.errorbar(
                    x=bar.get_x() + bar.get_width()/2,
                    y=bar.get_height(),
                    yerr=condition_isotype.iloc[i]['sem'],
                    color='black',
                    capsize=5
                )
        
        plt.title("Isotype Distribution by Condition")
        plt.xlabel("Isotype")
        plt.ylabel("Percentage (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "isotype_distribution_by_condition.pdf"))
        plt.close()
        
        # Statistical tests
        conditions = isotype_df['condition'].unique()
        
        if len(conditions) == 2:
            # t-test for each isotype
            test_results = []
            
            for isotype in isotype_df['isotype'].unique():
                isotype_data = isotype_df[isotype_df['isotype'] == isotype]
                
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
                test_df = test_df.sort_values('p_value')
                test_df.to_csv(os.path.join(output_dir, "isotype_statistical_tests.csv"), index=False)
    
    def _analyze_shm_by_condition(self, output_dir):
        """
        Analyze somatic hypermutation differences between conditions
        """
        if self.shm_data is None:
            print("Warning: No SHM data available.")
            return
        
        # Calculate mean mutation rate by condition and chain
        shm_by_condition = self.shm_data.groupby(['condition', 'chain'])['mutation_rate'].agg(['mean', 'sem']).reset_index()
        
        # Save data
        shm_by_condition.to_csv(os.path.join(output_dir, "shm_by_condition.csv"), index=False)
        
        # Plot mutation rate by condition and chain
        plt.figure(figsize=(10, 6))
        g = sns.barplot(data=self.shm_data, x='chain', y='mutation_rate', hue='condition')
        plt.title("Somatic Hypermutation Rate by Condition and Chain")
        plt.xlabel("Chain")
        plt.ylabel("Mutation Rate (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mutation_rate_by_condition.pdf"))
        plt.close()
        
        # For heavy chains only, plot by isotype and condition
        igh_data = self.shm_data[self.shm_data['chain'] == 'IGH']
        if not igh_data.empty and 'isotype' in igh_data.columns:
            # Remove rows with no isotype
            igh_data = igh_data[~igh_data['isotype'].isna()]
            
            if not igh_data.empty:
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=igh_data, x='isotype', y='mutation_rate', hue='condition')
                plt.title("Heavy Chain Mutation Rate by Isotype and Condition")
                plt.xlabel("Isotype")
                plt.ylabel("Mutation Rate (%)")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "igh_mutation_rate_by_isotype_condition.pdf"))
                plt.close()
        
        # Statistical tests
        conditions = self.shm_data['condition'].unique()
        
        if len(conditions) == 2:
            # t-test for each chain
            test_results = []
            
            for chain in self.shm_data['chain'].unique():
                chain_data = self.shm_data[self.shm_data['chain'] == chain]
                
                group1 = chain_data[chain_data['condition'] == conditions[0]]['mutation_rate'].values
                group2 = chain_data[chain_data['condition'] == conditions[1]]['mutation_rate'].values
                
                if len(group1) > 0 and len(group2) > 0:
                    stat, pval = stats.ttest_ind(group1, group2, equal_var=False)
                    
                    test_results.append({
                        'chain': chain,
                        'mean_mutation_rate_condition1': np.mean(group1),
                        'mean_mutation_rate_condition2': np.mean(group2),
                        'difference': np.mean(group1) - np.mean(group2),
                        'p_value': pval,
                        'significant': pval < 0.05
                    })
            
            if test_results:
                test_df = pd.DataFrame(test_results)
                test_df.to_csv(os.path.join(output_dir, "shm_statistical_tests.csv"), index=False)
                
            # If we have isotype information, test by isotype for heavy chains
            if not igh_data.empty:
                isotype_test_results = []
                
                for isotype in igh_data['isotype'].unique():
                    isotype_data = igh_data[igh_data['isotype'] == isotype]
                    
                    group1 = isotype_data[isotype_data['condition'] == conditions[0]]['mutation_rate'].values
                    group2 = isotype_data[isotype_data['condition'] == conditions[1]]['mutation_rate'].values
                    
                    if len(group1) > 0 and len(group2) > 0:
                        stat, pval = stats.ttest_ind(group1, group2, equal_var=False)
                        
                        isotype_test_results.append({
                            'isotype': isotype,
                            'mean_mutation_rate_condition1': np.mean(group1),
                            'mean_mutation_rate_condition2': np.mean(group2),
                            'difference': np.mean(group1) - np.mean(group2),
                            'p_value': pval,
                            'significant': pval < 0.05
                        })
                
                if isotype_test_results:
                    isotype_test_df = pd.DataFrame(isotype_test_results)
                    isotype_test_df.to_csv(os.path.join(output_dir, "shm_by_isotype_statistical_tests.csv"), index=False)
    
    def integrate_with_expression(self, gene_expression_file):
        """
        Integrate BCR data with gene expression (optional)
        
        Parameters:
        -----------
        gene_expression_file : str
            Path to gene expression file (10X Cell Ranger format)
        """
        print("Integrating BCR data with gene expression...")
        
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
            
            # Match barcodes with BCR data
            bcr_barcodes = self.clonotype_df['cell_id'].unique()
            common_barcodes = set(adata.obs.index).intersection(set(bcr_barcodes))
            
            print(f"Found {len(common_barcodes)} cells with both gene expression and BCR data")
            
            if not common_barcodes:
                print("Error: No common barcodes between gene expression and BCR data.")
                return None
            
            # Create BCR metadata for gene expression
            bcr_meta = {}
            
            for _, row in self.clonotype_df.iterrows():
                cell_id = row['cell_id']
                if cell_id in common_barcodes:
                    if cell_id not in bcr_meta:
                        bcr_meta[cell_id] = {
                            'has_bcr': True,
                            'clonotype_id': row['clonotype_id'],
                            'sample': row['sample'],
                            'isotype': row['isotype'] if 'isotype' in row and not pd.isna(row['isotype']) else None
                        }
                    
                    if row['chain'] == 'IGH':
                        bcr_meta[cell_id]['igh_v'] = row['v_gene']
                        bcr_meta[cell_id]['igh_d'] = row['d_gene']
                        bcr_meta[cell_id]['igh_j'] = row['j_gene']
                        bcr_meta[cell_id]['igh_cdr3'] = row['cdr3']
                        bcr_meta[cell_id]['isotype'] = row['isotype'] if 'isotype' in row and not pd.isna(row['isotype']) else None
                    elif row['chain'] in ['IGK', 'IGL']:
                        bcr_meta[cell_id]['light_chain'] = row['chain']
                        bcr_meta[cell_id]['light_v'] = row['v_gene']
                        bcr_meta[cell_id]['light_j'] = row['j_gene']
                        bcr_meta[cell_id]['light_cdr3'] = row['cdr3']
            
            # Add BCR metadata to anndata object
            bcr_df = pd.DataFrame.from_dict(bcr_meta, orient='index')
            
            # Add empty rows for cells without BCR data
            for cell in adata.obs.index:
                if cell not in bcr_df.index:
                    bcr_df.loc[cell] = pd.Series({
                        'has_bcr': False,
                        'clonotype_id': None,
                        'sample': None,
                        'isotype': None,
                        'igh_v': None,
                        'igh_d': None,
                        'igh_j': None,
                        'igh_cdr3': None,
                        'light_chain': None,
                        'light_v': None,
                        'light_j': None,
                        'light_cdr3': None
                    })
            
            # Sort to match adata order
            bcr_df = bcr_df.loc[adata.obs.index]
            
            # Add to adata.obs
            for col in bcr_df.columns:
                adata.obs[col] = bcr_df[col]
            
            # Save integrated object
            adata.write(os.path.join(integration_dir, "integrated_bcr_gene_expression.h5ad"))
            
            # Create visualization plots
            self._visualize_integrated_data(adata, integration_dir)
            
            # Extract B cell cluster signatures
            self._extract_bcell_signatures(adata, integration_dir)
            
            print("Integration with gene expression completed!")
            return adata
            
        except Exception as e:
            print(f"Error integrating with gene expression: {str(e)}")
            return None
    
    def _visualize_integrated_data(self, adata, output_dir):
        """
        Create visualization plots for integrated data
        """
        # UMAP colored by BCR presence
        sc.pl.umap(adata, color=['has_bcr'], save=os.path.join(output_dir, "umap_has_bcr.pdf"))
        
        # UMAP colored by isotype
        if 'isotype' in adata.obs.columns:
            sc.pl.umap(adata, color=['isotype'], save=os.path.join(output_dir, "umap_isotype.pdf"))
        
        # UMAP colored by clonotype (for top 10 clonotypes)
        if 'clonotype_id' in adata.obs.columns:
            # Get top 10 clonotypes
            top_clonotypes = adata.obs['clonotype_id'].value_counts().head(10).index
            
            # Create a new column for top clonotypes
            adata.obs['top_clonotype'] = adata.obs['clonotype_id']
            adata.obs.loc[~adata.obs['top_clonotype'].isin(top_clonotypes), 'top_clonotype'] = 'Other'
            
            # Plot
            sc.pl.umap(adata, color=['top_clonotype'], save=os.path.join(output_dir, "umap_top_clonotypes.pdf"))
        
        # Differential gene expression between BCR+ and BCR- cells
        if 'has_bcr' in adata.obs.columns and adata.obs['has_bcr'].sum() > 0:
            sc.tl.rank_genes_groups(adata, 'has_bcr', method='wilcoxon')
            sc.pl.rank_genes_groups(adata, n_genes=20, save=os.path.join(output_dir, "diff_genes_bcr.pdf"))
            
            # Save differentially expressed genes
            diff_genes = sc.get.rank_genes_groups_df(adata, group='True')
            diff_genes.to_csv(os.path.join(output_dir, "diff_genes_bcr.csv"))
        
        # If we have isotype information, perform differential expression by isotype
        if 'isotype' in adata.obs.columns:
            # Filter to cells with isotype information
            adata_with_isotype = adata[~adata.obs['isotype'].isna()].copy()
            
            if len(adata_with_isotype) > 10:  # Only if we have enough cells
                sc.tl.rank_genes_groups(adata_with_isotype, 'isotype', method='wilcoxon')
                sc.pl.rank_genes_groups(adata_with_isotype, n_genes=20, 
                                      save=os.path.join(output_dir, "diff_genes_by_isotype.pdf"))
                
                # Save differentially expressed genes by isotype
                for isotype in adata_with_isotype.obs['isotype'].unique():
                    if not pd.isna(isotype):
                        try:
                            diff_genes = sc.get.rank_genes_groups_df(adata_with_isotype, group=isotype)
                            diff_genes.to_csv(os.path.join(output_dir, f"diff_genes_{isotype}.csv"))
                        except:
                            print(f"Could not extract differential genes for {isotype}")
        
        # If we have cluster information, check BCR distribution across clusters
        if 'leiden' in adata.obs.columns:
            bcr_cluster = pd.crosstab(adata.obs['leiden'], adata.obs['has_bcr'])
            bcr_cluster.to_csv(os.path.join(output_dir, "bcr_distribution_by_cluster.csv"))
            
            # Plot
            plt.figure(figsize=(12, 8))
            bcr_cluster_percent = bcr_cluster.div(bcr_cluster.sum(axis=1), axis=0) * 100
            bcr_cluster_percent.plot(kind='bar', stacked=True)
            plt.title("BCR Distribution by Cluster")
            plt.xlabel("Cluster")
            plt.ylabel("Percentage")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "bcr_distribution_by_cluster.pdf"))
            plt.close()
    
    def _extract_bcell_signatures(self, adata, output_dir):
        """
        Extract B cell signature genes and correlation with isotype switching
        """
        # Define B cell signature genes
        b_cell_genes = [
            'CD19', 'MS4A1', 'CD79A', 'CD79B',  # Core B cell markers
            'IGHD', 'IGHM',  # Naive/immature
            'AICDA', 'IGHG1', 'IGHG2', 'IGHG3', 'IGHG4',  # Class-switching
            'IGHA1', 'IGHA2',  # IgA
            'XBP1', 'PRDM1', 'IRF4',  # Plasma cell
            'MKI67', 'PCNA',  # Proliferation
            'BCL6', 'LMO2',  # Germinal center
            'CD27', 'CD38',  # Memory/activation
            'FCER2', 'CR2'  # Other B cell markers
        ]
        
        # Check which genes are present in the dataset
        present_genes = [gene for gene in b_cell_genes if gene in adata.var_names]
        
        if not present_genes:
            print("None of the B cell signature genes found in the dataset.")
            return
        
        print(f"Found {len(present_genes)} B cell signature genes in the dataset.")
        
        # Extract expression of these genes
        b_cell_expr = pd.DataFrame(
            adata[:, present_genes].X.toarray(), 
            index=adata.obs.index, 
            columns=present_genes
        )
        
        # Add BCR and cell cluster information
        b_cell_expr = pd.concat([b_cell_expr, adata.obs[['has_bcr', 'isotype', 'leiden']]], axis=1)
        
        # Save expression data
        b_cell_expr.to_csv(os.path.join(output_dir, "b_cell_signature_expression.csv"))
        
        # Calculate correlation between gene expression and isotype
        if 'isotype' in b_cell_expr.columns:
            # Create dummy variables for isotypes
            isotype_dummies = pd.get_dummies(b_cell_expr['isotype'])
            
            # Calculate correlation
            correlation_data = []
            
            for gene in present_genes:
                for isotype in isotype_dummies.columns:
                    if pd.isna(isotype) or isotype == 'None':
                        continue
                        
                    # Calculate correlation
                    corr = b_cell_expr[gene].corr(isotype_dummies[isotype])
                    
                    correlation_data.append({
                        'gene': gene,
                        'isotype': isotype,
                        'correlation': corr
                    })
            
            # Create correlation dataframe
            corr_df = pd.DataFrame(correlation_data)
            corr_df.to_csv(os.path.join(output_dir, "gene_isotype_correlation.csv"), index=False)
            
            # Plot heatmap
            if not corr_df.empty:
                pivot_corr = corr_df.pivot(index='gene', columns='isotype', values='correlation')
                
                plt.figure(figsize=(10, 12))
                sns.heatmap(pivot_corr, cmap='RdBu_r', center=0, annot=True, fmt='.2f')
                plt.title("Correlation between Gene Expression and Isotype")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "gene_isotype_correlation_heatmap.pdf"))
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
        
        # Isotype distribution
        if self.isotype_distribution is not None:
            # Plot overall isotype distribution
            plt.figure(figsize=(10, 6))
            sns.barplot(data=self.isotype_distribution, x='isotype', y='percentage', hue='condition')
            plt.title("Isotype Distribution by Condition")
            plt.xlabel("Isotype")
            plt.ylabel("Percentage (%)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "isotype_distribution.pdf"))
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
        
        # Somatic hypermutation analysis
        if self.shm_data is not None:
            # Plot SHM by chain and isotype
            igh_data = self.shm_data[self.shm_data['chain'] == 'IGH']
            
            if not igh_data.empty and 'isotype' in igh_data.columns:
                igh_data = igh_data[~igh_data['isotype'].isna()]
                
                if not igh_data.empty:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=igh_data, x='isotype', y='mutation_rate')
                    plt.title("Somatic Hypermutation Rate by Isotype")
                    plt.xlabel("Isotype")
                    plt.ylabel("Mutation Rate (%)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "mutation_rate_by_isotype.pdf"))
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
            <title>scBCR-seq Analysis Report</title>
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
            <h1>Single Cell BCR-seq Analysis Report</h1>
            
            <div class="section summary">
                <h2>Analysis Summary</h2>
        """
        
        # Add general info
        if self.repertoire_df is not None:
            num_samples = len(self.repertoire_df['sample'].unique())
            num_clonotypes = len(self.repertoire_df)
            
            conditions = "None"
            if 'condition' in self.repertoire_df.columns:
                conditions = ", ".join(self.repertoire_df['condition'].unique())
            
            html_content += f"""
                <p>Number of samples analyzed: {num_samples}</p>
                <p>Total number of unique clonotypes: {num_clonotypes}</p>
                <p>Conditions: {conditions}</p>
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
                <h2>Isotype Distribution</h2>
        """
        
        if self.isotype_distribution is not None:
            html_content += """
                <table>
                    <tr>
                        <th>Sample</th>
                        <th>Isotype</th>
                        <th>Percentage</th>
                    </tr>
            """
            
            for _, row in self.isotype_distribution.iterrows():
                html_content += f"""
                    <tr>
                        <td>{row['sample']}</td>
                        <td>{row['isotype']}</td>
                        <td>{row['percentage']:.2f}%</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <div class="plot">
                    <img src="isotype_distribution.pdf" alt="Isotype Distribution">
                </div>
            """
        else:
            html_content += """
                <p>No isotype distribution data available.</p>
            """
        
        html_content += """
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
        """
        
        if self.shm_data is not None:
            html_content += """
                <div class="plot">
                    <h3>Somatic Hypermutation</h3>
                    <img src="mutation_rate_by_isotype.pdf" alt="Mutation Rate by Isotype">
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Analysis Details</h2>
                <p>This report summarizes the results of a single-cell BCR repertoire analysis. The analysis included the following steps:</p>
                <ul>
                    <li>Extraction and QC of BCR sequences from 10X Genomics data</li>
                    <li>V(D)J annotation and CDR3 identification</li>
                    <li>Clonotype assignment and repertoire compilation</li>
                    <li>Analysis of V(D)J usage, CDR3 length, motifs, and clonal expansion</li>
                    <li>Isotype distribution and somatic hypermutation analysis</li>
                    <li>B-cell lineage reconstruction</li>
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
def run_scbcr_analysis(data_dir, output_dir, sample_info_file=None, gene_expression_file=None):
    """
    Run the complete scBCR analysis pipeline
    
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
    print("Starting scBCR analysis pipeline...")
    
    # Load sample information if provided
    sample_info = None
    if sample_info_file and os.path.exists(sample_info_file):
        sample_info = pd.read_csv(sample_info_file)
        print(f"Loaded sample information for {len(sample_info)} samples")
    
    # Initialize analyzer
    analyzer = scBCRAnalyzer(data_dir, output_dir, sample_info)
    
    # Run analysis steps
    analyzer.extract_and_qc_bcr()
    analyzer.compile_clonotypes()
    analyzer.analyze_vdj_usage()
    analyzer.analyze_cdr3_features()
    analyzer.analyze_isotype_distribution()
    analyzer.analyze_somatic_hypermutation()
    analyzer.reconstruct_lineages()
    analyzer.analyze_clonal_expansion()
    analyzer.calculate_diversity_metrics()
    analyzer.calculate_repertoire_overlap()
    analyzer.comparative_analysis()
    
    # Optional: Integrate with gene expression
    if gene_expression_file:
        analyzer.integrate_with_expression(gene_expression_file)
    
    # Create final visualizations
    analyzer.visualize_results()
    
    print("scBCR analysis pipeline completed!")
    return analyzer


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Single-cell BCR analysis pipeline for 10X Genomics data')
    parser.add_argument('--data_dir', required=True, help='Directory containing the raw fastq.gz files')
    parser.add_argument('--output_dir', required=True, help='Directory for saving analysis results')
    parser.add_argument('--sample_info', help='Path to sample information file (CSV with sample,condition columns)')
    parser.add_argument('--gene_expression', help='Path to gene expression file for integration')
    
    args = parser.parse_args()
    
    # Run the analysis
    run_scbcr_analysis(args.data_dir, args.output_dir, args.sample_info, args.gene_expression)
