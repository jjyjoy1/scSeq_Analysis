#!/usr/bin/env python3
# Script to generate knockout efficiency summary report
# Filename: scripts/generate_ko_report.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from snakemake.utils import report

def parse_indel_file(indel_file, sample_name):
    """Parse indel percentage from CRISPResso output file"""
    try:
        with open(indel_file, 'r') as f:
            for line in f:
                if 'Indel' in line and '%' in line:
                    fields = line.strip().split()
                    for i, field in enumerate(fields):
                        if '%' in field:
                            try:
                                indel_pct = float(field.replace('%', ''))
                                return {'Sample': sample_name, 'Indel_percentage': indel_pct}
                            except ValueError:
                                continue
    except Exception as e:
        print(f"Error parsing {indel_file}: {e}")
        
    # Return 0% if file cannot be parsed
    return {'Sample': sample_name, 'Indel_percentage': 0.0}

def main(snakemake):
    # Get the sample names from the input files
    sample_names = [os.path.basename(f).replace('.indels.txt', '') 
                   for f in snakemake.input.indels]
    
    # Parse indel percentages from all samples
    indel_data = []
    for sample, indel_file in zip(sample_names, snakemake.input.indels):
        indel_data.append(parse_indel_file(indel_file, sample))
    
    # Create a dataframe with the results
    df = pd.DataFrame(indel_data)
    
    # Load the config to get target gene information
    config = snakemake.config
    
    # Add target gene information
    df['Target_gene'] = [config['samples'].get(sample, {}).get('target_gene', 'Unknown') 
                        for sample in df['Sample']]
    
    # Generate bar plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Sample', y='Indel_percentage', data=df, 
                    palette='viridis', hue='Target_gene')
    plt.title('CRISPR Knockout Efficiency by Sample')
    plt.ylabel('Indel Percentage (%)')
    plt.xlabel('Sample')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(os.path.dirname(snakemake.output.report), 'ko_efficiency_plot.png')
    plt.savefig(plot_path, dpi=300)
    
    # Calculate efficiency statistics
    efficiency_stats = df.groupby('Target_gene')['Indel_percentage'].agg(['mean', 'std', 'min', 'max']).reset_index()
    efficiency_stats = efficiency_stats.round(2)
    
    # Generate HTML report
    report_str = f"""
    # CRISPR Knockout Efficiency Summary
    
    ## Indel Percentage by Sample
    
    ![Knockout Efficiency](ko_efficiency_plot.png)
    
    ## Summary Statistics by Target Gene
    
    {efficiency_stats.to_html(index=False)}
    
    ## Raw Data
    
    {df.to_html(index=False)}
    """
    
    with open(snakemake.output.report, 'w') as f:
        f.write(report_str)

if __name__ == "__main__":
    main(snakemake)
