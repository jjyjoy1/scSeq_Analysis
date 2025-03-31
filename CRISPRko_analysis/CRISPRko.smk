# CRISPR Knockout (CRISPRko) Snakemake Pipeline
# Filename: Snakefile

import os
from os.path import join

# Configuration
configfile: "config.yaml"

# Define sample information
SAMPLES = config["samples"]
REFERENCE_GENOME = config["reference_genome"]
GUIDE_RNA_FILE = config["guide_rna_file"]
OUTPUT_DIR = config["output_dir"]

# Create output directories
for directory in ["fastqc", "trimmed", "aligned", "counts", "indels", "reports"]:
    os.makedirs(join(OUTPUT_DIR, directory), exist_ok=True)

# Final target rule
rule all:
    input:
        # QC reports
        expand(join(OUTPUT_DIR, "fastqc", "{sample}_R{read}_fastqc.html"), sample=SAMPLES, read=[1, 2]),
        # Trimmed reads
        expand(join(OUTPUT_DIR, "trimmed", "{sample}_R{read}.trimmed.fastq.gz"), sample=SAMPLES, read=[1, 2]),
        # Aligned reads
        expand(join(OUTPUT_DIR, "aligned", "{sample}.bam"), sample=SAMPLES),
        expand(join(OUTPUT_DIR, "aligned", "{sample}.bam.bai"), sample=SAMPLES),
        # Indel analysis
        expand(join(OUTPUT_DIR, "indels", "{sample}.indels.txt"), sample=SAMPLES),
        # Summary report
        join(OUTPUT_DIR, "reports", "knockout_efficiency_summary.html")

# Quality control with FastQC
rule fastqc:
    input:
        r1 = lambda wildcards: config["samples"][wildcards.sample]["r1"],
        r2 = lambda wildcards: config["samples"][wildcards.sample]["r2"]
    output:
        html_r1 = join(OUTPUT_DIR, "fastqc", "{sample}_R1_fastqc.html"),
        html_r2 = join(OUTPUT_DIR, "fastqc", "{sample}_R2_fastqc.html"),
        zip_r1 = join(OUTPUT_DIR, "fastqc", "{sample}_R1_fastqc.zip"),
        zip_r2 = join(OUTPUT_DIR, "fastqc", "{sample}_R2_fastqc.zip")
    log:
        join(OUTPUT_DIR, "logs", "fastqc", "{sample}.log")
    threads: 2
    shell:
        """
        fastqc --threads {threads} --outdir $(dirname {output.html_r1}) {input.r1} {input.r2} &> {log}
        """

# Trim adapters and low-quality bases with Trimmomatic
rule trimmomatic:
    input:
        r1 = lambda wildcards: config["samples"][wildcards.sample]["r1"],
        r2 = lambda wildcards: config["samples"][wildcards.sample]["r2"]
    output:
        r1 = join(OUTPUT_DIR, "trimmed", "{sample}_R1.trimmed.fastq.gz"),
        r2 = join(OUTPUT_DIR, "trimmed", "{sample}_R2.trimmed.fastq.gz"),
        r1_unpaired = join(OUTPUT_DIR, "trimmed", "{sample}_R1.unpaired.fastq.gz"),
        r2_unpaired = join(OUTPUT_DIR, "trimmed", "{sample}_R2.unpaired.fastq.gz")
    log:
        join(OUTPUT_DIR, "logs", "trimmomatic", "{sample}.log")
    threads: 8
    params:
        adapters = config["trimmomatic"]["adapters"],
        trimmomatic_params = config["trimmomatic"]["params"]
    shell:
        """
        trimmomatic PE -threads {threads} \
        {input.r1} {input.r2} \
        {output.r1} {output.r1_unpaired} \
        {output.r2} {output.r2_unpaired} \
        ILLUMINACLIP:{params.adapters}:2:30:10 \
        {params.trimmomatic_params} \
        &> {log}
        """

# Align reads to reference genome with BWA-MEM
rule bwa_mem:
    input:
        r1 = join(OUTPUT_DIR, "trimmed", "{sample}_R1.trimmed.fastq.gz"),
        r2 = join(OUTPUT_DIR, "trimmed", "{sample}_R2.trimmed.fastq.gz"),
        ref = REFERENCE_GENOME
    output:
        bam = join(OUTPUT_DIR, "aligned", "{sample}.bam")
    log:
        join(OUTPUT_DIR, "logs", "bwa_mem", "{sample}.log")
    threads: 16
    params:
        rg = r"@RG\tID:{sample}\tSM:{sample}\tPL:ILLUMINA"
    shell:
        """
        bwa mem -t {threads} -R '{params.rg}' {input.ref} {input.r1} {input.r2} 2> {log} | \
        samtools sort -@ {threads} -o {output.bam} - 2>> {log}
        """

# Index BAM files
rule index_bam:
    input:
        join(OUTPUT_DIR, "aligned", "{sample}.bam")
    output:
        join(OUTPUT_DIR, "aligned", "{sample}.bam.bai")
    log:
        join(OUTPUT_DIR, "logs", "index_bam", "{sample}.log")
    shell:
        """
        samtools index {input} {output} 2> {log}
        """

# Analyze indels around gRNA target sites using CRISPResso2
rule crispresso2:
    input:
        r1 = join(OUTPUT_DIR, "trimmed", "{sample}_R1.trimmed.fastq.gz"),
        r2 = join(OUTPUT_DIR, "trimmed", "{sample}_R2.trimmed.fastq.gz"),
        guides = GUIDE_RNA_FILE
    output:
        join(OUTPUT_DIR, "indels", "{sample}.indels.txt")
    params:
        output_dir = join(OUTPUT_DIR, "indels", "{sample}"),
        amplicon_seq = lambda wildcards: config["samples"][wildcards.sample]["amplicon"],
        window = config["crispresso"]["window_size"]
    log:
        join(OUTPUT_DIR, "logs", "crispresso", "{sample}.log")
    threads: 8
    shell:
        """
        CRISPResso --fastq_r1 {input.r1} \
                  --fastq_r2 {input.r2} \
                  --amplicon_seq {params.amplicon_seq} \
                  --guide_seq $(cat {input.guides}) \
                  --quantification_window_size {params.window} \
                  --base_editor_output \
                  --output_folder {params.output_dir} \
                  --n_processes {threads} \
                  --plot_window_size 20 \
                  --exclude_bp_from_left 5 \
                  --exclude_bp_from_right 5 2> {log}
        
        # Extract indel percentages for summary
        grep "Indel" {params.output_dir}/CRISPResso_quantification_of_editing_frequency.txt > {output} 2>> {log}
        """

# Generate a summary report of knockout efficiency
rule generate_report:
    input:
        indels = expand(join(OUTPUT_DIR, "indels", "{sample}.indels.txt"), sample=SAMPLES)
    output:
        report = join(OUTPUT_DIR, "reports", "knockout_efficiency_summary.html")
    script:
        "scripts/generate_ko_report.py"

