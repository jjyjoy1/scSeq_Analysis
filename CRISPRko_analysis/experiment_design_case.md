# CRISPR Knockout (CRISPRko) Experiment Design

## 1. Experimental Overview

This document outlines the design of a CRISPR/Cas9-mediated gene knockout experiment targeting two genes of interest: BRCA1 and TP53. The experiment aims to evaluate the efficiency of CRISPR-mediated gene knockout using next-generation sequencing (NGS) to quantify indel formation at the target sites.

## 2. Target Genes and Guide RNA Design

### Target Genes:
- **BRCA1**: Breast cancer type 1 susceptibility protein
- **TP53**: Tumor protein p53

### Guide RNA Design:
- Guide RNAs were designed using the CRISPOR tool (https://crispor.tefor.net/)
- Selection criteria:
  - High on-target score (>70)
  - Low off-target effects (specificity score >80)
  - Target site positioned in an early exon to ensure functional knockout
  - Avoid sites with known SNPs
  - GC content between 40-60%

### Guide RNA Sequences:
- BRCA1-gRNA: 5'-GCTAGCTAGCTATCGCTAGCT-3' (PAM: TGG)
- TP53-gRNA: 5'-GATCGATCGATCGTACGTACG-3' (PAM: CGG)

## 3. Cell Culture and Transfection

### Cell Lines:
- MCF-7 (breast cancer cell line) for BRCA1 targeting
- HCT116 (colorectal cancer cell line) for TP53 targeting

### Transfection Groups:
1. **BRCA1-knockout**: MCF-7 cells transfected with Cas9 and BRCA1-gRNA
2. **TP53-knockout**: HCT116 cells transfected with Cas9 and TP53-gRNA
3. **Control**: Cells transfected with Cas9 and non-targeting gRNA

### Transfection Method:
- Lipofectamine 3000 for delivery of Cas9 protein and gRNA
- Transfection efficiency monitored by co-transfection with GFP reporter

### Timeline:
- Day 0: Seed cells (2 × 10^5 cells per well in 6-well plates)
- Day 1: Transfect cells with Cas9 and gRNA
- Day 2: Check transfection efficiency using fluorescence microscopy
- Day 3: Begin selection with puromycin (if using a Cas9-puromycin vector)
- Day 7: Harvest cells for genomic DNA extraction and analysis

## 4. Genomic DNA Extraction and PCR Amplification

### Genomic DNA Extraction:
- Extract genomic DNA from harvested cells using DNeasy Blood & Tissue Kit (Qiagen)
- Quantify DNA concentration using Qubit fluorometric quantification
- Ensure high DNA quality (A260/280 ratio ~1.8-2.0)

### PCR Amplification:
- Design primers to amplify ~200-300 bp regions surrounding the gRNA target sites
- Include unique sample indices for multiplexed sequencing
- PCR conditions:
  - Initial denaturation: 95°C for 3 minutes
  - 30 cycles of: 95°C for 30s, 60°C for 30s, 72°C for 30s
  - Final extension: 72°C for 5 minutes
- Verify PCR products by agarose gel electrophoresis

### Primers:
- BRCA1-Forward: 5'-ACGTACGTACGTACGTACGT-3'
- BRCA1-Reverse: 5'-TGCATGCATGCATGCATGCA-3'
- TP53-Forward: 5'-ACGTACGTACGTACGTACGT-3'
- TP53-Reverse: 5'-TGCATGCATGCATGCATGCA-3'

## 5. Next-Generation Sequencing

### Library Preparation:
- Purify PCR products using AMPure XP beads
- Quantify libraries using Qubit and Bioanalyzer
- Pool libraries with unique indices in equimolar ratios

### Sequencing:
- Platform: Illumina MiSeq
- Read type: Paired-end 2 × 150 bp
- Coverage: Aim for >10,000× coverage per sample
- Include 10-15% PhiX control

## 6. Data Analysis

### Primary Analysis:
- Demultiplex samples based on indices
- Perform quality control using FastQC
- Trim adapters and low-quality bases using Trimmomatic

### Indel Analysis:
- Align reads to reference genome using BWA-MEM
- Analyze indels around the cut site using CRISPResso2
- Calculate knockout efficiency based on indel frequency

### Expected Outcomes:
- Control samples: <1% background indel rate
- Target samples: >70% indel formation for efficient knockout
- Common indels: 1-10 bp deletions near the cut site (3 bp upstream of the PAM sequence)

## 7. Validation of Knockout

### Protein-level Validation:
- Western blot analysis to confirm reduction in target protein levels
- Antibodies:
  - Anti-BRCA1 (Cell Signaling Technology, #9010)
  - Anti-p53 (Cell Signaling Technology, #2527)
  - Anti-GAPDH (loading control)

### Functional Validation:
- BRCA1 knockout: Assess sensitivity to PARP inhibitors (e.g., olaparib)
- TP53 knockout: Evaluate response to DNA-damaging agents (e.g., doxorubicin)

## 8. Timeline

| Week | Activities |
|------|------------|
| Week 1 | Guide RNA design and cloning |
| Week 2 | Cell culture and transfection |
| Week 3 | Selection and expansion of transfected cells |
| Week 4 | Genomic DNA extraction and PCR amplification |
| Week 5 | NGS library preparation and sequencing |
| Week 6 | Data analysis and validation |

## 9. Materials and Reagents

- **Cell Culture**:
  - DMEM (Gibco, #11965092)
  - FBS (Gibco, #16140071)
  - Penicillin-Streptomycin (Gibco, #15140122)
  - Lipofectamine 3000 (Invitrogen, #L3000015)

- **CRISPR Components**:
  - Cas9 protein (NEB, #M0386S)
  - tracrRNA (IDT, #1072532)
  - crRNA (custom synthesis by IDT)

- **Molecular Biology**:
  - DNeasy Blood & Tissue Kit (Qiagen, #69504)
  - Q5 High-Fidelity DNA Polymerase (NEB, #M0491S)
  - AMPure XP beads (Beckman Coulter, #A63880)
  - Qubit dsDNA HS Assay Kit (Invitrogen, #Q32851)

- **Sequencing**:
  - MiSeq Reagent Kit v3 (Illumina, #MS-102-3003)

## 10. Expected Results and Troubleshooting

### Expected Results:
- CRISPR efficiency: 70-90% indel formation at target sites
- Protein reduction: >80% reduction in target protein levels
- Functional changes: Increased sensitivity to specific drugs

### Potential Issues and Troubleshooting:
- **Low transfection efficiency**: Optimize cell density, DNA:lipid ratio, or try different transfection methods
- **Low editing efficiency**: Redesign gRNAs, check for SNPs at target site, or optimize Cas9:gRNA ratio
- **Off-target effects**: Validate using whole-genome sequencing or GUIDE-seq
- **No phenotypic effect**: Confirm knockout at protein level, consider genetic compensation or redundancy

