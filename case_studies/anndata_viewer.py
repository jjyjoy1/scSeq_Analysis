"""
Streamlit app for interactive exploration of AnnData objects
"""

import streamlit as st
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import base64
import io

# Set page configuration
st.set_page_config(page_title="AnnData Explorer", page_icon="🧬", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem;
    }
    .stPlotly, .stSelectbox {
        margin-bottom: 1rem;
    }
    .reportview-container .main .block-container {
        padding-top: 1rem;
    }
    h1, h2, h3 {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .css-1d391kg {
        padding-top: 0;
    }
    .stSidebar {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Function to download plot as PNG
def get_image_download_link(fig, filename, text):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Function to generate a unique color map from a categorical column
def generate_color_map(categories, cmap_name='tab20'):
    unique_categories = list(sorted(set(categories)))
    n_categories = len(unique_categories)
    
    if n_categories <= 20:
        # Use predefined colormap for small number of categories
        cmap = plt.cm.get_cmap(cmap_name, n_categories)
        colors = [cmap(i) for i in range(n_categories)]
    else:
        # Generate colors for large number of categories
        cmap = plt.cm.get_cmap('hsv', n_categories)
        colors = [cmap(i) for i in range(n_categories)]
    
    # Create a mapping from category to color
    color_map = {cat: colors[i] for i, cat in enumerate(unique_categories)}
    return color_map

# Function to customize scanpy plotting settings
def set_plotting_defaults():
    sc.settings.verbosity = 3
    sc.settings.set_figure_params(dpi=80, frameon=False)
    sc.settings.figdir = './figures/'

# Function to load AnnData
@st.cache_data
def load_anndata(file_path):
    try:
        adata = sc.read(file_path)
        return adata
    except Exception as e:
        st.error(f"Error loading AnnData file: {e}")
        return None

# Function to analyze AnnData object and return basic stats
def analyze_anndata(adata):
    stats = {
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "obs_keys": list(adata.obs.columns),
        "var_keys": list(adata.var.columns),
        "layers": list(adata.layers.keys()) if hasattr(adata, 'layers') else [],
        "obsm_keys": list(adata.obsm.keys()) if hasattr(adata, 'obsm') else [],
        "has_raw": hasattr(adata, 'raw') and adata.raw is not None,
        "has_umap": 'X_umap' in adata.obsm.keys() if hasattr(adata, 'obsm') else False,
        "has_tsne": 'X_tsne' in adata.obsm.keys() if hasattr(adata, 'obsm') else False,
        "has_pca": 'X_pca' in adata.obsm.keys() if hasattr(adata, 'obsm') else False,
        "available_colors": [col for col in adata.obs.columns if len(adata.obs[col].unique()) <= 100]
    }
    return stats

# Function to filter AnnData object
def filter_anndata(adata, filters):
    if not filters:
        return adata
    
    # Make a copy to avoid modifying original
    filtered_adata = adata.copy()
    
    # Apply all filters
    for column, value in filters.items():
        if value and column in filtered_adata.obs.columns:
            mask = filtered_adata.obs[column] == value
            if mask.sum() > 0:
                filtered_adata = filtered_adata[mask].copy()
            else:
                st.warning(f"No cells match filter {column}={value}")
    
    return filtered_adata

# Function to create UMAP visualization
def plot_umap(adata, color, title=None, use_raw=False):
    if 'X_umap' not in adata.obsm:
        st.warning("UMAP coordinates not found. Computing UMAP...")
        try:
            if 'X_pca' not in adata.obsm:
                sc.pp.pca(adata)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        except Exception as e:
            st.error(f"Error computing UMAP: {e}")
            return None
    
    fig = plt.figure(figsize=(10, 8))
    if use_raw and hasattr(adata, 'raw') and adata.raw is not None:
        sc.pl.umap(adata, color=color, use_raw=True, show=False, title=title)
    else:
        sc.pl.umap(adata, color=color, show=False, title=title)
    return fig

# Function to create t-SNE visualization
def plot_tsne(adata, color, title=None):
    if 'X_tsne' not in adata.obsm:
        st.warning("t-SNE coordinates not found. Computing t-SNE...")
        try:
            if 'X_pca' not in adata.obsm:
                sc.pp.pca(adata)
            sc.tl.tsne(adata)
        except Exception as e:
            st.error(f"Error computing t-SNE: {e}")
            return None
    
    fig = plt.figure(figsize=(10, 8))
    sc.pl.tsne(adata, color=color, show=False, title=title)
    return fig

# Function to create PCA visualization
def plot_pca(adata, color, title=None):
    if 'X_pca' not in adata.obsm:
        st.warning("PCA coordinates not found. Computing PCA...")
        try:
            sc.pp.pca(adata)
        except Exception as e:
            st.error(f"Error computing PCA: {e}")
            return None
    
    fig = plt.figure(figsize=(10, 8))
    sc.pl.pca(adata, color=color, show=False, title=title)
    return fig

# Function to display gene expression in different embeddings
def plot_gene_expression(adata, gene, embedding='umap'):
    if gene not in adata.var_names:
        st.error(f"Gene {gene} not found in the dataset.")
        return None
    
    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    
    if embedding == 'umap':
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        sc.pl.umap(adata, color=gene, show=False, title=f"{gene} expression (UMAP)")
    elif embedding == 'tsne':
        if 'X_tsne' not in adata.obsm:
            sc.tl.tsne(adata)
        sc.pl.tsne(adata, color=gene, show=False, title=f"{gene} expression (t-SNE)")
    elif embedding == 'pca':
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata)
        sc.pl.pca(adata, color=gene, show=False, title=f"{gene} expression (PCA)")
    
    return fig

# Function to create violin plot for gene expression
def plot_violin(adata, gene, groupby):
    if gene not in adata.var_names:
        st.error(f"Gene {gene} not found in the dataset.")
        return None
    
    if groupby not in adata.obs.columns:
        st.error(f"Column {groupby} not found in observations.")
        return None
    
    fig = plt.figure(figsize=(12, 6))
    sc.pl.violin(adata, gene, groupby=groupby, show=False)
    plt.tight_layout()
    return fig

# Function to create heatmap of top genes per group
def plot_heatmap(adata, groupby, n_genes=10):
    if groupby not in adata.obs.columns:
        st.error(f"Column {groupby} not found in observations.")
        return None
    
    # Rank genes for groups
    sc.tl.rank_genes_groups(adata, groupby=groupby, method='wilcoxon')
    
    fig = plt.figure(figsize=(12, 10))
    sc.pl.rank_genes_groups_heatmap(adata, n_genes=n_genes, show=False)
    plt.tight_layout()
    return fig

# Function to create dotplot
def plot_dotplot(adata, genes, groupby):
    if not all(gene in adata.var_names for gene in genes):
        missing_genes = [gene for gene in genes if gene not in adata.var_names]
        st.error(f"Some genes not found in the dataset: {missing_genes}")
        return None
    
    if groupby not in adata.obs.columns:
        st.error(f"Column {groupby} not found in observations.")
        return None
    
    fig = plt.figure(figsize=(12, 8))
    sc.pl.dotplot(adata, genes, groupby=groupby, show=False)
    plt.tight_layout()
    return fig

# Function to create scatter plot
def plot_scatter(adata, x, y, color=None):
    if x not in adata.var_names:
        st.error(f"Gene {x} not found in the dataset.")
        return None
    
    if y not in adata.var_names:
        st.error(f"Gene {y} not found in the dataset.")
        return None
    
    fig = plt.figure(figsize=(10, 8))
    if color and color in adata.obs.columns:
        sc.pl.scatter(adata, x=x, y=y, color=color, show=False)
    else:
        sc.pl.scatter(adata, x=x, y=y, show=False)
    
    return fig

# Function to display batch correction evaluation
def plot_batch_correction(adata, batch_key, cell_type_key=None):
    if batch_key not in adata.obs.columns:
        st.error(f"Batch column {batch_key} not found in observations.")
        return None
    
    figs = []
    
    # 1. Create UMAP colored by batch
    if 'X_umap' not in adata.obsm:
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    
    fig1 = plt.figure(figsize=(10, 8))
    sc.pl.umap(adata, color=batch_key, show=False, title=f"Batch Distribution ({batch_key})")
    figs.append(("Batch Distribution", fig1))
    
    # 2. Create UMAP colored by cell type if available
    if cell_type_key and cell_type_key in adata.obs.columns:
        fig2 = plt.figure(figsize=(10, 8))
        sc.pl.umap(adata, color=cell_type_key, show=False, title=f"Cell Types ({cell_type_key})")
        figs.append(("Cell Type Distribution", fig2))
    
    # 3. Calculate batch mixing metrics if scVI latent representation is available
    if 'X_scVI' in adata.obsm:
        from sklearn.metrics import silhouette_score
        
        # Calculate silhouette scores on PCA embedding
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata)
        
        batch_sil_pca = silhouette_score(
            adata.obsm['X_pca'], 
            adata.obs[batch_key].cat.codes, 
            metric='euclidean'
        )
        
        batch_sil_scvi = silhouette_score(
            adata.obsm['X_scVI'], 
            adata.obs[batch_key].cat.codes, 
            metric='euclidean'
        )
        
        # Create a bar plot comparing silhouette scores
        fig3 = plt.figure(figsize=(8, 6))
        embeddings = ['PCA', 'scVI']
        batch_sil_scores = [batch_sil_pca, batch_sil_scvi]
        
        plt.bar(embeddings, batch_sil_scores, color=['blue', 'orange'])
        plt.ylabel('Batch Silhouette Score')
        plt.title('Batch Mixing Evaluation (lower is better)')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add values on top of bars
        for i, v in enumerate(batch_sil_scores):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        plt.tight_layout()
        figs.append(("Batch Mixing Metrics", fig3))
        
        # If cell type key is available, also calculate cell type silhouette scores
        if cell_type_key and cell_type_key in adata.obs.columns:
            cell_sil_pca = silhouette_score(
                adata.obsm['X_pca'], 
                adata.obs[cell_type_key].cat.codes, 
                metric='euclidean'
            )
            
            cell_sil_scvi = silhouette_score(
                adata.obsm['X_scVI'], 
                adata.obs[cell_type_key].cat.codes, 
                metric='euclidean'
            )
            
            # Create a bar plot comparing cell type silhouette scores
            fig4 = plt.figure(figsize=(8, 6))
            cell_sil_scores = [cell_sil_pca, cell_sil_scvi]
            
            plt.bar(embeddings, cell_sil_scores, color=['blue', 'orange'])
            plt.ylabel('Cell Type Silhouette Score')
            plt.title('Cell Type Separation (higher is better)')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Add values on top of bars
            for i, v in enumerate(cell_sil_scores):
                plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
            
            plt.tight_layout()
            figs.append(("Cell Type Preservation Metrics", fig4))
    
    return figs

# Main Streamlit app
def main():
    st.title("🧬 AnnData Explorer")
    st.write("Interactive application for exploring single-cell RNA-seq data in AnnData format")
    
    # Sidebar for file selection
    st.sidebar.header("Data Loading")
    
    # Option to use default file path or select a file
    file_option = st.sidebar.radio("Select AnnData file:", ["Choose from disk", "Enter file path"])
    
    if file_option == "Choose from disk":
        uploaded_file = st.sidebar.file_uploader("Upload AnnData file (.h5ad)", type=["h5ad"])
        if uploaded_file:
            with open("temp_file.h5ad", "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_path = "temp_file.h5ad"
        else:
            st.info("Please upload an AnnData file (.h5ad)")
            return
    else:
        file_path = st.sidebar.text_input("Enter path to AnnData file (.h5ad)")
        if not file_path:
            st.info("Please enter a valid file path")
            return
    
    # Load the AnnData object
    try:
        adata = load_anndata(file_path)
        if adata is None:
            return
    except Exception as e:
        st.error(f"Error loading AnnData file: {e}")
        return
    
    # Analyze the AnnData object
    stats = analyze_anndata(adata)
    
    # Sidebar for filtering options
    st.sidebar.header("Filtering Options")
    
    # Create filters based on categorical columns in obs
    filters = {}
    categorical_cols = [col for col in adata.obs.columns 
                        if adata.obs[col].dtype.name == 'category' 
                        or len(adata.obs[col].unique()) < 100]
    
    if categorical_cols:
        st.sidebar.subheader("Filter by Categories")
        for col in categorical_cols:
            unique_values = sorted(adata.obs[col].unique())
            if len(unique_values) < 50:  # Only show if manageable number of options
                selected_value = st.sidebar.selectbox(
                    f"{col}:",
                    ["All"] + list(unique_values),
                    key=f"filter_{col}"
                )
                if selected_value != "All":
                    filters[col] = selected_value
    
    # Filter the data based on selections
    if filters:
        filtered_adata = filter_anndata(adata, filters)
        st.success(f"Filtered data: {filtered_adata.n_obs} cells, {filtered_adata.n_vars} genes")
    else:
        filtered_adata = adata
        st.success(f"Loaded data: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔍 Cell Embeddings", 
        "🧪 Gene Expression", 
        "📈 Differential Expression",
        "🧩 Batch Correction"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            st.write(f"Number of cells: {filtered_adata.n_obs}")
            st.write(f"Number of genes: {filtered_adata.n_vars}")
            
            if stats["has_raw"]:
                st.write("Raw data is available")
            
            if stats["layers"]:
                st.write(f"Available layers: {', '.join(stats['layers'])}")
            
            available_dims = []
            if stats["has_pca"]: available_dims.append("PCA")
            if stats["has_tsne"]: available_dims.append("t-SNE")
            if stats["has_umap"]: available_dims.append("UMAP")
            if "X_scVI" in filtered_adata.obsm: available_dims.append("scVI")
            
            if available_dims:
                st.write(f"Available embeddings: {', '.join(available_dims)}")
        
        with col2:
            st.subheader("Metadata Overview")
            if categorical_cols:
                selected_meta = st.selectbox(
                    "Select metadata to visualize:",
                    categorical_cols
                )
                
                # Calculate value counts
                value_counts = filtered_adata.obs[selected_meta].value_counts()
                
                # Create a bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                value_counts.plot(kind='bar', ax=ax)
                plt.title(f'Distribution of {selected_meta}')
                plt.ylabel('Count')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display download link for the figure
                st.markdown(
                    get_image_download_link(fig, f"{selected_meta}_distribution.png", "Download Plot"),
                    unsafe_allow_html=True
                )
        
        # Show data tables
        st.subheader("Data Preview")
        show_data_option = st.radio("Select data to preview:", ["Observations (obs)", "Variables (var)"])
        
        if show_data_option == "Observations (obs)":
            st.dataframe(filtered_adata.obs.head(100))
        else:
            st.dataframe(filtered_adata.var.head(100))
    
    # Tab 2: Cell Embeddings Visualization
    with tab2:
        st.header("Cell Embeddings Visualization")
        
        embedding_type = st.radio(
            "Select embedding type:",
            ["UMAP", "t-SNE", "PCA"],
            horizontal=True
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Select color by
            color_by_options = ["None"] + stats["available_colors"]
            color_by = st.selectbox("Color by:", color_by_options)
            
            if color_by != "None":
                if filtered_adata.obs[color_by].dtype.name == 'category':
                    # Show the color legend
                    categories = filtered_adata.obs[color_by].cat.categories
                    st.write(f"Categories in {color_by}:")
                    for cat in categories:
                        st.write(f"- {cat}")
                else:
                    st.write(f"Unique values: {len(filtered_adata.obs[color_by].unique())}")
            
            # Additional options
            use_raw = st.checkbox("Use raw data", value=False, disabled=not stats["has_raw"])
            
            # Plot title
            custom_title = st.text_input("Custom plot title:", f"{embedding_type} of filtered cells")
        
        with col1:
            # Create the embedding plot
            if embedding_type == "UMAP":
                fig = plot_umap(filtered_adata, color=color_by if color_by != "None" else None, 
                               title=custom_title, use_raw=use_raw and stats["has_raw"])
            elif embedding_type == "t-SNE":
                fig = plot_tsne(filtered_adata, color=color_by if color_by != "None" else None,
                               title=custom_title)
            else:  # PCA
                fig = plot_pca(filtered_adata, color=color_by if color_by != "None" else None,
                              title=custom_title)
            
            if fig:
                st.pyplot(fig)
                
                # Download options
                st.markdown(
                    get_image_download_link(
                        fig, 
                        f"{embedding_type}_{color_by if color_by != 'None' else 'no_color'}.png", 
                        "Download Plot"
                    ),
                    unsafe_allow_html=True
                )
    
    # Tab 3: Gene Expression
    with tab3:
        st.header("Gene Expression Analysis")
        
        tab3_col1, tab3_col2 = st.columns([1, 1])
        
        with tab3_col1:
            st.subheader("Gene Expression Visualization")
            
            # Gene input
            gene_input = st.text_input("Enter gene name:", "")
            
            if gene_input:
                if gene_input in filtered_adata.var_names:
                    # Select visualization type
                    viz_type = st.radio(
                        "Visualization type:",
                        ["UMAP", "t-SNE", "PCA", "Violin Plot"],
                        horizontal=True
                    )
                    
                    if viz_type in ["UMAP", "t-SNE", "PCA"]:
                        gene_fig = plot_gene_expression(filtered_adata, gene_input, embedding=viz_type.lower())
                        if gene_fig:
                            st.pyplot(gene_fig)
                            st.markdown(
                                get_image_download_link(
                                    gene_fig, 
                                    f"{gene_input}_{viz_type.lower()}.png", 
                                    "Download Plot"
                                ),
                                unsafe_allow_html=True
                            )
                    else:  # Violin Plot
                        groupby_col = st.selectbox(
                            "Group by:",
                            [col for col in filtered_adata.obs.columns if filtered_adata.obs[col].dtype.name == 'category']
                        )
                        violin_fig = plot_violin(filtered_adata, gene_input, groupby=groupby_col)
                        if violin_fig:
                            st.pyplot(violin_fig)
                            st.markdown(
                                get_image_download_link(
                                    violin_fig, 
                                    f"{gene_input}_violin_{groupby_col}.png", 
                                    "Download Plot"
                                ),
                                unsafe_allow_html=True
                            )
                else:
                    st.error(f"Gene {gene_input} not found in dataset.")
                    similar_genes = [gene for gene in filtered_adata.var_names if gene_input.lower() in gene.lower()][:5]
                    if similar_genes:
                        st.write("Did you mean one of these genes?")
                        for gene in similar_genes:
                            st.write(f"- {gene}")
        
        with tab3_col2:
            st.subheader("Gene-Gene Correlation")
            
            gene_x = st.text_input("Gene X:", "")
            gene_y = st.text_input("Gene Y:", "")
            
            color_scatter = st.selectbox(
                "Color by (optional):",
                ["None"] + [col for col in filtered_adata.obs.columns 
                           if filtered_adata.obs[col].dtype.name == 'category']
            )
            
            if gene_x and gene_y:
                if gene_x in filtered_adata.var_names and gene_y in filtered_adata.var_names:
                    scatter_fig = plot_scatter(
                        filtered_adata, 
                        x=gene_x, 
                        y=gene_y, 
                        color=color_scatter if color_scatter != "None" else None
                    )
                    
                    if scatter_fig:
                        st.pyplot(scatter_fig)
                        st.markdown(
                            get_image_download_link(
                                scatter_fig, 
                                f"{gene_x}_vs_{gene_y}.png", 
                                "Download Plot"
                            ),
                            unsafe_allow_html=True
                        )
                else:
                    if gene_x not in filtered_adata.var_names:
                        st.error(f"Gene {gene_x} not found in dataset.")
                    if gene_y not in filtered_adata.var_names:
                        st.error(f"Gene {gene_y} not found in dataset.")
    
    # Tab 4: Differential Expression
    with tab4:
        st.header("Differential Expression Analysis")
        
        groupby_options = [col for col in filtered_adata.obs.columns 
                          if filtered_adata.obs[col].dtype.name == 'category']
        
        if groupby_options:
            groupby_de = st.selectbox(
                "Group cells by:",
                groupby_options
            )
            
            # Check if rank_genes_groups has already been run
            if 'rank_genes_groups' not in filtered_adata.uns or \
               filtered_adata.uns['rank_genes_groups']['params']['groupby'] != groupby_de:
                if st.button("Run Differential Expression Analysis"):
                    with st.spinner("Running differential expression analysis..."):
                        try:
                            sc.tl.rank_genes_groups(filtered_adata, groupby=groupby_de, method='wilcoxon')
                            st.success("Differential expression analysis completed!")
                        except Exception as e:
                            st.error(f"Error in differential expression analysis: {e}")
            else:
                st.success(f"Differential expression results for {groupby_de} are available.")
            
            # If rank_genes_groups results are available
            if 'rank_genes_groups' in filtered_adata.uns:
                viz_type_de = st.radio(
                    "Visualization type:",
                    ["Heatmap", "Dotplot", "Rank Plot", "Volcano Plot"],
                    horizontal=True
                )
                
                if viz_type_de == "Heatmap":
                    n_genes = st.slider("Number of top genes per group:", 5, 50, 10)
                    with st.spinner("Generating heatmap..."):
                        try:
                            fig = plt.figure(figsize=(12, 10))
                            sc.pl.rank_genes_groups_heatmap(
                                filtered_adata, 
                                n_genes=n_genes, 
                                show=False
                            )
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.markdown(
                                get_image_download_link(
                                    fig, 
                                    f"DE_heatmap_{groupby_de}.png", 
                                    "Download Heatmap"
                                ),
                                unsafe_allow_html=True
                            )
                        except Exception as e:
                            st.error(f"Error generating heatmap: {e}")
                
                elif viz_type_de == "Dotplot":
                    n_genes_dot = st.slider("Number of top genes per group:", 3, 20, 5)
                    with st.spinner("Generating dotplot..."):
                        try:
                            fig = plt.figure(figsize=(12, 8))
                            sc.pl.rank_genes_groups_dotplot(
                                filtered_adata, 
                                n_genes=n_genes_dot, 
                                show=False
                            )
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.markdown(
                                get_image_download_link(
                                    fig, 
                                    f"DE_dotplot_{groupby_de}.png", 
                                    "Download Dotplot"
                                ),
                                unsafe_allow_html=True
                            )
                        except Exception as e:
                            st.error(f"Error generating dotplot: {e}")
                
                elif viz_type_de == "Rank Plot":
                    group_to_show = st.selectbox(
                        "Select group to show:", 
                        filtered_adata.obs[groupby_de].cat.categories
                    )
                    
                    with st.spinner("Generating rank plot..."):
                        try:
                            fig = plt.figure(figsize=(10, 8))
                            sc.pl.rank_genes_groups(
                                filtered_adata,
                                groups=[group_to_show],
                                show=False
                            )
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.markdown(
                                get_image_download_link(
                                    fig, 
                                    f"DE_rankplot_{groupby_de}_{group_to_show}.png", 
                                    "Download Rank Plot"
                                ),
                                unsafe_allow_html=True
                            )
                        except Exception as e:
                            st.error(f"Error generating rank plot: {e}")
                
                elif viz_type_de == "Volcano Plot":
                    group_to_show_volcano = st.selectbox(
                        "Select group:", 
                        filtered_adata.obs[groupby_de].cat.categories
                    )
                    
                    with st.spinner("Generating volcano plot..."):
                        try:
                            # Get results from .uns dictionary
                            try:
                                pvals = filtered_adata.uns['rank_genes_groups']['pvals']
                                logfoldchanges = filtered_adata.uns['rank_genes_groups']['logfoldchanges']
                                names = filtered_adata.uns['rank_genes_groups']['names']
                                
                                # Extract data for the selected group
                                group_idx = np.where(filtered_adata.uns['rank_genes_groups']['names'].dtype.names == group_to_show_volcano)[0][0]
                                
                                # Create a dataframe for plotting
                                plot_data = pd.DataFrame({
                                    'gene': names[group_idx],
                                    'log2_fc': logfoldchanges[group_idx],
                                    '-log10_pval': -np.log10(pvals[group_idx])
                                })
                                
                                # Create volcano plot
                                fig, ax = plt.subplots(figsize=(10, 8))
                                
                                # Add significance thresholds
                                ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
                                ax.axvline(-0.5, color='gray', linestyle='--', alpha=0.5)
                                ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
                                
                                # Plot points
                                scatter = ax.scatter(
                                    plot_data['log2_fc'], 
                                    plot_data['-log10_pval'],
                                    alpha=0.7,
                                    s=20,
                                    c=np.where(
                                        (plot_data['-log10_pval'] > -np.log10(0.05)) & 
                                        (abs(plot_data['log2_fc']) > 0.5), 
                                        'red', 'gray'
                                    )
                                )
                                
                                # Add labels for top genes
                                top_genes = plot_data.sort_values('-log10_pval', ascending=False).head(15)
                                for idx, row in top_genes.iterrows():
                                    ax.text(
                                        row['log2_fc'], 
                                        row['-log10_pval'], 
                                        row['gene'], 
                                        fontsize=9
                                    )
                                
                                plt.xlabel('Log2 Fold Change')
                                plt.ylabel('-Log10 P-value')
                                plt.title(f'Volcano Plot for {group_to_show_volcano} vs Rest')
                                plt.tight_layout()
                                
                                st.pyplot(fig)
                                st.markdown(
                                    get_image_download_link(
                                        fig, 
                                        f"volcano_plot_{groupby_de}_{group_to_show_volcano}.png", 
                                        "Download Volcano Plot"
                                    ),
                                    unsafe_allow_html=True
                                )
                            except Exception as e:
                                st.error(f"Error extracting differential expression data: {e}")
                        except Exception as e:
                            st.error(f"Error generating volcano plot: {e}")
                
                # Show DE results table
                st.subheader("Top Differentially Expressed Genes")
                
                group_for_table = st.selectbox(
                    "Select group for table:", 
                    filtered_adata.obs[groupby_de].cat.categories,
                    key="de_table_group"
                )
                
                n_genes_table = st.slider("Number of genes to show:", 5, 100, 20)
                
                try:
                    result = sc.get.rank_genes_groups_df(filtered_adata, group=group_for_table)
                    result = result.head(n_genes_table)
                    st.dataframe(result)
                    
                    # Add download button for table
                    csv = result.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="DE_genes_{groupby_de}_{group_for_table}.csv">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error retrieving differential expression table: {e}")
        else:
            st.warning("No categorical columns available for grouping cells. Add cell annotations first.")
    
    # Tab 5: Batch Correction Analysis
    with tab5:
        st.header("Batch Correction Analysis")
        
        batch_columns = [col for col in filtered_adata.obs.columns 
                        if filtered_adata.obs[col].dtype.name == 'category' and 
                        filtered_adata.obs[col].nunique() > 1 and 
                        filtered_adata.obs[col].nunique() < 100]
        
        if batch_columns:
            batch_key = st.selectbox("Select batch column:", batch_columns)
            
            cell_type_columns = [col for col in filtered_adata.obs.columns 
                                if col != batch_key and 
                                filtered_adata.obs[col].dtype.name == 'category' and 
                                filtered_adata.obs[col].nunique() > 1]
            
            cell_type_key = None
            if cell_type_columns:
                cell_type_key = st.selectbox(
                    "Select cell type column (optional):", 
                    ["None"] + cell_type_columns
                )
                if cell_type_key == "None":
                    cell_type_key = None
            
            if st.button("Analyze Batch Effects"):
                with st.spinner("Analyzing batch effects..."):
                    try:
                        batch_figs = plot_batch_correction(
                            filtered_adata,
                            batch_key=batch_key,
                            cell_type_key=cell_type_key
                        )
                        
                        if batch_figs:
                            for title, fig in batch_figs:
                                st.subheader(title)
                                st.pyplot(fig)
                                st.markdown(
                                    get_image_download_link(
                                        fig, 
                                        f"{title.lower().replace(' ', '_')}.png", 
                                        "Download Plot"
                                    ),
                                    unsafe_allow_html=True
                                )
                        
                    except Exception as e:
                        st.error(f"Error analyzing batch effects: {e}")
            
            # Additional batch visualization options
            st.subheader("Batch Distribution in Reduced Dimensions")
            
            if 'X_umap' in filtered_adata.obsm:
                try:
                    # Extract UMAP coordinates
                    umap1 = filtered_adata.obsm['X_umap'][:, 0]
                    umap2 = filtered_adata.obsm['X_umap'][:, 1]
                    
                    # Create scatter plot with colored batches
                    batch_values = filtered_adata.obs[batch_key].astype('category').cat.codes
                    batch_names = filtered_adata.obs[batch_key].astype('category').cat.categories
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(umap1, umap2, c=batch_values, cmap='tab10', alpha=0.7, s=10)
                    
                    # Add legend
                    legend1 = ax.legend(
                        handles=scatter.legend_elements()[0], 
                        labels=batch_names,
                        title=batch_key,
                        loc="upper right"
                    )
                    ax.add_artist(legend1)
                    
                    plt.xlabel('UMAP 1')
                    plt.ylabel('UMAP 2')
                    plt.title(f'UMAP colored by {batch_key}')
                    
                    st.pyplot(fig)
                    
                    # Option to split by batch
                    st.subheader("Split View by Batch")
                    if st.checkbox("Show Split View"):
                        # Create a facet plot - one subplot per batch
                        n_batches = len(batch_names)
                        n_cols = min(3, n_batches)
                        n_rows = (n_batches + n_cols - 1) // n_cols
                        
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
                        axes = axes.flatten() if n_batches > 1 else [axes]
                        
                        for i, batch_name in enumerate(batch_names):
                            batch_mask = filtered_adata.obs[batch_key] == batch_name
                            axes[i].scatter(
                                umap1[batch_mask], 
                                umap2[batch_mask], 
                                color=plt.cm.tab10(i % 10), 
                                alpha=0.7,
                                s=10
                            )
                            axes[i].set_title(f'{batch_name} (n={sum(batch_mask)})')
                            axes[i].set_xlabel('UMAP 1')
                            axes[i].set_ylabel('UMAP 2')
                        
                        # Hide empty subplots
                        for j in range(i + 1, len(axes)):
                            axes[j].axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Error creating batch visualization: {e}")
            
            # If scVI latent space is available
            if 'X_scVI' in filtered_adata.obsm:
                st.subheader("scVI Latent Space Analysis")
                
                try:
                    # Create UMAP of scVI latent space if not already present
                    if 'X_umap_scVI' not in filtered_adata.obsm:
                        # Use a temporary anndata object for this to avoid modifying the original
                        temp_adata = filtered_adata.copy()
                        
                        # Compute neighbors and UMAP on scVI latent space
                        sc.pp.neighbors(temp_adata, use_rep='X_scVI')
                        sc.tl.umap(temp_adata)
                        
                        # Copy the UMAP coordinates to the original object
                        filtered_adata.obsm['X_umap_scVI'] = temp_adata.obsm['X_umap']
                    
                    # Plot UMAP of scVI latent space
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Extract UMAP coordinates
                    umap1 = filtered_adata.obsm['X_umap_scVI'][:, 0]
                    umap2 = filtered_adata.obsm['X_umap_scVI'][:, 1]
                    
                    # Create scatter plot with colored batches
                    batch_values = filtered_adata.obs[batch_key].astype('category').cat.codes
                    batch_names = filtered_adata.obs[batch_key].astype('category').cat.categories
                    
                    scatter = ax.scatter(umap1, umap2, c=batch_values, cmap='tab10', alpha=0.7, s=10)
                    
                    # Add legend
                    legend1 = ax.legend(
                        handles=scatter.legend_elements()[0], 
                        labels=batch_names,
                        title=batch_key,
                        loc="upper right"
                    )
                    ax.add_artist(legend1)
                    
                    plt.xlabel('UMAP 1 (scVI)')
                    plt.ylabel('UMAP 2 (scVI)')
                    plt.title(f'UMAP of scVI latent space colored by {batch_key}')
                    
                    st.pyplot(fig)
                    
                    # If cell type is selected, also show that
                    if cell_type_key:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # Create scatter plot with colored cell types
                        cell_type_values = filtered_adata.obs[cell_type_key].astype('category').cat.codes
                        cell_type_names = filtered_adata.obs[cell_type_key].astype('category').cat.categories
                        
                        scatter = ax.scatter(umap1, umap2, c=cell_type_values, cmap='tab20', alpha=0.7, s=10)
                        
                        # Add legend
                        legend1 = ax.legend(
                            handles=scatter.legend_elements()[0], 
                            labels=cell_type_names,
                            title=cell_type_key,
                            loc="upper right"
                        )
                        ax.add_artist(legend1)
                        
                        plt.xlabel('UMAP 1 (scVI)')
                        plt.ylabel('UMAP 2 (scVI)')
                        plt.title(f'UMAP of scVI latent space colored by {cell_type_key}')
                        
                        st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Error analyzing scVI latent space: {e}")
            
        else:
            st.warning("No suitable batch columns found in the dataset.")
    
    # Add a footer
    st.markdown("---")
    st.markdown(
        "**AnnData Explorer** - An interactive tool for exploring single-cell RNA-seq data. " +
        "Created with Streamlit, Scanpy, and AnnData."
    )


if __name__ == "__main__":
    set_plotting_defaults()
    main()


