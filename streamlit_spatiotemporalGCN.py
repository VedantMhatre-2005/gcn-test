"""
Streamlit Application for Spatiotemporal GCN Visualization
Interactive visualization of the Unified Spatiotemporal GCN architecture
"""

import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import torch
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Spatiotemporal GCN Visualization",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 20px;
    }
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


def create_architecture_graph(num_layers=4, hidden_channels=64, num_nodes=8):
    """
    Create a NetworkX graph representing the spatiotemporal GCN architecture
    """
    G = nx.DiGraph()
    
    # Node categories and their properties
    node_types = {
        'input': {'color': '#FF6B6B', 'shape': 'o', 'size': 800},
        'embedding': {'color': '#4ECDC4', 'shape': 's', 'size': 700},
        'stconv': {'color': '#45B7D1', 'shape': 'D', 'size': 900},
        'temporal': {'color': '#96CEB4', 'shape': '^', 'size': 700},
        'spatial': {'color': '#FFEAA7', 'shape': 'v', 'size': 700},
        'pooling': {'color': '#DFE6E9', 'shape': 'p', 'size': 600},
        'output': {'color': '#A29BFE', 'shape': '8', 'size': 800},
        'gate': {'color': '#FD79A8', 'shape': 'h', 'size': 500}
    }
    
    # Build the graph structure
    node_id = 0
    layer_nodes = {}
    
    # Input layer
    G.add_node(node_id, type='input', label='Input\n(seq_len × nodes × features)', 
               layer=0, pos_idx=0)
    layer_nodes[0] = [node_id]
    prev_layer = [node_id]
    node_id += 1
    
    # Input embedding layer
    G.add_node(node_id, type='embedding', label='Input Conv\n(1×1)', 
               layer=1, pos_idx=0)
    layer_nodes[1] = [node_id]
    for prev in prev_layer:
        G.add_edge(prev, node_id)
    prev_layer = [node_id]
    node_id += 1
    
    # Spatiotemporal convolution blocks
    for layer_idx in range(num_layers):
        layer_num = 2 + layer_idx * 4
        current_layer_nodes = []
        
        # Each ST-Conv block has multiple sub-components
        # 1. Temporal convolution
        temporal_node = node_id
        G.add_node(temporal_node, type='temporal', 
                   label=f'ST-Block {layer_idx+1}\nTemporal Conv', 
                   layer=layer_num, pos_idx=1)
        current_layer_nodes.append(temporal_node)
        node_id += 1
        
        # 2. Batch norm (temporal)
        bn_temporal = node_id
        G.add_node(bn_temporal, type='gate', 
                   label=f'BatchNorm', 
                   layer=layer_num+1, pos_idx=1)
        G.add_edge(temporal_node, bn_temporal)
        node_id += 1
        
        # 3. Spatial graph convolution
        spatial_node = node_id
        G.add_node(spatial_node, type='spatial', 
                   label=f'Spatial GCN', 
                   layer=layer_num+1, pos_idx=2)
        G.add_edge(bn_temporal, spatial_node)
        current_layer_nodes.append(spatial_node)
        node_id += 1
        
        # 4. Batch norm (spatial)
        bn_spatial = node_id
        G.add_node(bn_spatial, type='gate', 
                   label=f'BatchNorm', 
                   layer=layer_num+2, pos_idx=2)
        G.add_edge(spatial_node, bn_spatial)
        node_id += 1
        
        # 5. Residual connection
        residual_node = node_id
        G.add_node(residual_node, type='gate', 
                   label=f'Residual +', 
                   layer=layer_num+3, pos_idx=1.5)
        G.add_edge(bn_spatial, residual_node)
        # Add skip connection from previous layer
        for prev in prev_layer:
            G.add_edge(prev, residual_node, style='dashed')
        current_layer_nodes.append(residual_node)
        node_id += 1
        
        # Connect input to this block
        for prev in prev_layer:
            G.add_edge(prev, temporal_node)
        
        layer_nodes[layer_num] = current_layer_nodes
        prev_layer = [residual_node]
    
    # Temporal pooling
    pooling_layer = 2 + num_layers * 4
    pooling_node = node_id
    G.add_node(pooling_node, type='pooling', 
               label='Temporal\nPooling', 
               layer=pooling_layer, pos_idx=0)
    for prev in prev_layer:
        G.add_edge(prev, pooling_node)
    prev_layer = [pooling_node]
    node_id += 1
    
    # Output projection
    output_node = node_id
    G.add_node(output_node, type='output', 
               label=f'Output Conv\n(pred_horizon)', 
               layer=pooling_layer+1, pos_idx=0)
    for prev in prev_layer:
        G.add_edge(prev, output_node)
    node_id += 1
    
    return G, node_types


def visualize_architecture(G, node_types, layout_type='hierarchical'):
    """
    Visualize the architecture graph
    """
    fig, ax = plt.subplots(figsize=(20, 14))
    
    # Layout
    if layout_type == 'hierarchical':
        # Group by layers for hierarchical layout
        layers = {}
        for node in G.nodes():
            layer = G.nodes[node]['layer']
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)
        
        pos = {}
        max_layer = max(layers.keys())
        for layer, nodes in layers.items():
            num_nodes = len(nodes)
            y_pos = (max_layer - layer) / max_layer  # Top to bottom
            for i, node in enumerate(nodes):
                x_pos = (i + 1) / (num_nodes + 1)
                # Small random offset for visual clarity
                x_offset = np.random.uniform(-0.02, 0.02)
                y_offset = np.random.uniform(-0.02, 0.02)
                pos[node] = (x_pos + x_offset, y_pos + y_offset)
    
    elif layout_type == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    
    else:  # circular
        pos = nx.circular_layout(G)
    
    # Draw edges
    for edge in G.edges(data=True):
        source, target = edge[0], edge[1]
        style = edge[2].get('style', 'solid')
        
        if style == 'dashed':
            # Skip connections
            arrow = FancyArrowPatch(
                pos[source], pos[target],
                arrowstyle='->', mutation_scale=20, linewidth=1.5,
                linestyle='--', color='gray', alpha=0.4,
                connectionstyle='arc3,rad=0.2'
            )
        else:
            # Regular connections
            arrow = FancyArrowPatch(
                pos[source], pos[target],
                arrowstyle='->', mutation_scale=20, linewidth=2,
                color='#34495e', alpha=0.6,
                connectionstyle='arc3,rad=0.1'
            )
        ax.add_patch(arrow)
    
    # Draw nodes
    for node_type, props in node_types.items():
        nodes_of_type = [n for n in G.nodes() if G.nodes[n]['type'] == node_type]
        if nodes_of_type:
            node_positions = [pos[n] for n in nodes_of_type]
            x, y = zip(*node_positions)
            ax.scatter(x, y, c=props['color'], s=props['size'], 
                      marker=props['shape'], edgecolors='black', linewidths=2,
                      alpha=0.9, zorder=3)
    
    # Draw labels
    for node in G.nodes():
        x, y = pos[node]
        label = G.nodes[node]['label']
        ax.text(x, y-0.05, label, fontsize=7, ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='gray', alpha=0.8))
    
    # Legend
    legend_elements = []
    for node_type, props in node_types.items():
        legend_elements.append(plt.scatter([], [], c=props['color'], s=200, 
                                          marker=props['shape'], 
                                          edgecolors='black', linewidths=2,
                                          label=node_type.capitalize()))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
             framealpha=0.9, title='Layer Types')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.1)
    ax.axis('off')
    ax.set_title('Unified Spatiotemporal GCN Architecture', 
                fontsize=20, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def load_model_info():
    """
    Load model information and training results
    """
    info = {
        'loaded': False,
        'config': {},
        'metrics': {}
    }
    
    # Try to load training history
    if Path('unified_stgcn_history.json').exists():
        with open('unified_stgcn_history.json', 'r') as f:
            history = json.load(f)
            info['config'] = history.get('config', {})
            info['metrics'] = history.get('test_metrics', {})
            info['train_losses'] = history.get('train_losses', [])
            info['val_losses'] = history.get('val_losses', [])
            info['val_maes'] = history.get('val_maes', [])
            info['loaded'] = True
    
    return info


def main():
    """
    Main Streamlit application
    """
    # Header
    st.markdown('<h1 class="main-header">🧠 Spatiotemporal GCN Architecture Visualization</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; padding: 10px; font-size: 1.2rem;'>
    Interactive visualization of the <b>Unified Spatiotemporal Graph Convolutional Network</b> 
    for traffic prediction on the Chembur network.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
        st.title("⚙️ Configuration")
        
        st.markdown("### Model Parameters")
        num_layers = st.slider("Number of ST-Conv Layers", 1, 8, 4)
        hidden_channels = st.slider("Hidden Channels", 16, 128, 64, step=16)
        num_nodes = st.slider("Number of Nodes", 4, 16, 8)
        
        st.markdown("### Visualization Settings")
        layout_type = st.selectbox(
            "Graph Layout",
            ['hierarchical', 'spring', 'kamada_kawai', 'circular'],
            index=0
        )
        
        show_stats = st.checkbox("Show Statistics", value=True)
        show_legend = st.checkbox("Show Architecture Details", value=True)
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.info("""
        This application visualizes the architecture of a 
        **Unified Spatiotemporal GCN** that replaces the traditional 
        GCN + CNN + MLP pipeline with an integrated architecture.
        
        **Key Features:**
        - Simultaneous spatial & temporal processing
        - Residual connections
        - Batch normalization
        - Integrated design
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="sub-header">🔷 Network Architecture Graph</p>', 
                   unsafe_allow_html=True)
        
        # Create and visualize architecture
        with st.spinner("Generating architecture graph..."):
            G, node_types = create_architecture_graph(num_layers, hidden_channels, num_nodes)
            fig = visualize_architecture(G, node_types, layout_type)
            st.pyplot(fig)
        
        # Architecture stats
        st.markdown("### 📊 Architecture Statistics")
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.metric("Total Nodes", G.number_of_nodes())
            st.metric("ST-Conv Blocks", num_layers)
        
        with stats_col2:
            st.metric("Total Edges", G.number_of_edges())
            st.metric("Hidden Channels", hidden_channels)
        
        with stats_col3:
            # Estimate parameters
            # Very rough estimate
            params_per_stconv = hidden_channels * hidden_channels * 3 * 2  # Temporal + Spatial weights
            total_params = (
                4 * hidden_channels +  # Input conv
                num_layers * params_per_stconv +  # ST-Conv blocks
                hidden_channels * 12  # Output conv
            )
            st.metric("Est. Parameters", f"{total_params:,}")
            st.metric("Graph Nodes", num_nodes)
    
    with col2:
        st.markdown('<p class="sub-header">📈 Model Performance</p>', 
                   unsafe_allow_html=True)
        
        # Load and display model info
        info = load_model_info()
        
        if info['loaded']:
            st.success("✅ Model information loaded")
            
            # Display metrics
            if info['metrics']:
                st.markdown("#### Test Set Performance")
                st.metric("MAE", f"{info['metrics'].get('mae', 0):.4f} vehicles")
                st.metric("RMSE", f"{info['metrics'].get('rmse', 0):.4f} vehicles")
                st.metric("MAPE", f"{info['metrics'].get('mape', 0):.2f}%")
            
            # Display config
            if info['config']:
                st.markdown("#### Model Configuration")
                config = info['config']
                st.json({
                    "Input Sequence": f"{config.get('seq_len', 60)} timesteps",
                    "Prediction Horizon": f"{config.get('pred_horizon', 12)} timesteps",
                    "Layers": config.get('num_layers', 4),
                    "Hidden Channels": config.get('hidden_channels', 64),
                    "Dropout": config.get('dropout', 0.2)
                })
            
            # Plot training curves
            if 'train_losses' in info and len(info['train_losses']) > 0:
                st.markdown("#### Training Progress")
                fig_loss, ax = plt.subplots(figsize=(8, 4))
                ax.plot(info['train_losses'], label='Train Loss', linewidth=2)
                ax.plot(info['val_losses'], label='Val Loss', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training History')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig_loss)
        else:
            st.warning("⚠️ No trained model found. Run `train_unified_stgcn.py` first.")
        
        # Architecture explanation
        if show_legend:
            st.markdown("---")
            st.markdown("#### 🔍 Layer Types")
            st.markdown("""
            - **🔴 Input**: Raw spatiotemporal traffic data
            - **🔷 Embedding**: Initial feature embedding
            - **💠 ST-Conv**: Spatiotemporal convolution block
            - **🔺 Temporal**: 1D temporal convolution
            - **🔻 Spatial**: Graph convolution (spatial)
            - **⬟ Gate**: BatchNorm / Residual connections
            - **⬡ Pooling**: Temporal pooling operation
            - **💜 Output**: Final prediction layer
            """)
    
    # Additional information
    st.markdown("---")
    st.markdown("### 🎯 Architecture Overview")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("""
        #### Traditional Architecture
        - **Step 1**: GCN (spatial)
        - **Step 2**: CNN (temporal)
        - **Step 3**: MLP (prediction)
        - ❌ Sequential processing
        """)
    
    with info_col2:
        st.markdown("""
        #### Unified ST-GCN
        - **Integrated**: ST-Conv blocks
        - **Simultaneous**: Spatial + Temporal
        - **Residual**: Skip connections
        - ✅ Parallel processing
        """)
    
    with info_col3:
        st.markdown("""
        #### Key Benefits
        - 🚀 Faster training
        - 🎯 Better accuracy
        - 💡 Simpler architecture
        - 🔗 Better information flow
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: gray;'>
    Built with Streamlit | Powered by PyTorch & NetworkX | 
    <a href='https://github.com'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
