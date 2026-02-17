"""
Streamlit Comparison Dashboard
Interactive visualization comparing GCN and QCNN performance
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import pandas as pd
from pathlib import Path
import time
import seaborn as sns

from chembur_network import ChemburTrafficNetwork
from spatiotemporal_gcn import SpatioTemporalGCN, TrafficPredictor
from qcnn_model import QCNN, QCNNPredictor
from traffic_data_generator import TrafficDataGenerator

# Set page config
st.set_page_config(
    page_title="GCN vs QCNN Comparison",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1f77b4, #9467bd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        font-weight: 600;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .quantum-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .classical-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = None
if 'network' not in st.session_state:
    st.session_state.network = None
if 'gcn_predictor' not in st.session_state:
    st.session_state.gcn_predictor = None
if 'qcnn_predictor' not in st.session_state:
    st.session_state.qcnn_predictor = None


@st.cache_resource
def load_network():
    """Load traffic network"""
    return ChemburTrafficNetwork()


@st.cache_data
def load_comparison_data():
    """Load comparison results from JSON"""
    try:
        with open('model_comparison.json', 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return None


@st.cache_data
def load_gcn_metrics():
    """Load GCN metrics if available"""
    try:
        # Try to load from training output
        # For now, return None if file doesn't exist
        return None
    except:
        return None


@st.cache_data
def load_qcnn_metrics():
    """Load QCNN metrics"""
    try:
        with open('qcnn_metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def plot_performance_comparison(comparison_data):
    """Create comparison bar charts"""
    models = comparison_data['models']
    
    if 'gcn' not in models or 'qcnn' not in models:
        st.warning("Both models needed for comparison. Please train both models.")
        return
    
    # Create comparison dataframe
    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'MSE'],
        'GCN': [models['gcn']['rmse'], models['gcn']['mae'], models['gcn']['mse']],
        'QCNN': [models['qcnn']['rmse'], models['qcnn']['mae'], models['qcnn']['mse']]
    })
    
    # Plotly bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Spatiotemporal GCN',
        x=metrics_df['Metric'],
        y=metrics_df['GCN'],
        marker_color='#4facfe',
        text=metrics_df['GCN'].round(4),
        textposition='auto',
    ))
    fig.add_trace(go.Bar(
        name='Quantum CNN',
        x=metrics_df['Metric'],
        y=metrics_df['QCNN'],
        marker_color='#f5576c',
        text=metrics_df['QCNN'].round(4),
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Performance Metrics Comparison',
        xaxis_title='Metric',
        yaxis_title='Value',
        barmode='group',
        height=400,
        template='plotly_white',
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_latency_comparison(comparison_data):
    """Plot inference latency comparison"""
    models = comparison_data['models']
    
    if 'gcn' not in models or 'qcnn' not in models:
        return
    
    latency_df = pd.DataFrame({
        'Model': ['GCN', 'QCNN'],
        'Avg Latency (ms)': [
            models['gcn']['avg_latency_ms'],
            models['qcnn']['avg_latency_ms']
        ],
        'Std Dev': [
            models['gcn']['std_latency_ms'],
            models['qcnn']['std_latency_ms']
        ]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=latency_df['Model'],
        y=latency_df['Avg Latency (ms)'],
        error_y=dict(type='data', array=latency_df['Std Dev']),
        marker_color=['#4facfe', '#f5576c'],
        text=latency_df['Avg Latency (ms)'].round(2),
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Inference Latency Comparison',
        xaxis_title='Model',
        yaxis_title='Latency (milliseconds)',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_node_comparison(comparison_data):
    """Plot per-node performance comparison"""
    models = comparison_data['models']
    node_names = comparison_data['node_names']
    
    if 'gcn' not in models or 'qcnn' not in models:
        return
    
    # Create dataframe
    node_df = pd.DataFrame({
        'Junction': node_names,
        'GCN MAE': models['gcn']['node_mae'],
        'QCNN MAE': models['qcnn']['node_mae']
    })
    
    # Calculate difference
    node_df['Difference'] = node_df['QCNN MAE'] - node_df['GCN MAE']
    
    # Create grouped bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='GCN',
        x=node_df['Junction'],
        y=node_df['GCN MAE'],
        marker_color='#4facfe',
    ))
    fig.add_trace(go.Bar(
        name='QCNN',
        x=node_df['Junction'],
        y=node_df['QCNN MAE'],
        marker_color='#f5576c',
    ))
    
    fig.update_layout(
        title='Per-Node MAE Comparison',
        xaxis_title='Junction',
        yaxis_title='Mean Absolute Error',
        barmode='group',
        height=500,
        template='plotly_white',
        xaxis={'tickangle': -45}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show difference table
    st.subheader("Detailed Node Comparison")
    node_df['Winner'] = node_df.apply(
        lambda row: 'GCN' if row['GCN MAE'] < row['QCNN MAE'] else 'QCNN',
        axis=1
    )
    st.dataframe(
        node_df.style.highlight_max(axis=1, subset=['GCN MAE', 'QCNN MAE'], color='#ffcccc')
                   .highlight_min(axis=1, subset=['GCN MAE', 'QCNN MAE'], color='#ccffcc'),
        use_container_width=True
    )


def plot_predictions_scatter(comparison_data, model_name):
    """Scatter plot of predictions vs actuals"""
    models = comparison_data['models']
    
    if model_name.lower() not in models:
        st.warning(f"{model_name} model not found.")
        return
    
    model_data = models[model_name.lower()]
    predictions = np.array(model_data['predictions'])
    actuals = np.array(model_data['actuals'])
    
    # Sample for visualization
    sample_size = min(500, len(predictions))
    indices = np.random.choice(len(predictions), sample_size, replace=False)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actuals[indices].flatten(),
        y=predictions[indices].flatten(),
        mode='markers',
        marker=dict(
            size=5,
            color=actuals[indices].flatten(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Actual Value")
        ),
        name='Predictions'
    ))
    
    # Add perfect prediction line
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title=f'{model_name} Predictions vs Actual Values',
        xaxis_title='Actual Traffic',
        yaxis_title='Predicted Traffic',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_training_curves(qcnn_metrics):
    """Plot training curves for QCNN"""
    if qcnn_metrics is None:
        st.warning("QCNN training metrics not found.")
        return
    
    train_losses = qcnn_metrics.get('train_losses', [])
    val_losses = qcnn_metrics.get('val_losses', [])
    
    if not train_losses:
        st.warning("No training history available.")
        return
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(train_losses) + 1)),
        y=train_losses,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='#4facfe', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=list(range(1, len(val_losses) + 1)),
        y=val_losses,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='#f5576c', width=2)
    ))
    
    fig.update_layout(
        title='QCNN Training Progress',
        xaxis_title='Epoch',
        yaxis_title='MSE Loss',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_architecture_comparison():
    """Visual comparison of architectures"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔷 Spatiotemporal GCN")
        st.markdown("""
        **Classical Architecture:**
        - Graph Convolutional Layers
        - Temporal CNN Layers
        - Fully Connected Layers
        - ~100K+ parameters
        - Fast inference (~10-50ms)
        - Traditional gradient descent
        """)
        
    with col2:
        st.markdown("### ⚛️ Quantum CNN")
        st.markdown("""
        **Quantum Architecture:**
        - Quantum Feature Encoding
        - Parameterized Quantum Circuits
        - Quantum Convolutional Gates
        - Measurement-based Pooling
        - Quantum-Classical Hybrid
        - Exponential state space
        """)


# Main Dashboard
def main():
    # Header
    st.markdown('<p class="main-header">⚛️ Quantum vs Classical: Traffic Prediction Comparison</p>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load data
    comparison_data = load_comparison_data()
    qcnn_metrics = load_qcnn_metrics()
    network = load_network()
    
    if comparison_data is None:
        st.error("⚠️ No comparison data found!")
        st.info("""
        Please run the following steps:
        1. Train the GCN model: `python train_model.py`
        2. Train the QCNN model: `python train_qcnn.py`
        3. Generate comparison: `python compare_models.py`
        4. Refresh this dashboard
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        st.markdown("### 📊 Models")
        models = comparison_data.get('models', {})
        gcn_available = 'gcn' in models
        qcnn_available = 'qcnn' in models
        
        st.markdown(f"**GCN:** {'✅ Loaded' if gcn_available else '❌ Not Found'}")
        st.markdown(f"**QCNN:** {'✅ Loaded' if qcnn_available else '❌ Not Found'}")
        
        st.markdown("---")
        
        st.markdown("### 🏙️ Network Info")
        st.markdown(f"**Nodes:** {len(comparison_data['node_names'])}")
        st.markdown(f"**Last Updated:** {comparison_data.get('timestamp', 'Unknown')}")
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Performance Metrics",
        "🎯 Predictions Analysis",
        "⚡ Architecture Details",
        "🔬 Model Insights"
    ])
    
    with tab1:
        st.header("Performance Overview")
        
        if gcn_available and qcnn_available:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Best RMSE",
                    f"{min(models['gcn']['rmse'], models['qcnn']['rmse']):.4f}",
                    delta=f"{'GCN' if models['gcn']['rmse'] < models['qcnn']['rmse'] else 'QCNN'}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Best MAE",
                    f"{min(models['gcn']['mae'], models['qcnn']['mae']):.4f}",
                    delta=f"{'GCN' if models['gcn']['mae'] < models['qcnn']['mae'] else 'QCNN'}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Fastest Inference",
                    f"{min(models['gcn']['avg_latency_ms'], models['qcnn']['avg_latency_ms']):.2f} ms",
                    delta=f"{'GCN' if models['gcn']['avg_latency_ms'] < models['qcnn']['avg_latency_ms'] else 'QCNN'}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Model-specific metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔷 Spatiotemporal GCN")
                gcn_col1, gcn_col2 = st.columns(2)
                with gcn_col1:
                    st.metric("RMSE", f"{models['gcn']['rmse']:.4f}")
                    st.metric("MAE", f"{models['gcn']['mae']:.4f}")
                with gcn_col2:
                    st.metric("Latency", f"{models['gcn']['avg_latency_ms']:.2f} ms")
                    st.metric("Eval Time", f"{models['gcn']['eval_time']:.2f} s")
            
            with col2:
                st.markdown("### ⚛️ Quantum CNN")
                qcnn_col1, qcnn_col2 = st.columns(2)
                with qcnn_col1:
                    st.metric("RMSE", f"{models['qcnn']['rmse']:.4f}")
                    st.metric("MAE", f"{models['qcnn']['mae']:.4f}")
                with qcnn_col2:
                    st.metric("Latency", f"{models['qcnn']['avg_latency_ms']:.2f} ms")
                    st.metric("Eval Time", f"{models['qcnn']['eval_time']:.2f} s")
        else:
            st.warning("Complete model comparison not available.")
    
    with tab2:
        st.header("Detailed Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            plot_performance_comparison(comparison_data)
        
        with col2:
            plot_latency_comparison(comparison_data)
        
        st.markdown("---")
        plot_node_comparison(comparison_data)
    
    with tab3:
        st.header("Predictions Analysis")
        
        model_choice = st.selectbox(
            "Select model to visualize:",
            ["GCN", "QCNN"] if gcn_available and qcnn_available else 
            (["GCN"] if gcn_available else ["QCNN"])
        )
        
        plot_predictions_scatter(comparison_data, model_choice)
        
        if qcnn_metrics and model_choice == "QCNN":
            st.markdown("---")
            st.subheader("QCNN Training History")
            plot_training_curves(qcnn_metrics)
    
    with tab4:
        st.header("Architecture Comparison")
        plot_architecture_comparison()
        
        st.markdown("---")
        
        if qcnn_metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ⚛️ Quantum Architecture Details")
                st.markdown(f"""
                - **Qubits:** {qcnn_metrics.get('num_qubits', 'N/A')}
                - **Quantum Layers:** {qcnn_metrics.get('num_layers', 'N/A')}
                - **Quantum Parameters:** {qcnn_metrics.get('quantum_parameters', 'N/A'):,}
                - **Classical Parameters:** {qcnn_metrics.get('classical_parameters', 'N/A'):,}
                - **Total Parameters:** {qcnn_metrics.get('total_parameters', 'N/A'):,}
                """)
            
            with col2:
                st.markdown("### 📊 Training Statistics")
                st.markdown(f"""
                - **Training Time:** {qcnn_metrics.get('training_time', 'N/A'):.2f}s
                - **Final Train Loss:** {qcnn_metrics.get('train_losses', [0])[-1]:.4f}
                - **Final Val Loss:** {qcnn_metrics.get('val_losses', [0])[-1]:.4f}
                - **Quantum/Classical Ratio:** {qcnn_metrics.get('quantum_parameters', 0) / qcnn_metrics.get('total_parameters', 1):.2%}
                """)
    
    with tab5:
        st.header("Model Insights & Analysis")
        
        st.markdown("### 🎯 Key Findings")
        
        if gcn_available and qcnn_available:
            gcn_rmse = models['gcn']['rmse']
            qcnn_rmse = models['qcnn']['rmse']
            
            diff_pct = abs((qcnn_rmse - gcn_rmse) / gcn_rmse * 100)
            
            if qcnn_rmse < gcn_rmse:
                st.success(f"✅ QCNN outperforms GCN by {diff_pct:.2f}% in RMSE")
            else:
                st.info(f"📊 GCN outperforms QCNN by {diff_pct:.2f}% in RMSE")
            
            # Latency analysis
            lat_diff = abs((models['qcnn']['avg_latency_ms'] - models['gcn']['avg_latency_ms']) / 
                          models['gcn']['avg_latency_ms'] * 100)
            
            if models['qcnn']['avg_latency_ms'] > models['gcn']['avg_latency_ms']:
                st.warning(f"⚠️ QCNN is {lat_diff:.2f}% slower than GCN (expected for quantum simulation)")
            
            st.markdown("### 💡 Insights")
            
            st.markdown("""
            **Quantum Advantages:**
            - Exploits superposition and entanglement
            - Potential for exponential speedup with real quantum hardware
            - Novel approach to graph-based learning
            
            **Classical Advantages:**
            - Faster on classical hardware
            - More mature optimization techniques
            - Larger model capacity
            
            **Future Directions:**
            - Test on actual quantum hardware
            - Hybrid quantum-classical ensembles
            - Variational quantum algorithms
            """)
        
        st.markdown("---")
        st.markdown("### 📚 References")
        st.markdown("""
        - Quantum Convolutional Neural Networks (Cong et al., 2019)
        - PennyLane: Automatic differentiation of hybrid quantum-classical computations
        - Graph Convolutional Networks (Kipf & Welling, 2017)
        """)


if __name__ == '__main__':
    main()
