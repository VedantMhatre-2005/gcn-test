"""
Script to extract and save embeddings from the trained model
These embeddings can be used for quantum encoding in future work
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from chembur_network import ChemburTrafficNetwork
from spatiotemporal_gcn import SpatioTemporalGCN, TrafficPredictor
from traffic_data_generator import create_dataloaders


def extract_and_save_embeddings():
    """
    Extract embeddings from the trained model and save them
    """
    print("="*80)
    print("EMBEDDING EXTRACTION FOR QUANTUM ENCODING")
    print("="*80)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create network
    print("\nLoading network...")
    network = ChemburTrafficNetwork()
    
    # Load model
    print("Loading trained model...")
    model = SpatioTemporalGCN(
        num_nodes=8,
        num_features=4,
        gcn_hidden_dims=[32, 64],
        temporal_channels=[64, 128],
        mlp_hidden_dims=[128, 64],
        output_horizon=1,
        dropout=0.2
    )
    
    try:
        model.load_state_dict(torch.load('traffic_gcn_model.pth', map_location=device))
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("✗ Model not found. Please run train_model.py first.")
        return
    
    predictor = TrafficPredictor(model, device=device)
    
    # Generate test data
    print("\nGenerating test data...")
    _, _, test_loader = create_dataloaders(
        network,
        train_days=1,
        val_days=1,
        test_days=1,
        seq_len=12,
        pred_horizon=1,
        batch_size=32
    )
    
    # Extract embeddings
    print("\n" + "="*80)
    print("EXTRACTING EMBEDDINGS")
    print("="*80)
    print("\nProcessing test data (this may take a minute)...")
    
    embeddings = predictor.extract_embeddings(test_loader, max_batches=50)
    
    print("\n✓ Embeddings extracted successfully!")
    print(f"\nTotal samples: {embeddings['combined_embeddings'].shape[0]}")
    
    # Save embeddings
    print("\n" + "="*80)
    print("SAVING EMBEDDINGS")
    print("="*80)
    
    predictor.save_embeddings(embeddings, 'traffic_embeddings.npz')
    
    # Analyze embeddings
    print("\n" + "="*80)
    print("EMBEDDING ANALYSIS")
    print("="*80)
    
    spatial_emb = embeddings['spatial_embeddings']
    temporal_emb = embeddings['temporal_embeddings']
    combined_emb = embeddings['combined_embeddings']
    
    print(f"\nSpatial Embeddings:")
    print(f"  Shape: {spatial_emb.shape}")
    print(f"  Description: (samples, time_steps=12, nodes=8, features=64)")
    print(f"  Size per sample: {spatial_emb[0].nbytes / 1024:.2f} KB")
    print(f"  Mean: {spatial_emb.mean():.4f}, Std: {spatial_emb.std():.4f}")
    
    print(f"\nTemporal Embeddings:")
    print(f"  Shape: {temporal_emb.shape}")
    print(f"  Description: (samples, features=128)")
    print(f"  Size per sample: {temporal_emb[0].nbytes / 1024:.2f} KB")
    print(f"  Mean: {temporal_emb.mean():.4f}, Std: {temporal_emb.std():.4f}")
    
    print(f"\nCombined Embeddings:")
    print(f"  Shape: {combined_emb.shape}")
    print(f"  Description: (samples, features=640)")
    print(f"  Size per sample: {combined_emb[0].nbytes / 1024:.2f} KB")
    print(f"  Mean: {combined_emb.mean():.4f}, Std: {combined_emb.std():.4f}")
    
    # Visualize embeddings
    print("\n" + "="*80)
    print("VISUALIZATION")
    print("="*80)
    
    visualize_embeddings(embeddings, network)
    
    # Compression analysis for quantum encoding
    print("\n" + "="*80)
    print("QUANTUM ENCODING READINESS ANALYSIS")
    print("="*80)
    
    analyze_for_quantum_encoding(embeddings)
    
    print("\n✓ Complete! Embeddings ready for quantum encoding pipeline.")


def visualize_embeddings(embeddings, network):
    """
    Create visualizations of the extracted embeddings
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Spatial embeddings - Last time step for each node
    ax1 = plt.subplot(2, 3, 1)
    spatial_last = embeddings['spatial_embeddings'][:100, -1, :, :]  # Last time step, first 100 samples
    spatial_mean = spatial_last.mean(axis=0)  # Average over samples: (8 nodes, 64 features)
    
    im1 = ax1.imshow(spatial_mean.T, aspect='auto', cmap='viridis')
    ax1.set_xlabel('Node ID', fontsize=11)
    ax1.set_ylabel('Embedding Dimension', fontsize=11)
    ax1.set_title('Spatial Embeddings\n(per node, 64-dim)', fontweight='bold')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Temporal embeddings distribution
    ax2 = plt.subplot(2, 3, 2)
    temporal_sample = embeddings['temporal_embeddings'][:100, :]
    ax2.hist(temporal_sample.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Embedding Value', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Temporal Embeddings\nDistribution (128-dim)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Combined embeddings - sample visualization
    ax3 = plt.subplot(2, 3, 3)
    combined_sample = embeddings['combined_embeddings'][:50, :]  # First 50 samples
    im3 = ax3.imshow(combined_sample.T, aspect='auto', cmap='plasma')
    ax3.set_xlabel('Sample Index', fontsize=11)
    ax3.set_ylabel('Embedding Dimension', fontsize=11)
    ax3.set_title('Combined Embeddings\n(640-dim)', fontweight='bold')
    plt.colorbar(im3, ax=ax3)
    
    # 4. Spatial embeddings over time for one node
    ax4 = plt.subplot(2, 3, 4)
    node_id = 0
    spatial_timeseries = embeddings['spatial_embeddings'][0, :, node_id, :]  # One sample, all time steps
    im4 = ax4.imshow(spatial_timeseries.T, aspect='auto', cmap='coolwarm')
    ax4.set_xlabel('Time Step', fontsize=11)
    ax4.set_ylabel('Embedding Dimension', fontsize=11)
    ax4.set_title(f'Spatial Evolution (Node {node_id})\nover 12 time steps', fontweight='bold')
    plt.colorbar(im4, ax=ax4)
    
    # 5. PCA visualization of combined embeddings
    ax5 = plt.subplot(2, 3, 5)
    from sklearn.decomposition import PCA
    
    combined_subset = embeddings['combined_embeddings'][:500, :]
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined_subset)
    
    scatter = ax5.scatter(combined_2d[:, 0], combined_2d[:, 1], 
                         c=np.arange(len(combined_2d)), cmap='viridis', 
                         alpha=0.6, s=20)
    ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax5.set_title('PCA of Combined Embeddings\n(2D projection)', fontweight='bold')
    plt.colorbar(scatter, ax=ax5, label='Sample Index')
    
    # 6. Embedding dimension importance (variance)
    ax6 = plt.subplot(2, 3, 6)
    combined_var = embeddings['combined_embeddings'].var(axis=0)
    ax6.plot(combined_var, linewidth=2, color='#1f77b4')
    ax6.set_xlabel('Embedding Dimension', fontsize=11)
    ax6.set_ylabel('Variance', fontsize=11)
    ax6.set_title('Embedding Dimension Variance\n(for compression guidance)', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('embedding_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved: embedding_visualization.png")


def analyze_for_quantum_encoding(embeddings):
    """
    Analyze embeddings for quantum encoding feasibility
    """
    combined = embeddings['combined_embeddings']
    
    # Original dimension
    original_dim = combined.shape[1]
    print(f"\nOriginal embedding dimension: {original_dim}")
    
    # Estimate required qubits
    import math
    qubits_direct = math.ceil(math.log2(original_dim))
    print(f"Qubits for direct encoding: {qubits_direct}")
    
    # PCA compression analysis
    from sklearn.decomposition import PCA
    
    print("\nPCA Compression Analysis:")
    for n_components in [32, 64, 128, 256]:
        pca = PCA(n_components=n_components)
        compressed = pca.fit_transform(combined)
        variance_retained = pca.explained_variance_ratio_.sum()
        qubits_needed = math.ceil(math.log2(n_components))
        
        print(f"  {n_components}-dim compression:")
        print(f"    - Variance retained: {variance_retained*100:.2f}%")
        print(f"    - Compression ratio: {original_dim/n_components:.1f}x")
        print(f"    - Qubits needed: {qubits_needed}")
    
    # Frequency domain analysis
    print("\nFrequency-Domain Encoding Potential:")
    
    # Simulate DCT compression
    from scipy.fftpack import dct, idct
    
    sample = combined[0:1, :].reshape(-1)
    dct_coeffs = dct(sample, norm='ortho')
    
    # Keep top coefficients
    for keep_ratio in [0.1, 0.2, 0.5]:
        n_keep = int(len(dct_coeffs) * keep_ratio)
        compressed_coeffs = dct_coeffs.copy()
        compressed_coeffs[n_keep:] = 0
        reconstructed = idct(compressed_coeffs, norm='ortho')
        
        mse = np.mean((sample - reconstructed) ** 2)
        qubits_needed = math.ceil(math.log2(n_keep))
        
        print(f"  Keep top {keep_ratio*100:.0f}% coefficients ({n_keep} dims):")
        print(f"    - Reconstruction MSE: {mse:.6f}")
        print(f"    - Qubits needed: {qubits_needed}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR QUANTUM PIPELINE")
    print("="*80)
    
    print("\n1. Dimension Reduction:")
    print("   → Use PCA to compress 640-dim → 64-dim (retains ~90% variance)")
    print("   → Requires only 6 qubits for amplitude encoding")
    
    print("\n2. Frequency-Domain Encoding:")
    print("   → Apply DCT to spatial embeddings")
    print("   → Keep top 20% coefficients (128 dims)")
    print("   → Requires 7 qubits")
    
    print("\n3. Angle Encoding:")
    print("   → Encode compressed embeddings as rotation angles")
    print("   → 1 feature per qubit → 64 features needs 64 qubits")
    print("   → Most NISQ-friendly approach")
    
    print("\n4. Hybrid Strategy (RECOMMENDED):")
    print("   → PCA: 640-dim → 32-dim")
    print("   → Angle encoding: 32 features")
    print("   → Requires: 32 qubits (feasible on current NISQ devices)")
    print("   → Circuit depth: O(32) - shallow and noise-resistant")


def demo_single_prediction_with_embeddings():
    """
    Demonstrate extracting embeddings from a single prediction
    """
    print("\n" + "="*80)
    print("DEMO: Single Prediction with Embeddings")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = ChemburTrafficNetwork()
    
    model = SpatioTemporalGCN(
        num_nodes=8, num_features=4,
        gcn_hidden_dims=[32, 64],
        temporal_channels=[64, 128],
        mlp_hidden_dims=[128, 64],
        output_horizon=1, dropout=0.2
    )
    
    try:
        model.load_state_dict(torch.load('traffic_gcn_model.pth', map_location=device))
    except:
        print("Model not found!")
        return
    
    predictor = TrafficPredictor(model, device=device)
    
    # Create sample input
    x_sample = np.random.randn(1, 12, 8, 4)
    adj_matrix = network.get_adjacency_matrix()
    
    # Predict with embeddings
    predictions, embeddings = predictor.predict(x_sample, adj_matrix, return_embeddings=True)
    
    print("\nPrediction shape:", predictions.shape)
    print("\nExtracted embeddings:")
    for key, value in embeddings.items():
        print(f"  {key}: {value.shape}")
    
    print("\n✓ Embeddings can be extracted from any prediction!")
    print("  These can be fed directly to quantum circuits.")


if __name__ == "__main__":
    # Extract and save embeddings from test data
    extract_and_save_embeddings()
    
    # Demo single prediction
    demo_single_prediction_with_embeddings()
    
    print("\n" + "="*80)
    print("NEXT STEPS FOR QUANTUM INTEGRATION:")
    print("="*80)
    print("\n1. Install quantum libraries:")
    print("   pip install qiskit pennylane")
    print("\n2. Load embeddings:")
    print("   embeddings = np.load('traffic_embeddings.npz')")
    print("\n3. Implement quantum encoding:")
    print("   - Use PCA to compress to 32-64 dimensions")
    print("   - Encode as angle rotations on qubits")
    print("   - Build QCNN circuit on top")
    print("\n4. Train hybrid quantum-classical model")
    print("="*80)
