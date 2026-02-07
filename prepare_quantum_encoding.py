"""
Example: Loading and Using Extracted Embeddings for Quantum Encoding

This script demonstrates how to:
1. Load the extracted embeddings
2. Compress them using PCA
3. Prepare them for quantum circuit encoding
4. Visualize the compressed representations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.fftpack import dct

def load_and_analyze_embeddings():
    """
    Load embeddings and prepare for quantum encoding
    """
    print("="*80)
    print("QUANTUM ENCODING PREPARATION")
    print("="*80)
    
    # Load embeddings
    print("\n1. Loading embeddings...")
    try:
        data = np.load('traffic_embeddings.npz')
        embeddings = {key: data[key] for key in data.files}
        print(f"✓ Loaded {len(embeddings)} embedding arrays")
    except FileNotFoundError:
        print("✗ Embeddings not found. Run extract_embeddings.py first.")
        return None
    
    # Get combined embeddings (most useful for quantum encoding)
    combined = embeddings['combined_embeddings']
    print(f"\nCombined embeddings shape: {combined.shape}")
    print(f"Total samples: {combined.shape[0]}")
    print(f"Embedding dimension: {combined.shape[1]}")
    
    return embeddings


def compress_for_quantum(embeddings, n_components=32):
    """
    Compress embeddings using PCA for quantum encoding
    """
    print("\n" + "="*80)
    print(f"2. COMPRESSING TO {n_components} DIMENSIONS")
    print("="*80)
    
    combined = embeddings['combined_embeddings']
    
    # Apply PCA
    print(f"\nApplying PCA compression...")
    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(combined)
    
    variance_retained = pca.explained_variance_ratio_.sum()
    print(f"✓ Compression complete")
    print(f"  Original: {combined.shape[1]} dimensions")
    print(f"  Compressed: {compressed.shape[1]} dimensions")
    print(f"  Variance retained: {variance_retained*100:.2f}%")
    print(f"  Compression ratio: {combined.shape[1]/n_components:.1f}x")
    
    return compressed, pca


def prepare_quantum_encoding(compressed_embeddings, encoding_type='angle'):
    """
    Prepare compressed embeddings for different quantum encoding schemes
    """
    print("\n" + "="*80)
    print(f"3. PREPARING QUANTUM ENCODING ({encoding_type.upper()})")
    print("="*80)
    
    n_samples, n_features = compressed_embeddings.shape
    
    if encoding_type == 'angle':
        # Angle encoding: each feature becomes a rotation angle
        # Normalize to [0, 2π]
        min_val = compressed_embeddings.min()
        max_val = compressed_embeddings.max()
        angles = 2 * np.pi * (compressed_embeddings - min_val) / (max_val - min_val)
        
        print(f"\n✓ Angle Encoding Prepared")
        print(f"  Features per sample: {n_features}")
        print(f"  Qubits needed: {n_features} (one per feature)")
        print(f"  Angle range: [0, 2π]")
        print(f"  Circuit depth: O({n_features}) gates")
        print(f"\n  Example encoding for first sample:")
        print(f"  Gates: RY(θ₁), RY(θ₂), ..., RY(θ₃₂)")
        print(f"  where θᵢ ∈ {angles[0][:5]}")
        
        return angles
    
    elif encoding_type == 'amplitude':
        # Amplitude encoding: encode in superposition state
        # Normalize each sample to unit vector
        norms = np.linalg.norm(compressed_embeddings, axis=1, keepdims=True)
        normalized = compressed_embeddings / (norms + 1e-8)
        
        import math
        qubits_needed = math.ceil(math.log2(n_features))
        
        print(f"\n✓ Amplitude Encoding Prepared")
        print(f"  Features per sample: {n_features}")
        print(f"  Qubits needed: {qubits_needed} (log₂({n_features}))")
        print(f"  Normalization: ‖x‖ = 1")
        print(f"  Circuit depth: O(2^{qubits_needed}) gates")
        print(f"  WARNING: Exponentially deep circuits!")
        
        return normalized
    
    elif encoding_type == 'basis':
        # Basis encoding: binary representation
        # Discretize and binary encode
        discretized = np.digitize(compressed_embeddings, 
                                  bins=np.linspace(compressed_embeddings.min(), 
                                                  compressed_embeddings.max(), 16))
        
        print(f"\n✓ Basis Encoding Prepared")
        print(f"  Features per sample: {n_features}")
        print(f"  Discretization levels: 16 (4 bits each)")
        print(f"  Qubits needed: {n_features * 4} total")
        print(f"  Circuit depth: O({n_features}) gates")
        
        return discretized


def visualize_compression_quality(original, compressed, pca):
    """
    Visualize compression quality
    """
    print("\n" + "="*80)
    print("4. VISUALIZING COMPRESSION QUALITY")
    print("="*80)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Explained variance
    ax1 = plt.subplot(1, 3, 1)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    ax1.plot(cumsum_var, linewidth=2, color='#1f77b4')
    ax1.axhline(y=0.99, color='red', linestyle='--', label='99% threshold')
    ax1.set_xlabel('Number of Components', fontsize=12)
    ax1.set_ylabel('Cumulative Variance Explained', fontsize=12)
    ax1.set_title('PCA Compression Quality', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Original vs compressed (first 100 samples)
    ax2 = plt.subplot(1, 3, 2)
    sample_idx = 0
    ax2.plot(original[sample_idx, :100], 'b-', label='Original (first 100 dims)', alpha=0.7)
    ax2.plot(compressed[sample_idx, :], 'r-', label=f'Compressed ({compressed.shape[1]} dims)', 
             linewidth=2)
    ax2.set_xlabel('Dimension Index', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Sample Embedding Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Compressed embeddings heatmap
    ax3 = plt.subplot(1, 3, 3)
    im = ax3.imshow(compressed[:50, :].T, aspect='auto', cmap='viridis')
    ax3.set_xlabel('Sample Index', fontsize=12)
    ax3.set_ylabel('Compressed Dimension', fontsize=12)
    ax3.set_title(f'Compressed Embeddings\n({compressed.shape[1]} dims)', fontweight='bold')
    plt.colorbar(im, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('quantum_encoding_preparation.png', dpi=300)
    print("✓ Visualization saved: quantum_encoding_preparation.png")


def generate_sample_quantum_circuit_pseudocode(angles):
    """
    Generate pseudocode for quantum circuit encoding
    """
    print("\n" + "="*80)
    print("5. SAMPLE QUANTUM CIRCUIT (Pseudocode)")
    print("="*80)
    
    n_features = angles.shape[1]
    sample_angles = angles[0, :]
    
    print(f"\n# Quantum Circuit for Traffic State Encoding")
    print(f"# Using {n_features} qubits (angle encoding)")
    print(f"\ndef encode_traffic_state(quantum_circuit, embedding_vector):")
    print(f"    '''")
    print(f"    Encode compressed traffic embedding into quantum state")
    print(f"    embedding_vector: {n_features}-dimensional numpy array")
    print(f"    '''")
    print(f"    # Initialize qubits")
    print(f"    qc = QuantumCircuit({n_features})")
    print(f"    ")
    print(f"    # Angle encoding: RY rotations")
    print(f"    for i in range({n_features}):")
    print(f"        angle = embedding_vector[i]")
    print(f"        qc.ry(angle, qubit=i)  # Rotate qubit i by angle")
    print(f"    ")
    print(f"    return qc")
    print(f"\n# Example for first traffic sample:")
    print(f"angles = {sample_angles[:5]} ... (truncated)")
    print(f"\n# Circuit:")
    for i in range(min(5, n_features)):
        print(f"qc.ry({sample_angles[i]:.4f}, qubit={i})")
    if n_features > 5:
        print(f"# ... ({n_features - 5} more qubits)")
    
    print(f"\n# This creates a quantum state |ψ⟩ that encodes the traffic pattern")
    print(f"# Next steps:")
    print(f"# 1. Apply QCNN layers (quantum convolution + pooling)")
    print(f"# 2. Measure output qubits")
    print(f"# 3. Use measurements for traffic classification/prediction")


def estimate_quantum_resources(n_qubits, circuit_depth):
    """
    Estimate quantum computing resources needed
    """
    print("\n" + "="*80)
    print("6. QUANTUM RESOURCE ESTIMATION")
    print("="*80)
    
    print(f"\n📊 For {n_qubits}-qubit circuit:")
    print(f"\nCircuit Parameters:")
    print(f"  Qubits: {n_qubits}")
    print(f"  Depth: ~{circuit_depth} gates")
    print(f"  Gate types: RY (rotation), CNOT (entanglement)")
    
    print(f"\nNISQ Device Compatibility:")
    devices = [
        ("IBM Quantum (free)", 5, "Limited", "❌"),
        ("IBM Quantum (paid)", 127, "High", "✅"),
        ("Rigetti Aspen", 80, "Medium", "✅" if n_qubits <= 80 else "❌"),
        ("IonQ", 32, "High", "✅" if n_qubits <= 32 else "❌"),
        ("Google Sycamore", 53, "Medium", "✅" if n_qubits <= 53 else "❌"),
    ]
    
    print(f"\n{'Device':<25} {'Qubits':<10} {'Fidelity':<12} {'Compatible'}")
    print("-" * 60)
    for device, qubits, fidelity, compatible in devices:
        print(f"{device:<25} {qubits:<10} {fidelity:<12} {compatible}")
    
    print(f"\nRecommendation:")
    if n_qubits <= 32:
        print("  ✅ NISQ-feasible with current hardware")
        print("  ✅ Can run on IonQ or IBM Quantum")
    elif n_qubits <= 53:
        print("  ⚠️  Requires larger NISQ devices")
        print("  ⚠️  Limited availability")
    else:
        print("  ❌ Beyond current NISQ capabilities")
        print("  ❌ Further compression needed")


def main():
    """
    Complete pipeline for quantum encoding preparation
    """
    # Load embeddings
    embeddings = load_and_analyze_embeddings()
    if embeddings is None:
        return
    
    # Compress using PCA
    n_components = 32  # NISQ-friendly dimension
    compressed, pca = compress_for_quantum(embeddings, n_components)
    
    # Prepare angle encoding (most NISQ-friendly)
    angles = prepare_quantum_encoding(compressed, encoding_type='angle')
    
    # Visualize quality
    visualize_compression_quality(embeddings['combined_embeddings'], 
                                  compressed, pca)
    
    # Generate circuit pseudocode
    generate_sample_quantum_circuit_pseudocode(angles)
    
    # Estimate resources
    estimate_quantum_resources(n_qubits=n_components, circuit_depth=n_components*2)
    
    # Save compressed embeddings for quantum pipeline
    print("\n" + "="*80)
    print("7. SAVING QUANTUM-READY EMBEDDINGS")
    print("="*80)
    
    np.savez_compressed('quantum_ready_embeddings.npz',
                       compressed_embeddings=compressed,
                       angle_encoded=angles,
                       pca_components=pca.components_,
                       pca_mean=pca.mean_,
                       explained_variance=pca.explained_variance_ratio_)
    
    print(f"\n✓ Saved quantum-ready embeddings to: quantum_ready_embeddings.npz")
    print(f"  - compressed_embeddings: {compressed.shape}")
    print(f"  - angle_encoded: {angles.shape}")
    print(f"  - PCA parameters for inference")
    
    print("\n" + "="*80)
    print("✅ QUANTUM ENCODING PREPARATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Install quantum framework: pip install qiskit")
    print("2. Implement quantum circuits using angle_encoded data")
    print("3. Build QCNN on top of encoded states")
    print("4. Train hybrid quantum-classical model")
    print("\nSee quantum_circuit_example.py for implementation")


if __name__ == "__main__":
    main()
