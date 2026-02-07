"""
Authentic Bloch Sphere Visualization using Qiskit
Uses Qiskit's native Bloch sphere tools for professional quantum visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
try:
    from qiskit.visualization import plot_bloch_vector, plot_bloch_multivector
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠️  Qiskit not installed. Please run: pip install qiskit qiskit-aer")


def embeddings_to_bloch_vector(embeddings, method='weighted'):
    """
    Convert 32-dimensional embeddings to Bloch vectors
    Same as before but optimized for Qiskit
    """
    n_samples = embeddings.shape[0]
    bloch_vectors = np.zeros((n_samples, 3))
    
    if method == 'average':
        for i in range(n_samples):
            angles = embeddings[i, :]
            x_sum = np.mean(np.sin(angles))
            y_sum = np.mean(np.sin(angles) * np.cos(angles))
            z_sum = np.mean(np.cos(angles))
            norm = np.sqrt(x_sum**2 + y_sum**2 + z_sum**2)
            if norm > 0:
                bloch_vectors[i] = [x_sum/norm, y_sum/norm, z_sum/norm]
    
    elif method == 'weighted':
        variance = np.var(embeddings, axis=0)
        weights = variance / variance.sum()
        
        for i in range(n_samples):
            angles = embeddings[i, :]
            x_sum = np.sum(weights * np.sin(angles))
            y_sum = np.sum(weights * np.sin(angles) * np.cos(angles))
            z_sum = np.sum(weights * np.cos(angles))
            norm = np.sqrt(x_sum**2 + y_sum**2 + z_sum**2)
            if norm > 0:
                bloch_vectors[i] = [x_sum/norm, y_sum/norm, z_sum/norm]
    
    elif method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        projected = pca.fit_transform(embeddings)
        for i in range(n_samples):
            norm = np.linalg.norm(projected[i])
            if norm > 0:
                bloch_vectors[i] = projected[i] / norm
    
    elif method == 'phase':
        for i in range(n_samples):
            vec = embeddings[i, :]
            theta = np.pi * (np.mean(vec) / (2*np.pi))
            phi = 2 * np.pi * (np.std(vec) / (2*np.pi))
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            bloch_vectors[i] = [x, y, z]
    
    return bloch_vectors


def bloch_vector_to_statevector(bloch_vector):
    """
    Convert Bloch vector [x, y, z] to quantum Statevector
    For Qiskit's plot_bloch_multivector
    """
    x, y, z = bloch_vector
    
    # Bloch vector to qubit state
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    
    # State: cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    
    return Statevector([alpha, beta])


def visualize_single_state_qiskit(bloch_vector, title="Traffic State", save_path=None):
    """
    Visualize a single traffic state using Qiskit's plot_bloch_vector
    """
    if not QISKIT_AVAILABLE:
        print("Qiskit not available!")
        return
    
    fig = plot_bloch_vector(bloch_vector, title=title, figsize=(8, 8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_multiple_states_qiskit(bloch_vectors, titles=None, save_path='qiskit_bloch_grid.png'):
    """
    Visualize multiple traffic states in a grid using Qiskit
    """
    if not QISKIT_AVAILABLE:
        print("Qiskit not available!")
        return
    
    print(f"\nCreating Qiskit Bloch sphere grid...")
    
    # Select key frames to show
    n_states = min(9, len(bloch_vectors))  # 3x3 grid
    indices = np.linspace(0, len(bloch_vectors)-1, n_states, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    for idx, (ax_idx, sample_idx) in enumerate(zip(range(n_states), indices)):
        vector = bloch_vectors[sample_idx]
        
        # Create Bloch sphere on this axis
        ax = axes[ax_idx]
        
        # Use Qiskit's plot_bloch_vector on specific axis
        from qiskit.visualization.bloch import Bloch
        bloch = Bloch(axes=ax)
        bloch.add_vectors(vector)
        bloch.vector_color = ['red']
        bloch.render()
        
        # Set title on the axis
        if titles:
            ax.set_title(titles[idx], fontsize=10, fontweight='bold')
        else:
            ax.set_title(f'State {sample_idx}', fontsize=10, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_states, 9):
        axes[idx].axis('off')
    
    plt.suptitle('Traffic States on Qiskit Bloch Spheres', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_qiskit_trajectory(bloch_vectors, traffic_levels=None, save_path='qiskit_trajectory.png'):
    """
    Show trajectory using Qiskit's Bloch class with custom rendering
    """
    if not QISKIT_AVAILABLE:
        print("Qiskit not available!")
        return
    
    print(f"\nCreating Qiskit Bloch trajectory...")
    
    from qiskit.visualization.bloch import Bloch
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create Bloch sphere
    bloch = Bloch(axes=ax)
    
    # Add all vectors as a trajectory
    sample_every = max(1, len(bloch_vectors) // 100)
    sampled_vectors = bloch_vectors[::sample_every]
    
    # Add vectors
    for vector in sampled_vectors:
        bloch.add_points(vector)
    
    bloch.point_color = ['blue']
    bloch.point_marker = ['o']
    bloch.point_size = [20]
    
    # Highlight start and end
    bloch.add_vectors([bloch_vectors[0]])
    bloch.add_vectors([bloch_vectors[-1]])
    
    bloch.vector_color = ['green', 'red']
    bloch.vector_width = 3
    
    ax.set_title('Traffic Embedding Trajectory on Qiskit Bloch Sphere', 
                fontsize=14, fontweight='bold', pad=20)
    
    bloch.render()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_qiskit_animation(bloch_vectors, traffic_levels=None, 
                            save_path='qiskit_animation.gif', fps=10):
    """
    Create animation using Qiskit Bloch sphere
    """
    if not QISKIT_AVAILABLE:
        print("Qiskit not available!")
        return
    
    print(f"\nCreating Qiskit animated Bloch sphere...")
    print("This may take a minute...")
    
    from qiskit.visualization.bloch import Bloch
    
    # Sample frames
    skip = max(1, len(bloch_vectors) // 200)
    bloch_vectors = bloch_vectors[::skip]
    if traffic_levels is not None:
        traffic_levels = traffic_levels[::skip]
    
    n_frames = len(bloch_vectors)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize Bloch sphere
    bloch = Bloch(axes=ax)
    
    # Trail length
    trail_length = 30
    
    def init():
        return fig,
    
    def update(frame):
        ax.clear()
        
        # Recreate Bloch sphere
        bloch = Bloch(axes=ax)
        
        # Current state
        current_vector = bloch_vectors[frame]
        
        # Color by traffic level
        if traffic_levels is not None:
            color_val = traffic_levels[frame] / traffic_levels.max()
            color = plt.cm.RdYlGn_r(color_val)
            color_rgb = color[:3]  # RGB only
            bloch.vector_color = [color_rgb]
        else:
            bloch.vector_color = ['red']
        
        bloch.vector_width = 3
        bloch.add_vectors([current_vector])
        
        # Skip trail for now to avoid Qiskit formatting issues
        # (Trail works in static visualizations but has issues with animation)
        
        # Add title with info
        theta = np.arccos(np.clip(current_vector[2], -1, 1))
        phi = np.arctan2(current_vector[1], current_vector[0])
        
        title = f'Quantum-Encoded Traffic State (Frame {frame}/{n_frames})\n'
        title += f'θ={np.degrees(theta):.1f}°, φ={np.degrees(phi):.1f}°'
        if traffic_levels is not None:
            title += f', Traffic={traffic_levels[frame]:.1f}'
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        
        # Rotate view
        ax.view_init(elev=20, azim=frame * 0.5)
        
        bloch.render()
        
        return fig,
    
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=1000//fps, blit=False, repeat=True)
    
    print("Saving animation...")
    anim.save(save_path, writer='pillow', fps=fps, dpi=100)
    print(f"✓ Saved: {save_path}")
    plt.close()


def demonstrate_quantum_state_evolution(embeddings, save_path='qiskit_state_evolution.png'):
    """
    Show how quantum state evolves over time using actual Statevectors
    """
    if not QISKIT_AVAILABLE:
        print("Qiskit not available!")
        return
    
    print(f"\nCreating quantum state evolution visualization...")
    
    # Convert to Bloch vectors
    bloch_vectors = embeddings_to_bloch_vector(embeddings[:200], method='weighted')
    
    # Select key states
    indices = [0, 50, 100, 150, 199]
    states = [bloch_vectors[i] for i in indices]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), subplot_kw={'projection': '3d'})
    
    from qiskit.visualization.bloch import Bloch
    
    for idx, (ax, state, frame) in enumerate(zip(axes, states, indices)):
        bloch = Bloch(axes=ax)
        bloch.add_vectors([state])
        bloch.vector_color = ['red']
        bloch.vector_width = 3
        bloch.render()
        
        # Calculate state parameters
        theta = np.arccos(np.clip(state[2], -1, 1))
        phi = np.arctan2(state[1], state[0])
        
        title = f't={frame*5}s\nθ={np.degrees(theta):.0f}°'
        ax.set_title(title, fontsize=10, fontweight='bold')
    
    plt.suptitle('Quantum State Evolution of Traffic Network', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def compare_qiskit_vs_matplotlib(bloch_vector, save_path='qiskit_vs_matplotlib.png'):
    """
    Side-by-side comparison of Qiskit vs matplotlib rendering
    """
    if not QISKIT_AVAILABLE:
        print("Qiskit not available!")
        return
    
    print(f"\nCreating Qiskit vs Matplotlib comparison...")
    
    from qiskit.visualization.bloch import Bloch
    
    fig = plt.figure(figsize=(16, 7))
    
    # Qiskit version
    ax1 = fig.add_subplot(121, projection='3d')
    bloch = Bloch(axes=ax1)
    bloch.add_vectors([bloch_vector])
    bloch.vector_color = ['red']
    bloch.vector_width = 3
    bloch.render()
    ax1.set_title('Qiskit Bloch Sphere\n(Authentic Quantum Visualization)', 
                 fontsize=12, fontweight='bold')
    
    # Matplotlib version (simple)
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Draw basic sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_surface(x, y, z, alpha=0.1, color='cyan')
    
    # Draw vector
    ax2.quiver(0, 0, 0, bloch_vector[0], bloch_vector[1], bloch_vector[2],
              color='red', arrow_length_ratio=0.1, linewidth=3)
    
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])
    ax2.set_title('Matplotlib Bloch Sphere\n(Custom Implementation)', 
                 fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    plt.suptitle('Comparison: Qiskit Native vs Custom Matplotlib', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """
    Main execution using Qiskit
    """
    print("="*80)
    print("AUTHENTIC QISKIT BLOCH SPHERE VISUALIZATION")
    print("="*80)
    
    if not QISKIT_AVAILABLE:
        print("\n❌ Qiskit is not installed!")
        print("\nPlease install Qiskit:")
        print("  pip install qiskit qiskit-aer")
        print("\nThen run this script again.")
        return
    
    print("\n✓ Qiskit is available!")
    
    # Load embeddings
    print("\nLoading embeddings...")
    try:
        data = np.load('quantum_ready_embeddings.npz')
        angle_encoded = data['angle_encoded']
        print(f"✓ Loaded {angle_encoded.shape[0]} samples")
    except FileNotFoundError:
        print("✗ quantum_ready_embeddings.npz not found!")
        print("  Run: python prepare_quantum_encoding.py first")
        return
    
    # Load traffic levels
    try:
        traffic_data = np.load('traffic_embeddings.npz')
        predictions = traffic_data['predictions']
        traffic_levels = predictions[:, 0, :].mean(axis=1)
        print(f"✓ Loaded traffic levels")
    except:
        traffic_levels = None
    
    # Convert to Bloch vectors
    print("\nConverting embeddings to Bloch vectors...")
    bloch_vectors = embeddings_to_bloch_vector(angle_encoded, method='weighted')
    print(f"✓ Converted {len(bloch_vectors)} states")
    
    print("\n" + "="*80)
    print("GENERATING QISKIT VISUALIZATIONS")
    print("="*80)
    
    # 1. Single state example
    print("\n1. Single traffic state...")
    visualize_single_state_qiskit(bloch_vectors[0], 
                                  title="Traffic State (t=0)", 
                                  save_path='qiskit_single_state.png')
    
    # 2. Grid of states
    print("\n2. Grid of traffic states...")
    visualize_multiple_states_qiskit(bloch_vectors, save_path='qiskit_bloch_grid.png')
    
    # 3. Trajectory
    print("\n3. Trajectory on Bloch sphere...")
    create_qiskit_trajectory(bloch_vectors, traffic_levels, 
                            save_path='qiskit_trajectory.png')
    
    # 4. State evolution
    print("\n4. Quantum state evolution...")
    demonstrate_quantum_state_evolution(angle_encoded, 
                                       save_path='qiskit_state_evolution.png')
    
    # 5. Comparison
    print("\n5. Qiskit vs Matplotlib comparison...")
    compare_qiskit_vs_matplotlib(bloch_vectors[100], 
                                save_path='qiskit_vs_matplotlib.png')
    
    # 6. Animation
    print("\n6. Creating animation with Qiskit...")
    create_qiskit_animation(bloch_vectors, traffic_levels,
                           save_path='qiskit_animation.gif', fps=15)
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print("\nGenerated Qiskit visualizations:")
    print("  1. qiskit_single_state.png - Single traffic state")
    print("  2. qiskit_bloch_grid.png - 3x3 grid of states")
    print("  3. qiskit_trajectory.png - Full trajectory")
    print("  4. qiskit_state_evolution.png - Evolution over time")
    print("  5. qiskit_vs_matplotlib.png - Quality comparison")
    print("  6. qiskit_animation.gif - Animated quantum state")
    print("\n✨ Using Qiskit's authentic quantum visualization tools!")
    print("="*80)


if __name__ == "__main__":
    main()
