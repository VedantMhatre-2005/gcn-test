"""
Bloch Sphere Visualization for Traffic Embeddings
Represents all 32-dimensional embeddings as a single fluctuating point on a Bloch sphere
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform


class Arrow3D(FancyArrowPatch):
    """3D arrow for Bloch sphere visualization"""
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        
        return np.min(zs)


def draw_bloch_sphere(ax):
    """Draw the Bloch sphere wireframe"""
    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, alpha=0.1, color='cyan', edgecolor='gray', linewidth=0.1)
    
    # Draw axes
    axis_length = 1.3
    ax.plot([-axis_length, axis_length], [0, 0], [0, 0], 'k-', linewidth=0.5, alpha=0.3)
    ax.plot([0, 0], [-axis_length, axis_length], [0, 0], 'k-', linewidth=0.5, alpha=0.3)
    ax.plot([0, 0], [0, 0], [-axis_length, axis_length], 'k-', linewidth=0.5, alpha=0.3)
    
    # Draw equator and meridians
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 0, 'gray', linewidth=0.5, alpha=0.3)
    ax.plot(np.cos(theta), 0, np.sin(theta), 'gray', linewidth=0.5, alpha=0.3)
    ax.plot(0, np.cos(theta), np.sin(theta), 'gray', linewidth=0.5, alpha=0.3)
    
    # Labels
    ax.text(1.4, 0, 0, '|+⟩', fontsize=14, color='black')
    ax.text(-1.4, 0, 0, '|−⟩', fontsize=14, color='black')
    ax.text(0, 1.4, 0, '|+i⟩', fontsize=14, color='black')
    ax.text(0, -1.4, 0, '|−i⟩', fontsize=14, color='black')
    ax.text(0, 0, 1.4, '|0⟩', fontsize=14, color='black')
    ax.text(0, 0, -1.4, '|1⟩', fontsize=14, color='black')
    
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')


def embeddings_to_bloch_vector(embeddings, method='average'):
    """
    Convert 32-dimensional embeddings to a single point on Bloch sphere
    
    Methods:
    - 'average': Average all 32 qubit states
    - 'pca': Use first principal component
    - 'weighted': Weight by variance importance
    - 'phase': Map to spherical coordinates
    """
    n_samples = embeddings.shape[0]
    bloch_vectors = np.zeros((n_samples, 3))
    
    if method == 'average':
        # Method 1: Average of all qubit states
        # Each embedding dimension becomes a rotation angle
        for i in range(n_samples):
            angles = embeddings[i, :]  # 32 angles
            
            # Convert each angle to Bloch vector and average
            x_sum = np.mean(np.sin(angles))
            y_sum = np.mean(np.sin(angles) * np.cos(angles))
            z_sum = np.mean(np.cos(angles))
            
            # Normalize
            norm = np.sqrt(x_sum**2 + y_sum**2 + z_sum**2)
            if norm > 0:
                bloch_vectors[i] = [x_sum/norm, y_sum/norm, z_sum/norm]
    
    elif method == 'pca':
        # Method 2: Project to most important direction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        projected = pca.fit_transform(embeddings)
        
        # Normalize each to unit sphere
        for i in range(n_samples):
            norm = np.linalg.norm(projected[i])
            if norm > 0:
                bloch_vectors[i] = projected[i] / norm
    
    elif method == 'weighted':
        # Method 3: Weighted by importance
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
    
    elif method == 'phase':
        # Method 4: Map to traffic phase space
        # Use statistical properties to define angles
        for i in range(n_samples):
            vec = embeddings[i, :]
            
            # Theta: related to overall magnitude (congestion level)
            theta = np.pi * (np.mean(vec) / (2*np.pi))
            
            # Phi: related to variance (traffic pattern diversity)
            phi = 2 * np.pi * (np.std(vec) / (2*np.pi))
            
            # Convert to Cartesian
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            
            bloch_vectors[i] = [x, y, z]
    
    return bloch_vectors


def visualize_static_bloch(embeddings, traffic_levels=None, method='average', save_path='bloch_static.png'):
    """
    Create static visualization with multiple traffic states
    """
    print(f"\nCreating static Bloch sphere visualization (method: {method})...")
    
    # Convert embeddings to Bloch vectors
    bloch_vectors = embeddings_to_bloch_vector(embeddings, method=method)
    
    # Sample subset for visualization
    n_show = min(200, len(bloch_vectors))
    indices = np.linspace(0, len(bloch_vectors)-1, n_show, dtype=int)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Bloch sphere
    draw_bloch_sphere(ax)
    
    # Plot trajectory
    x = bloch_vectors[indices, 0]
    y = bloch_vectors[indices, 1]
    z = bloch_vectors[indices, 2]
    
    # Color by time or traffic level
    if traffic_levels is not None:
        colors = traffic_levels[indices]
        label = 'Traffic Level'
    else:
        colors = indices
        label = 'Time Step'
    
    scatter = ax.scatter(x, y, z, c=colors, cmap='RdYlGn_r', 
                        s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Draw trajectory line
    ax.plot(x, y, z, 'b-', alpha=0.3, linewidth=1)
    
    # Highlight start and end
    ax.scatter([x[0]], [y[0]], [z[0]], c='green', s=200, marker='o', 
              edgecolors='black', linewidth=2, label='Start', zorder=10)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], c='red', s=200, marker='s', 
              edgecolors='black', linewidth=2, label='End', zorder=10)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label(label, fontsize=12)
    
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title(f'Traffic Embeddings on Bloch Sphere\nMapping Method: {method.capitalize()}', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_animated_bloch(embeddings, traffic_levels=None, method='average', 
                         save_path='bloch_animation.gif', fps=10):
    """
    Create animated Bloch sphere showing fluctuating quantum state
    """
    print(f"\nCreating animated Bloch sphere (method: {method})...")
    print("This may take a minute...")
    
    # Convert embeddings to Bloch vectors
    bloch_vectors = embeddings_to_bloch_vector(embeddings, method=method)
    
    # Sample for animation (use every Nth frame for speed)
    skip = max(1, len(bloch_vectors) // 200)
    bloch_vectors = bloch_vectors[::skip]
    if traffic_levels is not None:
        traffic_levels = traffic_levels[::skip]
    
    n_frames = len(bloch_vectors)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize
    draw_bloch_sphere(ax)
    
    # Initial state vector
    state_vec = bloch_vectors[0]
    
    # Create arrow for state vector
    arrow = Arrow3D(0, 0, 0, state_vec[0], state_vec[1], state_vec[2],
                   mutation_scale=20, lw=3, arrowstyle="-|>", color="red")
    ax.add_artist(arrow)
    
    # Point at tip
    point = ax.scatter([state_vec[0]], [state_vec[1]], [state_vec[2]], 
                      c='red', s=200, marker='o', edgecolors='black', linewidth=2)
    
    # Trail
    trail_length = 30
    trail, = ax.plot([], [], [], 'r-', alpha=0.5, linewidth=2)
    trail_points = ax.scatter([], [], [], c='red', s=20, alpha=0.3)
    
    # Text info
    info_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, 
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Title
    title = ax.set_title('Quantum-Encoded Traffic State\nFluctuating on Bloch Sphere', 
                        fontsize=14, fontweight='bold', pad=20)
    
    def init():
        return arrow, point, trail, trail_points, info_text, title
    
    def update(frame):
        # Update state vector
        state = bloch_vectors[frame]
        
        # Update arrow
        arrow._dxdydz = (state[0], state[1], state[2])
        
        # Update point
        point._offsets3d = ([state[0]], [state[1]], [state[2]])
        
        # Color by traffic level
        if traffic_levels is not None:
            color = plt.cm.RdYlGn_r(traffic_levels[frame] / traffic_levels.max())
            point.set_color([color])
            arrow.set_color(color)
        
        # Update trail
        start_idx = max(0, frame - trail_length)
        trail_x = bloch_vectors[start_idx:frame+1, 0]
        trail_y = bloch_vectors[start_idx:frame+1, 1]
        trail_z = bloch_vectors[start_idx:frame+1, 2]
        trail.set_data_3d(trail_x, trail_y, trail_z)
        trail_points._offsets3d = (trail_x, trail_y, trail_z)
        
        # Update text
        theta = np.arccos(np.clip(state[2], -1, 1))
        phi = np.arctan2(state[1], state[0])
        
        info = f'Frame: {frame}/{n_frames}\\n'
        info += f'θ = {np.degrees(theta):.1f}°\\n'
        info += f'φ = {np.degrees(phi):.1f}°\\n'
        if traffic_levels is not None:
            info += f'Traffic: {traffic_levels[frame]:.1f}'
        
        info_text.set_text(info)
        
        # Rotate view slightly for dynamic effect
        ax.view_init(elev=20, azim=frame * 0.5)
        
        return arrow, point, trail, trail_points, info_text, title
    
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=1000//fps, blit=False, repeat=True)
    
    # Save animation
    print("Saving animation (this takes time)...")
    anim.save(save_path, writer='pillow', fps=fps, dpi=100)
    print(f"✓ Saved: {save_path}")
    plt.close()


def compare_methods(embeddings, save_path='bloch_comparison.png'):
    """
    Compare different mapping methods side by side
    """
    print("\nCreating method comparison visualization...")
    
    methods = ['average', 'pca', 'weighted', 'phase']
    
    fig = plt.figure(figsize=(16, 12))
    
    for idx, method in enumerate(methods):
        ax = fig.add_subplot(2, 2, idx+1, projection='3d')
        
        # Draw Bloch sphere
        draw_bloch_sphere(ax)
        
        # Convert and plot
        bloch_vectors = embeddings_to_bloch_vector(embeddings[:100], method=method)
        
        x = bloch_vectors[:, 0]
        y = bloch_vectors[:, 1]
        z = bloch_vectors[:, 2]
        
        colors = np.arange(len(x))
        scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', 
                           s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.plot(x, y, z, 'gray', alpha=0.3, linewidth=1)
        
        ax.set_title(f'{method.capitalize()} Method', fontsize=12, fontweight='bold')
        ax.view_init(elev=20, azim=45)
    
    plt.suptitle('Comparison of Embedding-to-Bloch Mapping Methods', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def analyze_bloch_trajectory(bloch_vectors, embeddings_orig):
    """
    Analyze the Bloch sphere trajectory
    """
    print("\n" + "="*80)
    print("BLOCH SPHERE TRAJECTORY ANALYSIS")
    print("="*80)
    
    # Calculate statistics
    distances = np.sqrt(np.sum(bloch_vectors**2, axis=1))
    
    print(f"\nTrajectory Statistics:")
    print(f"  Number of states: {len(bloch_vectors)}")
    print(f"  Mean distance from origin: {distances.mean():.4f}")
    print(f"  Std distance from origin: {distances.std():.4f}")
    print(f"  Range: [{distances.min():.4f}, {distances.max():.4f}]")
    
    # Movement analysis
    movements = np.diff(bloch_vectors, axis=0)
    step_sizes = np.sqrt(np.sum(movements**2, axis=1))
    
    print(f"\nMovement Statistics:")
    print(f"  Mean step size: {step_sizes.mean():.4f}")
    print(f"  Max step size: {step_sizes.max():.4f}")
    print(f"  Total path length: {step_sizes.sum():.2f}")
    
    # Pole preferences
    z_values = bloch_vectors[:, 2]
    print(f"\nPole Distribution:")
    print(f"  Time at |0⟩ pole (z>0.5): {(z_values > 0.5).sum() / len(z_values) * 100:.1f}%")
    print(f"  Time at |1⟩ pole (z<-0.5): {(z_values < -0.5).sum() / len(z_values) * 100:.1f}%")
    print(f"  Time at equator (|z|<0.5): {(np.abs(z_values) < 0.5).sum() / len(z_values) * 100:.1f}%")


def main():
    """
    Main execution
    """
    print("="*80)
    print("BLOCH SPHERE VISUALIZATION OF TRAFFIC EMBEDDINGS")
    print("="*80)
    
    # Load quantum-ready embeddings
    print("\nLoading embeddings...")
    try:
        data = np.load('quantum_ready_embeddings.npz')
        angle_encoded = data['angle_encoded']
        print(f"✓ Loaded {angle_encoded.shape[0]} samples with {angle_encoded.shape[1]} dimensions")
    except FileNotFoundError:
        print("✗ quantum_ready_embeddings.npz not found!")
        print("  Run: python prepare_quantum_encoding.py first")
        return
    
    # Load traffic levels for color coding
    try:
        traffic_data = np.load('traffic_embeddings.npz')
        predictions = traffic_data['predictions']
        # Use mean traffic across all nodes as color
        traffic_levels = predictions[:, 0, :].mean(axis=1)
        print(f"✓ Loaded traffic levels for color coding")
    except:
        traffic_levels = None
        print("  (No traffic levels available for coloring)")
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Compare methods
    compare_methods(angle_encoded, 'bloch_comparison.png')
    
    # 2. Static visualization with best method
    best_method = 'weighted'
    visualize_static_bloch(angle_encoded, traffic_levels, method=best_method, 
                          save_path='bloch_static.png')
    
    # 3. Analyze trajectory
    bloch_vectors = embeddings_to_bloch_vector(angle_encoded, method=best_method)
    analyze_bloch_trajectory(bloch_vectors, angle_encoded)
    
    # 4. Create animation
    print("\n" + "="*80)
    print("CREATING ANIMATION")
    print("="*80)
    create_animated_bloch(angle_encoded, traffic_levels, method=best_method,
                         save_path='bloch_animation.gif', fps=15)
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. bloch_comparison.png - Comparison of 4 mapping methods")
    print("  2. bloch_static.png - Static trajectory on Bloch sphere")
    print("  3. bloch_animation.gif - Animated fluctuating quantum state")
    print("\nInterpretation:")
    print("  • The point on the sphere represents the 'quantum traffic state'")
    print("  • Movement shows how traffic patterns evolve")
    print("  • Position encodes compressed information from all 32 qubits")
    print("  • Color indicates traffic congestion level")
    print("="*80)


if __name__ == "__main__":
    main()
