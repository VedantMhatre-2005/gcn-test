"""
Generate MP4 Video of Bloch Sphere Visualization
Uses embeddings from Unified Spatiotemporal GCN to create animated Bloch sphere
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from sklearn.decomposition import PCA


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
    
    ax.plot_surface(x, y, z, alpha=0.08, color='cyan', edgecolor='gray', linewidth=0.1)
    
    # Draw axes
    axis_length = 1.3
    ax.plot([-axis_length, axis_length], [0, 0], [0, 0], 'k-', linewidth=1, alpha=0.4)
    ax.plot([0, 0], [-axis_length, axis_length], [0, 0], 'k-', linewidth=1, alpha=0.4)
    ax.plot([0, 0], [0, 0], [-axis_length, axis_length], 'k-', linewidth=1, alpha=0.4)
    
    # Draw equator and meridians
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 0, 'gray', linewidth=0.8, alpha=0.3)
    ax.plot(np.cos(theta), 0, np.sin(theta), 'gray', linewidth=0.8, alpha=0.3)
    ax.plot(0, np.cos(theta), np.sin(theta), 'gray', linewidth=0.8, alpha=0.3)
    
    # Labels - quantum basis states
    ax.text(1.5, 0, 0, '|+⟩', fontsize=16, color='blue', weight='bold')
    ax.text(-1.5, 0, 0, '|−⟩', fontsize=16, color='blue', weight='bold')
    ax.text(0, 1.5, 0, '|+i⟩', fontsize=16, color='green', weight='bold')
    ax.text(0, -1.5, 0, '|−i⟩', fontsize=16, color='green', weight='bold')
    ax.text(0, 0, 1.5, '|0⟩', fontsize=16, color='red', weight='bold')
    ax.text(0, 0, -1.5, '|1⟩', fontsize=16, color='red', weight='bold')
    
    ax.set_xlim([-1.6, 1.6])
    ax.set_ylim([-1.6, 1.6])
    ax.set_zlim([-1.6, 1.6])
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')


def embeddings_to_bloch_vectors(embeddings, method='pca_smooth'):
    """
    Convert high-dimensional embeddings to Bloch sphere coordinates
    
    Args:
        embeddings: (n_timesteps, n_features) array of embeddings
        method: 'pca_smooth', 'spherical', or 'trigonometric'
    
    Returns:
        bloch_vectors: (n_timesteps, 3) array of (x, y, z) coordinates on unit sphere
    """
    n_samples, n_features = embeddings.shape
    
    if method == 'pca_smooth':
        # Use PCA to reduce to 3D, then normalize to unit sphere
        # Add smoothing for better visualization
        pca = PCA(n_components=3)
        projected = pca.fit_transform(embeddings)
        
        # Apply smoothing filter
        from scipy.ndimage import gaussian_filter1d
        projected_smooth = np.zeros_like(projected)
        for i in range(3):
            projected_smooth[:, i] = gaussian_filter1d(projected[:, i], sigma=1.5)
        
        # Normalize to unit sphere
        bloch_vectors = np.zeros((n_samples, 3))
        for i in range(n_samples):
            norm = np.linalg.norm(projected_smooth[i])
            if norm > 0:
                bloch_vectors[i] = projected_smooth[i] / norm
            else:
                bloch_vectors[i] = [0, 0, 1]  # Default to |0⟩ state
    
    elif method == 'spherical':
        # Map to spherical coordinates using first two PCA components
        pca = PCA(n_components=2)
        projected = pca.fit_transform(embeddings)
        
        # Normalize to [0, 2π] and [0, π]
        theta = (projected[:, 0] - projected[:, 0].min()) / (projected[:, 0].max() - projected[:, 0].min()) * 2 * np.pi
        phi = (projected[:, 1] - projected[:, 1].min()) / (projected[:, 1].max() - projected[:, 1].min()) * np.pi
        
        # Convert to Cartesian
        bloch_vectors = np.zeros((n_samples, 3))
        bloch_vectors[:, 0] = np.sin(phi) * np.cos(theta)
        bloch_vectors[:, 1] = np.sin(phi) * np.sin(theta)
        bloch_vectors[:, 2] = np.cos(phi)
    
    elif method == 'trigonometric':
        # Use trigonometric mapping of embedding dimensions
        bloch_vectors = np.zeros((n_samples, 3))
        
        for i in range(n_samples):
            emb = embeddings[i, :]
            
            # Map to angles
            angles = np.tanh(emb)  # Normalize to [-1, 1]
            
            # Compute Bloch coordinates
            x = np.mean(np.sin(angles * np.pi))
            y = np.mean(np.cos(angles * np.pi) * np.sin(angles * np.pi))
            z = np.mean(np.cos(angles * np.pi))
            
            # Normalize
            norm = np.sqrt(x**2 + y**2 + z**2)
            if norm > 0:
                bloch_vectors[i] = [x/norm, y/norm, z/norm]
            else:
                bloch_vectors[i] = [0, 0, 1]
    
    return bloch_vectors


def create_bloch_sphere_video(data_file='unified_stgcn_bloch_data.npz', 
                              output_file='bloch_sphere_stgcn.mp4',
                              fps=15, method='pca_smooth'):
    """
    Create MP4 video of Bloch sphere visualization
    
    Args:
        data_file: Path to npz file with embeddings
        output_file: Path for output MP4 file
        fps: Frames per second for video
        method: Method for converting embeddings to Bloch coordinates
    """
    print("=" * 80)
    print("GENERATING BLOCH SPHERE VIDEO")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from {data_file}...")
    data = np.load(data_file)
    embeddings = data['embeddings']
    traffic_data = data['traffic_data']
    timestamps = data['timestamps']
    
    n_frames = len(embeddings)
    print(f"  Number of frames: {n_frames}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Duration: {n_frames} seconds")
    
    # Convert embeddings to Bloch vectors
    print(f"\nConverting embeddings to Bloch sphere coordinates (method: {method})...")
    bloch_vectors = embeddings_to_bloch_vectors(embeddings, method=method)
    print(f"  Generated {len(bloch_vectors)} Bloch vectors")
    
    # Create figure
    print("\nCreating animation...")
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    
    # Main plot - Bloch sphere
    ax_main = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Side plot - Traffic data
    ax_traffic = fig.add_subplot(2, 2, 2)
    ax_coords = fig.add_subplot(2, 2, 4)
    
    # Draw Bloch sphere
    draw_bloch_sphere(ax_main)
    
    # Initialize plots
    state_point, = ax_main.plot([], [], [], 'ro', markersize=15, label='Traffic State')
    state_trail, = ax_main.plot([], [], [], 'r-', linewidth=2, alpha=0.4, label='State Trajectory')
    state_vector = None
    
    # Traffic plot
    ax_traffic.set_xlim(0, n_frames)
    ax_traffic.set_ylim(0, np.max(traffic_data) * 1.1)
    ax_traffic.set_xlabel('Time (seconds)', fontsize=11)
    ax_traffic.set_ylabel('Total Traffic (vehicles)', fontsize=11)
    ax_traffic.set_title('Network-wide Traffic Flow', fontsize=12, weight='bold')
    ax_traffic.grid(True, alpha=0.3)
    
    traffic_line, = ax_traffic.plot([], [], 'b-', linewidth=2)
    traffic_point, = ax_traffic.plot([], [], 'ro', markersize=8)
    
    # Coordinates plot
    ax_coords.set_xlim(0, n_frames)
    ax_coords.set_ylim(-1.1, 1.1)
    ax_coords.set_xlabel('Time (seconds)', fontsize=11)
    ax_coords.set_ylabel('Coordinate Value', fontsize=11)
    ax_coords.set_title('Bloch Sphere Coordinates', fontsize=12, weight='bold')
    ax_coords.grid(True, alpha=0.3)
    
    x_line, = ax_coords.plot([], [], 'r-', linewidth=2, label='X', alpha=0.7)
    y_line, = ax_coords.plot([], [], 'g-', linewidth=2, label='Y', alpha=0.7)
    z_line, = ax_coords.plot([], [], 'b-', linewidth=2, label='Z', alpha=0.7)
    ax_coords.legend(loc='upper right', fontsize=9)
    
    # Title
    title_text = fig.suptitle('', fontsize=16, weight='bold', y=0.98)
    
    # Trajectory history
    trail_length = 15
    trail_x, trail_y, trail_z = [], [], []
    
    def init():
        state_point.set_data([], [])
        state_point.set_3d_properties([])
        state_trail.set_data([], [])
        state_trail.set_3d_properties([])
        traffic_line.set_data([], [])
        traffic_point.set_data([], [])
        x_line.set_data([], [])
        y_line.set_data([], [])
        z_line.set_data([], [])
        return state_point, state_trail, traffic_line, traffic_point, x_line, y_line, z_line
    
    def update(frame):
        nonlocal state_vector, trail_x, trail_y, trail_z
        
        # Get current Bloch vector
        x, y, z = bloch_vectors[frame]
        
        # Update state point
        state_point.set_data([x], [y])
        state_point.set_3d_properties([z])
        
        # Update trail
        trail_x.append(x)
        trail_y.append(y)
        trail_z.append(z)
        if len(trail_x) > trail_length:
            trail_x.pop(0)
            trail_y.pop(0)
            trail_z.pop(0)
        
        state_trail.set_data(trail_x, trail_y)
        state_trail.set_3d_properties(trail_z)
        
        # Remove old vector arrow if exists
        if state_vector is not None:
            state_vector.remove()
        
        # Draw new vector arrow from origin
        state_vector = Arrow3D(0, 0, 0, x, y, z,
                              mutation_scale=20, lw=3, arrowstyle='-|>',
                              color='red', alpha=0.8)
        ax_main.add_artist(state_vector)
        
        # Update traffic plot
        total_traffic = traffic_data[:frame+1].sum(axis=1)
        time_range = np.arange(frame+1)
        traffic_line.set_data(time_range, total_traffic)
        traffic_point.set_data([frame], [total_traffic[-1]])
        
        # Update coordinates plot
        x_line.set_data(time_range, bloch_vectors[:frame+1, 0])
        y_line.set_data(time_range, bloch_vectors[:frame+1, 1])
        z_line.set_data(time_range, bloch_vectors[:frame+1, 2])
        
        # Update title with time and state info
        title_text.set_text(
            f'Unified Spatiotemporal GCN - Bloch Sphere Visualization | '
            f'Time: {frame}s | '
            f'State: (x={x:.3f}, y={y:.3f}, z={z:.3f})'
        )
        
        # Rotate view slightly for dynamic effect
        ax_main.view_init(elev=20, azim=frame * 0.5)
        
        return state_point, state_trail, traffic_line, traffic_point, x_line, y_line, z_line
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        blit=False, interval=1000/fps, repeat=True)
    
    # Save as MP4
    print(f"\nSaving video to {output_file}...")
    print(f"  FPS: {fps}")
    print(f"  Total duration: {n_frames/fps:.1f} seconds")
    print("  This may take a few minutes...")
    
    writer = FFMpegWriter(fps=fps, bitrate=3000, 
                         metadata={'title': 'Bloch Sphere - Unified ST-GCN',
                                  'artist': 'Traffic Prediction GCN'})
    
    anim.save(output_file, writer=writer, dpi=150)
    
    print(f"\n✓ Video saved successfully!")
    print(f"  Output file: {output_file}")
    import os
    print(f"  File size: {np.round(os.path.getsize(output_file) / 1024 / 1024, 2)} MB")
    print("=" * 80)
    
    plt.close()


if __name__ == "__main__":
    import sys
    
    # Check if scipy is available
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        print("Warning: scipy not found. Install with: pip install scipy")
        print("Proceeding without smoothing filter...")
    
    # Parse arguments
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = 'bloch_sphere_stgcn.mp4'
    
    # Generate video
    create_bloch_sphere_video(
        data_file='unified_stgcn_bloch_data.npz',
        output_file=output_file,
        fps=15,
        method='pca_smooth'
    )
    
    print(f"\n🎬 Video generation complete!")
    print(f"   You can now play: {output_file}")
