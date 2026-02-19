"""
Large Traffic Data Generator for Extended Visualizations
Generates extensive synthetic traffic data suitable for 1-minute+ Bloch sphere visualizations
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from chembur_network import ChemburTrafficNetwork


class LargeTrafficDataGenerator:
    """
    Generate large-scale synthetic traffic data with realistic patterns
    Designed for extended visualizations (e.g., 1-minute Bloch sphere animations)
    """
    def __init__(self, network, time_interval=1):
        """
        Args:
            network: ChemburTrafficNetwork instance
            time_interval: Time interval in seconds between measurements (default: 1 second for finer granularity)
        """
        self.network = network
        self.num_nodes = network.graph.number_of_nodes()
        self.time_interval = time_interval  # 1 second for finer resolution
        self.steps_per_minute = 60 // time_interval  # 60 steps per minute
        self.steps_per_hour = 3600 // time_interval  # 3600 steps per hour
        
    def _get_base_traffic_pattern(self, hour, day_of_week, minute, second):
        """
        Get base traffic multiplier based on detailed time
        Includes micro-variations within minutes
        """
        is_weekend = day_of_week >= 5
        
        # Base hourly pattern
        if not is_weekend:
            if 7 <= hour < 10:  # Morning rush hour
                base = 0.9
                # Peak at 8:30 AM
                if hour == 8 and 20 <= minute < 40:
                    base = 0.95
            elif 10 <= hour < 17:  # Normal working hours
                base = 0.6
            elif 17 <= hour < 20:  # Evening rush hour
                base = 0.95
                # Peak at 6:00 PM
                if hour == 18 and minute < 30:
                    base = 1.0
            elif 20 <= hour < 23:  # Evening
                base = 0.5
            else:  # Night
                base = 0.2
        else:
            if 10 <= hour < 13:  # Late morning
                base = 0.6
            elif 13 <= hour < 18:  # Afternoon
                base = 0.7
            elif 18 <= hour < 22:  # Evening
                base = 0.65
            else:  # Night/early morning
                base = 0.25
        
        # Add micro-variations within a minute (simulate traffic light cycles)
        cycle_variation = 0.1 * np.sin(2 * np.pi * second / 60)
        
        return base + cycle_variation
    
    def _get_node_capacity_factor(self, node_id):
        """
        Get traffic capacity factor for each node
        """
        node_data = self.network.graph.nodes[node_id]
        if node_data['junction_type'] == 'major':
            return 1.2
        else:
            return 0.8
    
    def generate_traffic_sequence(self, num_minutes=120, start_hour=8, start_minute=0):
        """
        Generate detailed traffic data sequence with 1-second resolution
        
        Args:
            num_minutes: Number of minutes to simulate (default: 120 = 2 hours)
            start_hour: Starting hour of day (0-23)
            start_minute: Starting minute (0-59)
            
        Returns:
            traffic_data: (num_timesteps, num_nodes) - vehicle count at each node
            timestamps: (num_timesteps,) - timestamp for each measurement in seconds
            time_info: Dictionary with hour, minute, second for each timestep
        """
        total_steps = num_minutes * self.steps_per_minute
        
        traffic_data = np.zeros((total_steps, self.num_nodes))
        timestamps = np.zeros(total_steps)
        time_info = {
            'hour': np.zeros(total_steps, dtype=int),
            'minute': np.zeros(total_steps, dtype=int),
            'second': np.zeros(total_steps, dtype=int)
        }
        
        for step in range(total_steps):
            # Calculate current time
            total_seconds = step * self.time_interval
            current_second = (start_minute * 60 + total_seconds) % 60
            current_minute = (start_minute + (total_seconds // 60)) % 60
            current_hour = (start_hour + (start_minute + total_seconds // 60) // 60) % 24
            current_day = ((start_hour + (start_minute + total_seconds // 60) // 60) // 24) % 7
            
            timestamps[step] = total_seconds
            time_info['hour'][step] = current_hour
            time_info['minute'][step] = current_minute
            time_info['second'][step] = current_second
            
            # Get base traffic pattern
            base_multiplier = self._get_base_traffic_pattern(
                current_hour, current_day, current_minute, current_second
            )
            
            # Generate traffic for each node
            for node_id in range(self.num_nodes):
                # Node-specific capacity factor
                node_factor = self._get_node_capacity_factor(node_id)
                
                # Base traffic level
                base_traffic = base_multiplier * node_factor
                
                # Add smooth temporal variation (multiple frequencies)
                temporal_variation = (
                    0.10 * np.sin(2 * np.pi * step / (self.steps_per_minute * 5)) +  # 5-minute cycle
                    0.05 * np.sin(2 * np.pi * step / (self.steps_per_minute * 2)) +  # 2-minute cycle
                    0.03 * np.sin(2 * np.pi * step / self.steps_per_minute)          # 1-minute cycle
                )
                
                # Add random noise (smooth)
                if step == 0:
                    noise = np.random.normal(0, 0.03)
                else:
                    # Smooth noise (correlated with previous timestep)
                    prev_noise = (traffic_data[step-1, node_id] / node_capacity - base_traffic - temporal_variation)
                    noise = 0.8 * prev_noise + 0.2 * np.random.normal(0, 0.03)
                
                # Combine all factors
                traffic_level = base_traffic + temporal_variation + noise
                traffic_level = np.clip(traffic_level, 0, 1)
                
                # Convert to vehicle count
                node_capacity = self.network.graph.nodes[node_id]['junction_capacity']
                traffic_data[step, node_id] = traffic_level * node_capacity
        
        return traffic_data, timestamps, time_info
    
    def add_spatial_correlation(self, traffic_data):
        """
        Add spatial correlation based on adjacency
        Traffic at connected nodes should be correlated
        """
        adj_matrix = self.network.get_adjacency_matrix()
        
        # Smooth traffic using adjacency (weighted average with neighbors)
        smoothed_traffic = np.zeros_like(traffic_data)
        for t in range(traffic_data.shape[0]):
            current_traffic = traffic_data[t:t+1, :].T  # (num_nodes, 1)
            # Weight: 75% current + 25% neighbors
            smoothed = 0.75 * current_traffic + 0.25 * (adj_matrix @ current_traffic)
            smoothed_traffic[t, :] = smoothed.flatten()
        
        return smoothed_traffic
    
    def generate_for_bloch_visualization(self, duration_seconds=60):
        """
        Generate data specifically for Bloch sphere visualization
        Returns high-resolution data for smooth animation
        
        Args:
            duration_seconds: Duration of visualization in seconds (default: 60)
            
        Returns:
            traffic_data: High-resolution traffic data
            timestamps: Timestamps for each frame
            time_info: Detailed time information
        """
        num_minutes = int(np.ceil(duration_seconds / 60))
        traffic_data, timestamps, time_info = self.generate_traffic_sequence(
            num_minutes=num_minutes,
            start_hour=17,  # Start at evening rush hour for interesting patterns
            start_minute=30
        )
        
        # Add spatial correlation
        traffic_data = self.add_spatial_correlation(traffic_data)
        
        # Trim to exact duration
        num_steps = duration_seconds // self.time_interval
        traffic_data = traffic_data[:num_steps]
        timestamps = timestamps[:num_steps]
        for key in time_info:
            time_info[key] = time_info[key][:num_steps]
        
        print(f"Generated {num_steps} timesteps for {duration_seconds}-second visualization")
        print(f"  - Time resolution: {self.time_interval} second(s)")
        print(f"  - Number of nodes: {self.num_nodes}")
        print(f"  - Data shape: {traffic_data.shape}")
        
        return traffic_data, timestamps, time_info


class LargeTrafficDataset(Dataset):
    """
    PyTorch Dataset for large-scale traffic prediction
    """
    def __init__(self, traffic_data, adj_matrix, seq_len=60, pred_horizon=12):
        """
        Args:
            traffic_data: (num_timesteps, num_nodes) - traffic measurements
            adj_matrix: (num_nodes, num_nodes) - normalized adjacency matrix
            seq_len: Number of past time steps to use as input (default: 60 = 1 minute at 1-sec resolution)
            pred_horizon: Number of future time steps to predict (default: 12 = 12 seconds ahead)
        """
        self.traffic_data = traffic_data
        self.adj_matrix = adj_matrix
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.num_nodes = traffic_data.shape[1]
        
        # Create sequences
        self.samples = []
        for i in range(0, len(traffic_data) - seq_len - pred_horizon + 1, 5):  # Stride of 5 for efficiency
            # Input sequence
            x = traffic_data[i:i+seq_len, :]  # (seq_len, num_nodes)
            
            # Target sequence
            y = traffic_data[i+seq_len:i+seq_len+pred_horizon, :]  # (pred_horizon, num_nodes)
            
            self.samples.append((x, y))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        
        # Add features
        x_features = np.zeros((self.seq_len, self.num_nodes, 4))
        
        for t in range(self.seq_len):
            # Feature 1: Normalized traffic level
            x_features[t, :, 0] = x[t, :] / 100.0
            
            # Feature 2: Rate of change
            if t > 0:
                x_features[t, :, 1] = (x[t, :] - x[t-1, :]) / 100.0
            
            # Feature 3: Traffic level from previous timestep
            if t > 0:
                x_features[t, :, 2] = x[t-1, :] / 100.0
            
            # Feature 4: Time encoding (cyclic)
            x_features[t, :, 3] = np.sin(2 * np.pi * t / self.seq_len)
        
        return (
            torch.FloatTensor(x_features),
            torch.FloatTensor(y),
            torch.FloatTensor(self.adj_matrix)
        )


def create_large_dataloaders(network, train_minutes=180, val_minutes=30, test_minutes=30,
                             seq_len=60, pred_horizon=12, batch_size=16):
    """
    Create train, validation, and test dataloaders for large-scale data
    
    Args:
        network: ChemburTrafficNetwork instance
        train_minutes: Minutes of training data (default: 180 = 3 hours)
        val_minutes: Minutes of validation data (default: 30 minutes)
        test_minutes: Minutes of test data (default: 30 minutes)
        seq_len: Input sequence length in timesteps
        pred_horizon: Prediction horizon in timesteps
        batch_size: Batch size
    """
    generator = LargeTrafficDataGenerator(network, time_interval=1)
    
    print("=" * 80)
    print("GENERATING LARGE-SCALE TRAFFIC DATA")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Time resolution: {generator.time_interval} second(s)")
    print(f"  - Input sequence length: {seq_len} timesteps ({seq_len} seconds)")
    print(f"  - Prediction horizon: {pred_horizon} timesteps ({pred_horizon} seconds)")
    print(f"\nData splits:")
    print(f"  - Training: {train_minutes} minutes ({train_minutes * 60} timesteps)")
    print(f"  - Validation: {val_minutes} minutes ({val_minutes * 60} timesteps)")
    print(f"  - Test: {test_minutes} minutes ({test_minutes * 60} timesteps)")
    
    # Generate training data
    print("\nGenerating training data...")
    train_traffic, _, _ = generator.generate_traffic_sequence(num_minutes=train_minutes, start_hour=7)
    train_traffic = generator.add_spatial_correlation(train_traffic)
    
    # Generate validation data
    print("Generating validation data...")
    val_traffic, _, _ = generator.generate_traffic_sequence(num_minutes=val_minutes, start_hour=12)
    val_traffic = generator.add_spatial_correlation(val_traffic)
    
    # Generate test data
    print("Generating test data...")
    test_traffic, _, _ = generator.generate_traffic_sequence(num_minutes=test_minutes, start_hour=17)
    test_traffic = generator.add_spatial_correlation(test_traffic)
    
    # Get adjacency matrix
    adj_matrix = network.get_adjacency_matrix()
    
    # Normalize adjacency matrix
    adj_normalized = normalize_adjacency_matrix(adj_matrix)
    
    # Create datasets
    train_dataset = LargeTrafficDataset(train_traffic, adj_normalized, seq_len, pred_horizon)
    val_dataset = LargeTrafficDataset(val_traffic, adj_normalized, seq_len, pred_horizon)
    test_dataset = LargeTrafficDataset(test_traffic, adj_normalized, seq_len, pred_horizon)
    
    print(f"\nDataset sizes:")
    print(f"  - Training samples: {len(train_dataset):,}")
    print(f"  - Validation samples: {len(val_dataset):,}")
    print(f"  - Test samples: {len(test_dataset):,}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print("✓ Data loaders created successfully")
    print("=" * 80)
    
    return train_loader, val_loader, test_loader


def normalize_adjacency_matrix(adj):
    """
    Normalize adjacency matrix: Â = D^(-1/2) A D^(-1/2)
    where A = A + I (add self-loops)
    """
    adj_with_self_loops = adj + np.eye(adj.shape[0])
    degree = np.sum(adj_with_self_loops, axis=1)
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt @ adj_with_self_loops @ d_mat_inv_sqrt
    return adj_normalized


if __name__ == "__main__":
    # Test large data generation
    print("Testing Large Traffic Data Generator")
    print("=" * 80)
    
    from chembur_network import ChemburTrafficNetwork
    import matplotlib.pyplot as plt
    
    network = ChemburTrafficNetwork()
    generator = LargeTrafficDataGenerator(network, time_interval=1)
    
    # Generate 1-minute visualization data
    print("\nGenerating data for 60-second Bloch sphere visualization...")
    traffic_data, timestamps, time_info = generator.generate_for_bloch_visualization(duration_seconds=60)
    
    # Save for Bloch sphere visualization
    np.savez(
        'bloch_visualization_data.npz',
        traffic_data=traffic_data,
        timestamps=timestamps,
        hour=time_info['hour'],
        minute=time_info['minute'],
        second=time_info['second'],
        adj_matrix=network.get_adjacency_matrix()
    )
    print("✓ Saved to bloch_visualization_data.npz")
    
    # Plot sample
    plt.figure(figsize=(15, 8))
    for node_id in range(min(4, network.graph.number_of_nodes())):
        plt.subplot(2, 2, node_id + 1)
        plt.plot(timestamps, traffic_data[:, node_id])
        node_name = network.graph.nodes[node_id]['name']
        plt.title(f'Node {node_id}: {node_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Vehicle Count')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('large_traffic_patterns.png', dpi=300)
    print("✓ Traffic patterns saved to large_traffic_patterns.png")
    
    # Test dataloader creation
    print("\n" + "=" * 80)
    print("Testing dataloader creation...")
    train_loader, val_loader, test_loader = create_large_dataloaders(
        network,
        train_minutes=180,
        val_minutes=30,
        test_minutes=30,
        seq_len=60,
        pred_horizon=12,
        batch_size=16
    )
    
    # Show sample batch
    x, y, adj = next(iter(train_loader))
    print(f"\nSample batch shapes:")
    print(f"  - Input (x): {x.shape}")
    print(f"  - Target (y): {y.shape}")
    print(f"  - Adjacency (adj): {adj.shape}")
    
    print("\n✓ All tests passed!")
