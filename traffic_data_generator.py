"""
Traffic Data Generator for Chembur Network
Generates realistic synthetic traffic data with temporal patterns
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from chembur_network import ChemburTrafficNetwork


class TrafficDataGenerator:
    """
    Generate synthetic traffic data with realistic patterns:
    - Daily patterns (rush hours, normal hours)
    - Weekly patterns (weekday vs weekend)
    - Random variations
    - Spatial correlations (based on network structure)
    """
    def __init__(self, network, time_interval=5):
        """
        Args:
            network: ChemburTrafficNetwork instance
            time_interval: Time interval in seconds between measurements
        """
        self.network = network
        self.num_nodes = network.graph.number_of_nodes()
        self.time_interval = time_interval  # 5 seconds
        self.steps_per_hour = 3600 // time_interval  # 720 steps per hour
        
    def _get_base_traffic_pattern(self, hour, day_of_week):
        """
        Get base traffic multiplier based on hour and day
        """
        is_weekend = day_of_week >= 5
        
        # Weekday pattern
        if not is_weekend:
            if 7 <= hour < 10:  # Morning rush hour
                return 0.9
            elif 10 <= hour < 17:  # Normal working hours
                return 0.6
            elif 17 <= hour < 20:  # Evening rush hour
                return 0.95
            elif 20 <= hour < 23:  # Evening
                return 0.5
            else:  # Night
                return 0.2
        # Weekend pattern
        else:
            if 10 <= hour < 13:  # Late morning
                return 0.6
            elif 13 <= hour < 18:  # Afternoon
                return 0.7
            elif 18 <= hour < 22:  # Evening
                return 0.65
            else:  # Night/early morning
                return 0.25
    
    def _get_node_capacity_factor(self, node_id):
        """
        Get traffic capacity factor for each node
        """
        node_data = self.network.graph.nodes[node_id]
        # Major junctions tend to have more traffic
        if node_data['junction_type'] == 'major':
            return 1.2
        else:
            return 0.8
    
    def generate_traffic_sequence(self, num_days=7, start_hour=0):
        """
        Generate traffic data sequence
        
        Returns:
            traffic_data: (num_timesteps, num_nodes) - vehicle count at each node
            timestamps: (num_timesteps,) - timestamp for each measurement
        """
        steps_per_day = 24 * self.steps_per_hour
        total_steps = num_days * steps_per_day
        
        traffic_data = np.zeros((total_steps, self.num_nodes))
        timestamps = np.zeros(total_steps)
        
        for step in range(total_steps):
            # Calculate current time
            total_seconds = step * self.time_interval
            current_hour = (start_hour + (total_seconds // 3600)) % 24
            current_day = (total_seconds // (24 * 3600)) % 7
            timestamps[step] = total_seconds
            
            # Get base traffic pattern
            base_multiplier = self._get_base_traffic_pattern(current_hour, current_day)
            
            # Generate traffic for each node
            for node_id in range(self.num_nodes):
                # Node-specific capacity factor
                node_factor = self._get_node_capacity_factor(node_id)
                
                # Base traffic level
                base_traffic = base_multiplier * node_factor
                
                # Add smooth temporal variation (sine wave for smooth transitions)
                temporal_variation = 0.15 * np.sin(2 * np.pi * step / (self.steps_per_hour * 2))
                
                # Add random noise
                noise = np.random.normal(0, 0.05)
                
                # Combine all factors
                traffic_level = base_traffic + temporal_variation + noise
                traffic_level = np.clip(traffic_level, 0, 1)
                
                # Convert to vehicle count (assuming max capacity at node)
                node_capacity = self.network.graph.nodes[node_id]['junction_capacity']
                traffic_data[step, node_id] = traffic_level * node_capacity
        
        return traffic_data, timestamps
    
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
            # Weight: 70% current + 30% neighbors
            smoothed = 0.7 * current_traffic + 0.3 * (adj_matrix @ current_traffic)
            smoothed_traffic[t, :] = smoothed.flatten()
        
        return smoothed_traffic


class TrafficDataset(Dataset):
    """
    PyTorch Dataset for traffic prediction
    """
    def __init__(self, traffic_data, adj_matrix, seq_len=12, pred_horizon=1):
        """
        Args:
            traffic_data: (num_timesteps, num_nodes) - traffic measurements
            adj_matrix: (num_nodes, num_nodes) - normalized adjacency matrix
            seq_len: Number of past time steps to use as input
            pred_horizon: Number of future time steps to predict
        """
        self.traffic_data = traffic_data
        self.adj_matrix = adj_matrix
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.num_nodes = traffic_data.shape[1]
        
        # Create sequences
        self.samples = []
        for i in range(len(traffic_data) - seq_len - pred_horizon + 1):
            # Input sequence
            x = traffic_data[i:i+seq_len, :]  # (seq_len, num_nodes)
            
            # Target sequence
            y = traffic_data[i+seq_len:i+seq_len+pred_horizon, :]  # (pred_horizon, num_nodes)
            
            self.samples.append((x, y))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        
        # Add features: current traffic + rate of change
        x_features = np.zeros((self.seq_len, self.num_nodes, 4))
        
        for t in range(self.seq_len):
            # Feature 1: Normalized traffic level
            x_features[t, :, 0] = x[t, :] / 100.0  # Normalize by typical max
            
            # Feature 2: Rate of change (if not first timestep)
            if t > 0:
                x_features[t, :, 1] = (x[t, :] - x[t-1, :]) / 100.0
            
            # Feature 3: Traffic level from 1 step ago
            if t > 0:
                x_features[t, :, 2] = x[t-1, :] / 100.0
            
            # Feature 4: Hour of day (cyclic encoding)
            # Assume each sequence captures 1 minute (12 * 5 seconds)
            x_features[t, :, 3] = np.sin(2 * np.pi * (idx + t) / (12 * 60))  # Rough approximation
        
        return (
            torch.FloatTensor(x_features),
            torch.FloatTensor(y),
            torch.FloatTensor(self.adj_matrix)
        )


def create_dataloaders(network, train_days=5, val_days=1, test_days=1, 
                       seq_len=12, pred_horizon=1, batch_size=32):
    """
    Create train, validation, and test dataloaders
    """
    # Generate data
    generator = TrafficDataGenerator(network, time_interval=5)
    
    print("Generating traffic data...")
    print(f"  - Training days: {train_days}")
    print(f"  - Validation days: {val_days}")
    print(f"  - Test days: {test_days}")
    print(f"  - Time interval: {generator.time_interval} seconds")
    print(f"  - Sequence length: {seq_len} steps ({seq_len * 5} seconds)")
    print(f"  - Prediction horizon: {pred_horizon} step ({pred_horizon * 5} seconds)")
    
    # Training data
    train_traffic, _ = generator.generate_traffic_sequence(num_days=train_days)
    train_traffic = generator.add_spatial_correlation(train_traffic)
    
    # Validation data
    val_traffic, _ = generator.generate_traffic_sequence(num_days=val_days)
    val_traffic = generator.add_spatial_correlation(val_traffic)
    
    # Test data
    test_traffic, _ = generator.generate_traffic_sequence(num_days=test_days)
    test_traffic = generator.add_spatial_correlation(test_traffic)
    
    # Get adjacency matrix
    adj_matrix = network.get_adjacency_matrix()
    
    # Create datasets
    train_dataset = TrafficDataset(train_traffic, adj_matrix, seq_len, pred_horizon)
    val_dataset = TrafficDataset(val_traffic, adj_matrix, seq_len, pred_horizon)
    test_dataset = TrafficDataset(test_traffic, adj_matrix, seq_len, pred_horizon)
    
    print(f"\nDataset sizes:")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data generation
    from chembur_network import ChemburTrafficNetwork
    import matplotlib.pyplot as plt
    
    network = ChemburTrafficNetwork()
    generator = TrafficDataGenerator(network, time_interval=5)
    
    # Generate 2 days of data
    traffic_data, timestamps = generator.generate_traffic_sequence(num_days=2)
    traffic_data = generator.add_spatial_correlation(traffic_data)
    
    # Plot traffic for a few nodes
    plt.figure(figsize=(15, 8))
    hours = timestamps / 3600
    
    for node_id in range(min(4, network.graph.number_of_nodes())):
        plt.subplot(2, 2, node_id + 1)
        plt.plot(hours, traffic_data[:, node_id])
        node_name = network.graph.nodes[node_id]['name']
        plt.title(f'Node {node_id}: {node_name}')
        plt.xlabel('Time (hours)')
        plt.ylabel('Vehicle Count')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('traffic_patterns.png', dpi=300)
    print("Traffic patterns saved to traffic_patterns.png")
    
    # Test dataset creation
    train_loader, val_loader, test_loader = create_dataloaders(
        network, train_days=5, val_days=1, test_days=1
    )
    
    # Show a sample
    x, y, adj = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Input shape: {x.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Adjacency shape: {adj.shape}")
