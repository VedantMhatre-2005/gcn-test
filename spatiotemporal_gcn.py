"""
Spatiotemporal GCN for Traffic Prediction
This module implements a spatiotemporal GNN that combines:
1. GCN layers for capturing spatial dependencies (graph structure)
2. 1D-CNN for capturing temporal dependencies (time series)
3. MLP for generating final predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvolution(nn.Module):
    """
    Simple GCN layer: H' = σ(Â H W)
    where Â is the normalized adjacency matrix with self-loops
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Args:
            x: Node features (batch_size, num_nodes, in_features)
            adj: Normalized adjacency matrix (num_nodes, num_nodes)
        Returns:
            output: (batch_size, num_nodes, out_features)
        """
        # x: (batch, nodes, features)
        # adj: (nodes, nodes)
        support = torch.matmul(x, self.weight)  # (batch, nodes, out_features)
        output = torch.matmul(adj, support)  # (batch, nodes, out_features)
        if self.bias is not None:
            output = output + self.bias
        return output


class TemporalConvNet(nn.Module):
    """
    1D-CNN for capturing temporal dependencies
    Uses multiple 1D convolutional layers with different kernel sizes
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size,
                    stride=1, padding=(kernel_size-1)//2
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_inputs, seq_len)
        Returns:
            output: (batch_size, num_channels[-1], seq_len)
        """
        return self.network(x)


class SpatioTemporalGCN(nn.Module):
    """
    Spatiotemporal GCN combining GCN for spatial dependencies and 
    1D-CNN for temporal dependencies
    
    Architecture:
    1. Input: (batch, time_steps, num_nodes, features)
    2. For each time step:
       - Apply GCN to capture spatial dependencies
    3. Reshape and apply 1D-CNN to capture temporal dependencies
    4. Apply MLP to generate predictions
    """
    def __init__(self, num_nodes, num_features, gcn_hidden_dims, 
                 temporal_channels, mlp_hidden_dims, output_horizon, dropout=0.2):
        """
        Args:
            num_nodes: Number of nodes in the graph
            num_features: Number of input features per node
            gcn_hidden_dims: List of hidden dimensions for GCN layers
            temporal_channels: List of channel sizes for temporal CNN
            mlp_hidden_dims: List of hidden dimensions for MLP
            output_horizon: Number of future time steps to predict
            dropout: Dropout rate
        """
        super(SpatioTemporalGCN, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.output_horizon = output_horizon
        
        # GCN layers for spatial feature extraction
        self.gcn_layers = nn.ModuleList()
        in_dim = num_features
        for hidden_dim in gcn_hidden_dims:
            self.gcn_layers.append(GraphConvolution(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.gcn_out_dim = gcn_hidden_dims[-1]
        
        # Temporal CNN for time series feature extraction
        # Input: (batch, num_nodes * gcn_out_dim, seq_len)
        self.temporal_cnn = TemporalConvNet(
            num_inputs=num_nodes * self.gcn_out_dim,
            num_channels=temporal_channels,
            kernel_size=3
        )
        
        # MLP for final prediction
        mlp_layers = []
        # Flatten temporal features
        mlp_input_dim = num_nodes * self.gcn_out_dim + temporal_channels[-1]
        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(mlp_input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            mlp_input_dim = hidden_dim
        
        # Output layer: predict traffic for next 'output_horizon' steps for all nodes
        mlp_layers.append(nn.Linear(mlp_input_dim, num_nodes * output_horizon))
        self.mlp = nn.Sequential(*mlp_layers)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj, return_embeddings=False):
        """
        Args:
            x: Input features (batch_size, seq_len, num_nodes, num_features)
            adj: Normalized adjacency matrix (num_nodes, num_nodes)
            return_embeddings: If True, return embeddings along with predictions
        Returns:
            predictions: (batch_size, output_horizon, num_nodes)
            embeddings (optional): Dictionary of embeddings at different stages
        """
        batch_size, seq_len, num_nodes, num_features = x.shape
        
        # Apply GCN at each time step
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (batch, num_nodes, num_features)
            
            # Apply GCN layers
            h = x_t
            for gcn_layer in self.gcn_layers:
                h = F.relu(gcn_layer(h, adj))
                h = self.dropout(h)
            
            gcn_outputs.append(h)  # (batch, num_nodes, gcn_out_dim)
        
        # Stack temporal outputs
        gcn_outputs = torch.stack(gcn_outputs, dim=1)  # (batch, seq_len, num_nodes, gcn_out_dim)
        
        # Reshape for temporal CNN: (batch, num_nodes * gcn_out_dim, seq_len)
        gcn_flat = gcn_outputs.view(batch_size, seq_len, -1)  # (batch, seq_len, num_nodes*gcn_out_dim)
        gcn_flat = gcn_flat.transpose(1, 2)  # (batch, num_nodes*gcn_out_dim, seq_len)
        
        # Apply temporal CNN
        temporal_features = self.temporal_cnn(gcn_flat)  # (batch, temporal_channels[-1], seq_len)
        
        # Global pooling over time
        temporal_pooled = torch.mean(temporal_features, dim=2)  # (batch, temporal_channels[-1])
        
        # Take last spatial features
        last_spatial = gcn_outputs[:, -1, :, :].view(batch_size, -1)  # (batch, num_nodes*gcn_out_dim)
        
        # Concatenate spatial and temporal features
        combined = torch.cat([last_spatial, temporal_pooled], dim=1)
        
        # Apply MLP for prediction
        predictions = self.mlp(combined)  # (batch, num_nodes * output_horizon)
        
        # Reshape to (batch, output_horizon, num_nodes)
        predictions = predictions.view(batch_size, self.output_horizon, num_nodes)
        
        if return_embeddings:
            embeddings = {
                'spatial_embeddings': gcn_outputs,          # (batch, seq_len, num_nodes, gcn_out_dim)
                'temporal_embeddings': temporal_pooled,     # (batch, temporal_channels[-1])
                'temporal_features': temporal_features,     # (batch, temporal_channels[-1], seq_len)
                'last_spatial': last_spatial,               # (batch, num_nodes * gcn_out_dim)
                'combined_embeddings': combined             # (batch, num_nodes*gcn_out_dim + temporal_channels[-1])
            }
            return predictions, embeddings
        
        return predictions


class TrafficPredictor:
    """
    Wrapper class for traffic prediction with training and inference
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
    
    def train_model(self, train_loader, val_loader, num_epochs, learning_rate=0.001):
        """
        Train the model
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0
            for batch in train_loader:
                x, y, adj = batch
                x = x.to(self.device)
                y = y.to(self.device)
                adj = adj.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(x, adj)
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y, adj = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    adj = adj.to(self.device)
                    
                    predictions = self.model(x, adj)
                    loss = criterion(predictions, y)
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return train_losses, val_losses
    
    def predict(self, x, adj, return_embeddings=False):
        """
        Make predictions
        Args:
            x: Input features (batch_size, seq_len, num_nodes, num_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
            return_embeddings: If True, return embeddings along with predictions
        Returns:
            predictions: (batch_size, output_horizon, num_nodes)
            embeddings (optional): Dictionary of embeddings
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(x).to(self.device)
            adj = torch.FloatTensor(adj).to(self.device)
            
            if return_embeddings:
                predictions, embeddings = self.model(x, adj, return_embeddings=True)
                # Convert embeddings to numpy
                embeddings_np = {
                    k: v.cpu().numpy() for k, v in embeddings.items()
                }
                return predictions.cpu().numpy(), embeddings_np
            else:
                predictions = self.model(x, adj)
                return predictions.cpu().numpy()
    
    def measure_latency(self, x, adj, num_runs=100):
        """
        Measure prediction latency
        """
        import time
        
        self.model.eval()
        x = torch.FloatTensor(x).to(self.device)
        adj = torch.FloatTensor(adj).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(x, adj)
        
        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.model(x, adj)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'median_latency_ms': np.median(latencies)
        }
    
    def extract_embeddings(self, dataloader, max_batches=None):
        """
        Extract embeddings from a dataset
        Args:
            dataloader: PyTorch DataLoader
            max_batches: Maximum number of batches to process (None = all)
        Returns:
            Dictionary containing all embeddings and predictions
        """
        self.model.eval()
        
        all_spatial = []
        all_temporal = []
        all_combined = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                    
                x, y, adj = batch
                x = x.to(self.device)
                adj = adj.to(self.device)
                
                predictions, embeddings = self.model(x, adj, return_embeddings=True)
                
                all_spatial.append(embeddings['spatial_embeddings'].cpu().numpy())
                all_temporal.append(embeddings['temporal_embeddings'].cpu().numpy())
                all_combined.append(embeddings['combined_embeddings'].cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.numpy())
        
        return {
            'spatial_embeddings': np.concatenate(all_spatial, axis=0),
            'temporal_embeddings': np.concatenate(all_temporal, axis=0),
            'combined_embeddings': np.concatenate(all_combined, axis=0),
            'predictions': np.concatenate(all_predictions, axis=0),
            'targets': np.concatenate(all_targets, axis=0)
        }
    
    def save_embeddings(self, embeddings_dict, save_path='embeddings.npz'):
        """
        Save embeddings to disk
        Args:
            embeddings_dict: Dictionary of embeddings from extract_embeddings()
            save_path: Path to save embeddings
        """
        np.savez_compressed(save_path, **embeddings_dict)
        print(f"Embeddings saved to {save_path}")
        
        # Print summary
        print("\nEmbedding shapes:")
        for key, value in embeddings_dict.items():
            print(f"  {key}: {value.shape}")
    
    @staticmethod
    def load_embeddings(load_path='embeddings.npz'):
        """
        Load embeddings from disk
        Args:
            load_path: Path to load embeddings from
        Returns:
            Dictionary of embeddings
        """
        data = np.load(load_path)
        embeddings = {key: data[key] for key in data.files}
        print(f"Embeddings loaded from {load_path}")
        print("\nEmbedding shapes:")
        for key, value in embeddings.items():
            print(f"  {key}: {value.shape}")
        return embeddings


if __name__ == "__main__":
    # Test the model architecture
    num_nodes = 8
    num_features = 4  # Initial features per node
    seq_len = 12  # Look at past 12 time steps (1 minute if each step is 5 seconds)
    output_horizon = 1  # Predict next 5 seconds
    batch_size = 4
    
    # Create model
    model = SpatioTemporalGCN(
        num_nodes=num_nodes,
        num_features=num_features,
        gcn_hidden_dims=[32, 64],
        temporal_channels=[64, 128],
        mlp_hidden_dims=[128, 64],
        output_horizon=output_horizon,
        dropout=0.2
    )
    
    print("Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, num_nodes, num_features)
    adj = torch.eye(num_nodes) + torch.randn(num_nodes, num_nodes).abs()
    adj = adj / adj.sum(dim=1, keepdim=True)  # Normalize
    
    predictions = model(x, adj)
    print(f"\nInput shape: {x.shape}")
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Output shape: {predictions.shape}")
    print("\nModel test successful!")
