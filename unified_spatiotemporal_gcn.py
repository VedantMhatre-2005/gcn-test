"""
Unified Spatiotemporal GCN for Traffic Prediction
A single integrated architecture that processes spatial and temporal information simultaneously
without separate GCN, CNN, and MLP stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class STConvBlock(nn.Module):
    """
    Spatiotemporal Convolution Block
    Simultaneously processes spatial (graph) and temporal (time series) information
    """
    def __init__(self, in_channels, out_channels, num_nodes, kernel_size=3, dropout=0.2):
        super(STConvBlock, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Temporal convolution (across time)
        self.temporal_conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2)
        )
        
        # Spatial convolution (graph convolution)
        self.spatial_weight = nn.Parameter(torch.FloatTensor(out_channels, out_channels))
        
        # Batch normalization
        self.batch_norm_temporal = nn.BatchNorm2d(out_channels)
        self.batch_norm_spatial = nn.BatchNorm2d(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.spatial_weight)
    
    def forward(self, x, adj):
        """
        Args:
            x: Input tensor (batch_size, in_channels, num_nodes, seq_len)
            adj: Adjacency matrix (num_nodes, num_nodes) or (batch_size, num_nodes, num_nodes)
        Returns:
            output: (batch_size, out_channels, num_nodes, seq_len)
        """
        residual = self.residual(x)
        
        # Handle batched or single adjacency matrix
        if adj.dim() == 3:
            # Batched: take first one (they're all the same)
            adj = adj[0]
        
        # Temporal convolution
        # x: (batch, in_channels, num_nodes, seq_len)
        x_temporal = self.temporal_conv(x)  # (batch, out_channels, num_nodes, seq_len)
        x_temporal = self.batch_norm_temporal(x_temporal)
        x_temporal = F.relu(x_temporal)
        
        # Spatial graph convolution
        # Reshape for graph convolution: (batch, out_channels, num_nodes, seq_len)
        batch_size, channels, num_nodes, seq_len = x_temporal.shape
        
        # Apply graph convolution across nodes
        # x_temporal: (batch, channels, nodes, seq_len) -> (batch*seq_len, nodes, channels)
        x_reshaped = x_temporal.permute(0, 3, 2, 1).contiguous()  # (batch, seq_len, nodes, channels)
        x_reshaped = x_reshaped.view(batch_size * seq_len, num_nodes, channels)
        
        # Graph convolution: AXW
        x_spatial = torch.matmul(adj, x_reshaped)  # (batch*seq_len, nodes, channels)
        x_spatial = torch.matmul(x_spatial, self.spatial_weight)  # (batch*seq_len, nodes, channels)
        
        # Reshape back: (batch, seq_len, nodes, channels) -> (batch, channels, nodes, seq_len)
        x_spatial = x_spatial.view(batch_size, seq_len, num_nodes, channels)
        x_spatial = x_spatial.permute(0, 3, 2, 1).contiguous()  # (batch, channels, nodes, seq_len)
        
        x_spatial = self.batch_norm_spatial(x_spatial)
        x_spatial = F.relu(x_spatial)
        
        # Combine and add residual
        output = x_spatial + residual
        output = self.dropout(output)
        
        return output


class UnifiedSpatioTemporalGCN(nn.Module):
    """
    Unified Spatiotemporal GCN
    A single integrated architecture that replaces separate GCN, CNN, and MLP stages
    
    Architecture:
    - Input embedding layer
    - Multiple STConv blocks (spatiotemporal convolution)
    - Output projection layer
    
    This is a cleaner, more unified architecture compared to the separate GCN->CNN->MLP pipeline
    """
    def __init__(self, num_nodes, num_features, hidden_channels, num_layers, 
                 output_horizon, kernel_size=3, dropout=0.2):
        """
        Args:
            num_nodes: Number of nodes in the graph
            num_features: Number of input features per node
            hidden_channels: Number of hidden channels in STConv blocks
            num_layers: Number of STConv blocks
            output_horizon: Number of future time steps to predict
            kernel_size: Temporal kernel size
            dropout: Dropout rate
        """
        super(UnifiedSpatioTemporalGCN, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.output_horizon = output_horizon
        
        # Input embedding
        self.input_conv = nn.Conv2d(num_features, hidden_channels, kernel_size=1)
        
        # Spatiotemporal convolution blocks
        self.st_blocks = nn.ModuleList([
            STConvBlock(hidden_channels, hidden_channels, num_nodes, kernel_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        # Temporal pooling + prediction
        self.output_conv = nn.Conv2d(hidden_channels, output_horizon, kernel_size=1)
        
    def forward(self, x, adj, return_embeddings=False):
        """
        Args:
            x: Input features (batch_size, seq_len, num_nodes, num_features)
            adj: Normalized adjacency matrix (num_nodes, num_nodes)
            return_embeddings: If True, return intermediate embeddings
        Returns:
            predictions: (batch_size, output_horizon, num_nodes)
            embeddings (optional): List of embeddings from each STConv block
        """
        batch_size, seq_len, num_nodes, num_features = x.shape
        
        # Reshape to (batch, features, nodes, seq_len) for conv2d
        x = x.permute(0, 3, 2, 1)  # (batch, features, nodes, seq_len)
        
        # Input embedding
        x = self.input_conv(x)  # (batch, hidden_channels, nodes, seq_len)
        x = F.relu(x)
        
        # Apply spatiotemporal convolution blocks
        embeddings = []
        for st_block in self.st_blocks:
            x = st_block(x, adj)
            if return_embeddings:
                embeddings.append(x.clone())
        
        # Output projection
        # x: (batch, hidden_channels, nodes, seq_len)
        # Apply temporal pooling
        x_pooled = torch.mean(x, dim=3, keepdim=True)  # (batch, hidden_channels, nodes, 1)
        
        # Project to output horizon
        predictions = self.output_conv(x_pooled)  # (batch, output_horizon, nodes, 1)
        predictions = predictions.squeeze(-1)  # (batch, output_horizon, nodes)
        
        if return_embeddings:
            return predictions, embeddings
        return predictions


class SpatioTemporalGCNPredictor:
    """
    Wrapper for training and inference with UnifiedSpatioTemporalGCN
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (x, y, adj) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            adj = adj.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(x, adj)
            
            # Calculate loss
            # predictions: (batch, output_horizon, nodes)
            # y: (batch, output_horizon, nodes)
            loss = criterion(predictions, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, data_loader, criterion):
        """
        Evaluate on validation/test set
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for x, y, adj in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                adj = adj.to(self.device)
                
                # Forward pass
                predictions = self.model(x, adj)
                
                # Calculate loss
                loss = criterion(predictions, y)
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        mae = np.mean(np.abs(all_predictions - all_targets))
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        mape = np.mean(np.abs((all_predictions - all_targets) / (all_targets + 1e-8))) * 100
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    
    def predict(self, x, adj):
        """
        Make predictions
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            adj = adj.to(self.device)
            predictions = self.model(x, adj)
            return predictions.cpu().numpy()
    
    def get_embeddings(self, x, adj):
        """
        Extract embeddings from all layers
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            adj = adj.to(self.device)
            predictions, embeddings = self.model(x, adj, return_embeddings=True)
            return [emb.cpu().numpy() for emb in embeddings]


def normalize_adjacency_matrix(adj):
    """
    Normalize adjacency matrix: Â = D^(-1/2) A D^(-1/2)
    where A = A + I (add self-loops)
    """
    # Add self-loops
    adj_with_self_loops = adj + np.eye(adj.shape[0])
    
    # Compute degree matrix
    degree = np.sum(adj_with_self_loops, axis=1)
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    # Normalize: D^(-1/2) A D^(-1/2)
    adj_normalized = d_mat_inv_sqrt @ adj_with_self_loops @ d_mat_inv_sqrt
    
    return adj_normalized


if __name__ == "__main__":
    # Test the model
    print("Testing Unified Spatiotemporal GCN")
    print("=" * 80)
    
    # Parameters
    batch_size = 4
    seq_len = 12
    num_nodes = 8
    num_features = 4
    hidden_channels = 32
    num_layers = 3
    output_horizon = 1
    
    # Create model
    model = UnifiedSpatioTemporalGCN(
        num_nodes=num_nodes,
        num_features=num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        output_horizon=output_horizon,
        kernel_size=3,
        dropout=0.2
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy data
    x = torch.randn(batch_size, seq_len, num_nodes, num_features)
    adj = torch.eye(num_nodes) + torch.randn(num_nodes, num_nodes) * 0.1
    adj = (adj + adj.T) / 2  # Make symmetric
    adj = torch.abs(adj)
    adj = adj / adj.sum(dim=1, keepdim=True)  # Normalize
    
    # Forward pass
    predictions = model(x, adj)
    print(f"\nInput shape: {x.shape}")
    print(f"Adjacency shape: {adj.shape}")
    print(f"Output shape: {predictions.shape}")
    
    # Test with embeddings
    predictions, embeddings = model(x, adj, return_embeddings=True)
    print(f"\nNumber of embedding layers: {len(embeddings)}")
    for i, emb in enumerate(embeddings):
        print(f"  Layer {i+1} embedding shape: {emb.shape}")
    
    print("\n✓ Model test successful!")
