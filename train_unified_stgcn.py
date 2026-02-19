"""
Training Script for Unified Spatiotemporal GCN
Trains the unified architecture on large-scale traffic data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path

from chembur_network import ChemburTrafficNetwork
from unified_spatiotemporal_gcn import UnifiedSpatioTemporalGCN, SpatioTemporalGCNPredictor, normalize_adjacency_matrix
from large_traffic_data_generator import create_large_dataloaders, LargeTrafficDataGenerator


def train_unified_stgcn():
    """
    Complete training pipeline for Unified Spatiotemporal GCN
    """
    print("=" * 80)
    print("UNIFIED SPATIOTEMPORAL GCN FOR TRAFFIC PREDICTION")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # ========================================
    # STEP 1: Load Network
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 1: Loading Chembur Traffic Network")
    print("=" * 80)
    
    network = ChemburTrafficNetwork()
    network.get_network_info()
    
    adj_matrix = network.get_adjacency_matrix()
    adj_normalized = normalize_adjacency_matrix(adj_matrix)
    
    print(f"\nAdjacency Matrix:")
    print(f"  - Shape: {adj_matrix.shape}")
    print(f"  - Number of edges: {np.sum(adj_matrix > 0)}")
    print(f"  - Normalized: Yes")
    
    # ========================================
    # STEP 2: Generate Data
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 2: Generating Large-Scale Traffic Data")
    print("=" * 80)
    
    # Model hyperparameters
    num_nodes = network.graph.number_of_nodes()
    num_features = 4
    seq_len = 60  # 60 seconds lookback at 1-second resolution
    pred_horizon = 12  # Predict 12 seconds ahead
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_large_dataloaders(
        network,
        train_minutes=180,  # 3 hours of training data
        val_minutes=30,     # 30 minutes of validation data
        test_minutes=30,    # 30 minutes of test data
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        batch_size=16
    )
    
    # ========================================
    # STEP 3: Create Model
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 3: Creating Unified Spatiotemporal GCN")
    print("=" * 80)
    
    # Model configuration
    hidden_channels = 64
    num_layers = 4
    kernel_size = 3
    dropout = 0.2
    
    print(f"\nModel Configuration:")
    print(f"  - Architecture: Unified Spatiotemporal GCN")
    print(f"  - Number of nodes: {num_nodes}")
    print(f"  - Input features: {num_features}")
    print(f"  - Input sequence length: {seq_len} timesteps (60 seconds)")
    print(f"  - Prediction horizon: {pred_horizon} timesteps (12 seconds)")
    print(f"  - Hidden channels: {hidden_channels}")
    print(f"  - Number of ST-Conv layers: {num_layers}")
    print(f"  - Temporal kernel size: {kernel_size}")
    print(f"  - Dropout rate: {dropout}")
    
    model = UnifiedSpatioTemporalGCN(
        num_nodes=num_nodes,
        num_features=num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        output_horizon=pred_horizon,
        kernel_size=kernel_size,
        dropout=dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # Create predictor
    predictor = SpatioTemporalGCNPredictor(model, device=device)
    
    # ========================================
    # STEP 4: Training Setup
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 4: Training Setup")
    print("=" * 80)
    
    # Training hyperparameters
    num_epochs = 100
    learning_rate = 0.001
    weight_decay = 1e-5
    
    print(f"\nTraining Configuration:")
    print(f"  - Number of epochs: {num_epochs}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Weight decay: {weight_decay}")
    print(f"  - Optimizer: Adam")
    print(f"  - Loss function: MSE")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # ========================================
    # STEP 5: Training Loop
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 5: Training")
    print("=" * 80)
    
    train_losses = []
    val_losses = []
    val_maes = []
    val_rmses = []
    val_mapes = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    print("\nStarting training...\n")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = predictor.train_epoch(train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        
        # Validate
        val_metrics = predictor.evaluate(val_loader, criterion)
        val_losses.append(val_metrics['loss'])
        val_maes.append(val_metrics['mae'])
        val_rmses.append(val_metrics['rmse'])
        val_mapes.append(val_metrics['mape'])
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_metrics['loss']:.6f} | "
              f"Val MAE: {val_metrics['mae']:.4f} | "
              f"Val RMSE: {val_metrics['rmse']:.4f} | "
              f"Val MAPE: {val_metrics['mape']:.2f}% | "
              f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_mae': val_metrics['mae'],
                'val_rmse': val_metrics['rmse'],
                'val_mape': val_metrics['mape'],
            }, 'unified_stgcn_best.pth')
            patience_counter = 0
            print(f"  → New best model saved! (Val Loss: {val_metrics['loss']:.6f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    print(f"\n✓ Training completed in {total_time / 60:.2f} minutes")
    
    # ========================================
    # STEP 6: Evaluation on Test Set
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 6: Test Set Evaluation")
    print("=" * 80)
    
    # Load best model
    checkpoint = torch.load('unified_stgcn_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    
    # Evaluate on test set
    test_metrics = predictor.evaluate(test_loader, criterion)
    
    print(f"\nTest Set Performance:")
    print(f"  - Test Loss: {test_metrics['loss']:.6f}")
    print(f"  - Test MAE: {test_metrics['mae']:.4f} vehicles")
    print(f"  - Test RMSE: {test_metrics['rmse']:.4f} vehicles")
    print(f"  - Test MAPE: {test_metrics['mape']:.2f}%")
    
    # ========================================
    # STEP 7: Save Training History
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 7: Saving Results")
    print("=" * 80)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_maes': val_maes,
        'val_rmses': val_rmses,
        'val_mapes': val_mapes,
        'test_metrics': test_metrics,
        'config': {
            'num_nodes': num_nodes,
            'num_features': num_features,
            'seq_len': seq_len,
            'pred_horizon': pred_horizon,
            'hidden_channels': hidden_channels,
            'num_layers': num_layers,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'num_epochs': len(train_losses),
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
        }
    }
    
    with open('unified_stgcn_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("✓ Training history saved to unified_stgcn_history.json")
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(val_maes, label='Val MAE', linewidth=2, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (vehicles)')
    axes[0, 1].set_title('Validation Mean Absolute Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE
    axes[1, 0].plot(val_rmses, label='Val RMSE', linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE (vehicles)')
    axes[1, 0].set_title('Validation Root Mean Squared Error')
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAPE
    axes[1, 1].plot(val_mapes, label='Val MAPE', linewidth=2, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAPE (%)')
    axes[1, 1].set_title('Validation Mean Absolute Percentage Error')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('unified_stgcn_training_curves.png', dpi=300)
    print("✓ Training curves saved to unified_stgcn_training_curves.png")
    
    # ========================================
    # STEP 8: Generate Data for Bloch Sphere Visualization
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 8: Generating Data for 1-Minute Bloch Sphere Visualization")
    print("=" * 80)
    
    generator = LargeTrafficDataGenerator(network, time_interval=1)
    traffic_data, timestamps, time_info = generator.generate_for_bloch_visualization(duration_seconds=60)
    
    # Extract embeddings from the model
    print("\nExtracting embeddings from trained model...")
    
    # Create input sequences
    adj_tensor = torch.FloatTensor(adj_normalized).to(device)
    
    all_embeddings = []
    seq_inputs = []
    
    # Generate sequences from the 60-second data
    for i in range(0, len(traffic_data) - seq_len, 1):  # Every second
        x = traffic_data[i:i+seq_len, :]
        
        # Add features
        x_features = np.zeros((seq_len, num_nodes, 4))
        for t in range(seq_len):
            x_features[t, :, 0] = x[t, :] / 100.0
            if t > 0:
                x_features[t, :, 1] = (x[t, :] - x[t-1, :]) / 100.0
                x_features[t, :, 2] = x[t-1, :] / 100.0
            x_features[t, :, 3] = np.sin(2 * np.pi * t / seq_len)
        
        seq_inputs.append(x_features)
    
    # Process in batches
    batch_size = 16
    for i in range(0, len(seq_inputs), batch_size):
        batch = seq_inputs[i:i+batch_size]
        batch_tensor = torch.FloatTensor(np.array(batch)).to(device)
        
        embeddings = predictor.get_embeddings(batch_tensor, adj_tensor)
        
        # Use last layer embeddings and flatten
        last_layer_emb = embeddings[-1]  # (batch, channels, nodes, seq_len)
        # Average over nodes and time
        flattened = last_layer_emb.mean(axis=(2, 3))  # (batch, channels)
        
        all_embeddings.append(flattened)
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    print(f"✓ Generated embeddings shape: {all_embeddings.shape}")
    
    # Save for Bloch sphere visualization
    np.savez(
        'unified_stgcn_bloch_data.npz',
        embeddings=all_embeddings,
        traffic_data=traffic_data,
        timestamps=timestamps,
        hour=time_info['hour'],
        minute=time_info['minute'],
        second=time_info['second'],
        adj_matrix=adj_matrix
    )
    print("✓ Saved to unified_stgcn_bloch_data.npz")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"\nModel: Unified Spatiotemporal GCN")
    print(f"  - Parameters: {total_params:,}")
    print(f"  - Layers: {num_layers} ST-Conv blocks")
    print(f"\nBest Performance:")
    print(f"  - Validation Loss: {best_val_loss:.6f}")
    print(f"  - Test MAE: {test_metrics['mae']:.4f} vehicles")
    print(f"  - Test RMSE: {test_metrics['rmse']:.4f} vehicles")
    print(f"  - Test MAPE: {test_metrics['mape']:.2f}%")
    print(f"\nFiles Saved:")
    print(f"  - unified_stgcn_best.pth (model checkpoint)")
    print(f"  - unified_stgcn_history.json (training history)")
    print(f"  - unified_stgcn_training_curves.png (plots)")
    print(f"  - unified_stgcn_bloch_data.npz (Bloch sphere visualization data)")
    print("\n" + "=" * 80)
    print("✓ ALL DONE!")
    print("=" * 80)


if __name__ == "__main__":
    train_unified_stgcn()
