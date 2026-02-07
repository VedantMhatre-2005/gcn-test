"""
Main script to train and evaluate the Spatiotemporal GCN for traffic prediction
Measures prediction latency and evaluates model performance
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

from chembur_network import ChemburTrafficNetwork
from spatiotemporal_gcn import SpatioTemporalGCN, TrafficPredictor
from traffic_data_generator import create_dataloaders


def train_and_evaluate():
    """
    Complete pipeline for training and evaluation
    """
    print("="*80)
    print("TRAFFIC PREDICTION WITH SPATIOTEMPORAL GCN - CHEMBUR NETWORK")
    print("="*80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create network
    print("\n" + "="*80)
    print("STEP 1: Creating Chembur Traffic Network")
    print("="*80)
    network = ChemburTrafficNetwork()
    network.get_network_info()
    network.visualize_network('chembur_network.png')
    
    # Model hyperparameters
    num_nodes = network.graph.number_of_nodes()
    num_features = 4  # Features per node (traffic, rate_of_change, prev_traffic, time)
    seq_len = 12  # Look at past 60 seconds (12 * 5 seconds)
    pred_horizon = 1  # Predict next 5 seconds
    
    print("\n" + "="*80)
    print("STEP 2: Creating Spatiotemporal GCN Model")
    print("="*80)
    print(f"\nModel Configuration:")
    print(f"  - Number of nodes: {num_nodes}")
    print(f"  - Input features per node: {num_features}")
    print(f"  - Input sequence length: {seq_len} steps (60 seconds)")
    print(f"  - Prediction horizon: {pred_horizon} step (5 seconds)")
    print(f"  - GCN hidden dimensions: [32, 64]")
    print(f"  - Temporal CNN channels: [64, 128]")
    print(f"  - MLP hidden dimensions: [128, 64]")
    
    model = SpatioTemporalGCN(
        num_nodes=num_nodes,
        num_features=num_features,
        gcn_hidden_dims=[32, 64],
        temporal_channels=[64, 128],
        mlp_hidden_dims=[128, 64],
        output_horizon=pred_horizon,
        dropout=0.2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Create predictor
    predictor = TrafficPredictor(model, device=device)
    
    # Generate data
    print("\n" + "="*80)
    print("STEP 3: Generating Synthetic Traffic Data")
    print("="*80)
    train_loader, val_loader, test_loader = create_dataloaders(
        network,
        train_days=5,
        val_days=1,
        test_days=1,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        batch_size=32
    )
    
    # Training
    print("\n" + "="*80)
    print("STEP 4: Training the Model")
    print("="*80)
    num_epochs = 50
    learning_rate = 0.001
    
    print(f"\nTraining Configuration:")
    print(f"  - Number of epochs: {num_epochs}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Optimizer: Adam")
    print(f"  - Loss function: MSE")
    
    print("\nStarting training...\n")
    train_start = time.time()
    train_losses, val_losses = predictor.train_model(
        train_loader, val_loader, num_epochs, learning_rate
    )
    train_end = time.time()
    training_time = train_end - train_start
    
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    print("\nTraining curves saved to training_curves.png")
    
    # Evaluation on test set
    print("\n" + "="*80)
    print("STEP 5: Evaluating on Test Set")
    print("="*80)
    
    predictor.model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y, adj = batch
            x = x.to(device)
            y = y.to(device)
            adj = adj.to(device)
            
            predictions = predictor.model(x, adj)
            test_predictions.append(predictions.cpu().numpy())
            test_targets.append(y.cpu().numpy())
    
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    
    # Calculate metrics
    mse = np.mean((test_predictions - test_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(test_predictions - test_targets))
    mape = np.mean(np.abs((test_predictions - test_targets) / (test_targets + 1e-8))) * 100
    
    print(f"\nTest Set Metrics:")
    print(f"  - MSE: {mse:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - MAPE: {mape:.2f}%")
    
    # Visualize predictions vs actual
    plt.figure(figsize=(15, 10))
    node_ids_to_plot = [0, 1, 3, 7]  # Major junctions
    
    for idx, node_id in enumerate(node_ids_to_plot):
        plt.subplot(2, 2, idx + 1)
        
        # Plot first 100 predictions
        n_samples = min(100, len(test_predictions))
        actual = test_targets[:n_samples, 0, node_id]
        predicted = test_predictions[:n_samples, 0, node_id]
        
        time_steps = np.arange(n_samples) * 5  # 5 seconds per step
        
        plt.plot(time_steps, actual, 'b-', label='Actual', linewidth=1.5, alpha=0.7)
        plt.plot(time_steps, predicted, 'r--', label='Predicted', linewidth=1.5, alpha=0.7)
        
        node_name = network.graph.nodes[node_id]['name'].split()[0]
        plt.title(f'Node {node_id}: {node_name}', fontweight='bold')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Vehicle Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png', dpi=300)
    print("\nPredictions visualization saved to predictions_vs_actual.png")
    
    # Latency measurement
    print("\n" + "="*80)
    print("STEP 6: Measuring Prediction Latency")
    print("="*80)
    
    # Get a single sample for latency measurement
    x_sample, _, adj = next(iter(test_loader))
    x_single = x_sample[0:1].numpy()  # Single sample
    adj_np = adj[0].numpy()
    
    print(f"\nMeasuring latency with:")
    print(f"  - Input shape: {x_single.shape}")
    print(f"  - Device: {device}")
    print(f"  - Number of runs: 100")
    
    latency_stats = predictor.measure_latency(x_single, adj_np, num_runs=100)
    
    print(f"\n{'Latency Statistics':<30} {'Value'}")
    print("-" * 50)
    print(f"{'Mean Latency:':<30} {latency_stats['mean_latency_ms']:.3f} ms")
    print(f"{'Std Deviation:':<30} {latency_stats['std_latency_ms']:.3f} ms")
    print(f"{'Min Latency:':<30} {latency_stats['min_latency_ms']:.3f} ms")
    print(f"{'Max Latency:':<30} {latency_stats['max_latency_ms']:.3f} ms")
    print(f"{'Median Latency:':<30} {latency_stats['median_latency_ms']:.3f} ms")
    
    # Real-time feasibility
    prediction_window = pred_horizon * 5  # seconds
    latency_seconds = latency_stats['mean_latency_ms'] / 1000
    feasibility_ratio = latency_seconds / prediction_window
    
    print(f"\n{'Real-time Feasibility Analysis'}")
    print("-" * 50)
    print(f"{'Prediction window:':<30} {prediction_window} seconds")
    print(f"{'Prediction latency:':<30} {latency_seconds:.4f} seconds")
    print(f"{'Latency/Window ratio:':<30} {feasibility_ratio:.4%}")
    
    if feasibility_ratio < 0.1:
        status = "EXCELLENT - Real-time capable with margin"
    elif feasibility_ratio < 0.5:
        status = "GOOD - Real-time capable"
    elif feasibility_ratio < 1.0:
        status = "ACCEPTABLE - Near real-time"
    else:
        status = "NEEDS OPTIMIZATION - Too slow for real-time"
    
    print(f"{'Status:':<30} {status}")
    
    # Summary Report
    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT")
    print("="*80)
    
    print(f"\nNetwork Configuration:")
    print(f"  - Location: Chembur, Mumbai")
    print(f"  - Number of junctions: {num_nodes}")
    print(f"  - Number of roads: {network.graph.number_of_edges()}")
    print(f"  - Network type: Unidirectional")
    
    print(f"\nModel Architecture:")
    print(f"  - Spatial: GCN with 2 layers (32, 64 hidden units)")
    print(f"  - Temporal: 1D-CNN with 2 layers (64, 128 channels)")
    print(f"  - Output: MLP with 2 layers (128, 64 hidden units)")
    print(f"  - Total parameters: {total_params:,}")
    
    print(f"\nTraining Results:")
    print(f"  - Training time: {training_time/60:.2f} minutes")
    print(f"  - Final training loss: {train_losses[-1]:.4f}")
    print(f"  - Final validation loss: {val_losses[-1]:.4f}")
    
    print(f"\nTest Performance:")
    print(f"  - RMSE: {rmse:.4f} vehicles")
    print(f"  - MAE: {mae:.4f} vehicles")
    print(f"  - MAPE: {mape:.2f}%")
    
    print(f"\nInference Latency:")
    print(f"  - Mean: {latency_stats['mean_latency_ms']:.3f} ms")
    print(f"  - Median: {latency_stats['median_latency_ms']:.3f} ms")
    print(f"  - Real-time status: {status}")
    
    print("\n" + "="*80)
    print("All outputs saved:")
    print("  - chembur_network.png: Network visualization")
    print("  - traffic_patterns.png: Sample traffic patterns")
    print("  - training_curves.png: Training/validation loss")
    print("  - predictions_vs_actual.png: Model predictions")
    print("="*80)
    
    # Save model
    torch.save(model.state_dict(), 'traffic_gcn_model.pth')
    print("\nModel saved to: traffic_gcn_model.pth")
    
    return {
        'model': model,
        'predictor': predictor,
        'network': network,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        },
        'latency': latency_stats,
        'training_time': training_time
    }


if __name__ == "__main__":
    results = train_and_evaluate()
    print("\n✓ Complete pipeline executed successfully!")
