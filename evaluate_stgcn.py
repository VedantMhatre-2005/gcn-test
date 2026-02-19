"""
Quick Evaluation Script for Trained Unified Spatiotemporal GCN
Loads the saved model and evaluates it on test data
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json

from chembur_network import ChemburTrafficNetwork
from unified_spatiotemporal_gcn import UnifiedSpatioTemporalGCN, SpatioTemporalGCNPredictor, normalize_adjacency_matrix
from large_traffic_data_generator import create_large_dataloaders, LargeTrafficDataGenerator


def evaluate_trained_model():
    """
    Load and evaluate the trained model
    """
    print("=" * 80)
    print("EVALUATING TRAINED UNIFIED SPATIOTEMPORAL GCN")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load network
    print("\nLoading Chembur Traffic Network...")
    network = ChemburTrafficNetwork()
    adj_matrix = network.get_adjacency_matrix()
    adj_normalized = normalize_adjacency_matrix(adj_matrix)
    
    # Model parameters (must match training)
    num_nodes = 8
    num_features = 4
    seq_len = 60
    pred_horizon = 12
    hidden_channels = 64
    num_layers = 4
    
    # Create model
    print("Creating model architecture...")
    model = UnifiedSpatioTemporalGCN(
        num_nodes=num_nodes,
        num_features=num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        output_horizon=pred_horizon,
        kernel_size=3,
        dropout=0.2
    )
    
    # Load checkpoint
    print("\nLoading trained model...")
    checkpoint = torch.load('unified_stgcn_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint['epoch']+1}")
    print(f"  Training Loss: {checkpoint['train_loss']:.6f}")
    print(f"  Validation Loss: {checkpoint['val_loss']:.6f}")
    
    predictor = SpatioTemporalGCNPredictor(model, device=device)
    
    # Generate test data
    print("\nGenerating test data...")
    _, _, test_loader = create_large_dataloaders(
        network,
        train_minutes=180,
        val_minutes=30,
        test_minutes=30,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        batch_size=16
    )
    
    # Evaluate
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    
    criterion = nn.MSELoss()
    test_metrics = predictor.evaluate(test_loader, criterion)
    
    print(f"\nTest Set Performance:")
    print(f"  - Test Loss: {test_metrics['loss']:.6f}")
    print(f"  - Test MAE: {test_metrics['mae']:.4f} vehicles")
    print(f"  - Test RMSE: {test_metrics['rmse']:.4f} vehicles")
    print(f"  - Test MAPE: {test_metrics['mape']:.2f}%")
    
    # Generate embeddings for Bloch sphere visualization
    print("\n" + "=" * 80)
    print("GENERATING BLOCH SPHERE VISUALIZATION DATA")
    print("=" * 80)
    
    generator = LargeTrafficDataGenerator(network, time_interval=1)
    # Generate 120 seconds so we have enough data for 60-step sequences
    traffic_data, timestamps, time_info = generator.generate_for_bloch_visualization(duration_seconds=120)
    
    print("\nExtracting embeddings from trained model...")
    
    adj_tensor = torch.FloatTensor(adj_normalized).to(device)
    
    all_embeddings = []
    seq_inputs = []
    
    # Create 60 sequences (one per second for 1-minute visualization)
    for i in range(0, min(60, len(traffic_data) - seq_len + 1)):
        x = traffic_data[i:i+seq_len, :]
        
        x_features = np.zeros((seq_len, num_nodes, 4))
        for t in range(seq_len):
            x_features[t, :, 0] = x[t, :] / 100.0
            if t > 0:
                x_features[t, :, 1] = (x[t, :] - x[t-1, :]) / 100.0
                x_features[t, :, 2] = x[t-1, :] / 100.0
            x_features[t, :, 3] = np.sin(2 * np.pi * t / seq_len)
        
        seq_inputs.append(x_features)
    
    print(f"  Created {len(seq_inputs)} sequences for visualization")
    
    batch_size = 16
    for i in range(0, len(seq_inputs), batch_size):
        batch = seq_inputs[i:i+batch_size]
        batch_tensor = torch.FloatTensor(np.array(batch)).to(device)
        
        embeddings = predictor.get_embeddings(batch_tensor, adj_tensor)
        last_layer_emb = embeddings[-1]
        flattened = last_layer_emb.mean(axis=(2, 3))
        
        all_embeddings.append(flattened)
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    print(f"✓ Generated embeddings shape: {all_embeddings.shape}")
    
    # Save for Bloch sphere visualization (60 embeddings for 60-second animation)
    np.savez(
        'unified_stgcn_bloch_data.npz',
        embeddings=all_embeddings[:60],  # 60 embeddings for 60-second visualization
        traffic_data=traffic_data[:60],   # First 60 seconds of traffic
        timestamps=timestamps[:60],
        hour=time_info['hour'][:60],
        minute=time_info['minute'][:60],
        second=time_info['second'][:60],
        adj_matrix=adj_matrix
    )
    print("✓ Saved to unified_stgcn_bloch_data.npz (60 timesteps for 1-minute visualization)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✓ Model successfully evaluated!")
    print(f"✓ Test MAE: {test_metrics['mae']:.4f} vehicles")
    print(f"✓ Test MAPE: {test_metrics['mape']:.2f}%")
    print(f"✓ Bloch sphere data generated for 60 timesteps (1-minute visualization)")
    print("\nFiles generated:")
    print("  - unified_stgcn_bloch_data.npz (embeddings + traffic data)")
    print("\nYou can now:")
    print("  1. Run the Streamlit app: streamlit run streamlit_spatiotemporalGCN.py")
    print("  2. Use unified_stgcn_bloch_data.npz for Bloch sphere visualization")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_trained_model()
