"""
Training Script for Quantum CNN (QCNN) Traffic Prediction
Trains a pure quantum architecture and compares with classical GCN
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path

from chembur_network import ChemburTrafficNetwork
from qcnn_model import QCNN, QCNNPredictor
from traffic_data_generator import create_dataloaders


def train_qcnn():
    """
    Complete pipeline for training and evaluating QCNN
    """
    print("=" * 80)
    print("QUANTUM CNN FOR TRAFFIC PREDICTION - CHEMBUR NETWORK")
    print("=" * 80)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    print("Note: QCNN runs quantum simulations which may be slower than classical models")
    
    # Create network
    print("\n" + "=" * 80)
    print("STEP 1: Loading Chembur Traffic Network")
    print("=" * 80)
    network = ChemburTrafficNetwork()
    network.get_network_info()
    
    # Get adjacency matrix
    adj_matrix = network.get_adjacency_matrix()
    print(f"\nAdjacency Matrix Shape: {adj_matrix.shape}")
    print(f"Number of edges: {np.sum(adj_matrix > 0)}")
    
    # Model hyperparameters
    num_nodes = network.graph.number_of_nodes()
    num_features = 4  # traffic, rate_of_change, prev_traffic, time
    seq_len = 12  # 60 seconds lookback (12 * 5 seconds)
    n_qubits = 8  # Number of qubits (matches number of nodes)
    n_layers = 2  # Number of quantum convolutional layers
    
    print("\n" + "=" * 80)
    print("STEP 2: Creating Quantum CNN Model")
    print("=" * 80)
    print(f"\nQuantum Model Configuration:")
    print(f"  - Number of nodes: {num_nodes}")
    print(f"  - Input features per node: {num_features}")
    print(f"  - Input sequence length: {seq_len} steps (60 seconds)")
    print(f"  - Number of qubits: {n_qubits}")
    print(f"  - Quantum convolutional layers: {n_layers}")
    print(f"  - Quantum gates per layer: ~{4 * (n_qubits - 1)}")
    
    # Create QCNN model
    model = QCNN(
        num_nodes=num_nodes,
        num_features=num_features,
        seq_len=seq_len,
        n_qubits=n_qubits,
        n_layers=n_layers
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    quantum_params = model.q_params.numel()
    classical_params = total_params - quantum_params
    
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Quantum parameters: {quantum_params:,}")
    print(f"  - Classical parameters: {classical_params:,}")
    print(f"  - Quantum/Classical ratio: {quantum_params/total_params:.2%}")
    
    # Create predictor
    predictor = QCNNPredictor(model, device=device)
    
    # Generate training data
    print("\n" + "=" * 80)
    print("STEP 3: Generating Synthetic Traffic Data")
    print("=" * 80)
    train_loader, val_loader, test_loader = create_dataloaders(
        network,
        train_days=5,
        val_days=1,
        test_days=1,
        seq_len=seq_len,
        pred_horizon=1,
        batch_size=16  # Smaller batch size for quantum simulation
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Training
    print("\n" + "=" * 80)
    print("STEP 4: Training Quantum CNN")
    print("=" * 80)
    num_epochs = 30  # Fewer epochs due to quantum simulation overhead
    learning_rate = 0.01
    
    print(f"\nTraining Configuration:")
    print(f"  - Number of epochs: {num_epochs}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Optimizer: Adam")
    print(f"  - Loss function: MSE")
    print("\nWarning: Quantum simulation may take longer than classical training!")
    print("This is expected behavior for quantum circuit simulation.\n")
    
    start_time = time.time()
    train_losses, val_losses = predictor.train_model(
        train_loader,
        val_loader,
        adj_matrix,
        epochs=num_epochs,
        lr=learning_rate
    )
    training_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {training_time:.2f} seconds")
    print(f"  Average time per epoch: {training_time / num_epochs:.2f} seconds")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('QCNN Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('qcnn_training_curves.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: qcnn_training_curves.png")
    
    # Evaluation
    print("\n" + "=" * 80)
    print("STEP 5: Evaluating QCNN on Test Set")
    print("=" * 80)
    
    start_time = time.time()
    results = predictor.evaluate(test_loader, adj_matrix)
    eval_time = time.time() - start_time
    
    print(f"\nOverall Performance:")
    print(f"  - MSE: {results['mse']:.4f}")
    print(f"  - MAE: {results['mae']:.4f}")
    print(f"  - RMSE: {results['rmse']:.4f}")
    print(f"  - Evaluation time: {eval_time:.2f} seconds")
    
    print(f"\nPer-Node Performance:")
    node_names = [network.graph.nodes[i]['name'] for i in range(num_nodes)]
    for i, name in enumerate(node_names):
        print(f"  {name}:")
        print(f"    - MSE: {results['node_mse'][i]:.4f}")
        print(f"    - MAE: {results['node_mae'][i]:.4f}")
    
    # Measure inference latency
    print("\n" + "=" * 80)
    print("STEP 6: Measuring Inference Latency")
    print("=" * 80)
    
    # Get a single batch for timing
    test_iter = iter(test_loader)
    sample_seq, sample_target = next(test_iter)
    
    # Warm-up run
    _ = predictor.predict(sample_seq[:1], adj_matrix)
    
    # Timing runs
    num_runs = 10
    latencies = []
    
    for _ in range(num_runs):
        start = time.time()
        _ = predictor.predict(sample_seq[:1], adj_matrix)
        latencies.append((time.time() - start) * 1000)  # Convert to ms
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    print(f"\nInference Latency (single prediction):")
    print(f"  - Average: {avg_latency:.2f} ms")
    print(f"  - Std Dev: {std_latency:.2f} ms")
    print(f"  - Min: {np.min(latencies):.2f} ms")
    print(f"  - Max: {np.max(latencies):.2f} ms")
    
    # Save model
    print("\n" + "=" * 80)
    print("STEP 7: Saving Model and Results")
    print("=" * 80)
    
    torch.save(model.state_dict(), 'qcnn_traffic_model.pth')
    print("✓ Model saved to: qcnn_traffic_model.pth")
    
    # Save metrics
    metrics = {
        'model_type': 'QCNN',
        'num_qubits': n_qubits,
        'num_layers': n_layers,
        'total_parameters': total_params,
        'quantum_parameters': quantum_params,
        'classical_parameters': classical_params,
        'training_time': training_time,
        'evaluation_time': eval_time,
        'avg_inference_latency_ms': avg_latency,
        'std_inference_latency_ms': std_latency,
        'test_mse': float(results['mse']),
        'test_mae': float(results['mae']),
        'test_rmse': float(results['rmse']),
        'node_mse': results['node_mse'].tolist(),
        'node_mae': results['node_mae'].tolist(),
        'node_names': node_names,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    with open('qcnn_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("✓ Metrics saved to: qcnn_metrics.json")
    
    # Plot predictions vs actual
    plt.figure(figsize=(15, 10))
    
    # Select random samples to visualize
    num_samples = min(100, len(results['predictions']))
    sample_indices = np.random.choice(len(results['predictions']), num_samples, replace=False)
    
    for node in range(num_nodes):
        plt.subplot(3, 3, node + 1)
        plt.scatter(
            results['actuals'][sample_indices, node],
            results['predictions'][sample_indices, node],
            alpha=0.6,
            s=30
        )
        plt.plot(
            [results['actuals'][:, node].min(), results['actuals'][:, node].max()],
            [results['actuals'][:, node].min(), results['actuals'][:, node].max()],
            'r--',
            linewidth=2
        )
        plt.xlabel('Actual Traffic', fontsize=10)
        plt.ylabel('Predicted Traffic', fontsize=10)
        plt.title(f"{node_names[node][:20]}\nMAE: {results['node_mae'][node]:.2f}", fontsize=9)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qcnn_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    print("✓ Prediction plots saved to: qcnn_predictions_vs_actual.png")
    
    # Summary
    print("\n" + "=" * 80)
    print("QUANTUM CNN TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   Architecture: Pure Quantum CNN")
    print(f"   Qubits: {n_qubits}")
    print(f"   Quantum Layers: {n_layers}")
    print(f"   Test RMSE: {results['rmse']:.4f}")
    print(f"   Test MAE: {results['mae']:.4f}")
    print(f"   Avg Inference: {avg_latency:.2f} ms")
    print(f"\n📁 Generated Files:")
    print(f"   - qcnn_traffic_model.pth (trained model)")
    print(f"   - qcnn_metrics.json (performance metrics)")
    print(f"   - qcnn_training_curves.png (training visualization)")
    print(f"   - qcnn_predictions_vs_actual.png (prediction quality)")
    print(f"\n🚀 Next Steps:")
    print(f"   Run comparison dashboard: streamlit run comparison_dashboard.py")
    print("=" * 80)


if __name__ == '__main__':
    train_qcnn()
