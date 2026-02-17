"""
Model Comparison Script
Evaluates and compares GCN and QCNN models on the same test data
"""

import torch
import numpy as np
import json
import time
from pathlib import Path

from chembur_network import ChemburTrafficNetwork
from spatiotemporal_gcn import SpatioTemporalGCN, TrafficPredictor
from qcnn_model import QCNN, QCNNPredictor
from traffic_data_generator import create_dataloaders


def load_gcn_model(network, device='cpu'):
    """Load trained GCN model"""
    num_nodes = network.graph.number_of_nodes()
    
    model = SpatioTemporalGCN(
        num_nodes=num_nodes,
        num_features=4,
        gcn_hidden_dims=[32, 64],
        temporal_channels=[64, 128],
        mlp_hidden_dims=[128, 64],
        output_horizon=1,
        dropout=0.2
    )
    
    try:
        model.load_state_dict(torch.load('traffic_gcn_model.pth', map_location=device))
        predictor = TrafficPredictor(model, device=device)
        return predictor, True
    except FileNotFoundError:
        print("Warning: GCN model file not found. Please train the GCN model first.")
        return None, False


def load_qcnn_model(network, device='cpu'):
    """Load trained QCNN model"""
    num_nodes = network.graph.number_of_nodes()
    
    model = QCNN(
        num_nodes=num_nodes,
        num_features=4,
        seq_len=12,
        n_qubits=8,
        n_layers=2
    )
    
    try:
        model.load_state_dict(torch.load('qcnn_traffic_model.pth', map_location=device))
        predictor = QCNNPredictor(model, device=device)
        return predictor, True
    except FileNotFoundError:
        print("Warning: QCNN model file not found. Please train the QCNN model first.")
        return None, False


def evaluate_model(predictor, test_loader, adj_matrix, model_name, device='cpu'):
    """Evaluate a model on test data"""
    print(f"\nEvaluating {model_name}...")
    
    # Test performance
    start_time = time.time()
    results = predictor.evaluate(test_loader, adj_matrix)
    eval_time = time.time() - start_time
    
    # Measure inference latency
    test_iter = iter(test_loader)
    sample_seq, _, _ = next(test_iter)
    
    # Warm-up
    _ = predictor.predict(sample_seq[:1], adj_matrix)
    
    # Timing
    num_runs = 20
    latencies = []
    for _ in range(num_runs):
        start = time.time()
        _ = predictor.predict(sample_seq[:1], adj_matrix)
        latencies.append((time.time() - start) * 1000)
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    print(f"✓ {model_name} Evaluation Complete")
    print(f"  - Test RMSE: {results['rmse']:.4f}")
    print(f"  - Test MAE: {results['mae']:.4f}")
    print(f"  - Avg Latency: {avg_latency:.2f} ms")
    
    return {
        'model_name': model_name,
        'mse': float(results['mse']),
        'mae': float(results['mae']),
        'rmse': float(results['rmse']),
        'avg_latency_ms': avg_latency,
        'std_latency_ms': std_latency,
        'eval_time': eval_time,
        'node_mse': results['node_mse'].tolist(),
        'node_mae': results['node_mae'].tolist(),
        'predictions': results['predictions'].tolist(),
        'actuals': results['actuals'].tolist()
    }


def compare_models():
    """
    Compare GCN and QCNN models
    """
    print("=" * 80)
    print("MODEL COMPARISON: GCN vs QCNN")
    print("=" * 80)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load network
    print("\nLoading network...")
    network = ChemburTrafficNetwork()
    adj_matrix = network.get_adjacency_matrix()
    num_nodes = network.graph.number_of_nodes()
    node_names = [network.graph.nodes[i]['name'] for i in range(num_nodes)]
    
    # Generate test data
    print("Generating test data...")
    _, _, test_loader = create_dataloaders(
        network,
        train_days=5,
        val_days=1,
        test_days=1,
        seq_len=12,
        pred_horizon=1,
        batch_size=32
    )
    
    # Load models
    print("\n" + "=" * 80)
    print("Loading Models")
    print("=" * 80)
    
    gcn_predictor, gcn_loaded = load_gcn_model(network, device)
    qcnn_predictor, qcnn_loaded = load_qcnn_model(network, device)
    
    if not gcn_loaded and not qcnn_loaded:
        print("\n❌ No models found. Please train at least one model first.")
        return
    
    # Evaluate models
    print("\n" + "=" * 80)
    print("Evaluating Models")
    print("=" * 80)
    
    results = {}
    
    if gcn_loaded:
        results['gcn'] = evaluate_model(
            gcn_predictor, test_loader, adj_matrix, 'Spatiotemporal GCN', device
        )
    
    if qcnn_loaded:
        results['qcnn'] = evaluate_model(
            qcnn_predictor, test_loader, adj_matrix, 'Quantum CNN', device
        )
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    if gcn_loaded and qcnn_loaded:
        print("\n📊 Performance Comparison:")
        print(f"\n{'Metric':<30} {'GCN':<20} {'QCNN':<20} {'Winner':<10}")
        print("-" * 80)
        
        # RMSE
        gcn_rmse = results['gcn']['rmse']
        qcnn_rmse = results['qcnn']['rmse']
        rmse_winner = 'GCN' if gcn_rmse < qcnn_rmse else 'QCNN'
        print(f"{'Root Mean Squared Error':<30} {gcn_rmse:<20.4f} {qcnn_rmse:<20.4f} {rmse_winner:<10}")
        
        # MAE
        gcn_mae = results['gcn']['mae']
        qcnn_mae = results['qcnn']['mae']
        mae_winner = 'GCN' if gcn_mae < qcnn_mae else 'QCNN'
        print(f"{'Mean Absolute Error':<30} {gcn_mae:<20.4f} {qcnn_mae:<20.4f} {mae_winner:<10}")
        
        # Latency
        gcn_lat = results['gcn']['avg_latency_ms']
        qcnn_lat = results['qcnn']['avg_latency_ms']
        lat_winner = 'GCN' if gcn_lat < qcnn_lat else 'QCNN'
        print(f"{'Inference Latency (ms)':<30} {gcn_lat:<20.2f} {qcnn_lat:<20.2f} {lat_winner:<10}")
        
        # Improvements
        print(f"\n📈 Relative Differences:")
        rmse_diff = ((qcnn_rmse - gcn_rmse) / gcn_rmse) * 100
        mae_diff = ((qcnn_mae - gcn_mae) / gcn_mae) * 100
        lat_diff = ((qcnn_lat - gcn_lat) / gcn_lat) * 100
        
        print(f"  RMSE: QCNN is {abs(rmse_diff):.2f}% {'worse' if rmse_diff > 0 else 'better'} than GCN")
        print(f"  MAE:  QCNN is {abs(mae_diff):.2f}% {'worse' if mae_diff > 0 else 'better'} than GCN")
        print(f"  Latency: QCNN is {abs(lat_diff):.2f}% {'slower' if lat_diff > 0 else 'faster'} than GCN")
        
        # Per-node comparison
        print(f"\n📍 Per-Node Performance (MAE):")
        print(f"\n{'Junction':<40} {'GCN':<15} {'QCNN':<15} {'Diff':<10}")
        print("-" * 80)
        
        for i, name in enumerate(node_names):
            gcn_node = results['gcn']['node_mae'][i]
            qcnn_node = results['qcnn']['node_mae'][i]
            diff = qcnn_node - gcn_node
            diff_str = f"{'+' if diff > 0 else ''}{diff:.3f}"
            print(f"{name[:38]:<40} {gcn_node:<15.4f} {qcnn_node:<15.4f} {diff_str:<10}")
    
    # Save comparison results
    comparison_data = {
        'node_names': node_names,
        'models': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('model_comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=4)
    
    print("\n✓ Comparison saved to: model_comparison.json")
    print("\n🚀 Launch comparison dashboard: streamlit run comparison_dashboard.py")
    print("=" * 80)


if __name__ == '__main__':
    compare_models()
