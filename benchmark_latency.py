"""
Benchmark Inference Latency for Unified Spatiotemporal GCN
Measures mean latency, throughput, and performance metrics
"""

import torch
import numpy as np
import time
from tqdm import tqdm

from chembur_network import ChemburTrafficNetwork
from unified_spatiotemporal_gcn import UnifiedSpatioTemporalGCN, normalize_adjacency_matrix
from large_traffic_data_generator import create_large_dataloaders


def benchmark_model_latency(num_runs=100, batch_sizes=[1, 4, 8, 16]):
    """
    Benchmark the trained model's inference latency
    
    Args:
        num_runs: Number of inference runs for averaging
        batch_sizes: List of batch sizes to test
    """
    print("=" * 80)
    print("UNIFIED SPATIOTEMPORAL GCN - LATENCY BENCHMARK")
    print("=" * 80)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load network
    print("\nLoading Chembur Traffic Network...")
    network = ChemburTrafficNetwork()
    adj_matrix = network.get_adjacency_matrix()
    adj_normalized = normalize_adjacency_matrix(adj_matrix)
    adj_tensor = torch.FloatTensor(adj_normalized).to(device)
    
    # Model parameters
    num_nodes = 8
    num_features = 4
    seq_len = 60
    pred_horizon = 12
    hidden_channels = 64
    num_layers = 4
    
    print("\nCreating model architecture...")
    model = UnifiedSpatioTemporalGCN(
        num_nodes=num_nodes,
        num_features=num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        output_horizon=pred_horizon,
        kernel_size=3,
        dropout=0.2
    )
    
    # Load trained weights
    print("Loading trained model weights...")
    checkpoint = torch.load('unified_stgcn_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']+1}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    print("\n" + "=" * 80)
    print("LATENCY BENCHMARKING")
    print("=" * 80)
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'─' * 80}")
        print(f"Batch Size: {batch_size}")
        print(f"{'─' * 80}")
        
        # Create dummy input
        x = torch.randn(batch_size, seq_len, num_nodes, num_features).to(device)
        adj_batch = adj_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Warm-up runs
        print("  Running warm-up passes...")
        with torch.no_grad():
            for _ in range(10):
                _ = model(x, adj_batch)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Actual benchmark runs
        print(f"  Running {num_runs} inference passes...")
        latencies = []
        
        with torch.no_grad():
            for _ in tqdm(range(num_runs), desc="  Progress", ncols=70):
                start_time = time.time()
                
                predictions = model(x, adj_batch)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)
        
        # Calculate statistics
        latencies = np.array(latencies)
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Throughput (samples per second)
        throughput = 1000 / mean_latency * batch_size
        
        # Per-sample latency
        per_sample_latency = mean_latency / batch_size
        
        print(f"\n  Results:")
        print(f"    Mean latency:      {mean_latency:.3f} ms")
        print(f"    Std deviation:     {std_latency:.3f} ms")
        print(f"    Min latency:       {min_latency:.3f} ms")
        print(f"    Max latency:       {max_latency:.3f} ms")
        print(f"    Median (P50):      {p50_latency:.3f} ms")
        print(f"    P95:               {p95_latency:.3f} ms")
        print(f"    P99:               {p99_latency:.3f} ms")
        print(f"    Per-sample:        {per_sample_latency:.3f} ms/sample")
        print(f"    Throughput:        {throughput:.2f} samples/sec")
        
        results[batch_size] = {
            'mean_latency_ms': mean_latency,
            'std_latency_ms': std_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'p50_latency_ms': p50_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'per_sample_latency_ms': per_sample_latency,
            'throughput_samples_per_sec': throughput
        }
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n┌─────────────┬──────────────┬───────────────┬────────────────┐")
    print("│ Batch Size  │ Mean Latency │ Per-Sample    │ Throughput     │")
    print("├─────────────┼──────────────┼───────────────┼────────────────┤")
    
    for batch_size, metrics in results.items():
        print(f"│ {batch_size:11d} │ {metrics['mean_latency_ms']:10.3f} ms │ "
              f"{metrics['per_sample_latency_ms']:11.3f} ms │ "
              f"{metrics['throughput_samples_per_sec']:12.2f} s/s │")
    
    print("└─────────────┴──────────────┴───────────────┴────────────────┘")
    
    # Best configuration
    best_throughput_batch = max(results.items(), key=lambda x: x[1]['throughput_samples_per_sec'])
    best_latency_batch = min(results.items(), key=lambda x: x[1]['per_sample_latency_ms'])
    
    print(f"\n📊 Performance Insights:")
    print(f"  ✓ Best throughput: Batch size {best_throughput_batch[0]} "
          f"({best_throughput_batch[1]['throughput_samples_per_sec']:.2f} samples/sec)")
    print(f"  ✓ Lowest latency: Batch size {best_latency_batch[0]} "
          f"({best_latency_batch[1]['per_sample_latency_ms']:.3f} ms/sample)")
    
    # Real-time capability assessment
    single_sample_latency = results[1]['mean_latency_ms']
    prediction_interval_ms = 1000  # 1 second (for real-time traffic prediction)
    
    print(f"\n🚦 Real-Time Capability:")
    print(f"  - Single inference latency: {single_sample_latency:.3f} ms")
    print(f"  - Target prediction interval: {prediction_interval_ms} ms (1 second)")
    
    if single_sample_latency < prediction_interval_ms:
        speedup = prediction_interval_ms / single_sample_latency
        print(f"  ✓ REAL-TIME CAPABLE! ({speedup:.1f}x faster than required)")
        print(f"  - Can process {int(speedup)} predictions per second")
    else:
        print(f"  ✗ Not real-time capable for 1-second intervals")
    
    # Memory usage
    if device == 'cuda':
        print(f"\n💾 GPU Memory Usage:")
        print(f"  - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  - Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    print("\n" + "=" * 80)
    
    # Save results
    import json
    with open('latency_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ Results saved to latency_benchmark_results.json")
    
    return results


def quick_latency_test():
    """Quick test for immediate latency measurement"""
    print("=" * 80)
    print("QUICK LATENCY TEST")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    network = ChemburTrafficNetwork()
    adj_matrix = network.get_adjacency_matrix()
    adj_normalized = normalize_adjacency_matrix(adj_matrix)
    adj_tensor = torch.FloatTensor(adj_normalized).to(device)
    
    model = UnifiedSpatioTemporalGCN(
        num_nodes=8, num_features=4, hidden_channels=64,
        num_layers=4, output_horizon=12, kernel_size=3, dropout=0.2
    )
    
    checkpoint = torch.load('unified_stgcn_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Test with batch size 1
    x = torch.randn(1, 60, 8, 4).to(device)
    adj_batch = adj_tensor.unsqueeze(0)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, adj_batch)
    
    # Measure
    num_runs = 100
    latencies = []
    
    print(f"\nMeasuring latency over {num_runs} runs...")
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(x, adj_batch)
            if device == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.time() - start) * 1000)
    
    mean_lat = np.mean(latencies)
    
    print(f"\n{'=' * 80}")
    print(f"RESULT: Mean Inference Latency = {mean_lat:.3f} ms")
    print(f"{'=' * 80}")
    print(f"  Standard deviation: {np.std(latencies):.3f} ms")
    print(f"  Min: {np.min(latencies):.3f} ms")
    print(f"  Max: {np.max(latencies):.3f} ms")
    print(f"  Median: {np.median(latencies):.3f} ms")
    
    return mean_lat


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Quick test
        quick_latency_test()
    else:
        # Full benchmark
        benchmark_model_latency(
            num_runs=100,
            batch_sizes=[1, 4, 8, 16]
        )
