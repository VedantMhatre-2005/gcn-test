"""
Comparative Latency Benchmark: Original 3-Layer vs Unified ST-GCN
Measures and compares inference latency between the two architectures
"""

import torch
import numpy as np
import time
from tqdm import tqdm

from chembur_network import ChemburTrafficNetwork
from spatiotemporal_gcn import SpatioTemporalGCN as OriginalGCN
from unified_spatiotemporal_gcn import UnifiedSpatioTemporalGCN, normalize_adjacency_matrix


def benchmark_model(model, x, adj, num_runs=100, model_name="Model", device='cpu'):
    """
    Benchmark a single model
    
    Args:
        model: PyTorch model
        x: Input tensor
        adj: Adjacency tensor
        num_runs: Number of inference runs
        model_name: Name for display
        device: Device to run on
    
    Returns:
        Dictionary with latency statistics
    """
    model.eval()
    
    # Warm-up runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, adj)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark runs
    latencies = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc=f"  {model_name}", ncols=70):
            start_time = time.time()
            
            predictions = model(x, adj)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
    
    latencies = np.array(latencies)
    
    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'median': np.median(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99)
    }


def compare_architectures(num_runs=100):
    """
    Compare latency between original 3-layer architecture and Unified ST-GCN
    """
    print("=" * 80)
    print("LATENCY COMPARISON: 3-LAYER GCN vs UNIFIED ST-GCN")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load network
    print("\nLoading Chembur Traffic Network...")
    network = ChemburTrafficNetwork()
    adj_matrix = network.get_adjacency_matrix()
    adj_normalized = normalize_adjacency_matrix(adj_matrix)
    
    num_nodes = 8
    num_features = 4
    
    # ========================================
    # 1. ORIGINAL 3-LAYER ARCHITECTURE
    # ========================================
    print("\n" + "=" * 80)
    print("MODEL 1: ORIGINAL 3-LAYER ARCHITECTURE (GCN + CNN + MLP)")
    print("=" * 80)
    
    # Original model uses seq_len=12
    seq_len_original = 12
    pred_horizon_original = 1
    
    print("\nArchitecture:")
    print("  - Stage 1: GCN layers [32, 64]")
    print("  - Stage 2: Temporal CNN [64, 128]")
    print("  - Stage 3: MLP [128, 64]")
    print(f"  - Input: ({seq_len_original} timesteps, {num_nodes} nodes, {num_features} features)")
    
    original_model = OriginalGCN(
        num_nodes=num_nodes,
        num_features=num_features,
        gcn_hidden_dims=[32, 64],
        temporal_channels=[64, 128],
        mlp_hidden_dims=[128, 64],
        output_horizon=pred_horizon_original,
        dropout=0.2
    )
    
    # Load trained weights if available
    try:
        checkpoint = torch.load('traffic_gcn_model.pth', weights_only=False, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            original_model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Loaded trained weights from traffic_gcn_model.pth")
        else:
            original_model.load_state_dict(checkpoint)
            print("✓ Loaded trained weights from traffic_gcn_model.pth")
    except Exception as e:
        print(f"⚠ Could not load trained weights: {e}")
        print("  Using random initialization for comparison")
    
    original_model = original_model.to(device)
    original_params = sum(p.numel() for p in original_model.parameters())
    
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {original_params:,}")
    print(f"  - Model size: {original_params * 4 / 1024 / 1024:.2f} MB")
    
    # Prepare input for original model
    x_original = torch.randn(1, seq_len_original, num_nodes, num_features).to(device)
    adj_original = torch.FloatTensor(adj_normalized).to(device)
    
    # Benchmark original model
    print(f"\nBenchmarking original model ({num_runs} runs)...")
    original_results = benchmark_model(
        original_model, x_original, adj_original, 
        num_runs, "Original 3-Layer", device
    )
    
    # ========================================
    # 2. UNIFIED ST-GCN ARCHITECTURE
    # ========================================
    print("\n" + "=" * 80)
    print("MODEL 2: UNIFIED SPATIOTEMPORAL GCN")
    print("=" * 80)
    
    # Unified model uses seq_len=60
    seq_len_unified = 60
    pred_horizon_unified = 12
    
    print("\nArchitecture:")
    print("  - Integrated ST-Conv blocks [64 channels × 4 layers]")
    print("  - Simultaneous spatial + temporal processing")
    print(f"  - Input: ({seq_len_unified} timesteps, {num_nodes} nodes, {num_features} features)")
    
    unified_model = UnifiedSpatioTemporalGCN(
        num_nodes=num_nodes,
        num_features=num_features,
        hidden_channels=64,
        num_layers=4,
        output_horizon=pred_horizon_unified,
        kernel_size=3,
        dropout=0.2
    )
    
    # Load trained weights
    try:
        checkpoint = torch.load('unified_stgcn_best.pth', weights_only=False, map_location=device)
        unified_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded trained weights from epoch {checkpoint['epoch']+1}")
    except Exception as e:
        print(f"⚠ Could not load trained weights: {e}")
        print("  Using random initialization for comparison")
    
    unified_model = unified_model.to(device)
    unified_params = sum(p.numel() for p in unified_model.parameters())
    
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {unified_params:,}")
    print(f"  - Model size: {unified_params * 4 / 1024 / 1024:.2f} MB")
    
    # Prepare input for unified model
    x_unified = torch.randn(1, seq_len_unified, num_nodes, num_features).to(device)
    adj_unified = torch.FloatTensor(adj_normalized).unsqueeze(0).to(device)
    
    # Benchmark unified model
    print(f"\nBenchmarking unified model ({num_runs} runs)...")
    unified_results = benchmark_model(
        unified_model, x_unified, adj_unified,
        num_runs, "Unified ST-GCN", device
    )
    
    # ========================================
    # COMPARISON
    # ========================================
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    print("\n┌────────────────────────┬──────────────────┬──────────────────┬─────────────┐")
    print("│ Metric                 │ Original 3-Layer │ Unified ST-GCN   │ Improvement │")
    print("├────────────────────────┼──────────────────┼──────────────────┼─────────────┤")
    
    metrics = [
        ('Mean Latency (ms)', 'mean'),
        ('Std Deviation (ms)', 'std'),
        ('Min Latency (ms)', 'min'),
        ('Max Latency (ms)', 'max'),
        ('Median/P50 (ms)', 'median'),
        ('P95 (ms)', 'p95'),
        ('P99 (ms)', 'p99')
    ]
    
    for metric_name, key in metrics:
        orig_val = original_results[key]
        unified_val = unified_results[key]
        
        if orig_val > 0:
            improvement = ((orig_val - unified_val) / orig_val) * 100
            improvement_str = f"{improvement:+.1f}%"
        else:
            improvement_str = "N/A"
        
        print(f"│ {metric_name:22s} │ {orig_val:14.3f} │ {unified_val:14.3f} │ {improvement_str:11s} │")
    
    print("└────────────────────────┴──────────────────┴──────────────────┴─────────────┘")
    
    # Throughput comparison
    orig_throughput = 1000 / original_results['mean']
    unified_throughput = 1000 / unified_results['mean']
    throughput_improvement = ((unified_throughput - orig_throughput) / orig_throughput) * 100
    
    print(f"\n📊 Throughput:")
    print(f"  - Original 3-Layer: {orig_throughput:.2f} inferences/sec")
    print(f"  - Unified ST-GCN:   {unified_throughput:.2f} inferences/sec")
    print(f"  - Improvement:      {throughput_improvement:+.1f}%")
    
    # Parameter comparison
    param_reduction = ((original_params - unified_params) / original_params) * 100
    
    print(f"\n🔢 Model Size:")
    print(f"  - Original 3-Layer: {original_params:,} parameters")
    print(f"  - Unified ST-GCN:   {unified_params:,} parameters")
    print(f"  - Reduction:        {param_reduction:+.1f}%")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    speedup = original_results['mean'] / unified_results['mean']
    
    if unified_results['mean'] < original_results['mean']:
        print(f"\n✅ Unified ST-GCN is FASTER!")
        print(f"   - Speedup: {speedup:.2f}x")
        print(f"   - Latency reduction: {((original_results['mean'] - unified_results['mean']) / original_results['mean'] * 100):.1f}%")
    else:
        print(f"\n⚠️  Original 3-Layer is faster")
        print(f"   - Slowdown: {1/speedup:.2f}x")
    
    print(f"\n🎯 Key Insights:")
    
    if param_reduction > 0:
        print(f"   ✓ Unified ST-GCN uses {abs(param_reduction):.1f}% fewer parameters")
    else:
        print(f"   • Unified ST-GCN uses {abs(param_reduction):.1f}% more parameters")
    
    # Input size comparison
    orig_input_size = seq_len_original * num_nodes * num_features
    unified_input_size = seq_len_unified * num_nodes * num_features
    input_ratio = unified_input_size / orig_input_size
    
    print(f"   • Unified ST-GCN processes {input_ratio:.1f}x more input data")
    print(f"     (60 vs 12 timesteps)")
    
    if unified_results['mean'] < original_results['mean'] and input_ratio > 1:
        print(f"   🌟 Despite processing {input_ratio:.1f}x more data, Unified ST-GCN is still faster!")
    
    # Real-time capability
    print(f"\n⏱️  Real-Time Capability (1-second prediction interval):")
    print(f"   - Original 3-Layer: {1000/original_results['mean']:.0f} predictions/sec")
    print(f"   - Unified ST-GCN:   {1000/unified_results['mean']:.0f} predictions/sec")
    print(f"   - Both models: {'✓ Real-time capable' if max(original_results['mean'], unified_results['mean']) < 1000 else '✗ Not real-time'}")
    
    print("\n" + "=" * 80)
    
    # Save comparison results
    import json
    comparison = {
        'original_3layer': {
            'parameters': original_params,
            'seq_len': seq_len_original,
            'pred_horizon': pred_horizon_original,
            'latency_ms': original_results
        },
        'unified_stgcn': {
            'parameters': unified_params,
            'seq_len': seq_len_unified,
            'pred_horizon': pred_horizon_unified,
            'latency_ms': unified_results
        },
        'comparison': {
            'speedup': speedup,
            'parameter_reduction_percent': param_reduction,
            'throughput_improvement_percent': throughput_improvement,
            'latency_reduction_percent': ((original_results['mean'] - unified_results['mean']) / original_results['mean'] * 100)
        }
    }
    
    with open('architecture_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("\n✓ Comparison results saved to architecture_comparison.json")
    
    return comparison


if __name__ == "__main__":
    compare_architectures(num_runs=100)
