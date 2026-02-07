# Traffic Prediction with Spatiotemporal GCN - Chembur Network

This project implements a Graph Convolutional Network (GCN) for traffic prediction in a small network of traffic junctions in Chembur, Mumbai. The model predicts traffic flow for the next 5 seconds using a spatiotemporal architecture.

## Project Overview

### Network Structure
- **Location**: Chembur, Mumbai
- **Nodes**: 8 major traffic junctions
- **Edges**: Unidirectional roads connecting junctions
- **Node Attributes**: 
  - Junction capacity (max vehicles)
  - Signal cycle time
  - Junction type (major/minor)
  - Coordinates
- **Edge Attributes**:
  - Road length (meters)
  - Number of lanes
  - Speed limit (km/h)
  - Road quality score

### Model Architecture

The **Spatiotemporal GCN** combines three key components:

1. **GCN (Graph Convolutional Network)**: Captures spatial dependencies between connected junctions
   - 2 GCN layers with 32 and 64 hidden units
   - Processes graph structure to learn traffic patterns across the network

2. **1D-CNN (1D Convolutional Neural Network)**: Captures temporal dependencies in traffic flow
   - 2 convolutional layers with 64 and 128 channels
   - Extracts temporal patterns from historical traffic data

3. **MLP (Multi-Layer Perceptron)**: Generates final predictions
   - 2 fully connected layers with 128 and 64 hidden units
   - Outputs traffic predictions for next 5 seconds

### Features

- **Input**: 60 seconds of historical traffic data (12 time steps × 5 seconds)
- **Output**: Predicted traffic for next 5 seconds
- **Training**: Synthetic traffic data with realistic patterns (rush hours, normal hours)
- **Evaluation**: MSE, RMSE, MAE, and MAPE metrics
- **Latency Measurement**: Real-time prediction capability analysis

## File Structure

```
gcn-test1/
├── chembur_network.py          # Network graph definition
├── spatiotemporal_gcn.py       # Model architecture
├── traffic_data_generator.py   # Synthetic data generation
├── train_model.py              # Training and evaluation pipeline
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the complete pipeline:
```bash
python train_model.py
```

This will:
1. Create the Chembur traffic network
2. Build the spatiotemporal GCN model
3. Generate synthetic traffic data
4. Train the model
5. Evaluate performance on test data
6. Measure prediction latency
7. Generate visualizations

### Extract embeddings for quantum encoding:
```bash
python extract_embeddings.py
```

This will:
1. Extract spatial, temporal, and combined embeddings
2. Save embeddings to `traffic_embeddings.npz`
3. Analyze compression potential (PCA, DCT)
4. Generate embedding visualizations
5. Provide quantum encoding recommendations

### Individual Components

#### Visualize the network:
```python
from chembur_network import ChemburTrafficNetwork

network = ChemburTrafficNetwork()
network.get_network_info()
network.visualize_network()
```

#### Generate traffic data:
```python
from traffic_data_generator import TrafficDataGenerator

generator = TrafficDataGenerator(network, time_interval=5)
traffic_data, timestamps = generator.generate_traffic_sequence(num_days=7)
```

#### Test the model:
```python
from spatiotemporal_gcn import SpatioTemporalGCN

model = SpatioTemporalGCN(
    num_nodes=8,
    num_features=4,
    gcn_hidden_dims=[32, 64],
    temporal_channels=[64, 128],
    mlp_hidden_dims=[128, 64],
    output_horizon=1
)
```

## Outputs

The training script generates several outputs:

1. **chembur_network.png**: Visualization of the traffic network
2. **traffic_patterns.png**: Sample traffic patterns over time
3. **training_curves.png**: Training and validation loss curves
4. **predictions_vs_actual.png**: Model predictions vs actual traffic
5. **traffic_gcn_model.pth**: Saved model weights

The embedding extraction script generates:

6. **traffic_embeddings.npz**: Compressed numpy archive containing:
   - Spatial embeddings (batch, 12, 8, 64)
   - Temporal embeddings (batch, 128)
   - Combined embeddings (batch, 640)
   - Predictions and targets
7. **embedding_visualization.png**: Multi-panel visualization of embeddings including:
   - Spatial embedding heatmaps
   - Temporal distribution
   - PCA projection
   - Dimension variance analysis

## Model Performance

The model is evaluated on:
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Latency**: Prediction time for real-time feasibility

## Traffic Junctions (Nodes)

1. **Chembur Naka** - Major intersection
2. **Swami Vivekanand Road Junction** - Major junction
3. **Diamond Garden Junction** - Minor junction
4. **Chembur Station Junction** - Major junction
5. **RCF Colony Junction** - Minor junction
6. **Tilak Nagar Junction** - Minor junction
7. **Mahul Road Junction** - Minor junction
8. **Eastern Express Highway Junction** - Major junction

## Technical Details

### Input Features (per node)
- Current traffic level (normalized)
- Rate of change in traffic
- Previous time step traffic
- Time of day (cyclic encoding)

### Network Properties
- **Adjacency Matrix**: Normalized with self-loops for GCN
- **Graph Type**: Directed (unidirectional traffic)
- **Node Features**: Junction capacity, signal timing, type
- **Edge Features**: Road length, lanes, speed limit, quality

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Learning Rate**: 0.001
- **Epochs**: 50
- **Batch Size**: 32

## Real-time Performance

The model is designed for real-time traffic prediction with:
- Prediction window: 5 seconds
- Low latency inference (typically < 100ms on CPU)
- Suitable for deployment in traffic management systems

## Embeddings for Quantum Computing

The model generates rich embeddings at multiple stages:

### Extracted Embeddings
- **Spatial Embeddings**: (samples, 12 timesteps, 8 nodes, 64 features)
  - Captures spatial dependencies via GCN
  - Per-node 64-dimensional representations
- **Temporal Embeddings**: (samples, 128 features)
  - Captures temporal patterns via 1D-CNN
  - Global temporal context
- **Combined Embeddings**: (samples, 640 features)
  - Fused spatial-temporal representation
  - Ready for downstream quantum encoding

### Quantum Encoding Analysis
The embeddings have been analyzed for NISQ device compatibility:

| Compression Method | Dimensions | Variance Retained | Qubits Needed |
|-------------------|------------|-------------------|---------------|
| PCA 32-dim | 32 | 99.92% | 5 qubits |
| PCA 64-dim | 64 | 99.99% | 6 qubits |
| DCT Top 10% | 64 | ~99% | 6 qubits |
| DCT Top 20% | 128 | ~99% | 7 qubits |

**Recommended for NISQ devices:**
- PCA compress 640-dim → 32-dim
- Angle encoding (1 feature per qubit)
- Requires 32 qubits (feasible on current quantum hardware)
- Shallow circuit depth: O(32)

## Future Enhancements

- Integration with real traffic data from sensors
- Multi-step ahead predictions
- Attention mechanisms for important junctions
- Transfer learning to other city networks
- Real-time streaming data processing

## Dependencies

- Python 3.7+
- PyTorch 1.9+
- NetworkX 2.5+
- NumPy 1.19+
- Matplotlib 3.3+
- SciPy 1.5+

## License

This is an educational project for traffic prediction research.

## Author

Traffic Prediction System - Chembur Network Implementation
