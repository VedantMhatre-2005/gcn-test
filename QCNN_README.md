# Quantum CNN Implementation for Traffic Prediction

## Overview

This project implements a **Quantum Convolutional Neural Network (QCNN)** using PennyLane for traffic prediction on the Chembur network graph. It provides a comparison between a purely quantum architecture and the existing classical Spatiotemporal GCN implementation.

## Architecture Components

### 1. **QCNN Model** (`qcnn_model.py`)

The quantum architecture consists of:

- **Quantum Feature Encoding**: Encodes classical node features into quantum states using angle encoding (RY and RZ gates)
- **Graph Structure Encoding**: Uses CNOT gates to create entanglement based on graph adjacency
- **Quantum Convolutional Layers**: Parameterized two-qubit gates that create local entanglement patterns
- **Quantum Pooling**: Measurement-based dimensionality reduction
- **Classical Post-processing**: Fully connected layers for final predictions

#### Technical Details:
- **Qubits**: 8 (matching the number of nodes)
- **Quantum Layers**: 2 convolutional layers
- **Parameters**: ~100 quantum parameters + classical layers
- **Device**: PennyLane default.qubit simulator

### 2. **Training Script** (`train_qcnn.py`)

Trains the QCNN model with:
- Adam optimizer
- MSE loss function
- Batch size: 16 (smaller for quantum simulation)
- Training epochs: 30
- Learning rate: 0.01

Generates:
- `qcnn_traffic_model.pth` - Trained model weights
- `qcnn_metrics.json` - Performance metrics
- `qcnn_training_curves.png` - Training visualization
- `qcnn_predictions_vs_actual.png` - Prediction quality plots

### 3. **Comparison Tools** (`compare_models.py`)

Evaluates both GCN and QCNN on the same test data and compares:
- RMSE, MAE, MSE
- Inference latency
- Per-node performance
- Generates `model_comparison.json`

### 4. **Comparison Dashboard** (`comparison_dashboard.py`)

Interactive Streamlit dashboard featuring:
- **Overview Tab**: Side-by-side metrics comparison
- **Performance Metrics Tab**: Detailed charts and per-node analysis
- **Predictions Analysis Tab**: Scatter plots and training curves
- **Architecture Details Tab**: Model specifications
- **Model Insights Tab**: Key findings and analysis

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

Required packages:
- `pennylane>=0.33.0` - Quantum machine learning library
- `pennylane-lightning>=0.33.0` - Fast quantum simulator
- `torch>=1.9.0` - PyTorch
- `streamlit>=1.28.0` - Dashboard framework
- `plotly>=5.14.0` - Interactive plotting
- Plus standard dependencies (numpy, matplotlib, networkx, etc.)

## Quick Start

### Step 1: Train Classical GCN (if not already done)
```bash
python train_model.py
# or
run_dashboard.bat  # Windows
```

### Step 2: Train Quantum CNN
```bash
python train_qcnn.py
# or
train_qcnn.bat  # Windows
```

### Step 3: Generate Comparison
```bash
python compare_models.py
# or
run_comparison.bat  # Windows
```

### Step 4: Launch Comparison Dashboard
```bash
streamlit run comparison_dashboard.py
# or
run_comparison_dashboard.bat  # Windows
```

## File Structure

```
gcn-test/
├── qcnn_model.py                    # QCNN architecture
├── train_qcnn.py                    # QCNN training script
├── compare_models.py                # Model comparison utility
├── comparison_dashboard.py          # Interactive comparison dashboard
├── train_qcnn.bat                   # Windows: Train QCNN
├── run_comparison.bat               # Windows: Run comparison
├── run_comparison_dashboard.bat     # Windows: Launch dashboard
├── qcnn_traffic_model.pth          # Trained QCNN weights (generated)
├── qcnn_metrics.json               # QCNN performance metrics (generated)
├── model_comparison.json           # Comparison results (generated)
├── qcnn_training_curves.png        # Training visualization (generated)
└── qcnn_predictions_vs_actual.png  # Prediction plots (generated)

# Existing files (unchanged)
├── chembur_network.py              # Network graph definition
├── spatiotemporal_gcn.py          # Classical GCN model
├── train_model.py                  # GCN training script
├── dashboard.py                    # GCN dashboard
├── traffic_data_generator.py      # Data generation
└── requirements.txt                # Updated with PennyLane
```

## How QCNN Works

### 1. Feature Encoding
Each node's features (traffic, rate_of_change, prev_traffic, time) are encoded into quantum states:
```python
# Classical features → Quantum states
RY(feature[0], qubit_0)
RZ(feature[4], qubit_0)
...
```

### 2. Graph Structure Encoding
Graph connectivity creates entanglement:
```python
# For each edge in adjacency matrix
CNOT(node_i, node_j)
```

### 3. Quantum Convolution
Parameterized gates create local entanglement patterns:
```python
# For each adjacent qubit pair
RY(θ1, qubit_i)
RY(θ2, qubit_j)
CNOT(qubit_i, qubit_j)
RY(θ3, qubit_i)
RY(θ4, qubit_j)
CNOT(qubit_j, qubit_i)
```

### 4. Measurement & Post-processing
- Measure expectation values of Pauli-Z operators
- Feed through classical neural network for final prediction

## Performance Considerations

### Quantum Simulation Overhead
- **Expected**: QCNN is slower than GCN during training and inference
- **Reason**: Simulating quantum circuits on classical hardware is computationally expensive
- **Real quantum hardware**: Would provide significant speedup

### Model Comparison
The comparison dashboard shows:
- **Accuracy**: How prediction quality compares
- **Speed**: Inference latency differences
- **Scalability**: Parameter efficiency

## Key Differences: GCN vs QCNN

| Aspect | Spatiotemporal GCN | Quantum CNN |
|--------|-------------------|-------------|
| **Architecture** | Graph Conv + Temporal Conv | Quantum Circuits + Classical |
| **Parameters** | ~100K+ | ~100 quantum + ~10K classical |
| **State Space** | Linear | Exponential |
| **Hardware** | CPU/GPU | Quantum Simulator (classical) |
| **Inference Speed** | Fast (~10-50ms) | Slower (~100-500ms on simulator) |
| **Training** | Standard backprop | Quantum gradient descent |
| **Advantages** | Mature, fast, scalable | Novel, exponential expressivity |

## Quantum Advantages

1. **Exponential State Space**: 8 qubits can represent 2^8 = 256 states simultaneously
2. **Entanglement**: Captures complex correlations between nodes
3. **Quantum Interference**: Can find optimal solutions through interference
4. **Future Potential**: Real quantum hardware will provide speedup

## Future Enhancements

### Quantum Hardware
- Deploy on IBM Quantum, Rigetti, or IonQ devices
- Compare simulator vs real hardware performance

### Hybrid Approaches
- Ensemble classical + quantum models
- Use QCNN for feature extraction, GCN for prediction

### Advanced Techniques
- Variational Quantum Eigensolver (VQE) for optimization
- Quantum Approximate Optimization Algorithm (QAOA)
- Quantum Graph Neural Networks (QGNN)

## Troubleshooting

### Installation Issues
```bash
# If PennyLane installation fails
pip install --upgrade pip
pip install pennylane --no-cache-dir
```

### Memory Issues
```bash
# Reduce batch size in train_qcnn.py
batch_size=8  # Instead of 16
```

### Slow Training
```bash
# Reduce number of epochs
num_epochs=20  # Instead of 30

# Or use fewer qubits (in qcnn_model.py)
n_qubits=6  # Instead of 8
```

## References

1. **Quantum Convolutional Neural Networks**  
   Cong, I., Choi, S., & Lukin, M. D. (2019). Nature Physics.

2. **PennyLane: Automatic differentiation of hybrid quantum-classical computations**  
   Bergholm et al. (2018)

3. **Graph Convolutional Networks**  
   Kipf, T. N., & Welling, M. (2017). ICLR.

4. **Variational Quantum Algorithms**  
   Cerezo et al. (2021). Nature Reviews Physics.

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please refer to the main project README or create an issue in the repository.

---

**Note**: This implementation uses quantum circuit simulation on classical hardware. For production deployment, consider:
- Real quantum hardware for potential speedup
- Hybrid quantum-classical architectures
- Careful benchmarking against classical baselines
