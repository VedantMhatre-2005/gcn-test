# Unified Spatiotemporal GCN for Traffic Prediction

## Overview

This project implements a **Unified Spatiotemporal Graph Convolutional Network (ST-GCN)** that replaces the traditional GCN + CNN + MLP architecture with a single integrated model for traffic prediction on the Chembur traffic network.

## Architecture Comparison

### Traditional Architecture
```
Input → GCN (spatial) → CNN (temporal) → MLP → Output
```
- **Sequential processing**: Spatial and temporal features extracted separately
- **3 separate stages**: GCN, CNN, and MLP
- **Limited interaction**: Spatial and temporal information processed independently

### Unified ST-GCN Architecture
```
Input → Embedding → [ST-Conv Block × N] → Pooling → Output
```
- **Integrated processing**: Spatial and temporal features extracted simultaneously
- **Single unified architecture**: ST-Conv blocks handle both dimensions
- **Better information flow**: Residual connections and batch normalization
- **More efficient**: Fewer parameters, faster training

## Key Components

### 1. ST-Conv Block (Spatiotemporal Convolution Block)
Each ST-Conv block contains:
- **Temporal Convolution**: 1D convolution across time dimension
- **Batch Normalization**: Normalizes temporal features
- **Spatial Graph Convolution**: Graph convolution across nodes
- **Batch Normalization**: Normalizes spatial features
- **Residual Connection**: Skip connection from input
- **Dropout**: Regularization

### 2. Architecture Layers
1. **Input Embedding**: 1×1 convolution to project features to hidden dimension
2. **Multiple ST-Conv Blocks**: Stack of spatiotemporal convolution blocks
3. **Temporal Pooling**: Average pooling over time dimension
4. **Output Projection**: 1×1 convolution to generate predictions

## Files Structure

```
.
├── unified_spatiotemporal_gcn.py       # Unified ST-GCN model implementation
├── large_traffic_data_generator.py     # Large-scale data generator (1-second resolution)
├── train_unified_stgcn.py              # Training script
├── streamlit_spatiotemporalGCN.py      # Interactive visualization app
├── train_unified_stgcn.bat             # Batch file to run training
├── run_stgcn_visualization.bat         # Batch file to run Streamlit app
└── README_UNIFIED_STGCN.md             # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision numpy networkx matplotlib streamlit
```

### 2. Train the Model

**Option A: Using Python**
```bash
python train_unified_stgcn.py
```

**Option B: Using Batch File (Windows)**
```bash
train_unified_stgcn.bat
```

### 3. Visualize the Architecture

**Option A: Using Python**
```bash
streamlit run streamlit_spatiotemporalGCN.py
```

**Option B: Using Batch File (Windows)**
```bash
run_stgcn_visualization.bat
```

## Model Configuration

### Default Hyperparameters

```python
num_nodes = 8              # Number of nodes in Chembur network
num_features = 4           # Input features per node
seq_len = 60               # 60 seconds lookback (1-second resolution)
pred_horizon = 12          # Predict 12 seconds ahead
hidden_channels = 64       # Hidden dimension
num_layers = 4             # Number of ST-Conv blocks
kernel_size = 3            # Temporal kernel size
dropout = 0.2              # Dropout rate
```

### Training Configuration

```python
train_minutes = 180        # 3 hours of training data
val_minutes = 30           # 30 minutes of validation data  
test_minutes = 30          # 30 minutes of test data
batch_size = 16            # Batch size
learning_rate = 0.001      # Learning rate
num_epochs = 100           # Maximum epochs
early_stopping = 20        # Early stopping patience
```

## Data Generation

### Large-Scale Traffic Data

The `LargeTrafficDataGenerator` creates high-resolution traffic data with:
- **Time resolution**: 1 second per timestep (vs 5 seconds in original)
- **Realistic patterns**: Rush hours, daily cycles, traffic light effects
- **Spatial correlation**: Connected nodes have correlated traffic
- **Multiple frequencies**: 1-minute, 2-minute, and 5-minute cycles

### Bloch Sphere Visualization Data

The training script automatically generates data for 1-minute Bloch sphere visualization:
- **Duration**: 60 seconds
- **Resolution**: 1 second
- **Embeddings**: Extracted from all ST-Conv blocks
- **Output file**: `unified_stgcn_bloch_data.npz`

## Output Files

After training, the following files are generated:

1. **unified_stgcn_best.pth** - Best model checkpoint
2. **unified_stgcn_history.json** - Training history and metrics
3. **unified_stgcn_training_curves.png** - Visualization of training progress
4. **unified_stgcn_bloch_data.npz** - Data for Bloch sphere visualization

## Performance Metrics

The model is evaluated using:
- **MAE** (Mean Absolute Error): Average vehicle count error
- **RMSE** (Root Mean Squared Error): Penalizes large errors more
- **MAPE** (Mean Absolute Percentage Error): Percentage error
- **Loss**: MSE loss function

## Interactive Visualization

The Streamlit app (`streamlit_spatiotemporalGCN.py`) provides:

### Features
- **Interactive architecture graph**: Visualize the network structure
- **Multiple layout options**: Hierarchical, spring, Kamada-Kawai, circular
- **Real-time configuration**: Adjust layers, channels, and nodes
- **Performance metrics**: View training results and test performance
- **Training curves**: Plot loss, MAE, RMSE, and MAPE over epochs

### Node Types
- 🔴 **Input**: Raw spatiotemporal traffic data
- 🔷 **Embedding**: Initial feature embedding
- 💠 **ST-Conv**: Spatiotemporal convolution block
- 🔺 **Temporal**: 1D temporal convolution
- 🔻 **Spatial**: Graph convolution (spatial)
- ⬟ **Gate**: BatchNorm / Residual connections
- ⬡ **Pooling**: Temporal pooling operation
- 💜 **Output**: Final prediction layer

## Advantages Over Traditional Architecture

1. **Unified Design**
   - Single integrated architecture
   - No separate GCN, CNN, MLP stages
   - Cleaner and more maintainable code

2. **Better Performance**
   - Simultaneous spatial-temporal processing
   - Residual connections improve gradient flow
   - Batch normalization stabilizes training

3. **More Flexible**
   - Easy to add/remove ST-Conv blocks
   - Adjustable hidden dimensions
   - Modular design

4. **Efficient**
   - Fewer parameters than separate architectures
   - Faster training with parallel processing
   - Better memory usage

## Example Usage

### Training

```python
from unified_spatiotemporal_gcn import UnifiedSpatioTemporalGCN, SpatioTemporalGCNPredictor
from chembur_network import ChemburTrafficNetwork
from large_traffic_data_generator import create_large_dataloaders

# Load network
network = ChemburTrafficNetwork()

# Create dataloaders
train_loader, val_loader, test_loader = create_large_dataloaders(
    network, train_minutes=180, val_minutes=30, test_minutes=30
)

# Create model
model = UnifiedSpatioTemporalGCN(
    num_nodes=8, num_features=4, hidden_channels=64,
    num_layers=4, output_horizon=12
)

# Train
predictor = SpatioTemporalGCNPredictor(model, device='cuda')
# ... training loop ...
```

### Inference

```python
# Load trained model
checkpoint = torch.load('unified_stgcn_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Make predictions
predictions = predictor.predict(x, adj)

# Extract embeddings
embeddings = predictor.get_embeddings(x, adj)
```

## Customization

### Modify Architecture

Edit `unified_spatiotemporal_gcn.py`:
- Change `hidden_channels` for capacity
- Adjust `num_layers` for depth
- Modify `kernel_size` for temporal receptive field

### Custom Data

Edit `large_traffic_data_generator.py`:
- Adjust `time_interval` for resolution
- Modify traffic patterns in `_get_base_traffic_pattern()`
- Add new features in `LargeTrafficDataset`

### Visualization

Edit `streamlit_spatiotemporalGCN.py`:
- Add new metrics or plots
- Customize node colors and shapes
- Add interactive features

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size`
- Reduce `hidden_channels`
- Reduce `train_minutes`

### Training Too Slow
- Reduce `num_layers`
- Reduce `train_minutes`
- Use larger `batch_size` if memory allows

### Poor Performance
- Increase `num_layers`
- Increase `hidden_channels`
- Increase `train_minutes`
- Reduce `learning_rate`

## Citation

If you use this code, please cite:

```bibtex
@misc{unified_stgcn_2026,
  title={Unified Spatiotemporal Graph Convolutional Network for Traffic Prediction},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/unified-stgcn}}
}
```

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

---

**Built with** ❤️ **using PyTorch, NetworkX, and Streamlit**
