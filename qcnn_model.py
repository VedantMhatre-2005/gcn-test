"""
Quantum Convolutional Neural Network (QCNN) for Traffic Prediction
Implements a pure quantum architecture using PennyLane for graph-based traffic forecasting
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
from pennylane import numpy as pnp


class QuantumConvolutionalLayer:
    """
    Quantum Convolutional Layer that applies parameterized quantum gates
    to create entanglement patterns similar to classical convolutions
    """
    def __init__(self, n_qubits, params):
        self.n_qubits = n_qubits
        self.params = params
        
    def apply_two_qubit_gate(self, qubits, params):
        """Apply a parameterized two-qubit gate (convolution operation)"""
        q1, q2 = qubits
        # Rotation gates
        qml.RY(params[0], wires=q1)
        qml.RY(params[1], wires=q2)
        # Entangling gate
        qml.CNOT(wires=[q1, q2])
        # More rotations
        qml.RY(params[2], wires=q1)
        qml.RY(params[3], wires=q2)
        qml.CNOT(wires=[q2, q1])
        
    def apply(self):
        """Apply convolution across all adjacent qubits"""
        param_idx = 0
        # Apply to adjacent pairs (1D convolution pattern)
        for i in range(0, self.n_qubits - 1, 2):
            self.apply_two_qubit_gate(
                [i, i + 1],
                self.params[param_idx:param_idx + 4]
            )
            param_idx += 4
        
        # Shifted pattern for second layer
        for i in range(1, self.n_qubits - 1, 2):
            self.apply_two_qubit_gate(
                [i, i + 1],
                self.params[param_idx:param_idx + 4]
            )
            param_idx += 4


class QuantumPoolingLayer:
    """
    Quantum Pooling Layer that reduces dimensionality by measuring qubits
    """
    def __init__(self, qubits_to_measure):
        self.qubits_to_measure = qubits_to_measure
        
    def apply(self, params):
        """
        Apply conditional rotations before measurement
        Qubits are measured and traced out effectively
        """
        for idx, qubit in enumerate(self.qubits_to_measure):
            if qubit < len(params):
                qml.RY(params[qubit], wires=qubit)


class GraphQuantumEncoder:
    """
    Encodes graph node features into quantum states
    Uses amplitude encoding with normalization
    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        
    def encode_features(self, features):
        """
        Encode classical features into quantum state
        Uses angle encoding: features -> rotation angles
        """
        # Normalize features to [0, pi] range
        features = pnp.array(features, requires_grad=False)
        features = pnp.clip(features, -np.pi, np.pi)
        
        for i in range(min(len(features), self.n_qubits)):
            # RY and RZ gates for richer encoding
            qml.RY(features[i], wires=i)
            if i + self.n_qubits // 2 < len(features):
                qml.RZ(features[i + self.n_qubits // 2], wires=i)
                
    def encode_graph_structure(self, adjacency_matrix, node_idx):
        """
        Encode graph structure information
        Creates entanglement based on adjacency
        """
        neighbors = np.where(adjacency_matrix[node_idx] > 0)[0]
        for neighbor in neighbors:
            if neighbor < self.n_qubits and neighbor != node_idx:
                qml.CNOT(wires=[node_idx, neighbor])


class QCNN(nn.Module):
    """
    Complete Quantum Convolutional Neural Network for graph-based traffic prediction
    
    Architecture:
    1. Quantum Feature Encoding
    2. Quantum Convolutional Layers (parameterized unitary operations)
    3. Quantum Pooling (measurement-based dimensionality reduction)
    4. Classical Post-processing
    """
    def __init__(self, num_nodes, num_features, seq_len, n_qubits=8, n_layers=2):
        super(QCNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.seq_len = seq_len
        self.n_qubits = min(n_qubits, 12)  # Limit for simulation
        self.n_layers = n_layers
        
        # Calculate number of parameters needed
        # For each conv layer: 4 params per pair, multiple pairs
        params_per_conv = 4 * (self.n_qubits - 1)
        total_params = params_per_conv * n_layers + self.n_qubits  # +n_qubits for pooling
        
        # Initialize quantum parameters
        self.q_params = nn.Parameter(torch.randn(total_params) * 0.1)
        
        # Classical preprocessing
        self.feature_projection = nn.Linear(num_features * seq_len, self.n_qubits)
        
        # Classical post-processing
        self.classical_layers = nn.Sequential(
            nn.Linear(self.n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
        # Create quantum device and circuit
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        self.qnode = qml.QNode(self._quantum_circuit, self.dev, interface='torch')
        
    def _quantum_circuit(self, features, adj_matrix, node_idx, params):
        """
        Define the quantum circuit
        """
        # Encode features
        encoder = GraphQuantumEncoder(self.n_qubits)
        encoder.encode_features(features)
        encoder.encode_graph_structure(adj_matrix, node_idx % self.n_qubits)
        
        # Apply quantum convolutional layers
        param_idx = 0
        params_per_conv = 4 * (self.n_qubits - 1)
        
        for layer in range(self.n_layers):
            conv_layer = QuantumConvolutionalLayer(
                self.n_qubits,
                params[param_idx:param_idx + params_per_conv]
            )
            conv_layer.apply()
            param_idx += params_per_conv
        
        # Pooling (measure alternating qubits)
        pooling_params = params[param_idx:param_idx + self.n_qubits]
        pooling = QuantumPoolingLayer(range(0, self.n_qubits, 2))
        pooling.apply(pooling_params)
        
        # Measurement: return expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x, adj_matrix, batch_node_indices=None):
        """
        Forward pass through QCNN
        
        Args:
            x: (batch_size, seq_len, num_nodes, num_features)
            adj_matrix: (num_nodes, num_nodes) adjacency matrix
            batch_node_indices: specific nodes to process (optional)
            
        Returns:
            predictions: (batch_size, num_nodes) traffic predictions
        """
        batch_size = x.shape[0]
        
        # Reshape input: (batch, seq_len, nodes, features) -> (batch, nodes, seq*features)
        x = x.permute(0, 2, 1, 3)  # (batch, nodes, seq_len, features)
        x = x.reshape(batch_size, self.num_nodes, -1)  # (batch, nodes, seq*features)
        
        predictions = []
        
        for b in range(batch_size):
            node_predictions = []
            
            for node in range(self.num_nodes):
                # Project features to quantum dimension
                node_features = self.feature_projection(x[b, node])  # (n_qubits,)
                
                # Run quantum circuit
                q_params_np = self.q_params.detach().cpu().numpy()
                adj_np = adj_matrix.cpu().numpy() if torch.is_tensor(adj_matrix) else adj_matrix
                
                quantum_output = self.qnode(
                    node_features.detach().cpu().numpy(),
                    adj_np,
                    node,
                    q_params_np
                )
                
                # Convert to tensor
                quantum_features = torch.tensor(
                    np.array(quantum_output),
                    dtype=torch.float32,
                    device=x.device
                )
                
                # Classical post-processing
                prediction = self.classical_layers(quantum_features)
                node_predictions.append(prediction)
            
            predictions.append(torch.stack(node_predictions))
        
        predictions = torch.stack(predictions).squeeze(-1)  # (batch, nodes)
        return predictions


class QCNNPredictor:
    """
    Wrapper class for QCNN model with training and inference utilities
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
    def train_model(self, train_loader, val_loader, adj_matrix, epochs=50, lr=0.001):
        """
        Train the QCNN model
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        adj_tensor = torch.FloatTensor(adj_matrix).to(self.device)
        
        train_losses = []
        val_losses = []
        
        print("\nStarting QCNN Training...")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_train_loss = 0
            num_batches = 0
            
            for batch_idx, (seq, target, batch_adj) in enumerate(train_loader):
                seq = seq.to(self.device)
                target = target.to(self.device)
                # Squeeze target if pred_horizon=1: [batch, 1, nodes] -> [batch, nodes]
                if target.dim() == 3 and target.shape[1] == 1:
                    target = target.squeeze(1)
                
                optimizer.zero_grad()
                output = self.model(seq, adj_tensor)
                loss = criterion(output, target)
                
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_batches += 1
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.4f}")
            
            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            epoch_val_loss = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for seq, target, batch_adj in val_loader:
                    seq = seq.to(self.device)
                    target = target.to(self.device)
                    # Squeeze target if pred_horizon=1: [batch, 1, nodes] -> [batch, nodes]
                    if target.dim() == 3 and target.shape[1] == 1:
                        target = target.squeeze(1)
                    
                    output = self.model(seq, adj_tensor)
                    loss = criterion(output, target)
                    
                    epoch_val_loss += loss.item()
                    num_val_batches += 1
            
            avg_val_loss = epoch_val_loss / num_val_batches
            val_losses.append(avg_val_loss)
            
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return train_losses, val_losses
    
    def predict(self, seq, adj_matrix):
        """
        Make predictions
        """
        self.model.eval()
        with torch.no_grad():
            seq = seq.to(self.device)
            adj_tensor = torch.FloatTensor(adj_matrix).to(self.device)
            output = self.model(seq, adj_tensor)
        return output.cpu().numpy()
    
    def evaluate(self, test_loader, adj_matrix):
        """
        Evaluate model on test set
        """
        self.model.eval()
        adj_tensor = torch.FloatTensor(adj_matrix).to(self.device)
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for seq, target, batch_adj in test_loader:
                seq = seq.to(self.device)
                # Squeeze target if pred_horizon=1: [batch, 1, nodes] -> [batch, nodes]
                if target.dim() == 3 and target.shape[1] == 1:
                    target = target.squeeze(1)
                
                output = self.model(seq, adj_tensor)
                
                predictions.append(output.cpu().numpy())
                actuals.append(target.numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        
        # Per-node metrics
        node_mse = np.mean((predictions - actuals) ** 2, axis=0)
        node_mae = np.mean(np.abs(predictions - actuals), axis=0)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'predictions': predictions,
            'actuals': actuals,
            'node_mse': node_mse,
            'node_mae': node_mae
        }
