"""
Chembur Traffic Network Graph Definition
This module defines the traffic network structure for a small area in Chembur, Mumbai.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class ChemburTrafficNetwork:
    """
    Represents a small traffic network in Chembur with 8 major junctions.
    
    Junctions (Nodes):
    1. Chembur Naka - Major intersection
    2. Swami Vivekanand Road Junction
    3. Diamond Garden Junction
    4. Chembur Station Junction
    5. RCF Colony Junction
    6. Tilak Nagar Junction
    7. Mahul Road Junction
    8. Eastern Express Highway Junction
    
    Node Attributes:
    - junction_capacity: Maximum vehicles that can wait at junction (0-100)
    - signal_cycle_time: Traffic signal cycle time in seconds (60-120)
    - junction_type: Type of junction (major/minor)
    - coordinates: (x, y) position for visualization
    
    Edge Attributes (Unidirectional):
    - road_length: Length of road segment in meters
    - num_lanes: Number of lanes (1-3)
    - speed_limit: Speed limit in km/h
    - road_quality: Road condition score (0-1)
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for unidirectional traffic
        self._build_network()
        
    def _build_network(self):
        """Build the Chembur traffic network with nodes and edges."""
        
        # Define nodes with attributes
        nodes = {
            0: {  # Chembur Naka
                'name': 'Chembur Naka',
                'junction_capacity': 80,
                'signal_cycle_time': 120,
                'junction_type': 'major',
                'coordinates': (0, 0)
            },
            1: {  # Swami Vivekanand Road Junction
                'name': 'Swami Vivekanand Road Junction',
                'junction_capacity': 60,
                'signal_cycle_time': 90,
                'junction_type': 'major',
                'coordinates': (1, 1)
            },
            2: {  # Diamond Garden Junction
                'name': 'Diamond Garden Junction',
                'junction_capacity': 50,
                'signal_cycle_time': 75,
                'junction_type': 'minor',
                'coordinates': (2, 0.5)
            },
            3: {  # Chembur Station Junction
                'name': 'Chembur Station Junction',
                'junction_capacity': 70,
                'signal_cycle_time': 100,
                'junction_type': 'major',
                'coordinates': (1, -1)
            },
            4: {  # RCF Colony Junction
                'name': 'RCF Colony Junction',
                'junction_capacity': 40,
                'signal_cycle_time': 60,
                'junction_type': 'minor',
                'coordinates': (3, 1)
            },
            5: {  # Tilak Nagar Junction
                'name': 'Tilak Nagar Junction',
                'junction_capacity': 55,
                'signal_cycle_time': 80,
                'junction_type': 'minor',
                'coordinates': (2, -1.5)
            },
            6: {  # Mahul Road Junction
                'name': 'Mahul Road Junction',
                'junction_capacity': 45,
                'signal_cycle_time': 70,
                'junction_type': 'minor',
                'coordinates': (0, -2)
            },
            7: {  # Eastern Express Highway Junction
                'name': 'Eastern Express Highway Junction',
                'junction_capacity': 90,
                'signal_cycle_time': 120,
                'junction_type': 'major',
                'coordinates': (-1, 0)
            }
        }
        
        # Add nodes to graph
        for node_id, attrs in nodes.items():
            self.graph.add_node(node_id, **attrs)
        
        # Define edges (unidirectional roads) with attributes
        # Format: (source, destination, attributes)
        edges = [
            # From Chembur Naka
            (0, 1, {'road_length': 800, 'num_lanes': 2, 'speed_limit': 40, 'road_quality': 0.8}),
            (0, 3, {'road_length': 600, 'num_lanes': 2, 'speed_limit': 40, 'road_quality': 0.7}),
            (0, 7, {'road_length': 500, 'num_lanes': 3, 'speed_limit': 60, 'road_quality': 0.9}),
            
            # From Swami Vivekanand Road Junction
            (1, 2, {'road_length': 700, 'num_lanes': 2, 'speed_limit': 40, 'road_quality': 0.75}),
            (1, 0, {'road_length': 800, 'num_lanes': 2, 'speed_limit': 40, 'road_quality': 0.8}),
            (1, 4, {'road_length': 900, 'num_lanes': 1, 'speed_limit': 30, 'road_quality': 0.6}),
            
            # From Diamond Garden Junction
            (2, 1, {'road_length': 700, 'num_lanes': 2, 'speed_limit': 40, 'road_quality': 0.75}),
            (2, 4, {'road_length': 600, 'num_lanes': 2, 'speed_limit': 40, 'road_quality': 0.7}),
            
            # From Chembur Station Junction
            (3, 0, {'road_length': 600, 'num_lanes': 2, 'speed_limit': 40, 'road_quality': 0.7}),
            (3, 5, {'road_length': 550, 'num_lanes': 2, 'speed_limit': 40, 'road_quality': 0.65}),
            (3, 6, {'road_length': 1000, 'num_lanes': 2, 'speed_limit': 40, 'road_quality': 0.7}),
            
            # From RCF Colony Junction
            (4, 2, {'road_length': 600, 'num_lanes': 2, 'speed_limit': 40, 'road_quality': 0.7}),
            (4, 1, {'road_length': 900, 'num_lanes': 1, 'speed_limit': 30, 'road_quality': 0.6}),
            
            # From Tilak Nagar Junction
            (5, 3, {'road_length': 550, 'num_lanes': 2, 'speed_limit': 40, 'road_quality': 0.65}),
            (5, 6, {'road_length': 800, 'num_lanes': 2, 'speed_limit': 40, 'road_quality': 0.6}),
            
            # From Mahul Road Junction
            (6, 5, {'road_length': 800, 'num_lanes': 2, 'speed_limit': 40, 'road_quality': 0.6}),
            (6, 7, {'road_length': 700, 'num_lanes': 2, 'speed_limit': 50, 'road_quality': 0.75}),
            
            # From Eastern Express Highway Junction
            (7, 0, {'road_length': 500, 'num_lanes': 3, 'speed_limit': 60, 'road_quality': 0.9}),
            (7, 6, {'road_length': 700, 'num_lanes': 2, 'speed_limit': 50, 'road_quality': 0.75}),
        ]
        
        # Add edges to graph
        for source, dest, attrs in edges:
            self.graph.add_edge(source, dest, **attrs)
    
    def get_adjacency_matrix(self):
        """
        Get the adjacency matrix of the network.
        Returns normalized adjacency matrix for GCN.
        """
        # Get basic adjacency matrix
        adj = nx.adjacency_matrix(self.graph).todense()
        adj = np.array(adj, dtype=np.float32)
        
        # Add self-loops
        adj = adj + np.eye(adj.shape[0])
        
        # Normalize adjacency matrix (D^-1/2 * A * D^-1/2)
        degree = np.array(np.sum(adj, axis=1)).flatten()
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
        return adj_normalized
    
    def get_node_features(self):
        """
        Get node feature matrix.
        Features: [junction_capacity, signal_cycle_time, junction_type_encoded]
        """
        num_nodes = self.graph.number_of_nodes()
        features = np.zeros((num_nodes, 3))
        
        for node_id in range(num_nodes):
            node_data = self.graph.nodes[node_id]
            features[node_id, 0] = node_data['junction_capacity'] / 100.0  # Normalize
            features[node_id, 1] = node_data['signal_cycle_time'] / 120.0  # Normalize
            features[node_id, 2] = 1.0 if node_data['junction_type'] == 'major' else 0.0
        
        return features
    
    def get_edge_features(self):
        """
        Get edge feature matrix for all edges.
        Features: [road_length, num_lanes, speed_limit, road_quality]
        """
        edges = list(self.graph.edges())
        edge_features = np.zeros((len(edges), 4))
        
        for idx, (src, dst) in enumerate(edges):
            edge_data = self.graph[src][dst]
            edge_features[idx, 0] = edge_data['road_length'] / 1000.0  # Normalize
            edge_features[idx, 1] = edge_data['num_lanes'] / 3.0  # Normalize
            edge_features[idx, 2] = edge_data['speed_limit'] / 60.0  # Normalize
            edge_features[idx, 3] = edge_data['road_quality']
        
        return edge_features
    
    def visualize_network(self, save_path='chembur_network.png'):
        """Visualize the traffic network."""
        plt.figure(figsize=(12, 8))
        
        # Get positions from coordinates
        pos = {node: data['coordinates'] for node, data in self.graph.nodes(data=True)}
        
        # Draw the graph
        nx.draw_networkx_nodes(self.graph, pos, node_size=1000, 
                               node_color='lightblue', alpha=0.9)
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', 
                               arrows=True, arrowsize=20, width=2, alpha=0.6)
        
        # Add labels
        labels = {node: f"{node}\n{data['name'].split()[0]}" 
                  for node, data in self.graph.nodes(data=True)}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title('Chembur Traffic Network\n(Unidirectional Road Network)', 
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network visualization saved to {save_path}")
        plt.close()
    
    def get_network_info(self):
        """Print network information."""
        print("=" * 60)
        print("CHEMBUR TRAFFIC NETWORK INFORMATION")
        print("=" * 60)
        print(f"\nNumber of Junctions (Nodes): {self.graph.number_of_nodes()}")
        print(f"Number of Roads (Edges): {self.graph.number_of_edges()}")
        print(f"\nJunction Details:")
        for node_id in range(self.graph.number_of_nodes()):
            node_data = self.graph.nodes[node_id]
            print(f"  {node_id}. {node_data['name']}")
            print(f"     - Capacity: {node_data['junction_capacity']} vehicles")
            print(f"     - Signal Cycle: {node_data['signal_cycle_time']}s")
            print(f"     - Type: {node_data['junction_type']}")
        
        print(f"\nRoad Network Statistics:")
        road_lengths = [data['road_length'] for _, _, data in self.graph.edges(data=True)]
        print(f"  - Average road length: {np.mean(road_lengths):.0f}m")
        print(f"  - Total network length: {np.sum(road_lengths):.0f}m")
        print("=" * 60)


if __name__ == "__main__":
    # Create and visualize the network
    network = ChemburTrafficNetwork()
    network.get_network_info()
    network.visualize_network()
    
    # Display matrices
    print("\nAdjacency Matrix (Normalized):")
    print(network.get_adjacency_matrix())
    print("\nNode Features:")
    print(network.get_node_features())
