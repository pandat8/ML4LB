import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
import pathlib
import gzip
import pickle

"""
The GNN regression model for predicting k of the first LB iteration
"""

class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features, k_init=0, device=None):
        super().__init__()
        # if caller doesn't specify a device, prefer GPU when available
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move features to the specified device immediately
        self.constraint_features = torch.tensor(constraint_features, dtype=torch.float32, device=device)
        self.edge_index = torch.tensor(edge_indices.astype(np.int64), dtype=torch.long, device=device)
        self.edge_attr = torch.tensor(edge_features, dtype=torch.float32, device=device).unsqueeze(1)
        self.variable_features = torch.tensor(variable_features, dtype=torch.float32, device=device)

        self.k_init = k_init

    def __inc__(self, key, value):
        if key == 'edge_index':
            # Ensure the increment tensor is on the same device as the data
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]], device=self.edge_index.device)
        else:
            return super().__inc__(key, value)


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files, device='cpu'):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.device = device

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_kinit = sample

        # We note on which variables we were allowed to branch, the scores as well as the choice
        # taken by strong branching (relative to the candidates)

        variable_features = sample_observation.variable_features[:, -1:]
        graph = BipartiteNodeData(sample_observation.constraint_features, sample_observation.edge_features.indices,
                                  sample_observation.edge_features.values, variable_features,
                                  sample_kinit, device=self.device)

        # graph = BipartiteNodeData(sample_observation.constraint_features, sample_observation.edge_features.indices,
        #                           sample_observation.edge_features.values, sample_observation.variable_features,
        #                           sample_kinit)


        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = sample_observation.constraint_features.shape[0] + sample_observation.variable_features.shape[0]

        return graph



class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__('add') # mean
        emb_size = 64 # 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )

        self.post_conv_module = torch.nn.Sequential(
            # torch.nn.LayerNorm(emb_size)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size, bias=False),
        )

        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=0.2)

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_j) + self.feature_module_edge(edge_features)
                                          ) # + self.feature_module_edge(edge_features) + self.feature_module_right(node_features_i)
        return output

class GNNPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64 # 64
        cons_nfeats = 1
        edge_nfeats = 1
        var_nfeats = 1 # 10

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            # torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size, bias=False),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            # torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            # torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size, bias=False),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )
        self.pool_activation = torch.nn.Sequential(
            torch.nn.Sigmoid(),
        )

        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=0.2)

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        # ensure inputs are on the same device as the model parameters
        device = next(self.parameters()).device
        constraint_features = constraint_features.to(device)
        edge_indices = edge_indices.to(device)
        edge_features = edge_features.to(device)
        variable_features = variable_features.to(device)

        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features,
                                               constraint_features)

        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        # A final MLP on the variable features
        output = self.output_module(variable_features)  #.squeeze(-1)
        output = torch.mean(output, dim=0)  # sum
        output = self.pool_activation(output)

        return output