import torch
import torch_geometric.nn as nnGeo
import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import GATConv, LayerNorm, GATv2Conv
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels,hidden_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
    
        return x
        

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_channels, num_classes, heads = 1):
        super().__init__()
    
        self.gat1 = GATConv(input_dim, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels*heads, hidden_channels*heads, heads=heads)
        self.gat3 = GATConv(hidden_channels*heads, hidden_channels*heads, heads=heads)
        
    def forward(self, data):
    
        x, edge_index= data.x, data.edge_index        
        x, att_weight1 = self.gat1(x, edge_index, return_attention_weights=True) 
  
        x = x.relu()

        x, att_weight2 = self.gat2(x, edge_index, return_attention_weights=True)
  
        x = x.relu()
        
        x, att_weight3 = self.gat3(x, edge_index, return_attention_weights=True)
        
        return x, att_weight1, att_weight2, att_weight3
        
        
        