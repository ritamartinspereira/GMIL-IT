import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

class C17Dataset(Dataset):
    """
    Dataset class for loading graph data.
    
    Args:
        args: Command-line arguments or configurations.
        data_folder: Folder containing graph data files.
        h5_path: DataFrame with file paths and labels.
        mean: Mean values for feature normalization.
        std: Standard deviation values for feature normalization.
        dataset: Dataset name.
    """
    def __init__(self, args, data_folder, h5_path, mean, std, dataset):
        self.args = args
        self.data_folder = data_folder
        self.file_list = os.listdir(data_folder)
        self.df = h5_path.reset_index(drop=True)
        self.mean = mean
        self.std = std
        self.dataset = dataset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df['0'].iloc[idx]
        label = self.df['label'].iloc[idx]  
        
        if self.args.num_classes == 1:
            label_vec = 0 if label == 'negative' else 1
        else:
            class_map = {'negative': 0, 'itc': 1, 'micro': 2, 'macro': 3}
            if label in class_map:
                label_vec = class_map[label]
            else:
                raise ValueError(f"Unknown label: {label}")

        if self.args.num_classes == 1:
            label_vec = torch.tensor(label_vec, dtype=torch.float32).reshape(self.args.num_classes, 1)
        else:
            label_vec = torch.tensor(label_vec, dtype=torch.long)
        
        graph_path = os.path.join(self.data_folder, filename[:-3] + '.pt')
        try:
            graph_data = torch.load(graph_path)
            
            if isinstance(graph_data, dict):
                x = graph_data['x']
                edge_index = graph_data['edge_index']
            else:
                x = graph_data.x
                edge_index = graph_data.edge_index
                
            if self.mean is not None and self.std is not None:
                x = (x - self.mean) / self.std
                
            
            edge_index = edge_index.clone().detach().long()
            
            
            data = Data(x=x, edge_index=edge_index, y=label_vec)
            
            return data, filename
            
        except Exception as e:
            print(f"Error loading {graph_path}: {e}")
            # Return a dummy value to indicate error
            return -1, filename
        


class C16Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data_folder, h5_path, mean, std, dataset):
        self.args = args
        self.data_folder = data_folder
        self.file_list = os.listdir(data_folder)
        self.df = h5_path.reset_index(drop=True)
        self.mean = mean
        self.std = std
        self.dataset = dataset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df['0'].iloc[idx]
        label = self.df['label'].iloc[idx]  
        
        if self.args.num_classes==1:
            label_vec = label
        label_vec = torch.tensor(label_vec, dtype=torch.float32).reshape(self.args.num_classes, 1)

        basename = os.path.basename(filename)
        graph_path = os.path.join(self.data_folder, basename[:-3] + '.pt') 
        try:
            graph_data = torch.load(graph_path)

            if isinstance(graph_data, dict):
                x = graph_data['x']
                edge_index = graph_data['edge_index']
            else:
                x = graph_data.x
                edge_index = graph_data.edge_index
                
            if self.mean is not None and self.std is not None:
                x = (x - self.mean) / self.std

            edge_index = edge_index.clone().detach().long()
            data = Data(x=x, edge_index=edge_index, y=label_vec) 

            return data, basename
        
        except Exception as e:
            print(f"Error loading {graph_path}: {e}")
            # Return a dummy value to indicate error
            return -1, filename



def collate_fn_custom(batch):
    
    filtered_batch = [s for s in batch if s[0] != -1]
    
    if not filtered_batch:
        return None, None, None
    
    data_list = [s[0] for s in filtered_batch]
    file_names = [s[1] for s in filtered_batch]

    
    data_batch = Batch.from_data_list(data_list)
        
    return data_batch, file_names

