import os
import numpy as np
import torch



def compute_mean_features(fold_train_path, graphs_folder):
    total_features_sum = None
    sum_squared_diffs = None
    total_num_samples = 0
    number_nodes = 0
    for filename in fold_train_path['0']:
        graph_path = os.path.join(graphs_folder, filename[:-3] + '.pt')
        print(f"Loading from: {graph_path}")
        graph_data = torch.load(graph_path)
        x = graph_data.x.numpy()
        if total_features_sum is None:
            total_features_sum = np.sum(x, axis=0)
        else:
            total_features_sum += np.sum(x, axis=0)
                
        total_num_samples += x.shape[0]

    mean_features = total_features_sum / total_num_samples
    print('Mean computed...')
    for filename in fold_train_path['0']:
        graph_path = os.path.join(graphs_folder, filename[:-3] + '.pt')
        
        graph_data = torch.load(graph_path)
        x = graph_data.x.numpy()
        
        squared_diffs = (x - mean_features)**2
        if sum_squared_diffs is None:
            sum_squared_diffs = np.sum(squared_diffs, axis=0)
        else:
            sum_squared_diffs += np.sum(squared_diffs, axis=0)

    std_dev = np.sqrt(sum_squared_diffs/total_num_samples)
    print('Std deviation computed...')
    
    return mean_features, std_dev