import h5py
import csv
import torch
import torch.nn as nn
from torch_geometric.data import Data as geomData
from torch.utils.data import Dataset, ConcatDataset 
from torch.utils.data.dataset import random_split
from torch_geometric.loader import DataLoader
import torch_geometric.utils as utils
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import torch.nn.functional as F
from models import abmil
from models import dsmil
import pandas as pd
from sklearn.utils import shuffle
import sys, argparse,os, datetime
from tqdm import tqdm
from torch.autograd import profiler
import time
import pickle
import itertools
import skimage as ski
from skimage import graph
import joblib
import matplotlib.pyplot as plt
import json
import sys, argparse, os, copy, itertools, glob, datetime
from torch_geometric.utils import add_self_loops

import math
import PIL
from PIL import Image

from openslide import OpenSlide, OpenSlideError

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


class ConnectedComponents(Dataset):
    def __init__(self, h5_file_path, args) -> None:
        super(ConnectedComponents, self).__init__()
        self.h5_file_path = h5_file_path
        self.args = args
        
        
    def get_feats(self, csv_file_df):
        #gets features from one file
        feats_csv_path = csv_file_df.iloc[0]
        feat_path = os.path.join(self.args.feats_path, feats_csv_path)
        coordinates = h5py.File(feat_path)['coords'][()]
        feats =  h5py.File(feat_path)['features'][()]

        feats = np.array(feats)
        
        return feats, coordinates
    

    def perform_kmeans_clustering(self, features):
           
            kmeans_local = KMeans(n_clusters = self.args.num_clusters)
            clusts = kmeans_local.fit(features)
            clust_labels = kmeans_local.predict(features)
            cent_coord = kmeans_local.cluster_centers_ #features of the cluster center        
    
            clusters = [[] for _ in range(self.args.num_clusters)] 

            # Assign data points to their respective clusters based on cluster labels 
            for i, label in enumerate(clust_labels):
                clusters[label].append(i) #Contains the indices of data points assigned to each cluster

            return clust_labels, cent_coord, clusters
        
        
    def find_closest_centroid(self, features, kmeans):
        return kmeans.predict(features) 
    
    def get_binary_masks(self, cluster_labels, coordinates):

        min_x = np.min(coordinates[:, 0])
        min_y = np.min(coordinates[:, 1])
        max_x = np.max(coordinates[:, 0])
        max_y = np.max(coordinates[:, 1])

        height = max_y - min_y 
        width = max_x - min_x

        binary_masks = np.zeros((self.args.num_clusters, (height // self.args.patch_size) + 1 , (width // self.args.patch_size) + 1))

        print('Doing binary masks...')

        for i, (x, y) in enumerate(coordinates):
            adjusted_x = x - min_x 
            adjusted_y = y - min_y 

            patch_row = adjusted_y // self.args.patch_size
            patch_col = adjusted_x // self.args.patch_size

            cluster_id = cluster_labels[i]
            binary_masks[cluster_id, patch_row, patch_col] = 1

        return binary_masks, height, width    
    
    def create_WSI(self, features, coordinates, height, width):
    
        min_x = np.min(coordinates[:, 0])
        min_y = np.min(coordinates[:, 1])

        n_patches_height =  (height // self.args.patch_size) + 1 
        n_patches_width = (width // self.args.patch_size) + 1
        
        wsi_features = np.zeros((n_patches_height, n_patches_width, 1024))

        for i, feature_vector in enumerate(features):
            x, y = coordinates[i]
            adjusted_x = x - min_x 
            adjusted_y = y - min_y
            patch_row =  adjusted_y // self.args.patch_size
            patch_col =  adjusted_x // self.args.patch_size

            wsi_features[patch_row,patch_col] = feature_vector

        return wsi_features
    
    
    def connected_components(self, binary_mask, connectivity = 2):
        labeled_image, count = ski.measure.label(binary_mask,connectivity=connectivity, return_num=True)
        return labeled_image, count
    
    def join_labeled_imgs(self,labeled_images, height, width):
    
        regions_img = None
        last_non_zero_value = None

        for i, labeled_img in enumerate(labeled_images):
            labeled_img_copy = labeled_img.copy()
            if regions_img is None:
                regions_img = labeled_img_copy
            else:
                biggest_indices = np.unravel_index(np.argmax(regions_img), regions_img.shape)
                
                max_value = regions_img[biggest_indices[0], biggest_indices[1]]
                
                
                labeled_img_copy[labeled_img_copy != 0] += max_value
                regions_img = np.maximum(regions_img, labeled_img_copy)
       
        return regions_img
    
    def get_cent_coords(self, region_img, threshold = 0):
        cent_coords = {}
        nodes_to_remove = []
        region_properties = ski.measure.regionprops(region_img)
        nodes_to_remove.append(0)
        for region in region_properties:
            cent_coords[region.label] = region.centroid
            if region.area < threshold:
                nodes_to_remove.append(region.label)

        return cent_coords, nodes_to_remove   
    
    def create_coords_mask(self, cent_coords, regions_img):
        coords_mask = np.zeros((regions_img.shape[0], regions_img.shape[1], 3))
        
        for label, coords in cent_coords.items():
            region_indices = np.where(regions_img == label)
            row_indices = region_indices[0]
            col_indices = region_indices[1]
            coords_mask[row_indices, col_indices] = (coords[0], coords[1], 0)  
        return coords_mask
        
        
    def create_RAG(self, regions_img, coords_mask, nodes_to_remove):
        
        rag = graph.rag_mean_color(coords_mask, regions_img, mode = 'distance')
        print("Nodes:", len(rag.nodes()))
        print("Edges:", len(rag.edges()))
        
        rag.remove_nodes_from(nodes_to_remove)
        
        return rag
    
    
    def compute_region_features(self, regions_img, wsi_features):
        regions = np.unique(regions_img)
        region_features = {}
        for region_id in regions:
            if region_id == 0:
                continue
            region_indices = np.where(regions_img == region_id)
            feature_vectors = wsi_features[region_indices]
            mean_feature = np.mean(feature_vectors, axis=0)
            region_features[region_id] = mean_feature

        return region_features
    
    def create_pyg_graph(self, region_features, rag):
        edge_index = []
        edge_weights = []
        
        valid_nodes = {node: i for i, node in enumerate(rag.nodes())}
        
        node_features = []
        for node in rag.nodes():
            if node in region_features:
                node_features.append(region_features[node])
            else:
                raise ValueError(f"Node {node} in RAG is not in region_features")
        
        node_features_np = np.array(node_features)
        x = torch.tensor(node_features_np, dtype=torch.float)
        
        for (u, v, data) in rag.edges(data=True):
            if u in valid_nodes and v in valid_nodes:
                edge_index.append([valid_nodes[u], valid_nodes[v]])
                edge_weights.append(data['weight'])
            else:
                print(f"Skipping edge ({u}, {v}) as one of the nodes is not in valid_nodes")
            
        
        edge_index = np.array(edge_index).T 
        edge_index = torch.LongTensor(edge_index).to(device)
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.shape[0])
        edge_weights = np.array(edge_weights)  
        
        
        print('Edge index', edge_index.shape)
        print('Edge weights', edge_weights.shape)

        data = geomData(x = x, edge_index = edge_index)
        
        return data, edge_index

    
    def create_save_pyg_graph(self):
        for idx in tqdm(range(len(self.h5_file_path)), desc='Creating region graphs'):
            filename = os.path.basename(self.h5_file_path.iloc[idx]['0'])
            print(f'Creating graph for {filename[:-3]}')
            if os.path.isfile(os.path.join(self.args.graph_save_path, filename[:-3] + '.pt')):
                print(f'Already processed {filename[:3]}')
                continue
            labeled_images = [] # size (9, num_rows, num_cols)
            feats, coordinates = self.get_feats(self.h5_file_path.iloc[idx])
            cluster_labels_local, local_centroids, clusters_local = self.perform_kmeans_clustering(feats)
            binary_masks, height, width = self.get_binary_masks(cluster_labels_local, coordinates)
            wsi_features = self.create_WSI(feats, coordinates, height, width)
            for i, binary_mask in enumerate(binary_masks):
                labeled_image, _ = self.connected_components(binary_mask)
                labeled_images.append(labeled_image)

            regions_img = self.join_labeled_imgs(labeled_images,height, width)
            
            cent_coords, nodes_to_remove = self.get_cent_coords(regions_img)
            coords_mask = self.create_coords_mask(cent_coords, regions_img)
            rag = self.create_RAG(regions_img, coords_mask, nodes_to_remove)
            region_features = self.compute_region_features(regions_img, wsi_features)
            
            region_ids = sorted(region_features.keys())  # Sort keys for consistency
            region_feats = np.array([region_features[rid] for rid in region_ids])
        
            graph, edge_index = self.create_pyg_graph(region_features, rag)
            
            torch.save(graph, os.path.join(self.args.graph_save_path, filename[:-3] + '.pt'))


    
class PatchGraphs(Dataset):
    def __init__(self, csv_path, h5_file_path, save_path) -> None:
        super(PatchGraphs, self).__init__()
        self.csv_path = csv_path
        self.h5_file_path = h5_file_path
        self.save_path = save_path

        ####### Adjacent Patch     ######################################################################
    def NEW_pt2graph_adjacentpatch(self, coords, features, radius=1):
        edge_list = []
        maxxx = 0
        minnn = 100
        min_coords=[]
        count_isolated = 0
        count_edges = 0


        # Iterate through each patch
        coords = torch.tensor(coords, device=device)
        features = torch.tensor(features, device=device)

        for i, current_patch in enumerate(coords):
            adjacent_mask = torch.all(torch.abs(coords - current_patch) <= (256*radius), dim=1)

            if maxxx < torch.sum(adjacent_mask):
                maxxx = torch.sum(adjacent_mask) 

            if minnn >= torch.sum(adjacent_mask):
                minnn = torch.sum(adjacent_mask) 
                if minnn == 1:
                    count_isolated += 1

            adjacent_mask[i] = False
            num_true_entries = torch.sum(adjacent_mask)
            count_edges += torch.sum(adjacent_mask)

            neighbor_indices = torch.nonzero(adjacent_mask).squeeze().tolist()
            #print(neighbor_indices)

            #f neighbor_indices:  # Check if neighbor_indices is not empty
            if isinstance(neighbor_indices, int):
                edge_list.append([i, neighbor_indices])
            else:
                edge_list.extend([[i, idx] for idx in neighbor_indices])
        count_true_entries = torch.count_nonzero(adjacent_mask)

        '''print('\n\nNumber of isolated patches:', count_isolated)'''

        edge_index = torch.LongTensor(np.array(edge_list).T).to(device)
        edge_index, _ = add_self_loops(edge_index, num_nodes = coords.size(0))


        G = geomData(x=features,
                     edge_index=edge_index,
                     centroid=coords)
        return G

    def NEW_createDir_h5toPyG_adjacentpatch(self, visualize_index=0):
        print('\n\n Your choice: adjacent patches \n\n')
        h5_files = [f for f in os.listdir(self.h5_file_path) if f.endswith(".h5") and os.path.isfile(os.path.join(self.h5_file_path, f))]
        pbar = tqdm(h5_files)

        for h5_fname in pbar:
            pbar.set_description('Creating Graph for: %s' % (h5_fname))
            path_test = os.path.join(self.h5_file_path, h5_fname)
            try:
                with h5py.File(path_test, "r") as wsi_h5:
                    pt_file_path = os.path.join(self.save_path, h5_fname[:-3] + '.pt')

                    coords = np.array(wsi_h5['coords']).astype(int)
                    features = np.array(wsi_h5['features'])
                    G = self.NEW_pt2graph_adjacentpatch(coords, features)

                    print("Graph created!")
                    print("Saving graph to:", repr(pt_file_path))  # Show the full path


                    if os.path.exists(pt_file_path):
                        print("Warning: File already exists. Deleting it now.")
                        os.remove(pt_file_path)  # Delete it befo

                    torch.save(G, pt_file_path)  # Save
                    print("Graph saved successfully!")

                    data = torch.load(pt_file_path, map_location='cpu')  # Load
                    print("Graph loaded successfully!", data)

                    print(data)


            except Exception as e:
                print(f"{h5_fname} - Broken H5 (once): {e}")
                pbar.set_description('%s - Broken H5' % (h5_fname[:12]))
                exit()



def paint_regions(region_img, coords, filename, height, width, patch_size = 256):
    
    img_path = os.path.join('/home/catarinabarata/Data/externaldrive/Chamelyon_17/Training/center_0', filename[:-3] + '.tif') # por image path
    img, width, height = slide_to_scaled_pil_img(img_path)
    
    clustered_img = np.zeros((height // patch_size , width // patch_size, 3), dtype=np.uint8)
    
   
    min_x = np.min(coords[:, 0])
    min_y = np.min(coords[:, 1])
    
    unique_regions = np.unique(region_img) 
    unique_regions = unique_regions[unique_regions != 0]
    num_regions = len(unique_regions)
    print(num_regions)
    
    colormap = plt.get_cmap('nipy_spectral', num_regions + 1)
    colors_dict = {label: (np.array(colormap(i)[:3]) * 255).astype(np.uint8) for i, label in enumerate(unique_regions, 1)}
    #divide image into a grid
    for (x, y) in coords:
        patch_col = x // patch_size
        patch_row = y // patch_size

        # Calculate which patch the (x, y) coordinate belongs to
        rel_x = (x - min_x) // patch_size
        rel_y = (y - min_y) // patch_size
        
        index = region_img[rel_y, rel_x]
        
        if index == 0: # if it is background skip
            continue
        if index in colors_dict:
            clustered_img[patch_row, patch_col] = colors_dict[index]
    
    
    reshaped_clustered_img = PIL.Image.fromarray(clustered_img, mode="RGB")
    reshaped_clustered_img = reshaped_clustered_img.resize(img.size, PIL.Image.BILINEAR)

    overlayed_image = PIL.Image.blend(img, reshaped_clustered_img, alpha=0.5)
    overlayed_image.save(f"region_overlayUNI.png")
    
    

def slide_to_scaled_pil_img(img_path, SCALE_FACTOR = 32):
    
    img = OpenSlide(img_path)
    original_w, original_h = img.dimensions
    new_w = math.floor(original_w / SCALE_FACTOR)
    new_h = math.floor(original_h / SCALE_FACTOR)
    level = img.get_best_level_for_downsample(SCALE_FACTOR)
    whole_slide_image = img.read_region((0, 0), level, img.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), Image.BILINEAR)
    
    return img, original_w, original_h
    

def main():
    
    parser = argparse.ArgumentParser(description='Build graphs [patch, region]')
    parser.add_argument('--type_graph', default='patch', type=str, help='Type of graph to build [patch, region]')
    parser.add_argument('--h5_path', default=None, type=str, help='Path to the csv that contains all the h5 file paths')
    parser.add_argument('--feats_path', default=None, type=str, help='Path to the features')
    parser.add_argument('--graph_save_path', default=None, type=str, help='Path to save the graphs')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of classes')
    parser.add_argument('--num_clusters', default=9, type=int, help='Number of clusters for Kmeans')
    parser.add_argument('--patch_size', default=256, type=int, help='Size of patches extracted from the feature extrator')
    parser.add_argument('--dataset', default='C17', type=str, help='Name of the dataset')

    args = parser.parse_args()

    if args.h5_path is None:
        parser.error("--h5_path is required")
    
    if args.type_graph == 'region' and args.num_clusters is None:
        parser.error("--num_clusters is required for region graphs") 

    try:
        h5_csv_path = pd.read_csv(args.h5_path) # tem de ser /home/catarinabarata
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return


    if args.graph_save_path:
        os.makedirs(args.graph_save_path, exist_ok = True)
        
    if args.type_graph == 'region':
        dataset = ConnectedComponents(h5_csv_path, args)
        dataset.create_save_pyg_graph()
        

    elif args.type_graph == 'patch':
        dataset = PatchGraphs(h5_csv_path, args.feats_path, args.graph_save_path)
        dataset.NEW_createDir_h5toPyG_adjacentpatch()
    


if __name__ == '__main__':
    main()
