
import argparse
import json
import os
import datetime
import copy
import glob
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.evaluation import compute_mean_features
from sklearn.metrics import (balanced_accuracy_score, 
                             classification_report, confusion_matrix, 
                             precision_recall_fscore_support, precision_score, 
                             recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

# Import local modules
from models.GCN import GCNModel, GATModel
from models import abmil, dsmil
from utils.dataset import C17Dataset, collate_fn_custom



def parse_args():
    parser = argparse.ArgumentParser(description='Train graph-based MIL models')

    parser.add_argument('--mil_model', default='abmil', type=str, help='MIL model [admil, dsmil]')
    parser.add_argument('--gnn_model', default = 'gat', type = str, help = 'Graph model [gcn,gat]')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [1]')
    parser.add_argument('--input_dim', default=1024, type=int, help='Dimension of the feature size [1024]')
    parser.add_argument('--hidden_dim',default=256, type=int, help='Hidden size of GCN [256]')
    parser.add_argument('--mil_lr', default=0.0001, type=float, help='Initial learning rate [0.0001]')
    parser.add_argument('--gnn_lr', default=0.001, type=float, help='Initial learning rate [0.001]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [50]')
    parser.add_argument('--beta', type=int, default= 0.998, help = 'beta for class weighting')
    parser.add_argument('--mil_weight_decay', default=1e-4, type=float, help='Mil Weight decay [1e-4]')
    parser.add_argument('--gnn_weight_decay', default=5e-4, type=float, help='GNN Weight decay [5e-4]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name [Camelyon17]')
    parser.add_argument('--graph_folder', default=None, type=str, help='Path where graphs are stored')
    parser.add_argument('--non_linearity', default=0, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training. ')
    parser.add_argument('--config', type=str, default=None, help='Config file to store args')
    parser.add_argument('--class_weighting', type=bool, default=True, help='Class weighting [True]')
    parser.add_argument('--n_kfold', type=int, default=5, help='Number of cross validation folds [5]')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to save results')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to the csv with filename and labels')
    parser.add_argument('--gpu', type=int, default=1, help='Choose which GPU to use')



    # early stopping parameters
    parser.add_argument('--patience', type=int, default=10, help='epochs to wait after no improvement [10]')
    parser.add_argument('--min_delta', type=int, default=1e-5, help='min_delta for early stopping [1e-5]')

    args = parser.parse_args()
    
    # If config file is provided, update args
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            args.__dict__.update(config)
    
    return args

def setup_environment(args):
    """Set up training environment and seed."""
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu < torch.cuda.device_count() else 'cpu')
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return device

def create_output_dir(args):
    """Create output directory for saving models and logs."""
    timestamp = datetime.datetime.now().strftime("%d%m%Y")
    base_dir = os.path.join(args.output_dir, timestamp, f"{args.dataset}_{args.mil_model}_{args.gnn_model}")
    
    os.makedirs(base_dir, exist_ok=True)
    run_id = 1
    while os.path.exists(os.path.join(base_dir, f"run{run_id}")):
        run_id += 1

    run_dir = os.path.join(base_dir, f"run{run_id}")
    os.makedirs(run_dir)
    
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    log_path = os.path.join(run_dir, 'log.txt')
    
    return run_dir, log_path

def load_dataset(args):
    """Load CSV file with filenames and labels"""
 
    h5_path = pd.read_csv(args.data_dir)
    
    return h5_path

def setup_criterion(args, device):
    """Set up loss function with class weighting if needed."""
    if args.num_classes == 1:
        if args.class_weighting:
            pos_weights = torch.tensor(1.63).to(device) ## change the weight
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        else:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        if args.class_weighting:
            class_counts = np.array([318, 36, 59, 87])  # This is for the C17 data
            effective_num = (1 - args.beta) / (1 - args.beta ** class_counts)
            class_weights = torch.tensor(effective_num, dtype=torch.float32).to(device)
            print(f"Class weights: {class_weights}")
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    return criterion

def initialize_models(args, device):
    """Initialize MIL and GNN models."""

    if args.gnn_model.lower() == 'gat':
        gnn_model = GATModel(args.input_dim, args.hidden_dim, args.num_classes).to(device)
    else:
        gnn_model = GCNModel(args.input_dim, args.hidden_dim, args.num_classes).to(device)
    
    if args.mil_model.lower() == 'abmil':
        mil_model = abmil.Attention(args.hidden_dim, args.num_classes, 0).to(device)
    elif args.mil_model.lower() == 'dsmil':
        i_classifier = dsmil.FCLayer(in_size=args.hidden_dim, out_size=args.num_classes).to(device)
        b_classifier = dsmil.BClassifier(
            args.hidden_dim, args.num_classes, 0, 
            dropout_v=args.dropout_node, 
            nonlinear=args.non_linearity
        ).to(device)
        mil_model = dsmil.MILNet(i_classifier, b_classifier).to(device)
    else:
        raise ValueError(f"Unsupported MIL model: {args.mil_model}")
    
    return gnn_model, mil_model

def setup_optimizer(args, gnn_model, mil_model):

    optimizer = torch.optim.Adam([
        {
            'params': filter(lambda p: p.requires_grad, gnn_model.parameters()),
            'lr': args.gnn_lr,
            'weight_decay': args.gnn_weight_decay
        },
        {
            'params': filter(lambda p: p.requires_grad, mil_model.parameters()),
            'lr': args.mil_lr,
            'weight_decay': args.mil_weight_decay
        }
    ], betas=(0.5, 0.9))
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
    
    return optimizer, scheduler

def train_epoch(trainloader, mil_model, gnn_model, criterion, optimizer, args, log_path, device):
    """ 
    Train function for MIL and GNN models

    Args:
        trainloader: Training data, iterable over (batch_data, file_name).
        mil_model: MIL model.
        gnn_model: GNN model.
        criterion: Loss function.
        optimizer: Optimizer.
        args: Command-line arguments or configurations.
        log_path: Path to log training statistics.
        device: Device to perform computation on (e.g., CPU or CUDA).

    Returns:
        float: Average training loss for the epoch.
    """
    mil_model.train()
    gnn_model.train()
    total_loss = 0
    
    atten_max = 0
    atten_min = 0
    atten_mean = 0
    
    accumulation_steps = 16 # for gradient accumulation
    
    for i, (batch_data, filename) in enumerate(trainloader):
        if batch_data is None:  # Skip empty batches
            continue
            
        batch_data = batch_data.to(device)
        batch_data.x = batch_data.x.to(torch.float32)
        labels = batch_data.y
        
        # Forward pass through GNN model
        if args.gnn_model.lower() == 'gat':
            gnn_out, _, _, _ = gnn_model(batch_data) 
        else:
            gnn_out = gnn_model(batch_data)
        
        # Forward pass through MIL model
        if args.mil_model == 'dsmil':
            ins_prediction, bag_prediction, attention, atten_B = mil_model(gnn_out)
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_loss = criterion(bag_prediction.view(len(batch_data), -1), labels.view(len(batch_data), -1))
            max_loss = criterion(max_prediction.view(len(batch_data), -1), labels.view(len(batch_data), -1))
            loss = 0.5 * bag_loss + 0.5 * max_loss
        elif args.mil_model == 'abmil':
            bag_prediction, _, attention = mil_model(gnn_out)
            max_prediction = bag_prediction
            if args.num_classes == 1:
                loss = criterion(bag_prediction.view(1, -1), labels.view(1, -1).float())
            else:
                loss = criterion(bag_prediction, labels.long())
        
        # Backpropagation with gradient accumulation
        loss.backward()
        if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(trainloader)):
            optimizer.step()
            optimizer.zero_grad()
        
        # Track statistics
        total_loss = total_loss + loss.item()
        atten_max = atten_max + attention.max().item()
        atten_min = atten_min + attention.min().item()
        atten_mean = atten_mean + attention.mean().item()
        
        # Print progress
        sys.stdout.write('\r Training [%d/%d] loss: %.4f, attention max:%.4f, min:%.4f, mean:%.4f' 
                        % (i, len(trainloader), loss.item(), 
                           attention.max().item(), attention.min().item(), attention.mean().item()))
    
    # Calculate average statistics
    atten_max = atten_max / len(trainloader)
    atten_min = atten_min / len(trainloader)
    atten_mean = atten_mean / len(trainloader)
    
    # Log statistics
    with open(log_path, 'a+') as log_txt:
        log_txt.write('\n atten_max: ' + str(atten_max))
        log_txt.write('\n atten_min: ' + str(atten_min))
        log_txt.write('\n atten_mean: ' + str(atten_mean))
    
    return total_loss / len(trainloader)


def test(testloader, mil_model, gnn_model, criterion, optimizer, args, log_path, epoch, device):
    
    """ 
    Test function for MIL and GNN models

    Args:
        test_df: Test data, iterable over (batch_data, file_name).
        mil_model: MIL model.
        gnn_model: GNN model.
        criterion: Loss function.
        optimizer: Optimizer.
        args: Command-line arguments or configurations.
        log_path: Path to log training statistics.
        device: Device to perform computation on (e.g., CPU or CUDA).
        epoch: Current epoch number. Defaults to 0.

    Returns:
        float: Average training loss for the epoch.
        float: Accuracy 
        float: Balanced Accuracy
        float: AUC values
        int: Array test labels
        int: Array test predictions
    """
    
    
    mil_model.eval()
    gnn_model.eval()
    
    total_loss = 0
    test_labels = []
    test_predictions = []
        
    with torch.no_grad():
        for i, (batch_data, filename) in enumerate(testloader):
            
            # Forward pass
            batch_data = batch_data.to(device)
            batch_data.x = batch_data.x.to(torch.float32)
            labels = batch_data.y
            
            if args.gnn_model.lower() == 'gat':
                gnn_out, _, _, _ = gnn_model(batch_data) 
            else:
                gnn_out = gnn_model(batch_data)
        

            if args.mil_model == 'dsmil':
                ins_prediction, bag_prediction, _, _ = mil_model(gnn_out)
                max_prediction, _ = torch.max(ins_prediction, 0)  
                bag_loss = criterion(bag_prediction.view(len(batch_data), -1), labels.view(len(batch_data), -1).float())
                max_loss = criterion(max_prediction.view(len(batch_data), -1), labels.view(len(batch_data), -1).float())
                loss = 0.5*bag_loss + 0.5*max_loss
                
            elif args.mil_model in ['abmil', 'max', 'mean']:
                bag_prediction, _, _ =  mil_model(gnn_out)
                max_prediction = bag_prediction
                
                if args.num_classes == 1:
                    loss = criterion(bag_prediction.view(len(batch_data), -1), labels.view(len(batch_data), -1))
                else:
                    loss = criterion(bag_prediction.view(len(batch_data), args.num_classes), labels.long()) # MULTICLASS
                    
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing/Validation [%d/%d] loss: %.4f' % (i, len(testloader), loss.item()))
            test_labels.extend(labels.cpu().numpy())
            
              # Predictions and labels
            bag_pred_sigmoid = torch.sigmoid(bag_prediction) if args.num_classes == 1 else torch.softmax(bag_prediction, dim=-1)
            max_pred_sigmoid = torch.sigmoid(max_prediction) if args.num_classes == 1 else torch.softmax(max_prediction, dim=-1)

            avg_prediction = 0.5 * max_pred_sigmoid + 0.5 * bag_pred_sigmoid if args.average else bag_pred_sigmoid
            if args.num_classes == 1:
                test_predictions.extend([avg_prediction.squeeze().cpu().numpy()])
            else:                
                test_predictions.extend(avg_prediction.cpu().numpy())
        
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    
    auc_value, _, thresholds_optimal = compute_roc(test_labels, test_predictions, args.num_classes, pos_label=1, is_multiclass=(args.num_classes > 1))
    
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n *****************Threshold by optimal*****************')
    if args.num_classes==1:
        class_prediction = copy.deepcopy(test_predictions)
        class_prediction[test_predictions>=thresholds_optimal[0]] = 1 # the optimal threshold comes from ROC curve
        class_prediction[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction
        test_labels = np.squeeze(test_labels)
        
    else:
        # apply thresholding, there will be a optimal threshold for each class
        for i in range(args.num_classes):
            class_prediction = copy.deepcopy(test_predictions[:, i])
            class_prediction[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction
            
        # solve conflicts (a sample can have multiple 1s or multiple 0s with separate optimal thresholds)
        for j in range(len(test_predictions[0])):
            if test_predictions[j].sum() == 0:
                # the sample belongs to no class -> take the maximum probability
                max_idx = np.argmax(test_predictions[j, :])
                test_predictions[j, max_idx] = 1
            elif test_predictions[j].sum() > 1:
                # sample belongs to multiple classes -> also take the maximum probability
                max_idx = np.argmax(test_predictions[j, :])
                test_predictions[j, :] = 0 #put all other classes to 0
                test_predictions[j, max_idx] = 1
        
    print('\n')
    cm = confusion_matrix(test_labels, test_predictions.argmax(axis=1) if args.num_classes > 1 else test_predictions)
    print(cm)
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n'+str(cm))
        
    bag_score = 0
    avg_score = np.mean(test_predictions.argmax(axis=1) == test_labels) if args.num_classes > 1 else np.mean(test_predictions == test_labels)

    cls_report = classification_report(test_labels, test_predictions.argmax(axis=1) if args.num_classes > 1 else test_predictions, digits=4)
    balanced_acc = balanced_accuracy_score(test_labels, test_predictions.argmax(axis=1) if args.num_classes > 1 else test_predictions)
    print('\n  multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, sum(auc_value)/len(auc_value)*100))
    print('\n', cls_report)
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, sum(auc_value)/len(auc_value)*100))
        log_txt.write('\n' + cls_report)
        
    return total_loss / len(testloader), avg_score,balanced_acc, auc_value, test_labels, test_predictions

def compute_roc(labels, predictions, num_classes, pos_label=1, verbose=False, is_multiclass = False):

    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    
    if not isinstance(labels, np.ndarray) or not isinstance(predictions, np.ndarray):
        raise TypeError("Both labels and predictions must be numpy arrays.")
    if labels.shape[0] != predictions.shape[0]:
        print('labels', labels.shape[0])
        print('prediction', predictions.shape[0])
        raise ValueError("Labels and predictions must have the same number of samples.")
    if not np.all((0 <= predictions) & (predictions <= 1)):
        raise ValueError("Predictions must be probabilities in the range [0, 1].")
    
    if verbose:  
        print(f"\n Predictions shape: {predictions.shape}")
        print(f"Labels shape: {labels.shape}")
        
    if is_multiclass:
        labels = label_binarize(labels, classes=np.arange(num_classes))
    
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    
    for c in range(0, num_classes):
        label = labels[:, c]
        if sum(label)==0:
            continue
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
       
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold) # CHECK IF THIS WORKS IN MULTICLASS
        c_auc = roc_auc_score(label, prediction)
        
        thresholds_optimal.append(threshold_optimal)
            
        thresholds.append(threshold)
        aucs.append(c_auc)
        
        
    return aucs, thresholds, thresholds_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def run_cross_validation(args, device, dataset_df, log_path, output_dir):
    """Run k-fold cross-validation."""
    test_acc = []
    test_balanced_acc = []
    test_auc = []
    test_losses = []
    
    num_folds = args.n_kfold
    num_patients = 20 # this is for Camelyon17
    
    # Extract patient IDs from filenames
    dataset_df["patient_id"] = dataset_df["0"].apply(lambda x: x.split("_node_")[0])
    unique_patients = sorted(dataset_df["patient_id"].unique(), key=lambda x: int(x.split("_")[1]))
    
    patience = args.patience
    min_delta = args.min_delta
    
    for fold in range(num_folds):
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        best_val_loss = float('inf')
        counter = 0  
        
        fold_dir = os.path.join(output_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        
        start_idx = fold * num_patients
        end_idx = start_idx + num_patients
        test_patients = unique_patients[start_idx:end_idx]
        train_patients = [p for p in unique_patients if p not in test_patients]
        
       
        fold_test_path = dataset_df[dataset_df["patient_id"].isin(test_patients)]
        
        train_patients, val_patients = train_test_split(train_patients, test_size=0.2, random_state=args.seed)
        train_set = dataset_df[dataset_df["patient_id"].isin(train_patients)]
        val_set = dataset_df[dataset_df["patient_id"].isin(val_patients)]
        
        print(f'Fold {fold} train: {train_set.shape}, val: {val_set.shape}, test: {fold_test_path.shape}')
        print(f"Train patients: {sorted(set(train_patients))}")
        print(f"Validation patients: {sorted(set(val_patients))}")
        print(f"Test patients: {sorted(set(test_patients))}")
        
        mean, std = compute_mean_features(train_set, args.graph_folder)
        
        train_dataset = C17Dataset(args, args.graph_folder, train_set, mean, std, args.dataset)
        val_dataset = C17Dataset(args, args.graph_folder, val_set, mean, std, args.dataset)
        test_dataset = C17Dataset(args, args.graph_folder, fold_test_path, mean, std, args.dataset)
        
        trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_custom)
        valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_custom)
        testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_custom)

        gnn_model, mil_model = initialize_models(args, device)
        
        optimizer, scheduler = setup_optimizer(args, gnn_model, mil_model)
        
        criterion = setup_criterion(args, device)
        
        train_losses = []
        val_losses = []
        val_accs = []
        
        for epoch in range(args.num_epochs):
            print(f'Starting epoch {epoch+1}')
            
            train_loss = train_epoch(trainloader, mil_model, gnn_model, criterion, optimizer, args, log_path, device)
            print(f'\n Train Loss: {train_loss:.4f}')
            
            for i, param_group in enumerate(optimizer.param_groups):
                print(f"\n Learning rate of group {i}: {param_group['lr']}")
            
            val_loss, val_accuracy, val_baccuracy, val_aucs, _, _ = test(valloader, mil_model, gnn_model, criterion, optimizer, args, log_path, epoch, device)
            print(f'Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}, Validation balanced accuracy: {val_baccuracy:.4f}')
                    
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                
                save_name_mil = os.path.join(fold_dir, f'{args.mil_model}_best.pth')
                torch.save(mil_model.state_dict(), save_name_mil)
                
                save_name_gnn = os.path.join(fold_dir, f'{args.gnn_model}_best.pth')
                torch.save(gnn_model.state_dict(), save_name_gnn)
                
                counter = 0
            else:
                counter += 1
            
            scheduler.step()
            
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} for fold {fold}!")
                break
        
        
        mil_model.load_state_dict(torch.load(save_name_mil))
        gnn_model.load_state_dict(torch.load(save_name_gnn))
        
        test_loss, test_accuracy, test_baccuracy, aucs, _, _ = test(testloader, mil_model, gnn_model, criterion, optimizer, args, log_path, epoch, device)
        print(f'\r Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}, Test balanced accuracy: {test_baccuracy:.4f}, AUC: ' + 
              '|'.join(f'class-{i}>>{auc:.4f}' for i, auc in enumerate(aucs)))
        
        info = f'\r Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}, Test balanced accuracy: {test_baccuracy:.4f}, AUC: ' + \
               '|'.join(f'class-{i}>>{auc:.4f}' for i, auc in enumerate(aucs))
               
        with open(log_path, 'a+') as log_txt:
            log_txt.write(f'\n *****************Metrics of fold {fold}*****************')
            log_txt.write('\n' + info)
        
        test_acc.append(test_accuracy)
        test_balanced_acc.append(test_baccuracy)
        test_auc.append(aucs)
        test_losses.append(test_loss)
    
    avg_test_acc = np.mean(test_acc)
    avg_balanced_acc = np.mean(test_balanced_acc)
    avg_auc = np.mean(test_auc)
    avg_loss = np.mean(test_losses)
    
    std_dev_acc = np.std(test_acc, dtype=np.float32)
    std_dev_bacc = np.std(test_balanced_acc, dtype=np.float32)
    std_dev_auc = np.std(test_auc, dtype=np.float32)
    std_dev_loss = np.std(test_losses, dtype=np.float32)
    
    print('Average accuracy:', avg_test_acc)
    print('Average balanced test accuracy:', avg_balanced_acc)
    print('Average AUC:', avg_auc)
    print('Average loss:', avg_loss)
    
    print('Std accuracy:', std_dev_acc)
    print('Std balanced accuracy:', std_dev_bacc)
    print('Std AUC:', std_dev_auc)
    print('Std loss:', std_dev_loss)
    
    with open(log_path, 'a+') as log_txt:
        log_txt.write(f'\n *****************Average Metrics*****************')
        log_txt.write(f'\nAcc: {avg_test_acc} ± {std_dev_acc}')
        log_txt.write(f'\nBAcc: {avg_balanced_acc} ± {std_dev_bacc}')
        log_txt.write(f'\nAUC: {avg_auc} ± {std_dev_auc}')
        log_txt.write(f'\nLoss: {avg_loss} ± {std_dev_loss}')
    
    return {
        'accuracy': avg_test_acc,
        'balanced_accuracy': avg_balanced_acc,
        'auc': avg_auc,
        'loss': avg_loss
    }

def main():
    """Main function."""
    
    args = parse_args()
    
    
    device = setup_environment(args)
    
    
    output_dir, log_path = create_output_dir(args)
    
   
    dataset_df = load_dataset(args)
    
    
    with open(log_path, 'w') as log_txt:
        log_txt.write(f"Started training at {datetime.datetime.now()}\n")
        log_txt.write(f"Arguments: {json.dumps(vars(args), indent=2)}\n")
    
   
    metrics = run_cross_validation(args, device, dataset_df, log_path, output_dir)
    
    print(f"Training completed. Results saved to {output_dir}")

if __name__ == '__main__':
    main()
