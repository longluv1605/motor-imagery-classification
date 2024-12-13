import numpy as np
import torch
from utils.data import EEGDataset
from torchvision.transforms import v2
from utils.hypertuning import hyper_tuning_model
from models.eegnet import EEGNet
from utils.data import load_data

def prepare_data(data_path):
    X_train, X_test, y_train, y_test = load_data(data_path)
    
    # Train and test dataset
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)
    
    return train_dataset, test_dataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_path = 'data/bci-iv-2a'
    
    parameters = {
        "lr": [0.0001, 0.0005, 0.001, 0.005],
        "weight_decay": [0.001, 0.01],
        "step_size": [5, 10, 20],  # Discrete values
        "gamma": [0.1, 0.5, 0.9],
        "batch_size": [8, 16, 32, 64, 128],  # Add batch size options
    }
    
    train_dataset, test_dataset = prepare_data(data_path)
    
    best_params, best_acc, results = hyper_tuning_model(EEGNet, train_dataset, test_dataset, parameters, device, epochs=50, wandb_writer=None, show=False)
    print(f'best_params: {best_params}\nbest_acc: {best_acc}')