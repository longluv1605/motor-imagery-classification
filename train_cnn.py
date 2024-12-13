import torch
import torch.nn as nn
from utils.visualizer import visualize

from utils.data import load_data_and_mix
from utils.data import process_all_trials
from utils.data import EEGSpectralDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models.spectral_cnn import EEGCNN
from utils.cnn_controller import train_model

def prepare_data(data_path, test_size=0.2, batch_size=8):
    X, y = load_data_and_mix(data_path)
    
    # Các tham số
    fs = 256  # Tần số lấy mẫu
    bands = {"Theta": (4, 8), "Alpha": (8, 12), "Beta": (12, 30)}

    # Áp dụng cho X
    X = process_all_trials(X, fs, bands, output_size=(32, 32))
    
    # Chia data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
    
    # Train và test dataset
    train_dataset = EEGSpectralDataset(X_train, y_train)
    test_dataset = EEGSpectralDataset(X_test, y_test)

    # Train và test loader

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_path = 'data/bci-iv-2a'
    train_loader, test_loader = prepare_data(data_path)
    
    # Define model
    model = EEGCNN(input_channels=3, num_classes=4)
    
    # Prepare to train
    lr = 0.003
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train and evaluate
    train_losses, train_acc, test_losses, test_acc = train_model(model, train_loader, test_loader, criterion, optimizer, device=device, epochs=10)
    
    visualize(train_losses, test_losses, train_acc, test_acc)
    

if __name__ == "__main__":
    main()