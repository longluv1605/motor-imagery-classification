import torch
import torch.nn as nn
from utils.visualizer import visualize
from torchvision.transforms import v2
from utils.data import load_data
from utils.data import EEGDataset
from torch.utils.data import DataLoader
from models.eegnet import EEGNet
from utils.eeg_controller import train_model

def prepare_data(data_path, batch_size=16):
    X_train, X_test, y_train, y_test = load_data(data_path)
    
    # Train and test dataset
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)

    # Train and test loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    img, label = next(iter(train_loader))
    print(img.shape, label.shape)
    print(label)
    
    return train_loader, test_loader

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_path = 'data/bci-iv-2a'
    train_loader, test_loader = prepare_data(data_path)
    
    # Define model
    num_classes = 4
    eeg_net = EEGNet(num_classes=num_classes)
    
    # Prepare to train
    lr=0.001
    weight_decay=1e-4
    step_size=10
    gamma=0.5

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(eeg_net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Train and evaluate
    epochs = 200
    save_path = '/kaggle/working/eeg_net.pth'
    train_losses, train_acc, test_losses, test_acc = train_model(eeg_net, train_loader, test_loader, 
                                                                criterion, optimizer, scheduler, 
                                                                save_path, device, 
                                                                epochs=epochs, wandb_writer=None, show=True)
    visualize(train_losses, test_losses, train_acc, test_acc)
    