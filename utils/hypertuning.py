import torch
import torch.nn as nn
import itertools
from torch.utils.data import DataLoader
from utils.cnn_controller import train_model
import numpy as np

def hyper_tuning_model(model_class, train_dataset, test_dataset, parameters, device, epochs=50, wandb_writer=None, show=False):
    keys, values = zip(*parameters.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_params = None
    best_acc = 0
    results = []

    i = 0
    for params_set in combinations:
        batch_size = params_set['batch_size']
        lr = params_set['lr']
        weight_decay = params_set['weight_decay']
        step_size = params_set['step_size']
        gamma = params_set['gamma']

        print(f'Tuning: batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}, step_size={step_size}, gamma={gamma}\n')
        
        model = model_class(num_classes=4)
        
        train_loader = DataLoader(train_dataset, batch_size=params_set['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params_set['batch_size'], shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        save_path = f'/kaggle/working/eeg_net_set{i}.pth'
        i += 1
        train_losses, train_acc, test_losses, test_acc = train_model(model, train_loader, test_loader, 
                                                                     criterion, optimizer, scheduler, 
                                                                     save_path, device, 
                                                                     epochs=epochs, wandb_writer=wandb_writer, show=show)

        if np.max(test_acc) > best_acc:
            best_params = params_set
            best_acc = np.max(test_acc)

        results.append({
            'params': params_set,
            'train_losses': train_losses,
            'train_acc': train_acc,
            'test_losses': test_losses,
            'test_acc': test_acc,
        })
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>\n<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>\n")

    return best_params, best_acc, results
        