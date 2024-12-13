import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def evaluate_model(model, test_loader, criterion, device=None, report=False, show=False):
    if device is not None:
        model.to(device)

    model.eval() # Set the model to evaluation mode
    running_loss = 0.0

    true_labels = []
    predicted_labels = []

    if show:
        test_loader = tqdm(test_loader, unit='batch', desc='Evaluating')

    # Use torch.no_grad() to disable gradient calculation during evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            if device is not None:
                images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = torch.argmax(outputs, dim=-1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / len(test_loader)
    accuracy = accuracy_score(true_labels, predicted_labels)

    labels = list(range(4))
    names = ['Left Hand', 'Right Hand', 'Foot', 'Tongue']
    classification_rep = classification_report(true_labels, predicted_labels, labels=labels, target_names=names, zero_division=0)

    if show:
        print(f'\t---> Loss: {epoch_loss:.4f}\n\t---> Accuracy: {accuracy:.4f}')
    if report and show:
        print(f'\tClassification Report:\n{classification_rep}')
    return epoch_loss, accuracy


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, save_path, device=None, epochs=10, wandb_writer=None, report=False, show=True):
    if device is not None:
        model.to(device)
    model.train()

    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []

    best_acc = 0
    for epoch in range(epochs):
        running_loss = 0.0

        true_labels = []
        predicted_labels = []

        if show:
            train_loader = tqdm(train_loader, unit='batch', desc=f'Training epoch [{epoch+1}/{epochs}]')

        for images, labels in train_loader:
            if device is not None:
                images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            predicted = torch.argmax(outputs, dim=-1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
        
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        accuracy = accuracy_score(true_labels, predicted_labels)
        train_acc.append(accuracy)
        if show:
            print(f'\t---> Loss: {epoch_loss:.4f}\n\t---> Accuracy: {accuracy:.4f}')

        test_loss, test_ac = evaluate_model(model, test_loader, criterion, device, show=show, report=report)
        model.train()
        test_losses.append(test_loss)
        test_acc.append(test_ac)
        if wandb_writer is not None:
            wandb_writer.log({
                "Train loss": epoch_loss,
                "Train accuracy": accuracy,
                "Test loss": test_loss,
                "Test accuracy": test_ac
            })

        if test_ac > best_acc:
            best_acc = test_ac
            torch.save(model.state_dict(), save_path)
            if show:
                print('Model is saved to ', save_path)
            if wandb_writer is not None:
                wandb_writer.save(save_path)
        if show:
            print("====================================================================")

    if wandb_writer is not None:
        wandb_writer.finish()
    return train_losses, train_acc, test_losses, test_acc