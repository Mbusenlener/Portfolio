import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from OpenmiirNet import EEGCNN
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns


def train_validate(data_loader,test_data_loader, num_epochs,lr,device,lambda_l1=0.01):
    num_classes = 8
    num_epochs = num_epochs
    model = EEGCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=.00)
    train_losses, val_losses, val_accuracies, model_params, total_guesses, total_actuals = [], [], [], [], [], []
    print('Training beginning')
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            total_loss = loss + lambda_l1 * l1_penalty  # Add L1 penalty to the original loss
            total_loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')
        
        avg_train_loss = epoch_train_loss / len(data_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss, correct, total = 0, 0, 0
        guesses = []
        actuals = []
        with torch.no_grad():
            for inputs, labels in test_data_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                guesses.extend(predicted)
                actuals.extend(labels)

        avg_val_loss = epoch_val_loss / len(test_data_loader)
        val_accuracy = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        model_params.append(model.state_dict())
        total_guesses.append(guesses)
        total_actuals.append(actuals)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

    print('Training finished')

    return train_losses, val_losses, val_accuracies, model_params, total_guesses, total_actuals



