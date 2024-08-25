import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import LeaveOneGroupOut
import h5py
from OpenmiirNet import EEGCNN
import matplotlib.pyplot as plt
from train_convNet import train_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_metrics(train_losses, val_losses, val_accuracies, num_epochs,fold,run):
    epochs = range(1, num_epochs + 1)
    
    plt.figure(figsize=(12, 6))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    filename = f'plot_fold_{fold}_run_{run}.png'
    plt.savefig(filename)
    plt.show()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_classes = 8
num_epochs = 50
run=2
data = []
labels = []
file_path = 'processed_data/data.h5'
with h5py.File(file_path, 'r') as f:
    data = f['dataset'][:]
    labels = f['labels'][:]
data = np.array(data)
labels = np.array(labels)
groups = np.repeat(np.arange(9),40)
dataset = torch.utils.data.TensorDataset(torch.tensor(np.expand_dims(data, axis=2), dtype=torch.float32),torch.tensor(labels, dtype=torch.long))
logo = LeaveOneGroupOut()
results = []
train_loss_list = []
val_loss_list = []
model_param_list = []
total_guesses = []
total_actuals = []
for i, (train_index, test_index) in enumerate(logo.split(data,labels,groups)):
    training_data = Subset(dataset,train_index)
    testing_data = Subset(dataset,test_index)
    train_loader = DataLoader(training_data,batch_size=40, shuffle=True)
    test_loader = DataLoader(testing_data,batch_size=40,shuffle=True)
    train_losses, val_losses, val_accuracies, model_params, guesses, actuals = train_validate(train_loader,test_loader,num_epochs,0.01,device)
    results.append(val_accuracies)
    train_loss_list.append(train_losses)
    val_loss_list.append(val_losses)
    model_param_list.append(model_params)
    total_guesses.append(guesses)
    total_actuals.append(actuals)
max_accuracy_across_folds = 0
max_accuracy_epoch = 0
best_models = 0
best_guesses = []
best_actuals = []
for i in range(num_epochs):
    average_accuracy = 0
    for j in range(9):
        average_accuracy += results[j][i]
    average_accuracy /= 9.0
    if average_accuracy > max_accuracy_across_folds:
        max_accuracy_across_folds = average_accuracy
        max_accuracy_epoch = i+1
        best_models = np.array(model_param_list)[:,i]
        best_guesses = np.array(total_guesses)[:,i]
        best_actuals = np.array(total_actuals)[:,i]
        print(max_accuracy_across_folds)
        print(max_accuracy_epoch)

print("Best Average performance across all folds: ", max_accuracy_across_folds)
print("Epoch of best average accuracy across folds: ", max_accuracy_epoch)
print("Accuracy of each fold at this epoch: ")
results = np.array(results)
for i, accuracy in enumerate(results[:,max_accuracy_epoch-1]):
    print("Accuracy for fold ", i+1, ": ", accuracy)
best_guesses = np.array(best_guesses).flatten()
best_actuals = np.array(best_actuals).flatten()
print("Averaged confusion matrix across folds:")
cm = confusion_matrix(best_actuals,best_guesses)
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
filename = f'confusion_matrix_run{run}.png'
plt.savefig(filename)
plt.show()
for i, model in enumerate(best_models):
    model_file = f'BestModel{run}_Fold{i+1}.pt'
    torch.save(model,model_file)
plot_list = []
for i, (val_accuracies, train_losses, val_losses) in enumerate(zip(results,train_loss_list,val_loss_list)):
    plot_metrics(train_losses,val_losses,val_accuracies,num_epochs,i+1,run)




