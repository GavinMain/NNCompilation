#Version: MLP_02 uses PyTorch for improved performance
import numpy as np
import time
import torch
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as DataLoader

#Hyper Parameters
num_epochs = 10
lr = .001
batch_size = 64

#Other Parameters
log_file = "log.txt"

def train_model(model, train_loader, optimizer, criterion,learning_rate=lr, batch_size=batch_size):
    model.train()
    for batch, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return loss.item()

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return accuracy

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(32 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
if __name__ == "__main__":
    #Load + Format Data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape(-1, 1, 28, 28).astype('float32') / 255.0
    test_images = test_images.reshape(-1, 1, 28, 28).astype('float32') / 255.0
    train_labels = to_categorical(train_labels, num_classes=10)
    test_labels = to_categorical(test_labels, num_classes=10)

    #Convert to Tensor (for pytorch compatibility)
    train_images_tensor = torch.tensor(train_images)
    train_labels_tensor = torch.tensor(np.argmax(train_labels, axis=1))
    test_images_tensor = torch.tensor(test_images)
    test_labels_tensor = torch.tensor(np.argmax(test_labels, axis=1))

    #Put into Dataloader for simple batch loading
    train_dataset = DataLoader.TensorDataset(train_images_tensor, train_labels_tensor)
    test_dataset = DataLoader.TensorDataset(test_images_tensor, test_labels_tensor)
    
    train_loader = DataLoader.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    with open(log_file, 'a') as f:
        f.write("\nCNN_02:\n")
    
    # Train the model
    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        train_loss = train_model(model, train_loader, optimizer, criterion)
        test_acc = evaluate_model(model, test_loader)
        
        print(f"Epoch: {epoch} | Train Loss: {train_loss} | Test Accuracy: {test_acc} | Time: {time.time() - start_time}")

        with open(log_file, 'a') as f:
            f.write(f"Epoch: {epoch} | Train Loss: {train_loss} | Test Accuracy: {test_acc} | Time: {time.time() - start_time}\n")