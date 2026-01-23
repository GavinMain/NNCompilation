#Version: CNN_05 adds batch norm to improve model
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as DataLoader
import torchvision
import torchvision.transforms as transforms

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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm1 = nn.BatchNorm2d(num_features=16)
        self.norm2 = nn.BatchNorm2d(num_features=32)
        self.linear = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
if __name__ == "__main__":
    #Predefine transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    #Load data + apply Transforms
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    
    #Class Definitions from dataset
    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    image, label = train_dataset[0]

    print("Image shape:", image.shape)  
    print("Label index:", label)
    print("Label name:", classes[label])

    plt.imshow((image.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1))
    plt.title(f"Label: {classes[label]}")
    plt.axis("off")
    plt.show()


    #Create data loaders
    train_loader = DataLoader.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    with open(log_file, 'a') as f:
        f.write("\nCNN_05:\n")
    
    # Train the model
    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        train_loss = train_model(model, train_loader, optimizer, criterion)
        test_acc = evaluate_model(model, test_loader)
        
        print(f"Epoch: {epoch} | Train Loss: {train_loss} | Test Accuracy: {test_acc} | Time: {time.time() - start_time}")

        with open(log_file, 'a') as f:
            f.write(f"Epoch: {epoch} | Train Loss: {train_loss} | Test Accuracy: {test_acc} | Time: {time.time() - start_time}\n")