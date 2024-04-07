import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.utils import shuffle # for shuffling
import os
import cv2
import random
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import gc
import argparse
import wandb

# class labels
classesList = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]
dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"


# Loading pretrained ResNet50 model
model = models.resnet50(pretrained=True)

# Freezing all the parameters in the model
for param in model.parameters():
    param.requires_grad = False

# Getting the number of inputs for the final layer
num_features = model.fc.in_features  
model.fc = nn.Linear(num_features, 10) 

# resnet50 takes input dimension of 224*224
resize_width = 224
resize_height = 224

def load_data(train_dir, test_dir, batchSize):
    
    # Transformation
    transform = transforms.Compose([
        transforms.Resize((resize_width, resize_height)), # Resizing the image
        transforms.ToTensor(), # Converting image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
    ])

    # Dataset
    TrainDataset = datasets.ImageFolder(root=train_dir, transform=transform)
    class_to_idx = TrainDataset.class_to_idx

    # Initialize lists to hold indices for training and validation
    train_indices = []
    val_indices = []

    # Spliting indices for each class
    for class_name, class_index in class_to_idx.items():
        # Find indices of images in the current class
        class_indices = [i for i, (_, label) in enumerate(TrainDataset.samples) if label == class_index]
        # Split these indices into training and validation
        _train_indices, _val_indices = train_test_split(class_indices, test_size=0.2, random_state=42)
        # Append to the main list
        train_indices.extend(_train_indices)
        val_indices.extend(_val_indices)

    # creating subsets for training and validation
    # based on the indices we took from splitting 
    train_subset = Subset(TrainDataset, train_indices)
    val_subset = Subset(TrainDataset, val_indices)

    # Create data loaders
    trainData_loader = DataLoader(train_subset, batch_size=batchSize, shuffle=True, num_workers=2, pin_memory=True)
    valData_loader = DataLoader(val_subset, batch_size=batchSize, shuffle=True, num_workers=2, pin_memory=True)

    TestDataset = datasets.ImageFolder(root=test_dir, transform=transform)
    # DataLoader with shuffling
    TestData_loader = DataLoader(TestDataset,num_workers=2, batch_size=batchSize, pin_memory=True)
    
    return trainData_loader, valData_loader, TestData_loader

def train(model, criterion, optimizer, num_epochs, train_loader, val_loader):
    for epoch in range(num_epochs):
        # activating the model in train mode
        model.train()
        
        for ind, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Training Progress {epoch+1}')):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        find_accuracy(model, criterion, train_loader, "train")
        find_accuracy(model, criterion, val_loader, "validation")
        
def find_accuracy(model, criterion, dataLoader, dataName):
#     making the model in evaluation mode
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    print(f'{dataName} Loss: {val_loss/len(dataLoader)}, '
          f'{dataName} Accuracy: {100*correct/total}%\n')
    # wandb.log({f"{dataName}_loss": val_loss/len(dataLoader)})
    # wandb.log({f"{dataName}_accuracy": 100*correct/total})

def config_and_train(base_dir = "inaturalist_12K",  learning_rate = 1e-4, weight_decay=0.005, epochs = 10, batchSize = 32, optimiser_fn = "nadam"):
    trainDataLoader, valDataLoader, testDataLoader = load_data(train_dir = f'{base_dir}/train', test_dir = f'{base_dir}/val', batchSize = batchSize)
    
    if optimiser_fn == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimiser_fn == "nadam":
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimiser_fn == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        # stocastic gradient decent        
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model,device_ids = [0]).to(device)
    criterion = nn.CrossEntropyLoss()

    train(model, criterion, optimizer, epochs, trainDataLoader, valDataLoader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--wandb_entity", "-we",help = "Wandb Entity used to track experiments in the Weights & Biases dashboard.", default="cs23m065")
    # parser.add_argument("--wandb_project", "-wp",help="Project name used to track experiments in Weights & Biases dashboard", default="Assignment 2")
    parser.add_argument("--epochs","-e", help= "Number of epochs to train neural network", type= int, default=10)
    parser.add_argument("--batch_size","-b",help="Batch size used to train neural network", type =int, default=16)
    parser.add_argument("--optimizer","-o",help="batch size is used to train neural network", default= "nadam", choices=['nadam', 'adam', 'rmsprop'])
    parser.add_argument("--learning_rate","-lr", default=0.1, type=float)
    parser.add_argument("--weight_decay","-w_d", default=0.0,type=float)
    parser.add_argument("--base_dir", "-br", default="inaturalist_12K")
    
    args = parser.parse_args()
    config_and_train(base_dir=args.base_dir, learning_rate=args.learning_rate, weight_decay = args.weight_decay, epochs=args.epochs, batchSize= args.batch_size, optimiser_fn=args.optimizer)