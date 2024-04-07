import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.utils import shuffle # for shuffling
import os
import cv2
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb

# classes list
classesList = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]

# setting device
dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
resize_width = 500
resize_height= 500

def load_data(train_dir, test_dir, batchSize, data_augumentation = 'no'):
    
    # Transformation
    transform = transforms.Compose([
        transforms.Resize((resize_width, resize_height)),  # Resize to defined width and height
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
    ])
    
    tranform_aug = transforms.Compose([
        transforms.Resize((resize_width, resize_height)),  # Resize to defined width and height
        transforms.ToTensor(),              # Converting the image to a PyTorch tensor
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjusting brightness, contrast, saturation, and hue
        transforms.RandomRotation(10),      # Randomly rotating the image by a maximum of 10 degrees
        transforms.RandomResizedCrop(resize_width),  # Randomly crop and resize the image to the defined width
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
    ])

    # Dataset
    if data_augumentation == "no":
        # No augumentaion to the data
        TrainDataset = datasets.ImageFolder(root=train_dir, transform=transform)
    else:
        # Augumenting the data
        TrainDataset = datasets.ImageFolder(root=train_dir, transform=tranform_aug)
        
    class_to_idx = TrainDataset.class_to_idx

    # Initialize lists to hold indices for training and validation
    train_indices = []
    val_indices = []

    # Split indices for each class
    for class_name, class_index in class_to_idx.items():
        # Find indices of images in the current class
        class_indices = [i for i, (_, label) in enumerate(TrainDataset.samples) if label == class_index]

        # Split these indices into training and validation
        _train_indices, _val_indices = train_test_split(class_indices, test_size=0.2, random_state=42)

        # Append to the main list
        train_indices.extend(_train_indices)
        val_indices.extend(_val_indices)

    # Create subsets for training and validation
    train_subset = Subset(TrainDataset, train_indices)
    val_subset = Subset(TrainDataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batchSize, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batchSize, shuffle=True, num_workers=2, pin_memory=True)

    # same for validation
    TestDataset = datasets.ImageFolder(root=test_dir, transform=transform)
    # DataLoader with shuffling
    TestData_loader = DataLoader(TestDataset,num_workers=2, batch_size=batchSize, pin_memory=True)
    
    return train_loader, val_loader, TestData_loader
    
def getDimOfLastConv(num_filters, filter_sizes, use_batch_norm=False):
    layers = []
    
    # Initial convolution layer
    layers.append(nn.Conv2d(3, num_filters[0], kernel_size=filter_sizes[0], stride=1, padding=0))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(num_filters[0]))
    
    # Subsequent convolution layers
    for i in range(1, len(num_filters)):
        layers.append(nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=filter_sizes[i], stride=1, padding=0))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(num_filters[i]))
    
    conv_stack = nn.Sequential(*layers)
    
    image_tensor = torch.zeros([1, 3, resize_width, resize_height])
    x = conv_stack(image_tensor)
    flat = nn.Flatten()
    x = flat(x)
    return x.shape[-1]

class SimpleCNN(nn.Module):
  def __init__(self, num_filters, filter_sizes, activation_fn, num_neurons_dense, use_batch_norm, dropout_prob):
    super(SimpleCNN, self).__init__()

    layers = []
    
    # Initial convolution layer
    layers.append(nn.Conv2d(3, num_filters[0], kernel_size=filter_sizes[0], stride=1, padding=0))
    layers.append(activation_fn)
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    if use_batch_norm == 'true':
        layers.append(nn.BatchNorm2d(num_filters[0]))
    
    # Subsequent convolution layers
    for i in range(1, len(num_filters)):
        layers.append(nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=filter_sizes[i], stride=1, padding=0))
        layers.append(activation_fn)
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if use_batch_norm == 'true':
            layers.append(nn.BatchNorm2d(num_filters[i]))
    
    self.conv_stack = nn.Sequential(*layers)

    flattenNodes = getDimOfLastConv(num_filters, filter_sizes, use_batch_norm)
    self.flatten = nn.Flatten()
    self.dense = nn.Linear(flattenNodes, num_neurons_dense)
    self.dropout = nn.Dropout(dropout_prob)
    self.output = nn.Linear(num_neurons_dense, 10)

  def forward(self, x):
    x = self.conv_stack(x)  # Pass input through conv_stack
    x = self.flatten(x)  # Flatten the output of conv layers
    x = self.dense(x) # passing through dense layer
    x = self.dropout(x) #adding dropout
    x = self.output(x)
    return x


def train(model, criterion, optimizer, num_epochs, train_loader, val_loader):
    for epoch in range(num_epochs):
        model.train()
#         for inputs, labels in train_loader:
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
    
    wandb.log({f'{dataName}_accuracy': 100*correct/total})
    wandb.log({f'{dataName}_loss': val_loss/len(dataLoader)})
    
def train_model(learning_rate, num_filters, filter_sizes, activation_fn, optimiser_fn, num_neurons_dense, weight_decay, dropout, useBatchNorm, batchSize, num_epochs, data_augumentation = 'no', base_dir = "inaturalist_12k"):
    activation_dict = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'selu': nn.SELU(), 'silu': nn.SiLU(), 'gelu': nn.GELU(), 'mish': nn.Mish()}
    #SELU (Scaled Exponential Linear Unit) and SiLU (Sigmoid Linear Unit)
    trainDataLoader, valDataLoader, testDataLoader = load_data(train_dir = 'f{base_dir}/train', test_dir = 'f{base_dir}/val', batchSize = batchSize, data_augumentation = data_augumentation)
    
    #num_epochs = 10
    model = SimpleCNN(num_filters=num_filters, filter_sizes=filter_sizes, 
                  activation_fn=activation_dict[activation_fn], num_neurons_dense=num_neurons_dense,
                  dropout_prob=dropout, use_batch_norm = useBatchNorm)

    criterion = nn.CrossEntropyLoss()
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
    model = torch.nn.DataParallel(model,device_ids = [0,1]).to(device)


    train(model, criterion, optimizer, num_epochs, trainDataLoader, valDataLoader)
    
    return model

def generateImage_30(model, testDataLoader):

    # Identify the computing device used by the model
    compute_device = next(model.parameters()).device

    model.eval()  # Switch model to evaluation mode

    # Prepare to collect a limited number of image samples for each category
    samples_limit = 3
    category_samples = {category: [] for category in range(10)}  # Assuming categories are labeled 0-9

    # Ensure no gradient computations for efficiency
    with torch.no_grad():
        for batch_images, batch_labels in testDataLoader:
            batch_images, batch_labels = batch_images.to(compute_device), batch_labels.to(compute_device)  # Match model's device
            # Check if sufficient samples have been collected
            if all(len(samples) >= samples_limit for samples in category_samples.values()):
                break
            for image, label in zip(batch_images, batch_labels):
                current_label = label.item()
                if len(category_samples[current_label]) < samples_limit:
                    # Predict the label for each image
                    prediction = model(image.unsqueeze(0)).argmax(1).item()
                    # Store the CPU-based image and its predicted label
                    category_samples[current_label].append((image.cpu(), prediction))

    # Setting up the visualization
    figure, axes = plt.subplots(10, 3, figsize=(10, 33))  # Allocate a grid for the sample images

    for category_id, images in category_samples.items():
        for index, (image, predicted) in enumerate(images):
            plot_axis = axes[category_id, index]
            # Reformat image for plotting
            image_to_plot = image.numpy().transpose((1, 2, 0))
            normalize_mean = np.array([0.485, 0.456, 0.406])
            normalize_std = np.array([0.229, 0.224, 0.225])
            image_to_plot = normalize_std * image_to_plot + normalize_mean
            image_to_plot = np.clip(image_to_plot, 0, 1)
            plot_axis.imshow(image_to_plot)
            plot_axis.set_title(f'Real: {classesList[category_id]}, Guess: {classesList[predicted]}')
            plot_axis.axis('off')

    plt.tight_layout()

    # Save and display the image grid
    plt.savefig('predictions_overview.png', dpi=300)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", "-we",help = "Wandb Entity used to track experiments in the Weights & Biases dashboard.", default="cs23m065")
    parser.add_argument("--wandb_project", "-wp",help="Project name used to track experiments in Weights & Biases dashboard", default="Assignment 2")
    parser.add_argument("--epochs","-e", help= "Number of epochs to train neural network", type= int, default=10)
    parser.add_argument("--batch_size","-b",help="Batch size used to train neural network", type =int, default=16)
    parser.add_argument("--optimizer","-o",help="batch size is used to train neural network", default= "nadam", choices=['nadam', 'adam', 'rmsprop'])
    parser.add_argument("--learning_rate","-lr", default=0.1, type=float)
    parser.add_argument("--weight_decay","-w_d", default=0.0,type=float)
    parser.add_argument("--activation", "-a",choices=['relu', 'elu', 'selu', 'silu', 'gelu', 'mish'], default="relu")
    parser.add_argument("--num_filters", "-nf", nargs=5, type=int, default=[32, 32, 32, 32, 32])
    parser.add_argument("--filter_sizes", "-fs", nargs=5, type=int, default=[3, 3, 3, 3, 3])
    parser.add_argument("--batch_norm", "-bn", default="true", choices=["true", "false"])
    parser.add_argument("--dense_layer", "-dl", default=128, type=int)
    parser.add_argument("--augumentaion", "-a", default="yes", choices=["yes", "no"])
    parser.add_argument("--dropout", "-dp", default=0.2, type=float)
    parser.add_argument("--base_dir", "-br", default="inaturalist_12K")
    
    args = parser.parse_args()

    wandb.login()
    wandb.init(project=args.wandb_project,entity=args.wandb_project)
    model = train_model(learning_rate = args.learning_rate, num_filters = args.num_filters, filter_sizes=args.filter_sizes, 
                    activation_fn = args.activation, optimiser_fn = args.optimizer, num_neurons_dense = args.dense_layer, 
                    weight_decay = args.weight_decay, dropout = args.dropout, useBatchNorm = args.batch_norm, batchSize = args.batch_size, 
                    num_epochs = args.epochs, data_augumentation = args.augumentaion, base_dir= args.base_dir)
    wandb.finish()
   