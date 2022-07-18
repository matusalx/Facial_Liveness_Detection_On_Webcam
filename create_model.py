import os
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

from torch import nn
from collections import OrderedDict
from torch import optim
import argparse


import importlib
import create_dataset
from create_dataset import data_dir_train_all
from create_dataset import data_dir_valid_all
from torch.utils.data import random_split
import train


importlib.reload(create_dataset)


data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
train_data = create_dataset.CreateDatasetSpoof(data_dir_train_all, data_transform)
valid_data = create_dataset.CreateDatasetSpoof(data_dir_valid_all, data_transform)


##
train_dataset_1_size = int(len(train_data)*0.6)
train_dataset_2_size = len(train_data)-train_dataset_1_size
train_dataset_1, train_dataset_2 = random_split(train_data, [train_dataset_1_size, train_dataset_2_size])

valid_dataset_1_size = int(len(valid_data)*0.9)
valid_dataset_2_size = len(valid_data)-valid_dataset_1_size
valid_dataset_1, valid_dataset_2 = random_split(valid_data, [valid_dataset_1_size, valid_dataset_2_size])
##


train_dataloader = DataLoader(train_dataset_2, batch_size=32, shuffle=True)
val_dataloader = DataLoader(valid_dataset_2, batch_size=32, shuffle=True)

# print(len(train_data), len(valid_data))
# print(len(train_dataset_2), len(valid_dataset_2))


model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(512, 100)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(100, 2)),
    ('output', nn.LogSoftmax(dim=1))
    ]))
# fc params require grad True
model.fc = fc
criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

base_dir = os.getcwd()
checkpoint_path = os.path.join(base_dir, 'resnet_18_spoof_model_40.pt')
n_epochs = 1

train.train(model, optimizer, criterion, train_dataloader, val_dataloader, checkpoint_path, n_epochs=n_epochs)




