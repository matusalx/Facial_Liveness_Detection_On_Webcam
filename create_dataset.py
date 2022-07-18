import os
import torch
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

working_dir = os.getcwd()
dataset_folder = os.path.dirname(working_dir)


dataset_dir1 = os.path.join(dataset_folder, r'liveness_datasets/custom_spf_real/archive (1)')
dataset_dir2 = os.path.join(dataset_folder, r'liveness_datasets/LCC_FASD/archive/LCC_FASD')

dataset_dir_train1 = dataset_dir1 + r'/db_faces_mehdi/db_faces/train'
dataset_dir_train2 = dataset_dir1 + r'/indatacore_dataset_4_per_persons/indatacore_dataset_4_per_persons/train'
dataset_dir_train3 = dataset_dir1 + r'/soufiane_data/soufiane_data/train'
dataset_dir_train4 = dataset_dir2 + r'/LCC_FASD_training'

dataset_dir_valid1 = dataset_dir1 + r'/db_faces_mehdi/db_faces/test'
dataset_dir_valid2 = dataset_dir1 + r'/indatacore_dataset_4_per_persons/indatacore_dataset_4_per_persons/test'
dataset_dir_valid3 = dataset_dir1 + r'/soufiane_data/soufiane_data/test'
dataset_dir_valid4 = dataset_dir2 + r'/LCC_FASD_evaluation'

data_dir_train_all = [dataset_dir_train1, dataset_dir_train2, dataset_dir_train3, dataset_dir_train4]
data_dir_valid_all = [dataset_dir_valid1, dataset_dir_valid2, dataset_dir_valid3, dataset_dir_valid4]


class CreateDatasetSpoof(Dataset):
    def __init__(self, data_dirs, transform):
        self.data_dirs = data_dirs
        self.transform = transform

    def __len__(self):
        length = 0
        for x in self.data_dirs:
            length += len(os.listdir(x+'//real'))+len(os.listdir(x+'//spoof'))
        return length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        list_all_items = []
        for x in self.data_dirs:
            for dirname, dirnames, filenames in os.walk(x + '//real'):
                for filename in filenames:
                    list_all_items.append([os.path.join(dirname, filename), 'real'])
            for dirname, dirnames, filenames in os.walk(x + '//spoof'):
                for filename in filenames:
                    list_all_items.append([os.path.join(dirname, filename), 'spoof'])

        image = Image.open(list_all_items[idx][0])
        target = list_all_items[idx][1]
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'target': target}
        return sample
