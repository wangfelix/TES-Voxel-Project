from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random

class carlaDataset(Dataset):

    def __init__(self, X, image_size):
        'Initialization'
        self.X = X
        self.image_size = image_size

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.X[index]
        X = self.transform(image)
        return X
    
    transform = T.Compose([
        # T.ToPILImage(),
        T.ToTensor(),
        T.Resize((150,150))])

    