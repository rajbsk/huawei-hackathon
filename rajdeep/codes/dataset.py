import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import read_folder_files_regression, read_folder_files_prediction

class HuaweiRegressionDataset(Dataset):
    def __init__(self, opt):
        self.train_folder_path = opt["training_data_folder"]
        self.dev_folder_path = opt["dev_data_folder"]
        self.n_steps = opt["n_steps"]
        self.transform = False

        self.train_data = read_folder_files_regression(self.train_folder_path, self.n_steps)
        self.dev_data = read_folder_files_regression(self.dev_folder_path, self.n_steps)
        self.data = self.train_data + self.dev_data
        # self.data = self.train_data
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]

        return sample
    
class HuaweiPredictionDataset(Dataset):
    def __init__(self, opt):
        self.train_folder_path = opt["data_folder"]
        self.n_steps = opt["n_steps"]
        self.transform = False

        self.train_data = read_folder_files_prediction(self.train_folder_path, self.n_steps)
        self.data = self.train_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]

        return sample

