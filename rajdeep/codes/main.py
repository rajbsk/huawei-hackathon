from __future__ import print_function, division
import sys

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
import glob

from dataset import HuaweiDataset
from model import LSTMModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    opt_dataset = {"n_steps":10, "training_data_folder" : "../data/data_v1/training/", 
    "dev_data_folder" : "../data/data_v1/"}


    huawei_dataset = HuaweiDataset(opt_dataset)

    dataset_size = len(huawei_dataset)
    train_size = int(0.8*dataset_size)
    dev_size = int(0.5*(dataset_size - train_size))
    test_size = dataset_size - train_size - dev_size

    huawei_train_dataset, huawei_dev_dataset, huawei_test_dataset = torch.utils.data.random_split(huawei_dataset, [train_size, dev_size, test_size])

    opt_model = {"learning_rate": 0.0001, "input_size": 1, "hidden_size": 5, "epochs": 100, "batch_size": 256, "num_layers":1, "save_every":1,
     "model_directory": "", "weight_decay": 0, "model_name": "LSTMModel", "bidirectional": False, "device": device}

    HuaweiDatasetLoaderTrain = DataLoader(huawei_train_dataset, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0)
    HuaweiDatasetLoaderDev = DataLoader(huawei_dev_dataset, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0)
    HuaweiDatasetLoaderTest = DataLoader(huawei_test_dataset, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0)

    model = LSTMModel(opt_model)
    model.cuda(device)
    model.train_model(HuaweiDatasetLoaderTrain, HuaweiDatasetLoaderDev)

if __name__=="__main__":
    main()
