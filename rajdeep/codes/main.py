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

from dataset import HuaweiRegressionDataset, HuaweiPredictionDataset
from model_GRU import LSTMModel
from model_GRU_pred import GRUPred
from utils import normalize_sequence, split_sequence_prediction_test

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    

    opt_dataset = {"n_steps":10, "training_data_folder" : "../data/training/", 
    "dev_data_folder" : "../data/", "data_folder":"../data/training/"}

    # Regression
    huawei_dataset = HuaweiRegressionDataset(opt_dataset)

    dataset_size = len(huawei_dataset)
    train_size = int(0.8*dataset_size)
    dev_size = int(0.5*(dataset_size - train_size))
    test_size = dataset_size - train_size - dev_size


    huawei_train_dataset, huawei_dev_dataset, huawei_test_dataset = torch.utils.data.random_split(huawei_dataset, [train_size, dev_size, test_size])

    opt_model = {"learning_rate": 0.00001, "input_size": 1, "hidden_size": 5, "epochs": 100, "batch_size": 16, "num_layers":1, "save_every":1,
     "model_directory": "../models/", "weight_decay": 0, "model_name": "GRUModelReg", "bidirectional": False, "device": device}

    HuaweiDatasetLoaderTrain = DataLoader(huawei_train_dataset, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0)
    HuaweiDatasetLoaderDev = DataLoader(huawei_dev_dataset, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0)
    HuaweiDatasetLoaderTest = DataLoader(huawei_test_dataset, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0)

    model_reg = LSTMModel(opt_model)
    model_reg.cuda(device)
    model_reg.load_state_dict(torch.load(opt_model["model_directory"] + "GRUModelReg_90"))
    # model_reg.train_model(HuaweiDatasetLoaderTrain, HuaweiDatasetLoaderDev)


    # Prediction
    huawei_dataset = HuaweiPredictionDataset(opt_dataset)

    opt_dataset["data_folder"] = "../data/training/"
    huawei_dataset_test = HuaweiPredictionDataset(opt_dataset)

    dataset_size = len(huawei_dataset)
    train_size = int(0.9*dataset_size)
    dev_size = dataset_size - train_size

    huawei_train_dataset, huawei_dev_dataset = torch.utils.data.random_split(huawei_dataset, [train_size, dev_size])

    opt_model = {"learning_rate": 0.0001, "input_size": 1, "hidden_size": 5, "epochs": 1000, "batch_size": 16, "num_layers":1, "save_every":1,
     "model_directory": "../models/", "weight_decay": 0, "model_name": "GRUModelPred", "bidirectional": False, "device": device}

    HuaweiDatasetLoaderTrain = DataLoader(huawei_train_dataset, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0)
    HuaweiDatasetLoaderDev = DataLoader(huawei_dev_dataset, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0)

    

    model_pred = GRUPred(opt_model)
    model_pred.cuda(device)
    model_pred.load_state_dict(model_reg.state_dict())
    
    for param_tensor in model_pred.gru.parameters():
        param_tensor.requires_grad = False
    for param_tensor in model_pred.hidden_layer.parameters():
        param_tensor.requires_grad = False
    
    # model_pred.train_model(HuaweiDatasetLoaderTrain, HuaweiDatasetLoaderDev)
    model_pred.load_state_dict(torch.load(opt_model["model_directory"] + "GRUModelPred_390"))

    # Final prediction
    # read datafile and predict
    for i in range(1, 8):
        df = pd.read_csv("../data/dataset_"+str(i)+".csv")
        kpi = df.iloc[:, 1].tolist()
        model_features = torch.FloatTensor(split_sequence_prediction_test(kpi, 10)).permute(0, 2, 1).to(device)
        prediction_probabilities = model_pred(model_features).squeeze(1).detach().cpu().numpy()
        prediction = (np.around(prediction_probabilities)).astype(int)
        df["anomaly_label"] = prediction
        df.to_csv("../data/dataset_"+str(i)+".csv", index = False, header = True)

if __name__=="__main__":
    main()
