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
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
import glob

from dataset import HuaweiRegressionDataset, HuaweiPredictionDataset
from CNN_Autoencoder import CNNAutoencoder
from utils import normalize_sequence, split_sequence_prediction_test, normalize_sequence

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    opt_dataset = {"n_steps":35, "training_data_folder" : "../data/data_v2/training/", 
    "dev_data_folder" : "../data/data_v2/", "data_folder":"../data/data_v2/training/"}

    # Regression
    huawei_dataset = HuaweiRegressionDataset(opt_dataset)

    dataset_size = len(huawei_dataset)
    train_size = int(0.9*dataset_size)
    dev_size = int((dataset_size - train_size))


    huawei_train_dataset, huawei_dev_dataset = torch.utils.data.random_split(huawei_dataset, [train_size, dev_size])

    opt_model = {"device": device, "kernel_size": 3, "num_filters_1": 64, "num_filters_2": 16, "num_filters_3": 4, "length": opt_dataset["n_steps"],
    "output_layer_size": 1, "conv_stride": 1, "pool_size_1": 2, "pool_size_2": 2, "pool_strides_1": 2,
    "pool_strides_2": 2, "model_directory": "../models/", "model_name": "DeepANT", "batch_size": 64, "epochs": 20, "lr": 0.001,
    "save_every": 5}

    opt_model["padding"] = int(opt_model["kernel_size"]/2)

    HuaweiDatasetLoaderTrain = DataLoader(huawei_train_dataset, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0)
    HuaweiDatasetLoaderDev = DataLoader(huawei_dev_dataset, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0)

    model_reg = CNNAutoencoder(opt_model)
    model_reg.cuda(device)
    # model_reg.evaluate_model(HuaweiDatasetLoaderDev)
    # model_reg.train_model(HuaweiDatasetLoaderTrain, HuaweiDatasetLoaderDev)
    model_reg.load_state_dict(torch.load(opt_model["model_directory"] + "DeepANT_10"))

    # # Final prediction
    # # read datafile and predict
    file_indices = [1,2,3,4,5,6,7,8,9,10,11,12,13,100,101,102,103,105,106,100]
    thresholds = [i for i in range(0, 100, 5)]
    thresholds = [i/100 for i in thresholds]

    actual = []
    predicted = []
    print(os.listdir("../data/data_v2/"))
    for i in file_indices:
        df = pd.read_csv("../data/data_v2/dataset_"+str(i)+".csv")
        kpi = df.iloc[:, 1].tolist()
        normalized_kpi = normalize_sequence(kpi)
        model_features = torch.FloatTensor(split_sequence_prediction_test(normalized_kpi, opt_dataset["n_steps"])).permute(0, 2, 1).to(device)
        model_features = model_features.permute(0, 2, 1)
        # prediction_probabilities = model_pred(model_features).squeeze(1).detach().cpu().numpy()
        prediction_values = model_reg(model_features).squeeze(1).detach().cpu().numpy()
        prediction_difference = np.abs(prediction_values-np.array(normalized_kpi))
        
        prediction_values = np.where(prediction_difference<=0.2, prediction_difference, 1)
        prediction_values = np.where(prediction_values>0.2, prediction_values, 0)

        prediction_values = prediction_values.tolist()
        df["anomaly_label"] = prediction_values
        actual_values = df["anomaly_label"].tolist()

        prediction_values = [int(i) for i in prediction_values]
        actual_values = [int(i) for i in actual_values]
        df["anomaly_label"] = prediction_values

        df.to_csv("../data/data_v2/dataset_"+str(i)+".csv", index = False, header = True)

        actual += actual_values
        predicted += prediction_values

        # df["anomaly_label"] = prediction
        # df.to_csv("../data/dataset_"+str(i)+".csv", index = False, header = True)


if __name__=="__main__":
    main()
