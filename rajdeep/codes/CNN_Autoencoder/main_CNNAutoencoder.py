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
from sklearn.metrics import precision_recall_fscore_support, classification_report, precision_score, recall_score, f1_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
import glob

# from dataset import HuaweiRegressionDataset, HuaweiPredictionDataset
from daily_dataset import HuaweiRegressionDataset
from CNN_Autoencoder import CNNAutoencoder
from utils import normalize_sequence, split_sequence_prediction_test, normalize_sequence
from kmeans_pytorch import kmeans

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    opt_dataset = {"n_steps":65, "training_data_folder" : "../../data/data_v2/training/", 
    "dev_data_folder" : "../data/data_v2/", "data_folder":"../data/data_v2/training/", "training_file": "../../data/data_v2/training/daily_data.pkl"}

    # Regression
    huawei_dataset = HuaweiRegressionDataset(opt_dataset)

    dataset_size = len(huawei_dataset)
    train_size = int(0.9*dataset_size)
    dev_size = int((dataset_size - train_size))


    huawei_train_dataset, huawei_dev_dataset = torch.utils.data.random_split(huawei_dataset, [train_size, dev_size])

    opt_model = {"device": device, "kernel_size": 3, "num_filters_1": 64, "num_filters_2": 16, "num_filters_3": 4, "length": opt_dataset["n_steps"],
    "output_layer_size": 1, "conv_stride": 1, "pool_size_1": 2, "pool_size_2": 2, "pool_strides_1": 2,
    "pool_strides_2": 2, "model_directory": "../../models/", "model_name": "CNN_Autoencoder_daily", "batch_size": 8, "epochs": 30, "lr": 0.001,
    "save_every": 5}

    opt_model["padding"] = int(opt_model["kernel_size"]/2)

    HuaweiDatasetLoaderTrain = DataLoader(huawei_train_dataset, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0)
    HuaweiDatasetLoaderDev = DataLoader(huawei_dev_dataset, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0)

    model_reg = CNNAutoencoder(opt_model)
    model_reg.cuda(device)
    model_reg.load_state_dict(torch.load(opt_model["model_directory"] + "CNN_Autoencoder_daily_30"))
    # model_reg.evaluate_model(HuaweiDatasetLoaderDev)
    # model_reg.train_model(HuaweiDatasetLoaderTrain, HuaweiDatasetLoaderDev)
    

    # # Final prediction
    # # read datafile and predict
    file_indices = [1,2,3,4,5,6,7,8,9,10,11,12,13,100,101,102,103,105,106,100]
    thresholds = [i for i in range(0, 100, 5)]
    thresholds = [i/100 for i in thresholds]

    actual = []
    predicted = []
    for batch in HuaweiDatasetLoaderTrain:
        for instance in batch[1]:
            actual+=instance
        _, reconstructed, loss = model_reg.process_batch(batch)
        reconstructed = reconstructed.detach().cpu()
        input_data, input_labels = model_reg.get_batch_data(batch)
        reconstruction_error = torch.abs(input_data-reconstructed)
        output = torch.flatten(reconstruction_error).numpy()
        output = np.where(output<=0.03, output, 1)
        output = np.where(output>0.03, output, 0)
        predicted += output.tolist()
    # print(os.listdir("../../data/data_v2/"))
    # for i in file_indices:
    #     df = pd.read_csv("../../data/data_v2/dataset_"+str(i)+".csv")
    #     kpi = df.iloc[:, 1].tolist()
    #     normalized_kpi = normalize_sequence(kpi)
    #     model_features = torch.FloatTensor(split_sequence_prediction_test(normalized_kpi, opt_dataset["n_steps"])).permute(0, 2, 1).to(device)
    #     model_features = model_features.permute(0, 2, 1)
    #     # prediction_probabilities = model_pred(model_features).squeeze(1).detach().cpu().numpy()
    #     prediction_values, reconstruction = model_reg(model_features)
    #     reconstruction = reconstruction
    #     reconstruction_error = reconstruction - model_features
    #     reconstruction_error = reconstruction_error.squeeze(1)
    #     reconstruction_error = np.abs(reconstruction_error.detach().cpu().numpy())
    #     prediction_difference = np.sum(reconstruction_error, axis=1)
    #     prediction_values = np.where(prediction_difference<=0.017, prediction_difference, 1)
    #     prediction_values = np.where(prediction_difference>0.017, prediction_values, 0)

    #     df["anomaly_label"] = prediction_values
    #     actual_values = df["anomaly_label"].tolist()

    #     prediction_values = [int(i) for i in prediction_values]
    #     actual_values = [int(i) for i in actual_values]
    #     df["anomaly_label"] = prediction_values

    #     df.to_csv("../data/data_v2/dataset_"+str(i)+".csv", index = False, header = True)

    #     actual += actual_values
    #     predicted += prediction_values

    # files = (os.listdir("../../data/data_v2/training/"))
    # for file in files:
    #     if "csv" in file:
    #         try:
    #             df = pd.read_csv("../../data/data_v2/training/"+file)
    #             actual_values = df.iloc[:, 2].tolist()
    #             actual_values = [int(i) for i in actual_values]
    #             if len(np.unique(actual_values))==2:
    #                 kpi = df.iloc[:, 1].tolist()
    #                 normalized_kpi = normalize_sequence(kpi)
    #                 model_features = torch.FloatTensor(split_sequence_prediction_test(normalized_kpi, opt_dataset["n_steps"])).permute(0, 2, 1).to(device)
    #                 model_features = model_features.permute(0, 2, 1)
    #                 # prediction_probabilities = model_pred(model_features).squeeze(1).detach().cpu().numpy()
    #                 prediction_values, reconstruction = model_reg(model_features)
    #                 reconstruction = reconstruction
    #                 reconstruction_error = reconstruction - model_features
    #                 reconstruction_error = reconstruction_error.squeeze(1)
    #                 reconstruction_error = np.abs(reconstruction_error.detach().cpu().numpy())
    #                 prediction_difference = np.sum(reconstruction_error, axis=1)
    #                 prediction_values = np.where(prediction_difference<=0.1, prediction_difference, 1)
    #                 prediction_values = np.where(prediction_difference>0.1, prediction_values, 0)

    #                 prediction_values = prediction_values.tolist()
    #                 label_addition = [0 for i in range(len(df)-len(prediction_values))]
    #                 prediction_values += label_addition
                    
                    

    #                 prediction_values = [int(i) for i in prediction_values]
                    

    #                 predicted += prediction_values
    #                 actual += actual_values
    #                 # df.to_csv("../../data/data_v2/dataset_"+str(i)+".csv", index = False, header = True)
    #         except:
    #             pass
    
    print(precision_score(actual,predicted))
    print(recall_score(actual,predicted))
    print(f1_score(actual,predicted))
    print(classification_report(actual,predicted))


if __name__=="__main__":
    main()
