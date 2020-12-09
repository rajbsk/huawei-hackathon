import numpy as np
import pandas as pd
import os
from scipy import stats

def normalize_sequence(sequence):
	sequence_max = max(sequence)
	sequence_min = min(sequence)
	norm = [(float(i)-sequence_min)/(sequence_max-sequence_min) for i in sequence]
	# stan = stats.zscore(sequence)
	# return stan.tolist()
	return norm

# split a univariate sequence into samples
def split_sequence_regression(sequence, n_steps):
	X, y = list(), list()
	pretime_list = [0.0 for i in range(n_steps)]
	data = []
	for i in range(len(sequence)):
		# find the start of this pattern
		start_ix = i - n_steps
		# find the end of this pattern
		end_ix = i 
		
		pre_list = []
		if start_ix<0:
			pre_list = pretime_list[:abs(start_ix)]
		
		# gather input and output parts of the pattern
		seq_x, seq_y = pre_list+sequence[max(0, start_ix):end_ix], sequence[end_ix]
		data.append([seq_x, seq_y])
	return data

def split_sequence_prediction(sequence, labels, n_steps):
	X, y = list(), list()
	pretime_list = [0 for i in range(n_steps)]
	data = []
	for i in range(len(sequence)):
		# find the start of this pattern
		start_ix = i - n_steps + 1
		# find the end of this pattern
		end_ix = i + 1
		
		pre_list = []
		pre_list_labels = []
		if start_ix<0:
			pre_list = pretime_list[:abs(start_ix)]
			pre_list_labels = pretime_list[:abs(start_ix)]
		
		# gather input and output parts of the pattern
		try:
			seq_x = normalize_sequence(pre_list+sequence[max(0, start_ix):end_ix])
			seq_x_inverse = [1-x for x in seq_x]
			seq_y = (pre_list_labels + labels[max(0, start_ix):end_ix])
			data.append([seq_x, seq_y])
			data.append([seq_x_inverse, seq_y])
		except:
			pass

	return data

def split_sequence_prediction_test(sequence, n_steps):
	X, y = list(), list()

	data = []
	for i in range(0, len(sequence), n_steps):
		if len(sequence)-i < n_steps:
			break
		try:
			x_example = normalize_sequence(sequence[i:i+n_steps])
		except:
			x_example = [0 for i in range(n_steps)]
		y_example = sequence[i:i+n_steps]
		data.append([x_example])
	return data

def read_folder_files_regression(folder_name, steps):
	files = os.listdir(folder_name)
	data = []
	for file in files:
		if "csv" in file:
			dataframe = pd.read_csv(folder_name+file)
			kpi_values = dataframe.iloc[:, 1].tolist()
			normalized_kpi_values = normalize_sequence(kpi_values)
			data += split_sequence_regression(normalized_kpi_values, steps)
	return data

def read_folder_files_prediction(folder_name, steps):
	files = os.listdir(folder_name)
	data = []
	for file in files:
		if "training" in file:
			dataframe = pd.read_csv(folder_name+file)
			kpi_values = dataframe["kpi_value"].tolist()
			labels = dataframe["anomaly_label"].tolist()
			
		elif "real" in file or "synthetic" in file:
			dataframe = pd.read_csv(folder_name+file)
			kpi_values = dataframe["value"].tolist()
			labels = dataframe["is_anomaly"].tolist()
		
		data += split_sequence_prediction(kpi_values, labels, steps)

	return data
