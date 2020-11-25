import numpy as np
import pandas as pd
import os

def normalize_sequence(sequence):
	sequence_max = max(sequence)
	sequence_min = min(sequence)
	norm = [(float(i)-sequence_min)/(sequence_max-sequence_min) for i in sequence]
	return norm

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
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

def read_folder_files(folder_name, steps):
	files = os.listdir(folder_name)
	data = []
	for file in files:
		if "csv" in file:
			dataframe = pd.read_csv(folder_name+file)
			kpi_values = dataframe.iloc[:, 1].tolist()
			normalized_kpi_values = normalize_sequence(kpi_values)
			data += split_sequence(normalized_kpi_values, steps)
	return data
