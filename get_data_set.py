#!python3
#   get_data_set
#   Author: Helber Dussan
#   At:     Solving4x
#
import csv
import random
import pandas as pd

def get_dataset(data_set_name, number_cols):
    data_set = []
    numeric_cols = number_cols
    with open(data_set_name) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for ix in range(len(dataset) - 1):
            for iy in range(numeric_cols):
                dataset[ix][iy] = float(dataset[ix][iy])
        data_set = dataset
    data_set = data_set[:-1]
    return data_set

'''
  split: pecentage of data is going to be used for trainning
'''
def split_dataset(split, dataset, trainning = [], testing = []):
    for data in dataset:
        if(random.random() < split): 
            trainning.append(data)
        else:
            testing.append(data)  

def download_dataset(pathname, header_setup = None):
    df = pd.read_csv(pathname, header = header_setup)
    print(df.tail())
    return df

