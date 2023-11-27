import pandas as pd
import numpy as np
import datetime as dt
import sklearn 
import torch

def read(file):
    dataset = pd.read_csv('startup.data.csv')
    filtered_dataset = filter_unnamed(dataset)
    columns_to_normalize = [col for col in filtered_dataset.columns if 'age' in col or 'funding' in col or 'milestone' in col or 'avg_participants' in col]
    return filtered_dataset[columns_to_normalize]
    
def median_centered(data_column):
    return data_column - data_column.mean()

def filter_unnamed(dataframe):
    columns_to_keep = [col for col in dataframe.columns if 'Unnamed' not in col]
    return dataframe[columns_to_keep]

 


def main():
    print("Hello World!")

if __name__ == '__main__':
    main()
