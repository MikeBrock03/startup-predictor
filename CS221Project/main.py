import pandas as pd
import numpy as np
import datetime as dt
import sklearn 
import torch

def read(file):
    dataset = pd.read_csv('startup.data.csv')
    dtColumns = ['founded_at', 'closed_at', 'first_funding_at', 'last_funding_at']
    dataset[dtColumns] = dataset[dtColumns].apply(lambda _: pd.to_datetime(_, format='%m/%d/%Y'))
    filtered_dataset = filter_unnamed(dataset)
    columns_to_normalize = [col for col in filtered_dataset.columns if 'age' in col or 'funding' in col or 'milestone' in col or 'avg_participants' in col]
    return filtered_dataset[columns_to_normalize]
    
def median_centered(data_column):
    return data_column - data_column.median()

def filter_unnamed(dataframe):
    columns_to_keep = [col for col in dataframe.columns if 'Unnamed' not in col] + dtColumns
    return dataframe[columns_to_keep]
 


def main():
    print("Hello World!")

if __name__ == '__main__':
    main()
