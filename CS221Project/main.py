import pandas as pd
import numpy as np
import datetime as dt
import sklearn 
import torch


def read(file):
    dataset = pd.read_csv('startup.data.csv')
    dtColumns = ['founded_at', 'closed_at', 'first_funding_at', 'last_funding_at']
    columns_to_normalize = [col for col in dataset.columns if 'age' in col or 'funding' in col or 'milestone' in col or 'avg_participants' in col] + dtColumns
    dataset[dtColumns] = dataset[dtColumns].apply(lambda _: pd.to_datetime(_, format='%m/%d/%Y'))
    filtered_dataset = filter_unnamed(dataset)
    filtered_dataset[columns_to_normalize] = normalize(filtered_dataset[columns_to_normalize])
    
    def normalize(data_column):
        return data_column - data_column.median()

    def filter_unnamed(dataframe):
        columns_to_keep = [col for col in dataframe.columns if 'Unnamed' not in col]
        return dataframe[columns_to_keep]
    
    return filtered_dataset


def main():
    print(read('startup.data.csv'))

if __name__ == '__main__':
    main()
