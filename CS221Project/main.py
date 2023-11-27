import pandas as pd
import numpy as np
import datetime as dt
import sklearn 
import torch

def read(file):
    dataset = pd.read_csv('startup.data.csv')

    # median centering columns with date values
    dtColumns = ['founded_at', 'closed_at', 'first_funding_at', 'last_funding_at']
    dataset[dtColumns] = dataset[dtColumns].apply(lambda _: pd.to_datetime(_, format='%m/%d/%Y'))
    def medianCentering(column):
        median = column.median
        column = column - median
    
    columns_to_normalize +=  dtColumns
    dataset[columns_to_normalize] = dataset[columns_to_normalize].apply(medianCentering(column))
    

def main():
    print("Hello World!")

if __name__ == '__main__':
    main()
