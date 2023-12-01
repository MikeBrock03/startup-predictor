import os
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import torch


def read(file):
    dataset = pd.read_csv(file)
    dtColumns = ['founded_at', 'closed_at', 'first_funding_at', 'last_funding_at']
    columns_to_normalize = [col for col in dataset.columns if 'age' in col or 'funding' in col or 'milestone' in col or 'avg_participants' in col] + ['founded_at']
    booleanColumns = dataset.columns[(dataset.dtypes == int) & (dataset.nunique() == 2)].tolist()
    dataset[dtColumns] = dataset[dtColumns].apply(lambda _: pd.to_datetime(_, format='%m/%d/%Y'))

    def normalize(data_column):
        return data_column - data_column.median()

    def filter_unnamed(dataframe):
        columns_to_keep = [col for col in dataframe.columns if 'Unnamed' not in col]
        return dataframe[columns_to_keep]
    
    filtered_dataset = filter_unnamed(dataset)
    filtered_dataset[columns_to_normalize] = filtered_dataset[columns_to_normalize].apply(normalize)
    filtered_dataset = filtered_dataset.applymap(lambda x: x.days if isinstance(x, pd.Timedelta) else x)

    columns_to_keep = columns_to_normalize + booleanColumns
    preprocessed_df = filtered_dataset[columns_to_keep]

    #NEED TO RESOLVE ISSUE W NANS (age first/last milestone year, closed at)
    preprocessed_df = preprocessed_df.dropna(axis=1)

    return preprocessed_df

def splitDataset(df):
    X = df.drop('labels', axis=1)
    y = df['labels']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test

def logReg(X_train, y_train, X_val, y_val, X_test, y_test):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

def main():
    p = os.path.join(os.path.dirname(__file__), "startup_data.csv")
    preprocessed_df = read(p)

    X_train, y_train, X_val, y_val, X_test, y_test = splitDataset(preprocessed_df)

    logReg(X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == '__main__':
    main()
