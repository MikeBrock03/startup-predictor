import os
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import RandomOverSampler
from model import BinaryClassifier

# Fill milestone cols with a billion
# Normalize appropriate cols
# Keep cols that are bools
# Change dates to time since certain date
# Drop cols that are all null

def read(file):
    dataset = pd.read_csv(file)
    dtColumns = ['founded_at', 'closed_at', 'first_funding_at', 'last_funding_at']
    columns_to_normalize = [col for col in dataset.columns if 'age' in col or 'funding' in col or 'milestone' in col or 'avg_participants' in col] + ['founded_at', 'first_funding_gap']
    booleanColumns = dataset.columns[(dataset.dtypes == int) & (dataset.nunique() == 2)].tolist()
    dataset[dtColumns] = dataset[dtColumns].apply(lambda _: pd.to_datetime(_, format='%m/%d/%Y'))

    def normalize(data_column):
        return data_column - data_column.median()

    def filter_unnamed(dataframe):
        columns_to_keep = [col for col in dataframe.columns if 'Unnamed' not in col]
        return dataframe[columns_to_keep]
    
    filtered_dataset = filter_unnamed(dataset)
    filtered_dataset['first_funding_gap'] = filtered_dataset['first_funding_at'] - filtered_dataset['founded_at']
    filtered_dataset[columns_to_normalize] = filtered_dataset[columns_to_normalize].apply(normalize)
    filtered_dataset = filtered_dataset.applymap(lambda x: x.days if isinstance(x, pd.Timedelta) else x)

    columns_to_keep = columns_to_normalize + booleanColumns
    preprocessed_df = filtered_dataset[columns_to_keep]

    milestoneColumns = ['age_first_milestone_year','age_last_milestone_year']
    preprocessed_df[milestoneColumns] = preprocessed_df[milestoneColumns].fillna(1000000000000)

    return preprocessed_df

# splits dataset maintaining general distributions
# train into train, other and another time to split other into validation, test

def splitDataset(df):
    X = df.drop('labels', axis=1)
    y = df['labels']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train_resampled, y_train_resampled, X_val, y_val, X_test, y_test
    # return X_train, y_train, X_val, y_val, X_test, y_test

# this is all scikit learn
# train model using .fit function
# predict using .predict function
# extract the weights from the fully trained model for printing and interpreting to see what the model is doing
# evaluate using accuracy_score, precision_score, recall_score, classification_report
# we did not do hyperparam tuning for the baseline; it does not have any hyperparams
# we tested it twice (once on validation and once on test) to see how it would perform on unseen data

def logReg(X_train, y_train, X_val, y_val, X_test, y_test):
    model = LogisticRegression(random_state=42)
    
    model.fit(X_train, y_train)

    columns = X_train.columns

    weights = model.coef_[0]

    for column, weight in zip(columns, weights):
        print(f"{column}: {weight}")

    print(f"Intercept: {model.intercept_[0]}")

    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')
    print(classification_report(y_val, y_val_pred))

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# split data into tesnsors so pytorch can use it
# create data loaders for each dataset type (train, validation, test)

def prepare_data_NN(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    X_train_tensor = torch.tensor(X_train.values).float()
    y_train_tensor = torch.tensor(y_train.values).float()

    X_val_tensor = torch.tensor(X_val.values).float()
    y_val_tensor = torch.tensor(y_val.values).float()

    X_test_tensor = torch.tensor(X_test.values).float()
    y_test_tensor = torch.tensor(y_test.values).float()

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

# accepts training and validation data, list of learning rates, list of batch sizes, list of lists, with each list being a hidden layer configuration, and input size
# for the hidden layer configuration lists, each element is the number of nodes in that layer (first then second then third later etc)
# loop thorugh all possible combinaitons of hyperparams and train a model for each, evaluating on validation set
# compare validation accuracy and return the hyperparams that gave the best validation accuracy

def hyperparameter_tuning(train_loader, val_loader, learning_rates, batch_sizes, hidden_layer_configs, input_size):
    best_val_accuracy = 0
    best_hyperparams = {}

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for hidden_layers in hidden_layer_configs:
                # Initialize model
                model = BinaryClassifier(input_size, hidden_layers)
                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                # Train model
                trainNN(model, train_loader, val_loader, criterion, optimizer)

                # Evaluate on validation set
                val_loss, val_accuracy, _, _ = evalNN(model, val_loader, criterion)

                print('curr lr', lr, 'curr batch size', batch_size, 'curr hidden', hidden_layers)

                # Update best hyperparameters
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_hyperparams = (lr, batch_size, hidden_layers)

    return best_hyperparams

# use pytorch to set up the model
# appending the right number of nodes for each layer

# train the model using trainig data
# evaluate on validation set
# return the average test loss and the accuracy, precision, and recall on the test set
# when we call this in the triple nested loop in the hyperparameter tuning function, we get some metrics on how the model is doing on unseen data with that configuration
# once we pick the best hyperparams, we train the model again on the training and validation data, and print the validation loss
# the point of this is to evaluate the model on unseen data as we do gradient decent

def trainNN(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        # Training
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, labels.unsqueeze(1))
                total_val_loss += val_loss.item()

        print(f'Epoch {epoch+1}, Validation Loss: {total_val_loss / len(val_loader)}')

# evalNN evaluates on the test set
# this gives us the final metrics for the model
# in the hyperparameter tuning function, we call it on the validation data set to test to see how well it does on unseen data for each hyperparam configuration
# once we pick the best hyperparameters, we use the eval function to evaluate the model on the test set

def evalNN(model, test_loader, criterion):
    model.eval()
    total_test_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            test_loss = criterion(outputs, labels.unsqueeze(1))
            total_test_loss += test_loss.item()

        predictions = (outputs > 0.5).float()
        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.reshape(-1).tolist())

    average_test_loss = total_test_loss / len(test_loader)
    print(f'Test Loss: {average_test_loss}')

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions)

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print('Classification Report:\n', report)

    return average_test_loss, accuracy, precision, recall

# we load the data, preprocess it, and split it into train, validation, and test
# we call logReg on the data to get a baseline
# we convert the data into tensors and create data loaders for each dataset type for pytorch to read
# we then do hyperparameter tuning on the data to find the best hyperparams
# we train the model on the training and validation data using the best hyperparams

# we chose the numbers for configs from playing around with different numbers and seeing what worked best
# the conclusion was that with more layers the model did not preform much better if at all better
# for the less complex hidden layer configs, the higher learning rates worked better, and for the more complex ones, the lower learning rates worked better

def main():
    p = os.path.join(os.path.dirname(__file__), "startup_data.csv")
    preprocessed_df = read(p)

    X_train, y_train, X_val, y_val, X_test, y_test = splitDataset(preprocessed_df)

    logReg(X_train, y_train, X_val, y_val, X_test, y_test)
    
    train_loader, val_loader, test_loader = prepare_data_NN(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32)

    input_size = X_train.shape[1]
    print('input size is', input_size)
    print('x train columns are', X_train.columns)
    # customize num hidden layers and nodes in them here!!
    learning_rates = [0.1, 0.01, 0.001, 0.0001, .00001]
    batch_sizes = [16, 32]
    hidden_layer_configs = [[5], [64], [5, 4], [64, 32], [5, 4, 3], [64, 32, 16]]
    # best_lr, best_batch_size, best_hidden_layers = hyperparameter_tuning(train_loader, val_loader, learning_rates, batch_sizes, hidden_layer_configs, input_size)
    best_lr, best_batch_size, best_hidden_layers = .1, 32, [5,4]
    print('best lr,', best_lr, 'best batch', best_batch_size, 'best hidden', best_hidden_layers)

    train_loader, val_loader, test_loader = prepare_data_NN(X_train, y_train, X_val, y_val, X_test, y_test, best_batch_size)
    customNN = BinaryClassifier(input_size, best_hidden_layers)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(customNN.parameters(), best_lr)

    trainNN(customNN, train_loader, val_loader, criterion, optimizer, num_epochs=100)
    eval_loss = evalNN(customNN, test_loader, criterion)

    model_path = 'startup_predictor.pt'
    torch.save(customNN.state_dict(), model_path)

if __name__ == '__main__':
    main()
