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

    #NEED TO RESOLVE ISSUE W NANS (age first/last milestone year, closed at)
    #preprocessed_df = preprocessed_df.dropna(axis=1)

    milestoneColumns = ['age_first_milestone_year','age_last_milestone_year']
    preprocessed_df[milestoneColumns] = preprocessed_df[milestoneColumns].fillna(1000000000000)

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

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size=1):
        super(BinaryClassifier, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def trainNN(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
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

def main():
    p = os.path.join(os.path.dirname(__file__), "startup_data.csv")
    preprocessed_df = read(p)

    X_train, y_train, X_val, y_val, X_test, y_test = splitDataset(preprocessed_df)

    logReg(X_train, y_train, X_val, y_val, X_test, y_test)
    
    train_loader, val_loader, test_loader = prepare_data_NN(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32)

    input_size = X_train.shape[1]
    # customize num hidden layers and nodes in them here!!
    hidden_layers = [5, 4, 4]
    learning_rates = [0.1, 0.01, 0.001, 0.0001, .00001]
    batch_sizes = [16, 32, 64, 128]
    hidden_layer_configs = [[5], [5, 4], [64, 32], [5, 4, 4]]
    best_lr, best_batch_size, best_hidden_layers = hyperparameter_tuning(train_loader, val_loader, learning_rates, batch_sizes, hidden_layer_configs, input_size)
    print('best lr,', best_lr, 'best batch', best_batch_size, 'best hidden', best_hidden_layers)

    train_loader, val_loader, test_loader = prepare_data_NN(X_train, y_train, X_val, y_val, X_test, y_test, best_batch_size)
    customNN = BinaryClassifier(input_size, best_hidden_layers)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(customNN.parameters(), best_lr)

    trainNN(customNN, train_loader, val_loader, criterion, optimizer, num_epochs=10)
    eval_loss = evalNN(customNN, test_loader, criterion)

if __name__ == '__main__':
    main()
