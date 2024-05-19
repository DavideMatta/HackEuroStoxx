# %%
import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import optuna
import pandas as pd
from processing.pipeline import financial_data_pipeline
import pickle

# %%
data = yf.download("^STOXX50E", period="max")

# %%
pr_data = financial_data_pipeline.fit_transform(data)

# %%
pr_data.head()

# %%
X = pr_data.drop('Close_t1', axis=1)
y = pr_data['Close_t1']
y = y[4:].reset_index(drop=True).values.reshape(-1, 1, 1)
X, y

# %%
window_size = 5

# Initialize an empty list to store 3D arrays
X_3d_list = []

# Iterate through each window
for i in range(len(X) - window_size + 1):
    # Extract the data for the current window
    window_data = X.iloc[i:i+window_size]
    # Convert the window data to a 2D numpy array
    window_array = window_data.drop(columns=['Date']).to_numpy()
    # Append the 2D array with the date column included as the first feature
    window_array_with_time = np.insert(window_array, 0, window_data['Date'], axis=1)
    # Append the 2D array to the list
    X_3d_list.append(window_array_with_time)

# Stack the list of 3D arrays into a single 3D numpy array
X_3d = np.stack(X_3d_list)


X_3d[1], X_3d[-1]

# %%
X_train, X_remainder, y_train, y_remainder = train_test_split(X_3d, y, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_remainder, y_remainder, test_size=0.333, shuffle=False)
X_test, X_forecast, y_test, y_forecast = train_test_split(X_test, y_test, test_size=0.2, shuffle=False)

# %%
len(X_train), len(X_val), len(X_test), len(X_forecast), len(y_val)

# %%
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# %%
batch_size = 32

# %%
train_dataset = CustomDataset(X_train, y_train)
validation_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)
forecast_dataset = CustomDataset(X_forecast, y_forecast)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
forecast_dataloader = DataLoader(forecast_dataset, batch_size=batch_size, shuffle=False)

# %%
X_train[0]

# %%
for batch in train_dataloader:
    pass

# %%
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        self.layer_dim = num_layers
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out

# %%
input_dim = X_train.shape[2]
output_dim = 1

# %%


# %%
num_epochs = 10
num_trials = 5

# %%
# Define the objective function for hyperparameter optimization
def objective(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 512, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)

    # Define and initialize the model
    model = GRUModel(input_dim, hidden_dim, num_layers, output_dim, dropout)

    # Define optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            inputs = inputs.float()
            targets = targets.float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for inputs, targets in validation_dataloader:
                inputs = inputs.float()
                targets = targets.float()
                outputs = model(inputs)
                validation_loss += criterion(outputs, targets).item()

        validation_loss /= len(validation_dataloader)

        # Report intermediate results to Optuna
        trial.report(validation_loss, epoch)

        # Handle pruning based on the intermediate results
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return validation_loss

# Create a study object with TPE sampler
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=num_trials)

# Get the best hyperparameters
best_params = study.best_params




# %%
best_params

# %%
best_lr = best_params['learning_rate']
best_hd = best_params['hidden_dim']
best_nl = best_params['num_layers']
best_do = best_params['dropout']

# %%
X_tv = np.concatenate((X_train, X_val), axis=0)
y_tv = np.concatenate((y_train, y_val), axis=0)

tv_dataset = CustomDataset(X_tv, y_tv)
tv_dataloader = DataLoader(tv_dataset, batch_size=batch_size, shuffle=False)

# %%
cv_model = GRUModel(input_dim, best_hd, best_nl, output_dim, best_do)

# %%
cv_num_epochs = 10
cv_criterion = nn.MSELoss()
cv_optimizer = torch.optim.Adam(cv_model.parameters(), lr=best_lr)

# %%
# Training loop

best_loss = float('inf')  # Initialize with a very high loss
best_epoch = -1

# Assuming cv_model is your trained PyTorch model
for epoch in range(cv_num_epochs):
    cv_model.train()
    epoch_loss = 0.0
    for inputs, targets in tv_dataloader:
        cv_optimizer.zero_grad()
        inputs = inputs.float()
        targets = targets.float()
        outputs = cv_model(inputs)
        loss = cv_criterion(outputs, targets)
        loss.backward()
        cv_optimizer.step()
        epoch_loss += loss.item()

    # Calculate average epoch loss
    epoch_loss /= len(tv_dataloader)

    # Check if current epoch's loss is better than the best loss
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_epoch = epoch
        # Save the model state dictionary when the loss improves
        best_model_state_path = "best_cv_model.pkl"
        with open(best_model_state_path, 'wb') as f:
            pickle.dump(cv_model.state_dict(), f)

print(f"Best epoch: {best_epoch}, Best loss: {best_loss}")


# %%
cv_model.eval()
with torch.no_grad():
    test_loss = 0
    for inputs, targets in test_dataloader:
        inputs = inputs.float()
        targets = targets.float()
        outputs = cv_model(inputs)
        test_loss += cv_criterion(outputs, targets).item()

    test_loss /= len(test_dataloader)

print(f'Test Loss: {test_loss}')

# %%
cv_model.eval()
with torch.no_grad():
    forecast_loss = 0
    for inputs, targets in forecast_dataloader:
        inputs = inputs.float()
        targets = targets.float()
        outputs = cv_model(inputs)
        forecast_loss += cv_criterion(outputs, targets).item()

    forecast_loss /= len(forecast_dataloader)

print(f'Forecast Loss: {forecast_loss}')

# %%


# %%


# %%


# %%




