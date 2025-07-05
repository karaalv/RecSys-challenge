"""
This module contains the definition of our 
Feedforward Neural Network (FNN) model designed 
for binary classification tasks.
- The model is built using PyTorch implementing
    a simple architecture with one hidden layer.
- Early stopping is implemented to prevent overfitting.
- The model is trained using the Adam optimizer.
- The size of the hidden layer is adjustable and should 
    be set based using cross-validation.
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class FNNModel(nn.Module):
    """
    Feedforward Neural Network (FNN) model for binary 
    classification. This model consists of one hidden 
    layer and uses ReLU activation for the hidden layer
    and sigmoid activation for the output layer.
    """
    def __init__(
        self, 
        input_size, 
        hidden_size,
        learning_rate=0.001,
        batch_size=1024
    ):
        """
        Initializes the FNN model with one hidden layer.
        
        Parameters:
        - input_size: Number of input features.
        - hidden_size: Number of neurons in the hidden layer.
        - learning_rate: Learning rate for the optimizer.
        - batch_size: Size of the mini-batches for training.
        """
        super(FNNModel, self).__init__()
        # Model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # Layers of the model
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        # Model functions and optimizer
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate
        )
    
    def _to_tensor(self, data) -> torch.Tensor:
        """
        Converts input data to a PyTorch tensor.
        
        Parameters:
        - data: Input data (numpy array or list).
        
        Returns:
        - PyTorch tensor of the input data.
        """
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        return torch.tensor(data, dtype=torch.float32)
    
    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = self._to_tensor(x)  # Ensure input is a tensor
        x = F.relu(self.fc1(x)) # ReLU activation for hidden layer
        x = torch.sigmoid(self.fc2(x)) # Sigmoid activation for output layer
        return x
    
    def train_model(self, x_train, y_train, num_epochs=100, patience=10):
        """
        Trains the model using the provided training data.
        If the number of samples in the training data is more than
        the batch size, it uses DataLoader for batching.
        
        Parameters:
        - train_x: Training input features.
        - train_y: Training labels.
        - num_epochs: Number of epochs to train the model.
        - patience: Number of epochs with no improvement after 
            which training will be stopped.
        """
        print(
            f"Starting training...\n"
            f"Hidden layer size: {self.hidden_size}\n"
            f"Batch size: {self.batch_size}\n"
            f"Number of epochs: {num_epochs}\n"
        )
        
        best_loss = float('inf')
        epochs_without_improvement = 0
        x_train = self._to_tensor(x_train)
        y_train = self._to_tensor(y_train)
        
        for epoch in range(num_epochs):
            self.train()
            avg_epoch_loss = 0.0

            # Define mini-batch size and DataLoader
            dataset = TensorDataset(x_train, y_train)
            batch_size = min(self.batch_size, len(dataset))
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True
            )

            for x_batch, y_batch in dataloader:
                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass and compute loss
                outputs = self(x_batch)
                loss = self.loss_fn(outputs.squeeze(), y_batch.float())

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                # Accumulate loss for the epoch
                avg_epoch_loss += loss.item() * len(x_batch)

            # Average loss for the epoch
            avg_epoch_loss /= len(dataset)
            
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

    def predict_prob(self, x):
        """
        Makes predictions on the input data.
        
        Parameters:
        - x: Input features for prediction.

        Returns:
        - Predicted probabilities of the positive class.
        """
        self.eval()
        x = self._to_tensor(x)

        with torch.no_grad():
            outputs = self(x)
        return outputs.squeeze()
    
    def predict(self, x, threshold=0.5):
        """
        Makes predictions on the input data.
        
        Parameters:
        - x: Input features for prediction.
        - threshold: Threshold for binary classification.
        
        Returns:
        - Predicted labels (0 or 1).
        """
        self.eval()
        x = self._to_tensor(x)

        with torch.no_grad():
            outputs = self(x)
            predicted = (outputs.squeeze() >= threshold).float()
        return predicted
    
    def save_model(self, path):
        """
        Saves the model to the specified path.
        
        Parameters:
        - path: Path to save the model.
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Loads the model from the specified path.
        
        Parameters:
        - path: Path to load the model from.
        """
        self.load_state_dict(torch.load(path))
        self.eval()
        print(f"Model loaded from {path}")