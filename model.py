import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class DeepRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_layers=[128, 256, 128, 64, 32], dropout_rate=0.2):
        """
        Deep Neural Network for Regression
        
        Args:
            input_dim (int): Number of input features
            hidden_layers (list): Number of neurons in each hidden layer
            dropout_rate (float): Dropout probability for regularization
        """
        super(DeepRegressionModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Add more hidden layers
        for layer_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.BatchNorm1d(layer_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = layer_dim
        
        # Output layer (single neuron for regression)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)

