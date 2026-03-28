import torch
import torch.nn as nn

class SquareActivation(nn.Module):
    """
    Polynomial approximation for non-linear activation.
    f(x) = x^2
    Compatible with Homomorphic Encryption because it relies only on multiplication.
    """
    def forward(self, x):
        return x ** 2

class SecureCNN(nn.Module):
    def __init__(self):
        super(SecureCNN, self).__init__()
        # Minimalist CNN architecture for low HE latency
        # Input: 1 channel, 28x28
        # Output of conv1: 4 channels, 8x8 spatial dimensions (since kernel=7, stride=3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=7, stride=3, padding=0)
        
        # Non-linear activation (x^2)
        self.act1 = SquareActivation()
        
        # Fully connected layer maps flattened 4x8x8=256 features to 10 classes
        self.fc1 = nn.Linear(4 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = x.view(-1, 4 * 8 * 8)
        x = self.fc1(x)
        return x
