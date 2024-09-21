import torch.nn as nn
import torch 

class StormNETC1(nn.Module):
    def __init__(self, use_sst=True):
        super(StormNETC1, self).__init__()
        self.input_channels = 5 if use_sst else 4
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(self.input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.1),
            
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.1),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.1),
        )
        
        # Calculate the size of the flattened features
        self.feature_size = self._get_conv_output((self.input_channels, 241, 321))
        
        self.linear_layers = nn.Sequential(
            nn.Linear(self.feature_size, 64),
            nn.SiLU(),
            nn.Linear(64, 4)  # 4 classes
        )
    
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.cnn_layers(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def initialize_stormnetc1(use_sst=True):
    return StormNETC1(use_sst=use_sst)

print("StormNETC1 model defined and ready for use.")