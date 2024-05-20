import torch.nn as nn
import torch

KERNEL = 3
STRIDE = 2   # For max pooling
CHANNEL = [3,64, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512]
FC = [512*7*7, 4096, 4096 ]
POOL_POS = [2,4,6,9,12]


class my_relu(nn.Module):
    def __init(self):
        super().__init__()
    
    def forward(self, x):
       return torch.max(torch.tensor(0.0), x)
    
    
class custom_VGG16(nn.Module):
    
    def __init__(self, num_of_classes, CHANNEL=CHANNEL, FC=FC, KERNEL=KERNEL, STRIDE=STRIDE, POOL_POS=POOL_POS):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(4096, num_of_classes)
        
        for i in range(1, len(CHANNEL)):
            # conv 2d layers
            self.layers.append(nn.Conv2d(in_channels=CHANNEL[i-1], out_channels=CHANNEL[i], kernel_size=KERNEL, padding='same'))
            
            # activation layer
            self.layers.append(my_relu())
            
            # Max pool
            if i in POOL_POS:
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=STRIDE))
                
        # Fully connected Layers              
        for i in range(len(FC)-1):
            self.fc.append(nn.Linear(FC[i], FC[i+1]))
            self.fc.append(my_relu())
            
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = self.flatten(x)
    
        for layer in self.fc:
            x = layer(x)
        
        # classifier 
        x = self.classifier(x)
        return x
