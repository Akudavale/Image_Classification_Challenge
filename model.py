import torch
import torch.nn as nn
import torchvision

"""class Network(nn.Module):
    
    def __init__(self, input_shape = int, hidden_units=int, output_shape = int):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels= input_shape, out_channels=hidden_units, kernel_size=3, stride=1),
            nn.ReflectionPad2d(padding=2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels= hidden_units, out_channels=hidden_units, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units, out_channels=hidden_units, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*29*29,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=output_shape),

        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x
    
    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        
        torch.save(self.state_dict(), 'new__model.pkl')"""

class Network(nn.Module):
    def __init__(self):
        super().__init__()        
        self.model = torchvision.models.efficientnet_b4(pretrained = True)
        self.model.fc = nn.Sequential(nn.Linear(1792, 8))

    def forward(self, x):       
        x = self.model(x)
        return x

    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        
        torch.save(self.state_dict(), 'model.pkl')