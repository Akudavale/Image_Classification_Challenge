import torch
import torch.nn as nn
import torchvision

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