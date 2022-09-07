import torch
import torch.nn as nn
import torchvision.models as torch_models

class PlantClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Load the pretrained model 
        self.model = torch_models.mobilenet_v3_small(True)
        
        # Change the last classification layer
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)

    def forward(self, x):
        return self.model(x)