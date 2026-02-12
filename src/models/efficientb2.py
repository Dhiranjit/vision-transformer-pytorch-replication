from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import torch
from torch import nn


device = "cuda" if torch.cuda.is_available() else "cpu"

def create_effnet_b2():
    weights = EfficientNet_B2_Weights.DEFAULT
    model = efficientnet_b2(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False
    
    model.classifier[1] = nn.Linear(in_features=1408, out_features=10)
    model.to(device)

    print(f"EfficientNetB2 transfer learning model initialised")
    
    return model