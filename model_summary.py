import torch
from torchsummary import summary

from resnet_from_scratch.resnet_instantiate import resnet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# resnet18
summary(resnet18(in_channels=3, num_classes=3).to(device), (3, 224, 224))