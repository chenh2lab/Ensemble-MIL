import torch 
import torch.nn as nn
import torchvision.models as models

# model
model = models.vgg19_bn(pretrained = True)
IN_FEATURES = model.classifier[-1].in_features
final_fc = nn.Linear(IN_FEATURES, 9)
model.classifier[-1] = final_fc
