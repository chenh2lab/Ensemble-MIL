import torch 
import torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import cv2

from torchvision import models, transforms

# device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model (feature extractor)
model = models.resnet50(pretrained = True)
model.fc = nn.Identity()

# Configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = '/staging/biology/u1307362/TCGACOAD_tumor_patch_512_resnet101'
target_root = '/staging/biology/u1307362/TCGACOAD_tumor_pretrained_resnet50_feature_512_resnet101'

# transforms
transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.7731, 0.5610, 0.7280], std = [0.1541, 0.2230, 0.1530])])

# Feature extraction
model = model.to(device)
model.eval()
for files in os.listdir(root):
    features = []
    labels = []
    try:
        for jpg in os.listdir(os.path.join(root, files)):
            labels.append(jpg[:-4])
            img = cv2.imread(os.path.join(root, files, jpg))
            img = transform(img)
            img = img.reshape(1, 3, 224, 224) # img = torch.unsqueeze(img, 0)
            img = img.to(device)
            with torch.no_grad():
                feature = model(img)
                features.append(feature.cpu().detach().numpy().reshape(-1))
        features_df = pd.DataFrame(np.array(features), index = labels)
        features_df.to_csv(os.path.join(target_root, f'{files}.csv'))
    except:
        print(f'{files}: error')
