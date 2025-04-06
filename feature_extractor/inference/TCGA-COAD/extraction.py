import torch 
import torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from torchvision import models, transforms
from PIL import Image
from sklearn.decomposition import PCA

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model (feature extractor)
model = models.resnet50(pretrained = False)
model.fc = nn.Identity()

# load model parameter
backbone = torch.load('/home/u1307362/feature_extractor_training/simclr/SimCLR_resnet50.pt')
model.load_state_dict(backbone['model_state_dict'])
model = model.to(device)

# Configurations
tile_root = '/staging/biology/u1307362/TCGA-COAD_norm_tumor_patch/vgg19_bn/TCGA-HE-89K'
feature_root = '/staging/biology/u1307362/TCGA-COAD_norm_tumor_feature/simclr/vgg19_bn/TCGA-HE-89K'

# transforms
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                        ])

# Feature extraction without pca
model.eval()
for svs in os.listdir(tile_root):
    features, labels = [], []
    n = len(os.listdir(os.path.join(tile_root, svs)))
    if n > 0:
        for tile in os.listdir(os.path.join(tile_root, svs)):
            labels.append(tile[:-4])
            img = Image.open(os.path.join(tile_root, svs, tile))
            img_t = transform(img)
            img_t = torch.unsqueeze(img_t, 0)
            img_t = img_t.to(device)
            with torch.no_grad():
                feature = model(img_t)
                features.append(feature.cpu().detach().numpy().reshape(-1))
        outputs = np.array(features)
        df = pd.DataFrame(outputs, index = labels)
        df.to_csv(os.path.join(feature_root, f'{svs}.csv'))
    else:
        print(f'{svs}: 0 tumor tile')