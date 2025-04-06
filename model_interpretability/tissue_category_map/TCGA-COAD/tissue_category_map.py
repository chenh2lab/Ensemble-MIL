import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os, slideio

from PIL import Image

# define simple logging functionality
log_fw = open("log.txt", 'w') # open log file to save log outputs
def log(text):     # define a logging function to trace the training process
    print(text)
    log_fw.write(str(text)+'\n')
    log_fw.flush()

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
model = models.vgg19_bn(pretrained = False)
IN_FEATURES = model.classifier[-1].in_features
final_fc = nn.Linear(IN_FEATURES, 9)
model.classifier[-1] = final_fc
model = model.to(device)

# load model parameter
model.load_state_dict(torch.load('/staging/biology/u1307362/tumor_detection_model/vgg19_bn/TCGA-HE-89K/Vgg19_bn.pt'))

# transforms
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

# define sliding window algorithmn
def sliding_window(img, stepsize, windowsize):
    for y in range(0, img.size[1], stepsize):
        for x in range(0, img.size[0], stepsize):
            if (img.size[0] - x) > windowsize and (img.size[1] - y) > windowsize: 
                yield x, y, img.crop((x, y, x + windowsize, y + windowsize))

# root
wsi_root = '/staging/biology/u1307362/TCGA-COAD_WSI/mpp_diagnosis_slides/mpp'
map_root = '/staging/biology/u1307362/TCGA-COAD_sliding_window/vgg19_bn/TCGA-HE-89K'

# category
classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

# model prediction
for svs in os.listdir(wsi_root):
    log(f'Pcocessing : {svs}')
    color = {}

    # read WSI
    slide = slideio.open_slide(os.path.join(wsi_root, svs), 'SVS')
    scene = slide.get_scene(0)
    img = scene.read_block(size = (int(scene.rect[2] * 0.5), int(scene.rect[3] * 0.5)))
    img = Image.fromarray(img)

    # sliding window algorithm
    windows = sliding_window(img, stepsize = 84, windowsize = 224)

    # model
    model.eval()
    with torch.no_grad():
        for x, y, window in windows:
            window_tensor = transform(window)
            window_tensor = torch.unsqueeze(window_tensor, 0)
            window_tensor = window_tensor.to(device)
            outputs = model(window_tensor)
            _, predicted = torch.max(outputs, 1)
            if classes[predicted[0]] == 'ADI':
                color[(x, y)] = 'gray'
            elif classes[predicted[0]] == 'BACK':
                color[(x, y)] = 'white'
            elif classes[predicted[0]] == 'DEB':
                color[(x, y)] = 'purple'
            elif classes[predicted[0]] == 'LYM':
                color[(x, y)] = 'sky blue'
            elif classes[predicted[0]] == 'MUC':
                color[(x, y)] = 'sea blue'
            elif classes[predicted[0]] == 'MUS':
                color[(x, y)] = 'green'
            elif classes[predicted[0]] == 'NORM':
                color[(x, y)] = 'yellow'
            elif classes[predicted[0]] == 'STR':
                color[(x, y)] = 'orange'
            elif classes[predicted[0]] == 'TUM':
                color[(x, y)] = 'red'
    
    # color
    for loc in color:
        if color[loc] == 'gray':
            pad = Image.new('RGB', (224, 224), '#aaaaaa')
            img.paste(pad, (loc[0], loc[1], loc[0] + 224, loc[1] + 224))
        elif color[loc] == 'white':
            pad = Image.new('RGB', (224, 224), '#ffffff')
            img.paste(pad, (loc[0], loc[1], loc[0] + 224, loc[1] + 224))
        elif color[loc] == 'purple':
            pad = Image.new('RGB', (224, 224), '#9966cc')
            img.paste(pad, (loc[0], loc[1], loc[0] + 224, loc[1] + 224))
        elif color[loc] == 'sky blue':
            pad = Image.new('RGB', (224, 224), '#0b9fd5')
            img.paste(pad, (loc[0], loc[1], loc[0] + 224, loc[1] + 224))
        elif color[loc] == 'sea blue':
            pad = Image.new('RGB', (224, 224), '#8ebec4')
            img.paste(pad, (loc[0], loc[1], loc[0] + 224, loc[1] + 224))
        elif color[loc] == 'green':
            pad = Image.new('RGB', (224, 224), '#369b34')
            img.paste(pad, (loc[0], loc[1], loc[0] + 224, loc[1] + 224))
        elif color[loc] == 'yellow':
            pad = Image.new('RGB', (224, 224), '#fff381')
            img.paste(pad, (loc[0], loc[1], loc[0] + 224, loc[1] + 224))
        elif color[loc] == 'orange':
            pad = Image.new('RGB', (224, 224), '#ff9b2b')
            img.paste(pad, (loc[0], loc[1], loc[0] + 224, loc[1] + 224))
        elif color[loc] == 'red':
            pad = Image.new('RGB', (224, 224), '#d50202')
            img.paste(pad, (loc[0], loc[1], loc[0] + 224, loc[1] + 224))

    # save
    img.resize((int(scene.rect[2] * 0.05), int(scene.rect[3] * 0.05)))
    try:
        img.save(os.path.join(map_root, f'{svs[:-4]}.jpg'), format = 'JPEG')
    except:
        log(f'Error : {svs}')