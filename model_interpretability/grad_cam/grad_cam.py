import torch
import torch.nn as nn
import os
import random
import matplotlib.pyplot as plt
import numpy as np

from torchvision import models
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# random seed
seed = 890904
random.seed(seed)

# load model
model = models.vgg19_bn()
IN_FEATURES = model.classifier[-1].in_features
final_fc = nn.Linear(IN_FEATURES, 9)
model.classifier[-1] = final_fc
model = model.to(device)

model.load_state_dict(torch.load('/staging/biology/u1307362/tumor_detection_model/vgg19_bn/TCGA-HE-89K/Vgg19_bn.pt'))

# cam & target layer
target_layer = [model.features[-1]]
cam = GradCAM(model = model, target_layers = target_layer)

# transform
transforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                                      ])

# matplotlib setting
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18

# grad-cam for internal testing
root = '/staging/biology/u1307362/tissue_label_tcga/TCGA-HE-89K'
classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

fig1 = plt.figure(figsize = (16, 16))
fig2 = plt.figure(figsize = (16, 16))
for index, i in enumerate(classes):
    # selct tile
    imgs = [os.path.join(root, i, tile) for tile in os.listdir(os.path.join(root, i))]
    img = random.sample(imgs, 1)
    img = Image.open(img[0])

    # to float32
    img_float = np.float32(np.array(img)) / 255

    # plot original
    ax = fig1.add_subplot(3, 3, index + 1)
    ax.imshow(img)
    ax.set_title(f'{i}')

    # transfer to tensor
    img_t = transforms(img)
    img_t = torch.unsqueeze(img_t, 0)
    img_t = img_t.to(device)

    # prediction (make sure the prediction is correct)
    model.eval()
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
        print(classes[predicted[0]])

    # grad-cam
    grayscale_cam = cam(input_tensor = img_t, targets = None)
    grayscale_cam = grayscale_cam[0,:]
    cam_img = show_cam_on_image(img_float, grayscale_cam, use_rgb = True)

    # plot grad-cam
    ax = fig2.add_subplot(3, 3, index + 1)
    ax.imshow(cam_img)
    ax.set_title(f'{i}')
fig1.savefig('original_internal', bbox_inches = 'tight')
fig2.savefig('grad-cam_internal', bbox_inches = 'tight')

print('-----------------')

# grad-cam for external testing
root = '/staging/biology/u1307362/tissue_label_nct_non_norm/NCT-CRC-HE-100K-NONORM'

fig3 = plt.figure(figsize = (16, 16))
fig4 = plt.figure(figsize = (16, 16))
for index, i in enumerate(classes):
    # selct tile
    imgs = [os.path.join(root, i, tile) for tile in os.listdir(os.path.join(root, i))]
    # img = random.sample(imgs, 1)
    img = Image.open(imgs[27])

    # to float32
    img_float = np.float32(np.array(img)) / 255

    # plot original
    ax = fig3.add_subplot(3, 3, index + 1)
    ax.imshow(img)
    ax.set_title(f'{i}')

    # transfer to tensor
    img_t = transforms(img)
    img_t = torch.unsqueeze(img_t, 0)
    img_t = img_t.to(device)

    # prediction (make sure the prediction is correct)
    model.eval()
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
        print(classes[predicted[0]])

    # grad-cam
    grayscale_cam = cam(input_tensor = img_t, targets = None)
    grayscale_cam = grayscale_cam[0,:]
    cam_img = show_cam_on_image(img_float, grayscale_cam, use_rgb = True)

    # plot grad-cam
    ax = fig4.add_subplot(3, 3, index + 1)
    ax.imshow(cam_img)
    ax.set_title(f'{i}')
fig3.savefig('original_external', bbox_inches = 'tight')
fig4.savefig('grad-cam_external', bbox_inches = 'tight')
