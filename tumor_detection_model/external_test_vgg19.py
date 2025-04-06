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
import os, copy, random

from PIL import Image
from torch.utils.data.dataset import Dataset
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import KFold
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

# define simple logging functionality
log_fw = open("log.txt", 'w') # open log file to save log outputs
def log(text):     # define a logging function to trace the training process
    print(text)
    log_fw.write(str(text)+'\n')
    log_fw.flush()

# Set Seed for reproducibility
seed = 890904
random.seed(seed) # Python built-in random module
np.random.seed(seed) # Numpy
torch.manual_seed(seed) # Torch
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
test_transforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                                      ])

# Dataset
root = '/staging/biology/u1307362/tissue_label_nct_non_norm/NCT-CRC-HE-100K-NONORM'
test_data = datasets.ImageFolder(root = root, transform = test_transforms)
test_iterator = data.DataLoader(test_data, batch_size = 128, shuffle = False, num_workers = 4)

# prediction
classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
def get_predictions(model, iterator):
    model.eval()
    labels = []
    probs = []
    with torch.no_grad():
        for x, y in iterator:
            x = x.to(device)
            y_pred = model(x)
            y_prob = F.softmax(y_pred, dim = 1)
            labels.append(y.cpu())
            probs.append(y_prob.cpu())
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)
    pred_labels = torch.argmax(probs, 1)
    return labels, pred_labels, probs

# plot function & plot function definition
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16

def evaluated_metrics(labels, pred_labels):
    acc = metrics.accuracy_score(labels, pred_labels)
    cohen_kappa= metrics.cohen_kappa_score(labels, pred_labels)
    mcc = metrics.matthews_corrcoef(labels, pred_labels)
    log(f'acc: {acc}, matthews_corrcoef: {mcc}, cohen_kappa: {cohen_kappa}')

def plot_confusion_matrix(labels, pred_labels, classes, title):
    fig, ax = plt.subplots(figsize = (10, 10))
    cm = confusion_matrix(labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm_normalized, display_labels = classes)
    cm_display.plot(values_format = '.2f', cmap = 'Blues', ax = ax, colorbar = False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size = "5%", pad = 0.5)
    norm = plt.Normalize(vmin = 0, vmax = 1)
    sm = plt.cm.ScalarMappable(cmap = 'Blues', norm = norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax = cax)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer = True))
    plt.suptitle(title, x = 0.5, y = 0.9)
    plt.subplots_adjust(top = 0.85)
    ax.set_xlabel('Predicted label', labelpad = 30)
    ax.set_ylabel('True label', labelpad = 20)

# model
model = models.vgg19_bn(pretrained = False)
IN_FEATURES = model.classifier[-1].in_features
final_fc = nn.Linear(IN_FEATURES, 9)
model.classifier[-1] = final_fc
model = model.to(device)

# testing
log('start testing')
for fold in range(5):
    log(f'FOLD {fold}')

    model.load_state_dict(torch.load(f'/staging/biology/u1307362/tumor_detection_model/vgg19_bn/TCGA-HE-89K/fold_{fold}_VGG19_bn.pt'))
    labels, pred_labels, probs = get_predictions(model, test_iterator)

    evaluated_metrics(labels, pred_labels)

    plot_confusion_matrix(labels, pred_labels, classes, title = 'Confusion matrix')
    plt.savefig(f'fold_{fold}_confusion_matrix', bbox_inches = 'tight')

    labels = np.array(labels)
    pred_labels = np.array(pred_labels)
    probs = np.array(probs)

    np.save(f'/staging/biology/u1307362/tumor_detection_model/vgg19_bn/NCT-CRC-HE-100K-NONORM/labels_fold_{fold}.npy', labels)
    np.save(f'/staging/biology/u1307362/tumor_detection_model/vgg19_bn/NCT-CRC-HE-100K-NONORM/pred_labels_fold_{fold}.npy', pred_labels)
    np.save(f'/staging/biology/u1307362/tumor_detection_model/vgg19_bn/NCT-CRC-HE-100K-NONORM/probs_fold_{fold}.npy', probs)