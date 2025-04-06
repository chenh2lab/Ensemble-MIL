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
from torch.utils.data.dataset import Dataset, Subset
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
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
train_transforms = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                                      ])

test_transforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                                      ])

# Dataset & Dataloder
class ImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(ImageDataset, self).__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

root = '/staging/biology/u1307362/tissue_label_tcga/TCGA-HE-89K'
all_data = ImageDataset(root = root)
labels = [label for _, label in all_data]

# training & validation function definition
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for x, y in iterator:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

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

# loss function
criterion = nn.CrossEntropyLoss().to(device)

# evaluation function & plot function
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

# training & validation
batch_size = 128
n_epochs = 100
k_folds = 5
classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

# Define the K-fold Cross Validator
kfold = StratifiedKFold(n_splits = k_folds, shuffle = True, random_state = seed)

# Get indices for each fold
indices = list(range(len(all_data)))
folds = list(kfold.split(indices, labels))

for fold, (train_index, test_index) in enumerate(folds):
    log(f'FOLD {fold}')
    log('---------------------------------------------')
    best_valid_loss = float('inf')

    # Create train and test subsets
    train_subsampler = Subset(all_data, train_index)
    test_data = Subset(all_data, test_index)

    # Split train set into train and validation sets
    n_valid_examples = int(len(train_subsampler) * 0.125)
    n_train_examples = len(train_subsampler) - n_valid_examples

    train_data, valid_data = data.random_split(train_subsampler, [n_train_examples, n_valid_examples])

    # Apply transforms
    train_data.dataset.transform = train_transforms
    valid_data.dataset.transform = test_transforms
    test_data.dataset.transform = test_transforms

    train_iterator = data.DataLoader(train_data, batch_size = batch_size, num_workers = 4)
    valid_iterator = data.DataLoader(valid_data, batch_size = batch_size, num_workers = 4)
    test_iterator = data.DataLoader(test_data, batch_size = batch_size, num_workers = 4)

    # model
    model = models.vgg19_bn(pretrained = True)
    IN_FEATURES = model.classifier[-1].in_features
    final_fc = nn.Linear(IN_FEATURES, 9)
    model.classifier[-1] = final_fc
    model = model.to(device)

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr = 5e-5, weight_decay = 0.0001)

    # training & validation
    log('start training')
    for epoch in range(n_epochs):
        log(f'Epoch [{epoch + 1}/{n_epochs}]')

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        log(f'Training Loss: {train_loss:.3f}\t Training Acc: {train_acc:.3f}')

        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        log(f'Valid Loss: {valid_loss:.3f}\t Valid Acc: {valid_acc:.3f}')

        if valid_loss < best_valid_loss:
            best_valid_loss_fold = valid_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'/staging/biology/u1307362/tumor_detection_model/vgg19_bn/TCGA-HE-89K/fold_{fold}_VGG19_bn.pt')

    log(f'Best checkpoint at {best_epoch} epoch')
    
    # testing
    log('start testing')
    model_test = models.vgg19_bn(pretrained = True)
    IN_FEATURES = model_test.classifier[-1].in_features
    final_fc = nn.Linear(IN_FEATURES, 9)
    model_test.classifier[-1] = final_fc
    model_test = model_test.to(device)

    model_test.load_state_dict(torch.load(f'/staging/biology/u1307362/tumor_detection_model/vgg19_bn/TCGA-HE-89K/fold_{fold}_VGG19_bn.pt'))
    labels, pred_labels, probs = get_predictions(model_test, test_iterator)
    evaluated_metrics(labels, pred_labels)

    plot_confusion_matrix(labels, pred_labels, classes, title = 'Confusion matrix')
    plt.savefig(f'fold_{fold}_confusion_matrix', bbox_inches = 'tight')

    labels = np.array(labels)
    pred_labels = np.array(pred_labels)
    probs = np.array(probs)

    np.save(f'/staging/biology/u1307362/tumor_detection_model/vgg19_bn/TCGA-HE-89K/labels_fold_{fold}.npy', labels)
    np.save(f'/staging/biology/u1307362/tumor_detection_model/vgg19_bn/TCGA-HE-89K/pred_labels_fold_{fold}.npy', pred_labels)
    np.save(f'/staging/biology/u1307362/tumor_detection_model/vgg19_bn/TCGA-HE-89K/probs_fold_{fold}.npy', probs)