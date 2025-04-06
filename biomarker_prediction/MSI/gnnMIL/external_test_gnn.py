import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.models as models

import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DataListLoader

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os, random

from PIL import Image
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV

# define simple logging functionality
log_fw = open("log.txt", 'w') # open log file to save log outputs
def log(text):     # define a logging function to trace the training process
    print(text)
    log_fw.write(str(text)+'\n')
    log_fw.flush()

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ID 'MSS(86), MSI-H(24)'
fea_src = '/staging/biology/u1307362/CPTAC-COAD_norm_tumor_feature/simclr/vgg19_bn/TCGA-HE-89K'
adj_src = '/staging/biology/u1307362/adjacency/CPTAC-COAD'
label_dict = np.load('/home/u1307362/biomarker_label/MSI/CPTAC-COAD.npy', allow_pickle = True).item()

x_ids, y_label = [], []
for csv in os.listdir(fea_src):
    try:
        if label_dict[csv[:7]] == 0:
            x_ids.append(csv[:-4])
            y_label.append(0)
        elif label_dict[csv[:7]] == 1:
            x_ids.append(csv[:-4])
            y_label.append(1)
    except:
        log(f'{csv[:7]} : key error')

prefix_dict = {}
for csv in x_ids:
    prefix = csv[:7]
    if prefix not in prefix_dict:
        prefix_dict[prefix] = csv

x_ids = list(prefix_dict.values())

# create IDs - Data dictionary
ids_data_dict = {}
for i in x_ids:
    # x
    df = pd.read_csv(os.path.join(fea_src, i + '.csv'))
    feature_array = df[[f'{n}' for n in range(2048)]].values
    feature_tensor = torch.from_numpy(feature_array)
    # edge_index
    adjacency_array = np.load(os.path.join(adj_src, i + '.npy'))
    adjacency_tensor = torch.from_numpy(adjacency_array)
    adjacency_index = adjacency_tensor.nonzero().t().contiguous()
    # y (label)
    label = int(label_dict[i[:7]])

    ids_data_dict[i] = Data(x = feature_tensor.float(), edge_index = adjacency_index.long(), y = label)

# Gnn model
class GnnClassifier(nn.Module):
    def __init__(self):
        super(GnnClassifier, self).__init__()

        self.input_dim = 2048
        self.hid_dim1 = 512
        self.hid_dim2 = 128
        self.hid_dim3 = 32
        self.output_dim = 2

        # GCN layer
        self.conv1 = pyg_nn.GCNConv(self.input_dim, self.hid_dim1)
        self.conv2 = pyg_nn.GCNConv(self.hid_dim1, self.hid_dim2)
        self.conv3 = pyg_nn.GCNConv(self.hid_dim2, self.hid_dim3)

        # GAT layer
        self.gat = pyg_nn.GATConv(self.hid_dim3, self.hid_dim3, heads = 1, concat = True)

        # fc layer
        self.fc = nn.Linear(self.hid_dim3, self.output_dim)

    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GCN layer
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # GAT layer
        x = F.relu(self.gat(x, edge_index))

        # global pooling
        x = x.mean(dim = 0, keepdim = True)

        # fc
        out = self.fc(x)

        return out

# prediction function definition
def get_predictions(model, iterator):
    model.eval()
    labels = []
    probs = []
    with torch.no_grad():
        for graph in iterator:
            graph = graph.to(device)
            y_pred = model(graph)
            y_prob = F.softmax(y_pred, dim = 1)
            labels.append(graph.y.cpu())
            probs.append(y_prob.cpu())
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)
    pred_labels = torch.argmax(probs, 1)
    return labels, pred_labels, probs

# plot function definition
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18

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

# evaluation metrics function definition
def evaluated_metrics(labels, pred_labels, probs):
    balanced_acc = metrics.balanced_accuracy_score(labels, pred_labels)
    sen = metrics.recall_score(labels, pred_labels)
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    AUC = auc(fpr, tpr)
    cm = confusion_matrix(labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    spe = tn / (tn + fp)
    f1 = metrics.f1_score(labels, pred_labels)
    log(f'balanced-Acc: {np.round(balanced_acc, 3)}, sensitivity: {np.round(sen, 3)}, specificity: {np.round(spe, 3)}, F1-score: {np.round(f1, 3)}, auc: {np.round(AUC, 3)}')

# configuration
batch_size = 1
k_folds = 5
classes = ['MSS', 'MSI-H']

for fold in range(k_folds):
    log(f'FOLD {fold}')

    # data & dataloader
    test_data = [ids_data_dict[i] for i in x_ids]
    test_iterator = DataLoader(test_data, batch_size = batch_size, num_workers = 4)

    # model
    model = GnnClassifier().to(device)
    model.load_state_dict(torch.load(f'/staging/biology/u1307362/MSI/gnnMIL/simclr/fold_{fold}_gnn.pt'))

    labels, pred_labels, probs = get_predictions(model, test_iterator)
    evaluated_metrics(labels, pred_labels, probs)

    # confusion matrix
    plot_confusion_matrix(labels, pred_labels, classes, title = 'MSI prediction')
    plt.savefig(f'fold_{fold}_confusion_matrix', bbox_inches = 'tight')

    # save prediction
    labels = np.array(labels)
    pred_labels = np.array(pred_labels)
    probs = np.array(probs)
    pred = dict(zip(x_ids, pred_labels))

    np.save(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/MSI/gnnMIL/simclr/labels_fold_{fold}.npy', labels)
    np.save(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/MSI/gnnMIL/simclr/pred_labels_fold_{fold}.npy', pred_labels)
    np.save(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/MSI/gnnMIL/simclr/probs_fold_{fold}.npy', probs)
    np.save(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/MSI/gnnMIL/simclr/pred_fold_{fold}.npy', pred)

# import labels
labels_0 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/MSI/gnnMIL/simclr/labels_fold_0.npy')
labels_1 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/MSI/gnnMIL/simclr/labels_fold_1.npy')
labels_2 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/MSI/gnnMIL/simclr/labels_fold_2.npy')
labels_3 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/MSI/gnnMIL/simclr/labels_fold_3.npy')
labels_4 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/MSI/gnnMIL/simclr/labels_fold_4.npy')

# import probs
probs_0 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/MSI/gnnMIL/simclr/probs_fold_0.npy')
probs_1 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/MSI/gnnMIL/simclr/probs_fold_1.npy')
probs_2 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/MSI/gnnMIL/simclr/probs_fold_2.npy')
probs_3 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/MSI/gnnMIL/simclr/probs_fold_3.npy')
probs_4 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/MSI/gnnMIL/simclr/probs_fold_4.npy')

# roc curve
plt.figure(figsize = (10, 10))
fpr, tpr, _ = roc_curve(labels_0, probs_0[:, 1])
plt.plot(fpr, tpr, label = 'AUC =' + str(round(auc(fpr, tpr), 2)))
fpr, tpr, _ = roc_curve(labels_1, probs_1[:, 1])
plt.plot(fpr, tpr, label = 'AUC =' + str(round(auc(fpr, tpr), 2)))
fpr, tpr, _ = roc_curve(labels_2, probs_2[:, 1])
plt.plot(fpr, tpr, label = 'AUC =' + str(round(auc(fpr, tpr), 2)))
fpr, tpr, _ = roc_curve(labels_3, probs_3[:, 1])
plt.plot(fpr, tpr, label = 'AUC =' + str(round(auc(fpr, tpr), 2)))
fpr, tpr, _ = roc_curve(labels_4, probs_4[:, 1])
plt.plot(fpr, tpr, label = 'AUC =' + str(round(auc(fpr, tpr), 2)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.title('MSI-H prediction')
plt.savefig('ROC_AUC', bbox_inches = 'tight')