import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

# define simple logging functionality
log_fw = open("log.txt", 'w') # open log file to save log outputs
def log(text):     # define a logging function to trace the training process
    print(text)
    log_fw.write(str(text)+'\n')
    log_fw.flush()

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# attention model
class AttnClassifier(nn.Module):
    def __init__(self, input_dim, attn_dim, output_dim, lambda_l1, dropout = True, p_dropout = 0.5):
        super(AttnClassifier, self).__init__()

        att_v = [nn.Linear(input_dim, attn_dim), nn.Tanh()]
        att_u = [nn.Linear(input_dim, attn_dim), nn.Sigmoid()]

        if dropout:
            att_v.append(nn.Dropout(p_dropout))
            att_u.append(nn.Dropout(p_dropout))

        self.attention_V = nn.Sequential(*att_v)
        self.attention_U = nn.Sequential(*att_u)
        self.attention_weights = nn.Linear(attn_dim, 1)

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # batch size must be 1 !!!
        x = x.squeeze(0)   # (N, input_dim)

        alpha_V = self.attention_V(x)   # (N, 1042)
        alpha_U = self.attention_U(x)   # (N, 1024)
        alpha = self.attention_weights(alpha_V * alpha_U)   # element wise muliplication (N, 1)
        alpha = torch.transpose(alpha, 1, 0) # (1, N)
        alpha = F.softmax(alpha, dim = 1)

        M = torch.mm(alpha, x)   # (1, input_dim)
        out = self.fc(M) # (1, 2)
        return out, alpha

    def l1_regularization(self, lambda_l1):
        l1_reg = torch.tensor(0., requires_grad = True).to(next(self.parameters()).device)
        for param in self.parameters():
            l1_reg = l1_reg + torch.norm(param, p = 1)
        return lambda_l1 * l1_reg

# ID 'BRAF wildtype(97), mutation(13)'
root = '/staging/biology/u1307362/CPTAC-COAD_norm_tumor_feature/simclr/vgg19_bn/TCGA-HE-89K'
label_dict = np.load('/home/u1307362/biomarker_label/BRAF/CPTAC-COAD.npy', allow_pickle = True).item()

x_ids, y_label = [], []
for csv in os.listdir(root):
    try:
        if label_dict[csv[:7]] == 0:
            x_ids.append(csv)
            y_label.append(0)
        elif label_dict[csv[:7]] == 1:
            x_ids.append(csv)
            y_label.append(1)
    except:
        log(f'{csv[:7]} : key error')

prefix_dict = {}
for csv in x_ids:
    prefix = csv[:7]
    if prefix not in prefix_dict:
        prefix_dict[prefix] = csv

x_ids = list(prefix_dict.values())

# dataset & dataloader
class TumorFeatureDataset(Dataset):
    def __init__(self, feature_folder, ids):
        self.feature_folder = feature_folder
        self.ids = ids
        self.slide_list = [os.path.join(feature_folder, x) for x in ids]

        log(f'{len(self.slide_list)} slides in external dataset')

    def __getitem__(self, idx):
        fname = self.slide_list[idx]
        df = pd.read_csv(fname)
        feature_array = df[[f'{n}' for n in range(2048)]].values
        feature_tensor = torch.from_numpy(feature_array)

        label = int(label_dict[fname.split("/")[-1][:7]])
        return feature_tensor.float(), label

    def __len__(self):
        return len(self.slide_list)

# prediction
def get_predictions(model, iterator):
    model.eval()
    labels = []
    probs = []
    with torch.no_grad():
        for x, y in iterator:
            x = x.to(device)
            y_pred, attn = model(x)
            y_prob = F.softmax(y_pred, dim = 1)
            labels.append(y.cpu())
            probs.append(y_prob.cpu())
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)
    pred_labels = torch.argmax(probs, 1)
    return labels, pred_labels, probs

# plot function definition
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18

# function definition
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
classes = ['wildtype', 'mutation']

for fold in range(k_folds):
    log(f'FOLD {fold}')

    # dataset & dataloader
    test_data = TumorFeatureDataset(feature_folder = root, ids = x_ids)
    test_iterator = data.DataLoader(test_data, batch_size = batch_size, num_workers = 4)

    # model
    model = AttnClassifier(input_dim = 2048, attn_dim = 1024, output_dim = 2, lambda_l1 = 1e-4).to(device)
    model.load_state_dict(torch.load(f'/staging/biology/u1307362/BRAF/attMIL/simclr/fold_{fold}_attention.pt'))

    labels, pred_labels, probs = get_predictions(model, test_iterator)
    evaluated_metrics(labels, pred_labels, probs)

    # confusion matrix
    plot_confusion_matrix(labels, pred_labels, classes, title = 'BRAF prediction')
    plt.savefig(f'fold_{fold}_confusion_matrix', bbox_inches = 'tight')

    # save prediction
    labels = np.array(labels)
    pred_labels = np.array(pred_labels)
    probs = np.array(probs)
    pred = dict(zip(x_ids, pred_labels))

    np.save(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/attMIL/simclr/labels_fold_{fold}.npy', labels)
    np.save(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/attMIL/simclr/pred_labels_fold_{fold}.npy', pred_labels)
    np.save(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/attMIL/simclr/probs_fold_{fold}.npy', probs)
    np.save(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/attMIL/simclr/pred_fold_{fold}.npy', pred)

# import labels
labels_0 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/attMIL/simclr/labels_fold_0.npy')
labels_1 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/attMIL/simclr/labels_fold_1.npy')
labels_2 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/attMIL/simclr/labels_fold_2.npy')
labels_3 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/attMIL/simclr/labels_fold_3.npy')
labels_4 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/attMIL/simclr/labels_fold_4.npy')

# import probs
probs_0 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/attMIL/simclr/probs_fold_0.npy')
probs_1 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/attMIL/simclr/probs_fold_1.npy')
probs_2 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/attMIL/simclr/probs_fold_2.npy')
probs_3 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/attMIL/simclr/probs_fold_3.npy')
probs_4 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/attMIL/simclr/probs_fold_4.npy')

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
plt.title('BRAF V600E mutation prediction')
plt.savefig('ROC_AUC', bbox_inches = 'tight')