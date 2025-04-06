import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV

# define simple logging functionality
log_fw = open("log.txt", 'w') # open log file to save log outputs
def log(text):     # define a logging function to trace the training process
    print(text)
    log_fw.write(str(text)+'\n')
    log_fw.flush()

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed = 12494
set_seed(seed)

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

# split training, testing ID 'BRAF wildtype(377), mutation(51)'
root = '/staging/biology/u1307362/TCGA-COAD_norm_tumor_feature/simclr/vgg19_bn/TCGA-HE-89K'
label_dict = np.load('/home/u1307362/biomarker_label/BRAF/TCGA-COAD.npy', allow_pickle = True).item()

x_ids, y_label = [], []
for csv in os.listdir(root):
    x_ids.append(csv)
    if label_dict[csv[:12]] == 0:
        y_label.append(0)
    elif label_dict[csv[:12]] == 1:
        y_label.append(1)

# dataset & dataloader
class TumorFeatureDataset(Dataset):
    def __init__(self, feature_folder, ids, mode = None):
        """
        mode: 'train', 'val', 'test'
        """
        self.feature_folder = feature_folder
        self.ids = ids
        
        if mode == 'train':
            self.slide_list = [os.path.join(feature_folder, x) for x in ids]
        elif mode == 'val':
            self.slide_list = [os.path.join(feature_folder, x) for x in ids]
        elif mode == 'test':
            self.slide_list = [os.path.join(feature_folder, x) for x in ids]

        log(f'{len(self.slide_list)} slides in {mode} dataset')

    def __getitem__(self, idx):
        fname = self.slide_list[idx]
        df = pd.read_csv(fname)
        feature_array = df[[f'{n}' for n in range(2048)]].values
        feature_tensor = torch.from_numpy(feature_array)

        label = int(label_dict[fname.split("/")[-1][:12]])
        return feature_tensor.float(), label

    def __len__(self):
        return len(self.slide_list)

# loss
criterion = nn.CrossEntropyLoss().to(device)

# function definition
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

# training
def train(model, iterator, optimizer, criterion, lambda_l1, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred, attn = model(x)
        loss = criterion(y_pred, y)
        l1_loss = model.l1_regularization(lambda_l1)
        loss += l1_loss
        acc = calculate_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# validation
def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for x, y in iterator:
            x = x.to(device)
            y = y.to(device)
            y_pred, attn = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

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
n_epochs = 100
lambda_l1 = 1e-4
k_folds = 5
classes = ['wildtype', 'mutation']

# Define the K-fold Cross Validator
kfold = StratifiedKFold(n_splits = k_folds, shuffle = True, random_state = seed)

for fold, (train_index, test_index) in enumerate(kfold.split(x_ids, y_label)):
    log(f'FOLD {fold}')
    log('---------------------------------------------')
    best_valid_loss = float('inf')

    # split train & val
    x = [x_ids[i] for i in train_index]
    y = [y_label[i] for i in train_index]
    train_x_ids, val_x_ids, train_y_label, val_y_label = train_test_split(x, y, test_size = (1/8), stratify = y)

    # test
    test_x_ids = [x_ids[i] for i in test_index]
    test_y_label = [y_label[i] for i in test_index]

    # training class weight adjustment
    train_x_ids_wildtype, train_x_ids_mutation = [], []
    for index, i in enumerate(train_y_label):
        if i == 0:
            train_x_ids_wildtype.append(train_x_ids[index])
        elif i == 1:
            train_x_ids_mutation.append(train_x_ids[index])
    train_x_ids = train_x_ids_wildtype + train_x_ids_mutation * 7

    # dataset & dataloader
    train_data = TumorFeatureDataset(feature_folder = root, mode = 'train', ids = train_x_ids)
    val_data = TumorFeatureDataset(feature_folder = root, mode = 'val', ids = val_x_ids)
    test_data = TumorFeatureDataset(feature_folder = root, mode = 'test', ids = test_x_ids)

    train_iterator = data.DataLoader(train_data, batch_size = batch_size, num_workers = 4)
    valid_iterator = data.DataLoader(val_data, batch_size = batch_size, num_workers = 4)
    test_iterator = data.DataLoader(test_data, batch_size = batch_size, num_workers = 4)

    # model
    model = AttnClassifier(input_dim = 2048, attn_dim = 1024, output_dim = 2, lambda_l1 = lambda_l1).to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5, weight_decay = 1e-2)
    # warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch * 3 + 1/25.0), last_epoch = -1, verbose = True)
    # mainscheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [30, 60, 90], gamma = (0.1 ** (1/3)), last_epoch = -1, verbose = True)

    log('start training & validation')
    for epoch in range(n_epochs):
        log(f'Epoch [{epoch + 1}/{n_epochs}]')
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, lambda_l1, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

        # if epoch <= 8:
        #     warmupscheduler.step()
        # mainscheduler.step()

        log(f'Training Loss: {train_loss:.3f}\t Training Acc: {train_acc:.3f}')
        log(f'Valid Loss: {valid_loss:.3f}\t Valid Acc: {valid_acc:.3f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'/staging/biology/u1307362/BRAF/attMIL/simclr/fold_{fold}_attention.pt')

    log(f'Best checkpoint at {best_epoch} epoch')

    log('start testing')
    # model_test
    model_test = AttnClassifier(input_dim = 2048, attn_dim = 1024, output_dim = 2, lambda_l1 = lambda_l1).to(device)
    model_test.load_state_dict(torch.load(f'/staging/biology/u1307362/BRAF/attMIL/simclr/fold_{fold}_attention.pt'))

    labels, pred_labels, probs = get_predictions(model_test, test_iterator)
    evaluated_metrics(labels, pred_labels, probs)

    # confusion matrix
    plot_confusion_matrix(labels, pred_labels, classes, title = 'BRAF prediction')
    plt.savefig(f'fold_{fold}_confusion_matrix', bbox_inches = 'tight')

    # save prediction
    labels = np.array(labels)
    pred_labels = np.array(pred_labels)
    probs = np.array(probs)
    pred = dict(zip(test_x_ids, pred_labels))

    np.save(f'/staging/biology/u1307362/BRAF/attMIL/simclr/labels_fold_{fold}.npy', labels)
    np.save(f'/staging/biology/u1307362/BRAF/attMIL/simclr/pred_labels_fold_{fold}.npy', pred_labels)
    np.save(f'/staging/biology/u1307362/BRAF/attMIL/simclr/probs_fold_{fold}.npy', probs)
    np.save(f'/staging/biology/u1307362/BRAF/attMIL/simclr/pred_fold_{fold}.npy', pred)

# import labels
labels_0 = np.load('/staging/biology/u1307362/BRAF/attMIL/simclr/labels_fold_0.npy')
labels_1 = np.load('/staging/biology/u1307362/BRAF/attMIL/simclr/labels_fold_1.npy')
labels_2 = np.load('/staging/biology/u1307362/BRAF/attMIL/simclr/labels_fold_2.npy')
labels_3 = np.load('/staging/biology/u1307362/BRAF/attMIL/simclr/labels_fold_3.npy')
labels_4 = np.load('/staging/biology/u1307362/BRAF/attMIL/simclr/labels_fold_4.npy')

# import probs
probs_0 = np.load('/staging/biology/u1307362/BRAF/attMIL/simclr/probs_fold_0.npy')
probs_1 = np.load('/staging/biology/u1307362/BRAF/attMIL/simclr/probs_fold_1.npy')
probs_2 = np.load('/staging/biology/u1307362/BRAF/attMIL/simclr/probs_fold_2.npy')
probs_3 = np.load('/staging/biology/u1307362/BRAF/attMIL/simclr/probs_fold_3.npy')
probs_4 = np.load('/staging/biology/u1307362/BRAF/attMIL/simclr/probs_fold_4.npy')

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