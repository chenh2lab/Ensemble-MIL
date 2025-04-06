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
import os, copy, random, math

from torch.utils.data.dataset import Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from einops import repeat, rearrange

# define simple logging functionality
log_fw = open("log.txt", 'w') # open log file to save log outputs
def log(text):     # define a logging function to trace the training process
    print(text)
    log_fw.write(str(text)+'\n')
    log_fw.flush()

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# utils
def exists(val):
    return val is not None

# baseaggregator
class BaseAggregator(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

# feedforward
class FeedForward(nn.Module):
    def __init__(self, dim=512, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention
class Attention(nn.Module):
    def __init__(self, dim=512, heads=8, dim_head=512 // 8, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, register_hook=False):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        # save self-attention maps
        self.save_attention_map(attn)
        if register_hook:
            attn.register_hook(self.save_attn_gradients)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def get_self_attention(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        return attn

# prenorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)

# transformer block
class TransformerBlocks(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                    ]
                )
            )

    def forward(self, x, register_hook=False):
        for attn, ff in self.layers:
            x = attn(x, register_hook=register_hook) + x
            x = ff(x) + x
        return x

# transformer model
class Transformer(BaseAggregator):
    def __init__(
        self,
        *,
        num_classes,
        input_dim=2048,
        dim=512,
        depth=2,
        heads=8,
        mlp_dim=512,
        pool='cls',
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        pos_enc=None,
    ):
        super(BaseAggregator, self).__init__()
        assert pool in {
            'cls', 'mean'
        }, 'pool type must be either cls (class token) or mean (mean pooling)'

        self.projection = nn.Sequential(nn.Linear(input_dim, heads*dim_head, bias=True), nn.ReLU())
        self.mlp_head = nn.Sequential(nn.LayerNorm(mlp_dim), nn.Linear(mlp_dim, num_classes))
        self.transformer = TransformerBlocks(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)
        
        self.pos_enc = pos_enc

    def forward(self, x, coords=None, register_hook=False):
        b, _, _ = x.shape

        x = self.projection(x)

        if self.pos_enc:
            x = x + self.pos_enc(coords)

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x)
        x = self.transformer(x, register_hook=register_hook)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(self.norm(x))

# split training, testing ID 'BRAF wildtype(97), mutation(13)'
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
            y_pred = model(x)
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
    model = Transformer(num_classes = 2).to(device)
    model.load_state_dict(torch.load(f'/staging/biology/u1307362/BRAF/transMIL/simclr/fold_{fold}_transformer.pt'))

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

    np.save(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/transMIL/simclr/labels_fold_{fold}.npy', labels)
    np.save(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/transMIL/simclr/pred_labels_fold_{fold}.npy', pred_labels)
    np.save(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/transMIL/simclr/probs_fold_{fold}.npy', probs)
    np.save(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/transMIL/simclr/pred_fold_{fold}.npy', pred)

# import labels
labels_0 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/transMIL/simclr/labels_fold_0.npy')
labels_1 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/transMIL/simclr/labels_fold_1.npy')
labels_2 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/transMIL/simclr/labels_fold_2.npy')
labels_3 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/transMIL/simclr/labels_fold_3.npy')
labels_4 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/transMIL/simclr/labels_fold_4.npy')

# import probs
probs_0 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/transMIL/simclr/probs_fold_0.npy')
probs_1 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/transMIL/simclr/probs_fold_1.npy')
probs_2 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/transMIL/simclr/probs_fold_2.npy')
probs_3 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/transMIL/simclr/probs_fold_3.npy')
probs_4 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/BRAF/transMIL/simclr/probs_fold_4.npy')

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