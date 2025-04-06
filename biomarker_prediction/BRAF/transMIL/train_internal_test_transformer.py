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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.25):
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
        dropout=0.25,
        emb_dropout=0.25,
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
    
    def l1_regularization(self, lambda_l1):
        l1_reg = torch.tensor(0., requires_grad=True).to(next(self.parameters()).device)
        l1_reg = l1_reg + sum(torch.norm(param, p=1) for param in self.parameters())
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

# learning rate schedule
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, 
    warmup_steps: int, 
    total_training_steps: int, 
    num_cycles: float = 0.5, 
    last_epoch: int = -1):

    def lr_lambda(step):
        # Warmup
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # decadence
        progress = float(step - warmup_steps) / float(
            max(1, total_training_steps - warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# function definition
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

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
lambda_l1 = 1e-5
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
    model = Transformer(num_classes = 2).to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5, weight_decay = 1e-5)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps = 8 * len(train_iterator), total_training_steps = 100 * len(train_iterator))
    
    log('start training & validation')
    for epoch in range(n_epochs):
        log(f'Epoch [{epoch + 1}/{n_epochs}]')

        # training
        epoch_train_loss, epoch_train_acc = 0, 0
        model.train()
        for step, (x, y) in enumerate(train_iterator):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            l1_loss = model.l1_regularization(lambda_l1)
            loss += l1_loss
            acc = calculate_accuracy(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_train_loss += loss.item()
            epoch_train_acc += acc.item()

        train_loss = epoch_train_loss / len(train_iterator)
        train_acc = epoch_train_acc / len(train_iterator)
        log(f'Training Loss: {train_loss:.3f}\t Training Acc: {train_acc:.3f}')

        # validation
        epoch_valid_loss, epoch_valid_acc = 0, 0
        model.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(valid_iterator):
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                acc = calculate_accuracy(y_pred, y)
                epoch_valid_loss += loss.item()
                epoch_valid_acc += acc.item()

        valid_loss = epoch_valid_loss / len(valid_iterator)
        valid_acc = epoch_valid_acc / len(valid_iterator)
        log(f'Valid Loss: {valid_loss:.3f}\t Valid Acc: {valid_acc:.3f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'/staging/biology/u1307362/BRAF/transMIL/simclr/fold_{fold}_transformer.pt')

    log(f'Best checkpoint at {best_epoch} epoch')

    log('start testing')
    # model_test
    model_test = Transformer(num_classes = 2).to(device)
    model_test.load_state_dict(torch.load(f'/staging/biology/u1307362/BRAF/transMIL/simclr/fold_{fold}_transformer.pt'))
    
    # testing
    labels, probs = [], []
    model_test.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(test_iterator):
            x = x.to(device)
            y = y.to(device)
            y_pred = model_test(x)
            y_prob = F.softmax(y_pred, dim = 1)
            labels.append(y.cpu())
            probs.append(y_prob.cpu())
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)
    pred_labels = torch.argmax(probs, 1)

    evaluated_metrics(labels, pred_labels, probs)
    
    # confusion matrix
    plot_confusion_matrix(labels, pred_labels, classes, title = 'BRAF prediction')
    plt.savefig(f'fold_{fold}_confusion_matrix', bbox_inches = 'tight')

    # save prediction
    labels = np.array(labels)
    pred_labels = np.array(pred_labels)
    probs = np.array(probs)
    pred = dict(zip(test_x_ids, pred_labels))

    np.save(f'/staging/biology/u1307362/BRAF/transMIL/simclr/labels_fold_{fold}.npy', labels)
    np.save(f'/staging/biology/u1307362/BRAF/transMIL/simclr/pred_labels_fold_{fold}.npy', pred_labels)
    np.save(f'/staging/biology/u1307362/BRAF/transMIL/simclr/probs_fold_{fold}.npy', probs)
    np.save(f'/staging/biology/u1307362/BRAF/transMIL/simclr/pred_fold_{fold}.npy', pred)

# import labels
labels_0 = np.load('/staging/biology/u1307362/BRAF/transMIL/simclr/labels_fold_0.npy')
labels_1 = np.load('/staging/biology/u1307362/BRAF/transMIL/simclr/labels_fold_1.npy')
labels_2 = np.load('/staging/biology/u1307362/BRAF/transMIL/simclr/labels_fold_2.npy')
labels_3 = np.load('/staging/biology/u1307362/BRAF/transMIL/simclr/labels_fold_3.npy')
labels_4 = np.load('/staging/biology/u1307362/BRAF/transMIL/simclr/labels_fold_4.npy')

# import probs
probs_0 = np.load('/staging/biology/u1307362/BRAF/transMIL/simclr/probs_fold_0.npy')
probs_1 = np.load('/staging/biology/u1307362/BRAF/transMIL/simclr/probs_fold_1.npy')
probs_2 = np.load('/staging/biology/u1307362/BRAF/transMIL/simclr/probs_fold_2.npy')
probs_3 = np.load('/staging/biology/u1307362/BRAF/transMIL/simclr/probs_fold_3.npy')
probs_4 = np.load('/staging/biology/u1307362/BRAF/transMIL/simclr/probs_fold_4.npy')

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