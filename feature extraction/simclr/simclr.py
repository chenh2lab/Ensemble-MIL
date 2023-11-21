import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.distributed as dist
import torch.multiprocessing as mp

from info_nce import InfoNCE, info_nce
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

import numpy as np
import matplotlib.pyplot as plt
import os, argparse, random

# define simple logging functionality
log_fw = open("SimCLR_log.txt", 'w') # open log file to save log outputs
def log(text):     # define a logging function to trace the training process
    print(text)
    log_fw.write(str(text)+'\n')
    log_fw.flush()

# device (multi-gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
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
set_seed(890904)

# data augmentation
class Augment:
    def __init__(self, img_size, s = 1):
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        blur = transforms.GaussianBlur((3, 3), (0.1, 2.0))

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size = img_size),
            transforms.RandomHorizontalFlip(p = 0.5),  # with 0.5 probability
            transforms.RandomApply([color_jitter], p = 0.8),
            transforms.RandomApply([blur], p = 0.5),
            transforms.RandomGrayscale(p = 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.7731, 0.5610, 0.7280], std = [0.1541, 0.2230, 0.1530])])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

transform = Augment(img_size = 224)

# datasets & dataloader
batch_size = 256
root = '/staging/biology/u1307362/TCGACOAD_tumor_patch_512_resnet101'

train_data = datasets.ImageFolder(root = root,  transform = transform)
train_iterator = data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = False)

# model
class ProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.projection = nn.Sequential(
            nn.Linear(in_features = in_features, out_features = hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Linear(in_features = hidden_features, out_features = out_features),
            nn.BatchNorm1d(out_features))

    def forward(self,x):
        x = self.projection(x)
        return x

class PreModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        #PRETRAINED MODEL
        self.pretrained = models.resnet50(pretrained = True)
        self.pretrained.fc = nn.Identity()

        for p in self.pretrained.parameters():
            p.requires_grad = True

        self.projector = ProjectionHead(2048, 2048, 128)

    def forward(self,x):
        out = self.pretrained(x)
        xp = self.projector(torch.squeeze(out))
        return xp

model = PreModel('resnet50')
model = nn.DataParallel(model, device_ids = [0, 1]).to(device)

# loss
criterion = InfoNCE().to(device)

# optimizer
optimizer = optim.AdamW(model.module.parameters(), lr = 0.3, weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(train_iterator), eta_min = 0.0, last_epoch = -1, verbose = True)

# training & validation
epochs = 800
train_loss = []
best_train_loss = float('inf')

log('start training')
for epoch in range(epochs):
    log(f'Epoch [{epoch + 1}/{epochs}]')

    model.train()
    train_loss_epoch = 0
    for (x_i, x_j), _ in train_iterator:
        optimizer.zero_grad()
        x_i = x_i.squeeze().to(device).float()
        x_j = x_j.squeeze().to(device).float()

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()

        train_loss_epoch += loss.item()

    train_loss.append(train_loss_epoch / len(train_iterator))
    log(f'Training Loss: {(train_loss_epoch / len(train_iterator)):.3f}')

    if epoch >= 80:
        scheduler.step()

    if (train_loss_epoch / len(train_iterator)) < best_train_loss:
        best_train_loss = (train_loss_epoch / len(train_iterator))
        torch.save(model.state_dict(), 'SimCLR.pt')
        torch.save({'model_state_dict': model.module.pretrained.state_dict()}, 'SimCLR_resnet50.pt')
    print(f'Epoch [{epoch}/{epochs}]\t Training Loss: {(train_loss_epoch / len(train_iterator)):.3f}')
log('finish training')
print(train_loss)

# plot
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

x = range(1, 801)
y_train = train_loss
plt.plot(x, y_train, color = 'royalblue', label = 'training')
plt.xlabel('Epoch', fontsize = 10)
plt.ylabel('Loss', fontsize = 10)
plt.title('SimCLR loss curve', fontsize = 14)
plt.legend(loc = 'best')
plt.savefig('SimCLR_loss_curve', bbox_inches = 'tight')
