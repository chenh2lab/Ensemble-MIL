import torch 
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

# define simple logging functionality
log_fw = open("log.txt", 'w') # open log file to save log outputs
def log(text):
    print(text)
    log_fw.write(str(text)+'\n')
    log_fw.flush()

# import KRAS groud truth
label_dict = np.load('/home/u1307362/biomarker_label/KRAS/CPTAC-COAD.npy', allow_pickle = True).item()

# plot configuration
classes = ['wildtype', 'mutation']

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

# majority voting
for fold in range(5):
    log(f'FOLD {fold}')
    log('---------------------------------------------')

    # import kras biomarker prediction (hard voting)
    attn = np.load(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/attMIL/simclr/pred_fold_{fold}.npy', allow_pickle = True).item()
    gnn = np.load(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/gnnMIL/simclr/pred_fold_{fold}.npy', allow_pickle = True).item()
    tran = np.load(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/transMIL/simclr/pred_fold_{fold}.npy', allow_pickle = True).item()
    ensemble_label, ensemble_pred_label = [], []
    for a, g, t in zip(attn, gnn, tran):
        if a[:-4] == g == t[:-4]:
            ensemble_label.append(label_dict[a[0:7]])
            total = attn[a] + gnn[g] + tran[t]
            if total == 0:
                ensemble_pred_label.append(0)
            elif total == 1:
                ensemble_pred_label.append(0)
            elif total == 2:
                ensemble_pred_label.append(1)
            elif total == 3:
                ensemble_pred_label.append(1)

    # plot confusion matrix
    plot_confusion_matrix(ensemble_label, ensemble_pred_label, classes, title = 'KRAS prediction')
    plt.savefig(f'fold_{fold}_hard_confusion_matrix.png', bbox_inches = 'tight')

    # import kras biomarker prediction (soft voting)
    attn = np.load(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/attMIL/simclr/probs_fold_{fold}.npy')
    gnn = np.load(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/gnnMIL/simclr/probs_fold_{fold}.npy')
    tran = np .load(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/transMIL/simclr/probs_fold_{fold}.npy')
    avg_probs = (attn + gnn + tran) / 3
    ensemble_pred_label = np.argmax(avg_probs, axis = 1)

    # calculate ensemble result
    evaluated_metrics(ensemble_label, ensemble_pred_label, avg_probs)

    # plot confusion matrix
    plot_confusion_matrix(ensemble_label, ensemble_pred_label, classes, title = 'KRAS prediction')
    plt.savefig(f'fold_{fold}_soft_confusion_matrix.png', bbox_inches = 'tight')

    # save
    np.save(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/ensemble/simclr/labels_fold_{fold}.npy', ensemble_label)
    np.save(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/ensemble/simclr/pred_labels_fold_{fold}.npy', ensemble_pred_label)
    np.save(f'/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/ensemble/simclr/avg_probs_fold_{fold}.npy', avg_probs)

# import labels
labels_0 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/ensemble/simclr/labels_fold_0.npy')
labels_1 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/ensemble/simclr/labels_fold_1.npy')
labels_2 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/ensemble/simclr/labels_fold_2.npy')
labels_3 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/ensemble/simclr/labels_fold_3.npy')
labels_4 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/ensemble/simclr/labels_fold_4.npy')

# import probs
probs_0 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/ensemble/simclr/avg_probs_fold_0.npy')
probs_1 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/ensemble/simclr/avg_probs_fold_1.npy')
probs_2 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/ensemble/simclr/avg_probs_fold_2.npy')
probs_3 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/ensemble/simclr/avg_probs_fold_3.npy')
probs_4 = np.load('/staging/biology/u1307362/external_validation/CPTAC-COAD/KRAS/ensemble/simclr/avg_probs_fold_4.npy')

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
plt.title('KRAS mutation prediction')
plt.savefig('ROC_AUC', bbox_inches = 'tight')