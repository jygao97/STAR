import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def pad_matrix(batch_dataset):
    maxlen = len(batch_dataset[0][1])
    batch_num = len(batch_dataset)
    batch_words = np.zeros((batch_num, maxlen))
    for i, sample in enumerate(batch_dataset):
        for j, word in enumerate(sample[1]):
            batch_words[i][j] = word
    return batch_words


def calPerformance(predicted, truth):
    predicted = (predicted >= 0.5).float()
    truth = truth.float()
    Nt = (predicted + truth == 2.0).sum().item()
    Nr = predicted.sum().item()
    Ny = truth.sum().item()
    return Nt, Nr, Ny


def calLoss(predicted, truth, bceLossFunc, l2LossFunc, args):
    accuLoss = bceLossFunc(predicted[:, args.leafIds],
                           truth[:, args.leafIds]).sum(dim=1).mean()
    accuLoss += args.weightI * bceLossFunc(
        predicted[:, args.rootIds], truth[:, args.rootIds]).sum(dim=1).mean()
    for i in args.son:
        accuLoss += args.weightV * l2LossFunc(
            predicted[:, i], predicted[:, args.son[i]].max(dim=1)[0])
    return accuLoss


def batch_pr(predicted, truth):
    predicted = np.array(predicted.cpu())
    truth = np.array(truth.cpu())
    return average_precision_score(truth, predicted, average='micro')
