import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import _pickle as pickle
import time
import random
from model import STAR
from utils import pad_matrix, calPerformance, calLoss, batch_pr

torch.manual_seed(2019)  # cpu
torch.cuda.manual_seed(2019)  #gpu
np.random.seed(2019)  #numpy
random.seed(2019)  #random and transforms
torch.backends.cudnn.deterministic = True  # cudnn

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=30)
parser.add_argument('-device', type=int, default=0)
parser.add_argument('-batch_size', type=int, default=256)
parser.add_argument('-word_embd_size', type=int, default=128)
parser.add_argument('-time_embd_size', type=int, default=128)
parser.add_argument('-location_embd_size', type=int, default=64)
parser.add_argument('-st_embd_size', type=int, default=128)
parser.add_argument('-filter_num', type=int, default=80)
parser.add_argument('-filter_size', type=str, default='1,2,3,4,5')
parser.add_argument('-weightI', type=float, default=1.5)
parser.add_argument('-weightV', type=float, default=0.1)
parser.add_argument('-weightS', type=float, default=0.01)
parser.add_argument('-drop_ratio', type=float, default=0.4)
parser.add_argument('-l_rate', type=float, default=0.001)
parser.add_argument('-description', type=str, default='')
parser.add_argument('-data_ratio', type=float, default=1.0)
parser.add_argument('-Mid_l', type=str, default='512,512,512')
parser.add_argument('-Mid_g', type=str, default='256,256,256')
parser.add_argument('-data_folder', type=str, default='data')
parser.add_argument('-model_file', type=str, default='STAR.model')
args = parser.parse_args()
args.hid_l = [int(i) for i in args.Mid_l.split(',')]
args.hid_g = [int(i) for i in args.Mid_g.split(',')]
device = torch.device("cuda:{}".format(str(args.device))
                      if torch.cuda.is_available() else "cpu")
print("availabel device: {}".format(device))

if len(args.description) != 0:
    args.model_file = '{}_{}'.format(args.model_file, args.description)
args.best_loss_model = '{}_{}'.format(args.model_file, 'best_loss')

args.filter_size = [int(i) for i in args.filter_size.split(',')]
print(args)
vocab = pickle.load(open(os.path.join(args.data_folder, 'vocab.pkl'), 'rb'))
args.vocab_size = len(vocab)
print("vocab_size: {}".format(args.vocab_size))

levelCnt, label2Id, father, son = pickle.load(
    open(os.path.join(args.data_folder, 'taxonomy'), 'rb'))
id2label = {}
for k in label2Id:
    v = label2Id[k]
    id2label[v] = k
time_dict, smoothT = pickle.load(
    open(os.path.join(args.data_folder, 'time_refined.pkl'), 'rb'))
location_dict, smoothL, _ = pickle.load(
    open(os.path.join(args.data_folder, 'location_refined.pkl'), 'rb'))
args.leafIds = sorted(list(set(range(len(label2Id))) - set(son.keys())))
args.rootIds = sorted(list(son.keys()))
args.son = son
args.leafNum = len(args.leafIds)
args.nodeNum = [levelCnt[1], levelCnt[2], levelCnt[3]]

args.timeNum = smoothT.shape[0]
args.locationNum = smoothL.shape[0]

start_time = time.time()
labels = pickle.load(open(os.path.join(args.data_folder, 'label'), 'rb'))
dataset_train, dataset_valid, dataset_test = pickle.load(
    open(os.path.join(args.data_folder, 'split_tid_wordid.pkl'), 'rb'))
label_train = np.array([labels[i[0]] for i in dataset_train])
label_valid = np.array([labels[i[0]] for i in dataset_valid])
label_test = np.array([labels[i[0]] for i in dataset_test])
time_train = np.array([time_dict[i[0]] for i in dataset_train])
time_valid = np.array([time_dict[i[0]] for i in dataset_valid])
time_test = np.array([time_dict[i[0]] for i in dataset_test])
location_train = np.array([location_dict[i[0]] for i in dataset_train])
location_valid = np.array([location_dict[i[0]] for i in dataset_valid])
location_test = np.array([location_dict[i[0]] for i in dataset_test])

print("dataset load {} s".format(time.time() - start_time))

n_batches = int(args.data_ratio * np.ceil(
    float(len(label_train)) / float(args.batch_size)))
print('n_batches:{}'.format(n_batches))
samples = list(range(n_batches))
random.shuffle(samples)
recommend = STAR(args).to(device)
smoothL = torch.FloatTensor(smoothL).to(device)
smoothT = torch.FloatTensor(smoothT).to(device)

bceloss = nn.BCELoss(reduction='none')
l2loss = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(recommend.parameters(), lr=args.l_rate)

best_valid_loss = 100000
for i in range(args.epochs):
    recommend.train()
    start_time = time.time()
    losses = []
    s_time = time.time()
    for index in samples:
        batch_words = pad_matrix(
            dataset_train[args.batch_size * index:args.batch_size * (
                index + 1)])
        batch_label = label_train[args.batch_size * index:args.batch_size * (
            index + 1)]
        time_data = time_train[args.batch_size * index:args.batch_size * (
            index + 1)]
        location_data = location_train[args.batch_size * index:
                                       args.batch_size * (index + 1)]
        t_batch_words = torch.LongTensor(batch_words).to(device)
        t_batch_label = torch.FloatTensor(batch_label).to(device)
        t_time = torch.LongTensor(time_data).to(device)
        t_location = torch.LongTensor(location_data).to(device)
        optimizer.zero_grad()
        woutput, stoutput, output = recommend(t_batch_words, t_time,
                                              t_location)
        loss = calLoss(output, t_batch_label, bceloss, l2loss, args)
        loss += calLoss(woutput, t_batch_label, bceloss, l2loss, args)
        loss += calLoss(stoutput, t_batch_label, bceloss, l2loss, args)
        loss += args.weightS * torch.norm(
            smoothT.mm(recommend.state_dict()['timeEmbedding.weight']))
        loss += args.weightS * torch.norm(
            smoothL.mm(recommend.state_dict()['locationEmbedding.weight']))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    with torch.no_grad():
        recommend.eval()
        valid_losses = []
        valid_sum_Nt = 0
        valid_sum_Nr = 0
        valid_sum_Ny = 0
        valid_batches = int(
            np.ceil(float(len(label_valid)) / float(args.batch_size)))
        for index in range(valid_batches):
            batch_words = pad_matrix(
                dataset_valid[args.batch_size * index:args.batch_size * (
                    index + 1)])
            batch_label = label_valid[args.batch_size * index:args.batch_size *
                                      (index + 1)]
            time_data = time_valid[args.batch_size * index:args.batch_size * (
                index + 1)]
            location_data = location_valid[args.batch_size * index:
                                           args.batch_size * (index + 1)]
            t_batch_words = torch.LongTensor(batch_words).to(device)
            t_batch_label = torch.FloatTensor(batch_label).to(device)
            t_time = torch.LongTensor(time_data).to(device)
            t_location = torch.LongTensor(location_data).to(device)
            woutput, stoutput, output = recommend(t_batch_words, t_time,
                                                  t_location)
            loss = calLoss(output, t_batch_label, bceloss, l2loss, args)
            loss += calLoss(woutput, t_batch_label, bceloss, l2loss, args)
            loss += calLoss(stoutput, t_batch_label, bceloss, l2loss, args)
            loss += args.weightS * torch.norm(
                smoothT.mm(recommend.state_dict()['timeEmbedding.weight']))
            loss += args.weightS * torch.norm(
                smoothL.mm(recommend.state_dict()['locationEmbedding.weight']))
            valid_losses.append(loss.item())
            Nt, Nr, Ny = calPerformance(output[:, args.leafIds],
                                        t_batch_label[:, args.leafIds])
            valid_sum_Nt += Nt
            valid_sum_Nr += Nr
            valid_sum_Ny += Ny
    duration = time.time() - start_time
    train_loss = np.mean(losses)
    valid_loss = np.mean(valid_losses)
    if valid_sum_Nr == 0:
        precision = -1
    else:
        precision = float(valid_sum_Nt) / float(valid_sum_Nr)
    recall = float(valid_sum_Nt) / float(valid_sum_Ny)
    F1 = float(2 * valid_sum_Nt) / float(valid_sum_Nr + valid_sum_Ny)
    print("epoch: {} train:{} duration: {}".format(i + 1, train_loss,
                                                   duration))
    print("valid: {} precision:{} recall:{} F1:{}".format(
        valid_loss, precision, recall, F1))
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        state = {
            'net': recommend.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': i + 1
        }
        torch.save(state, args.best_loss_model)

print("best_valid_loss {}".format(best_valid_loss))

checkpoint = torch.load(args.best_loss_model)
save_epoch = checkpoint['epoch']
print("last saved model of best valid loss is in epoch {}".format(save_epoch))
recommend.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])
recommend.eval()
final_f1 = 0
final_pr = 0
with torch.no_grad():
    test_losses = []
    test_sum_Nt = 0
    test_sum_Nr = 0
    test_sum_Ny = 0
    test_pr = []
    test_batches = int(
        np.ceil(float(len(label_test)) / float(args.batch_size)))
    for index in range(test_batches):
        batch_words = pad_matrix(
            dataset_test[args.batch_size * index:args.batch_size * (
                index + 1)])
        batch_label = label_test[args.batch_size * index:args.batch_size * (
            index + 1)]
        time_data = time_test[args.batch_size * index:args.batch_size * (
            index + 1)]
        location_data = location_test[args.batch_size * index:args.batch_size *
                                      (index + 1)]
        t_batch_words = torch.LongTensor(batch_words).to(device)
        t_batch_label = torch.FloatTensor(batch_label).to(device)
        t_time = torch.LongTensor(time_data).to(device)
        t_location = torch.LongTensor(location_data).to(device)
        woutput, stoutput, output = recommend(t_batch_words, t_time,
                                              t_location)
        loss = calLoss(output, t_batch_label, bceloss, l2loss, args)
        loss += calLoss(woutput, t_batch_label, bceloss, l2loss, args)
        loss += calLoss(stoutput, t_batch_label, bceloss, l2loss, args)
        loss += args.weightS * torch.norm(
            smoothT.mm(recommend.state_dict()['timeEmbedding.weight']))
        loss += args.weightS * torch.norm(
            smoothL.mm(recommend.state_dict()['locationEmbedding.weight']))
        test_losses.append(loss.item())

        Nt, Nr, Ny = calPerformance(output[:, args.leafIds],
                                    t_batch_label[:, args.leafIds])
        pr = batch_pr(output[:, args.leafIds], t_batch_label[:, args.leafIds])
        test_pr.append(pr)
        test_sum_Nt += Nt
        test_sum_Nr += Nr
        test_sum_Ny += Ny
    precision = float(test_sum_Nt) / float(test_sum_Nr)
    recall = float(test_sum_Nt) / float(test_sum_Ny)
    pr = np.mean(test_pr)
    F1 = float(2 * test_sum_Nt) / float(test_sum_Nr + test_sum_Ny)
    final_f1 = F1
    final_pr = pr
    print("### Performance on testset ###")
    print("loss {}  F1 {} pr {}".format(np.mean(test_losses), F1, pr))
    print("precision {} recall {}".format(precision, recall))
