import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import time
import random
import heapq
from collections import defaultdict


class STAR(nn.Module):
    def __init__(self, args):
        super(STAR, self).__init__()
        self.args = args
        # the network for textual information
        self.embedding = nn.Embedding(
            args.vocab_size, args.word_embd_size, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, args.filter_num, (size, args.word_embd_size))
            for size in args.filter_size
        ])
        self.dropout = nn.Dropout(p=args.drop_ratio)
        self.wg0 = nn.Linear(
            len(args.filter_size) * args.filter_num, args.hid_g[0])
        self.wg1 = nn.Linear(
            len(args.filter_size) * args.filter_num + args.hid_g[0],
            args.hid_g[1])
        self.wg2 = nn.Linear(
            len(args.filter_size) * args.filter_num + args.hid_g[1],
            args.hid_g[2])
        self.wl0 = nn.Linear(args.hid_g[0], args.hid_l[0])
        self.wl1 = nn.Linear(args.hid_g[1], args.hid_l[1])
        self.wl2 = nn.Linear(args.hid_g[2], args.hid_l[2])
        self.Woutput0 = nn.Linear(args.hid_l[0], args.nodeNum[0])
        self.Woutput1 = nn.Linear(args.hid_l[1], args.nodeNum[1])
        self.Woutput2 = nn.Linear(args.hid_l[2], args.nodeNum[2])

        # the network for spatio-temperal information
        self.timeEmbedding = nn.Embedding(args.timeNum, args.time_embd_size)
        self.locationEmbedding = nn.Embedding(args.locationNum,
                                              args.location_embd_size)
        self.project = nn.Linear(args.time_embd_size + args.location_embd_size,
                                 args.st_embd_size)
        self.dropout2 = nn.Dropout(p=args.drop_ratio)
        self.stg0 = nn.Linear(args.st_embd_size, args.hid_g[0])
        self.stg1 = nn.Linear(args.st_embd_size + args.hid_g[0], args.hid_g[1])
        self.stg2 = nn.Linear(args.st_embd_size + args.hid_g[1], args.hid_g[2])
        self.stl0 = nn.Linear(args.hid_g[0], args.hid_l[0])
        self.stl1 = nn.Linear(args.hid_g[1], args.hid_l[1])
        self.stl2 = nn.Linear(args.hid_g[2], args.hid_l[2])
        self.SToutput0 = nn.Linear(args.hid_l[0], args.nodeNum[0])
        self.SToutput1 = nn.Linear(args.hid_l[1], args.nodeNum[1])
        self.SToutput2 = nn.Linear(args.hid_l[2], args.nodeNum[2])

        # the fusion module
        self.ratio0 = torch.nn.Parameter(torch.randn(args.nodeNum[0]))
        self.ratio1 = torch.nn.Parameter(torch.randn(args.nodeNum[1]))
        self.ratio2 = torch.nn.Parameter(torch.randn(args.nodeNum[2]))

    def forward(self, x, time, location):
        x = self.embedding(x)  # (B*W*E)
        x = x.unsqueeze(1)  # (B*1*W*E)
        x = [F.relu(conv(x)).squeeze(3)
             for conv in self.convs]  #[(B*FN*W),...]*len(FS)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)  # B*(len(FS)*FN)
        x = self.dropout(x)  # the initial text embedding
        AG0 = F.relu(self.wg0(x))
        AL0 = F.relu(self.wl0(AG0))
        woutput0 = torch.sigmoid(self.Woutput0(AL0))
        AG1 = F.relu(self.wg1(torch.cat([AG0, x], dim=1)))
        AL1 = F.relu(self.wl1(AG1))
        woutput1 = torch.sigmoid(self.Woutput1(AL1))
        AG2 = F.relu(self.wg2(torch.cat([AG1, x], dim=1)))
        AL2 = F.relu(self.wl2(AG2))
        woutput2 = torch.sigmoid(self.Woutput2(AL2))

        time = self.timeEmbedding(time)
        location = self.locationEmbedding(location)
        time_location = F.relu(
            self.project(torch.cat([time, location], dim=1)))
        time_location = self.dropout2(time_location)
        BG0 = F.relu(self.stg0(time_location))
        BL0 = F.relu(self.stl0(BG0))
        stoutput0 = torch.sigmoid(self.SToutput0(BL0))
        BG1 = F.relu(self.stg1(torch.cat([BG0, time_location], dim=1)))
        BL1 = F.relu(self.stl1(BG1))
        stoutput1 = torch.sigmoid(self.SToutput1(BL1))
        BG2 = F.relu(self.stg2(torch.cat([BG1, time_location], dim=1)))
        BL2 = F.relu(self.stl2(BG2))
        stoutput2 = torch.sigmoid(self.SToutput2(BL2))

        weight0 = torch.sigmoid(self.ratio0)
        weight1 = torch.sigmoid(self.ratio1)
        weight2 = torch.sigmoid(self.ratio2)

        output0 = weight0 * woutput0 + (1 - weight0) * stoutput0
        output1 = weight1 * woutput1 + (1 - weight1) * stoutput1
        output2 = weight2 * woutput2 + (1 - weight2) * stoutput2

        return torch.cat(
            [woutput0, woutput1, woutput2], dim=1), torch.cat(
                [stoutput0, stoutput1, stoutput2], dim=1), torch.cat(
                    [output0, output1, output2], dim=1)
