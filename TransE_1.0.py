import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import random
import torch.optim as optim
import time
import os
import numpy as np


# data preprocessing
data_path = 'data/FB15k/'
model_path = 'mytraining.pt'

print("loading data...")
# read entity2id, relation2id into a dict，and create id2relation dict and id2relation dict
def read_ID(filename):
    f = open(data_path + filename)
    object2id = {}
    id2object = {}
    line = f.readline()
    while line:
        temp = line.strip('\n').split('\t')
        object2id[temp[0]] = int(temp[1])
        id2object[int(temp[1])] = temp[0]
        line = f.readline()
    f.close()
    return object2id, id2object

entity2id, id2entity = read_ID('entity2id.txt')
relation2id, id2relation = read_ID('relation2id.txt')

# read train, valid, test, and convert strings into id
def convert2id(filename):
    fw = open(data_path + filename[0:-4] + '2id.txt', 'w')
    fr = open(data_path + filename)
    for line in fr.readlines():
        head, tail, relation = line.strip('\n').split('\t')
        new_line = str(entity2id[head]) + '\t' + str(entity2id[tail]) + '\t' + str(relation2id[relation]) + '\n'
        fw.write(new_line)
    fr.close()
    fw.close()
# to avoid generating the files repeatedly, comment the following 3 lines
# convert2id('train.txt')
# convert2id('valid.txt')
# convert2id('test.txt')

#print("finish converting data")

# create dataloader, imitate code on https://github.com/nasusu/ProjE-PyTorch/blob/master/ProjE.ipynb
RELNUM = len(relation2id)  # relation number of dataset
ENTNUM = len(entity2id) # entity number of datset
EMBDIM = 100 # embedding dim
MARGIN = 1 # margin
EPOCHES = 1000

trainTriple = [[int(line.strip().split('\t')[0]), int(line.strip().split('\t')[1]), int(line.strip().split('\t')[2])] for line in open(data_path + 'train2id.txt', encoding='utf-8').readlines()]
testTriple = [[int(line.strip().split('\t')[0]), int(line.strip().split('\t')[1]), int(line.strip().split('\t')[2])] for line in open(data_path + 'test2id.txt', encoding='utf-8').readlines()]
validTriple = [[int(line.strip().split('\t')[0]), int(line.strip().split('\t')[1]), int(line.strip().split('\t')[2])] for line in open(data_path + 'valid2id.txt', encoding='utf-8').readlines()]

def tri2pair(triList):
    hrtPair = {}
    trhPair = {}
    for h, t, r in triList:
        if h not in hrtPair:
            hrtPair[h] = {}
        if r not in hrtPair[h]:
            hrtPair[h][r] = set()
        hrtPair[h][r].add(t)
        
        if t not in trhPair:
            trhPair[t] = {}
        if r not in trhPair[t]:
            trhPair[t][r] = set()
        trhPair[t][r].add(h)
    
    return hrtPair, trhPair

trainHrtPair, trainTrhPair = tri2pair(trainTriple)
allHrtPair, allTrhPair = tri2pair(trainTriple + testTriple + validTriple)

# print(trainHrtPair[0])
# print(trainHrtPair[0][0], trainHrtPair[0][1])

class dset(Dataset):
    def __init__(self, triple, hrtPair, trhPair, train):
        self.hrtPair = hrtPair
        self.trhPair = trhPair
        self.triple = triple
        self.train = train
        
    def __getitem__(self, index):
        hrtPos = self.hrtPair[self.triple[index][0]][self.triple[index][2]]
        trhPos = self.trhPair[self.triple[index][1]][self.triple[index][2]]

        tailsPos = hrtPos - set([self.triple[index][1]])
        headsPos = trhPos - set([self.triple[index][0]])

        hrtNeg = set(range(ENTNUM)) - hrtPos
        trhNeg = set(range(ENTNUM)) - trhPos

        if self.train:
            # sample negative tail       
            hrtSampleNeg = random.sample(hrtNeg, 1)
            
            # sample negative head      
            trhSampleNeg = random.sample(trhNeg, 1)

            return self.triple[index][0], self.triple[index][1], self.triple[index][2], hrtSampleNeg, trhSampleNeg
        else:

            # add filter
            if((self.triple[index][0] in trainHrtPair.keys()) and (self.triple[index][2] in trainHrtPair[self.triple[index][0]].keys())):
                filter_hrtNeg = hrtNeg - trainHrtPair[self.triple[index][0]][self.triple[index][2]]
                tailsNeg = [self.triple[index][1]] + list(filter_hrtNeg)
            else:
                tailsNeg = [self.triple[index][1]] + list(hrtNeg)
            if((self.triple[index][1] in trainTrhPair.keys()) and (self.triple[index][2] in trainTrhPair[self.triple[index][1]].keys())):
                filter_trhNeg = trhNeg - trainTrhPair[self.triple[index][1]][self.triple[index][2]]
                headsNeg = [self.triple[index][0]] + list(filter_trhNeg)
            else:
                headsNeg = [self.triple[index][0]] + list(trhNeg)

            return self.triple[index][0], self.triple[index][1], self.triple[index][2], list(tailsNeg), list(headsNeg)

    def __len__(self):
        return len(self.triple)

trainData = dset(trainTriple, trainHrtPair, trainTrhPair, True)
testData = dset(testTriple, allHrtPair, allTrhPair, False)
validData = dset(validTriple, allHrtPair, allTrhPair, False)

train_loader = DataLoader(trainData, batch_size=512, shuffle=True)
valid_loader = DataLoader(validData, batch_size=1, shuffle=False)
test_loader = DataLoader(testData, batch_size=1, shuffle=False)

print("defining model...")
# imitate code on https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch
class TransE(nn.Module):

    def __init__(self):
        super(TransE,self).__init__()
        self.ent_embeddings=nn.Embedding(ENTNUM,EMBDIM)
        self.rel_embeddings=nn.Embedding(RELNUM,EMBDIM)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def _calc(self, h, t, r):
        # return torch.abs(h + r - t).pow(2)
        return torch.abs(h + r - t)

    def forward(self, pos_hID, pos_rID, pos_tID, neg_tID, neg_hID):
        p_h = self.ent_embeddings(pos_hID)
        p_r = self.rel_embeddings(pos_rID)
        p_t = self.ent_embeddings(pos_tID)
        n_r = self.rel_embeddings(pos_rID)
        # head 为true，则替换head使之成为corrupted triple，反之则替换tail
        l = [True, False]
        head = random.sample(l, 1)[0]

        # print(repalce_head)
        # print()
        if head:
            n_h = self.ent_embeddings(neg_hID)
            n_t = self.ent_embeddings(pos_tID)
        else:
            n_h = self.ent_embeddings(pos_hID)
            n_t = self.ent_embeddings(neg_tID)
        pos_score = torch.sum(self._calc(p_h, p_r, p_t), 1)
        neg_score = torch.sum(self._calc(n_h, n_r, n_t), 1)

        return pos_score, neg_score

model = TransE()
# if there is a trained model, load it, or else comment this line
print("loading model...")
model.load_state_dict(torch.load(model_path))

# loss function & optimizer
criterion = nn.MarginRankingLoss(MARGIN, False)
y = Variable(torch.Tensor([-1]))
# loss = criterion(p_score,n_score,y)
optimizer = optim.SGD(model.parameters(), 0.01)

# train function
def train():
    for epoch in range(EPOCHES):
        total_loss = 0

        for i, (pos_hID, pos_tID, pos_rID, neg_tID, neg_hID) in enumerate(train_loader):
            pos_score, neg_score = model(pos_hID, pos_rID, pos_tID, neg_tID[0], neg_hID[0])
            loss = criterion(pos_score, neg_score, y)
            total_loss += loss.data[0]
            model.zero_grad()
            #print(i, loss)
            loss.backward()
            optimizer.step()
        if epoch % 80 == 0:
            torch.save(model.state_dict(), model_path + str(epoch//80))
        torch.save(model.state_dict(), model_path)
        print("epoch:", epoch, "loss: ", total_loss)
print("start training...")
train()

# save model
#torch.save(the_model.state_dict(), PATH)

# hit@10 & mean rank, replace head only in this function
print("start evaluating...")
ranks = []
for i, (hID, tID, rID, neg_tails, neg_heads) in enumerate(valid_loader):
    heads_len = len(neg_heads)
    heads = torch.LongTensor(neg_heads)    
    tails = torch.LongTensor(list(tID)*(heads_len))
    relations = torch.LongTensor(list(rID)*(heads_len))
    scores, _ = trained_model(heads, relations, tails, heads, heads)
    #当前triple的score
    current_score = scores[0]
    sorted_scores, indexes = scores.sort()
    sorted_scores, indexes = sorted_scores.tolist(), indexes.tolist()
    rank = indexes.index(0) + 1
    ranks.append(rank)
    if(i%1000 == 0):
        print(i, rank)

mean_rank = sum(ranks) / len(ranks)
hit10 = sum([1 for rank in ranks if rank <=10]) / len(ranks)
print("mean_rank", mean_rank)
print("hit@10", hit10)
