import gzip
from pathlib import Path
import numpy as np
import torch
from data.dataset import TextSamplerDataset
from torch.utils.data import DataLoader
from block_recurrent_transformer.model import BlockRecurrentTransformer
import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from copy import deepcopy


## hyper-parameters start here
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
SEQ_LEN = 2048
device = "cuda:1"
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATE_EVERY = 4
VALIDATE_EVERY = 100
BLOCK_LEN = 512
WINDOW_WIDTH = 100
STATE_NUM = 64
## hyper-parameters end here


def cycle(loader):
    while True:
        for data in loader:
            yield data


paths = ['/home/archen/pg19/train_processed/10.txt', '/home/archen/pg19/train_processed/11.txt','/home/archen/pg19/train_processed/12.txt','/home/archen/pg19/train_processed/13.txt','/home/archen/pg19/train_processed/15.txt']
data = []
for path in paths:
    with open(path, 'r') as f:
        tmp = eval(f.read())
        data += tmp

np_train, np_valid = np.split(np.array(data),[1000000])
data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)
train_dataset = TextSamplerDataset(data_train, SEQ_LEN, device)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN, device)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

model = BlockRecurrentTransformer(
    seq_size = SEQ_LEN,
    block_size=BLOCK_LEN, 
    window_size=WINDOW_WIDTH, 
    d_input = 100,
    d_ff = 256,
    head_num = 12,
    head_dim = 64,
    state_num = STATE_NUM,
    dropout = 0.1,
    transformer_num = 9,
    # word_num = 50625,
    word_num = 32000,
    device = device
)

optim = Adam(model.model.parameters(), lr = LEARNING_RATE)

# training


def train(model, train_loader, val_loader, optim, block_len, epoch):
    model.train()
    text = next(train_loader)
    seq_len = text.shape[1]
    segments = seq_len // block_len
    
    total_loss = 0.0
    for i in range(len(model.model)-1):
        model.model[i].clear()
    for ind in range(segments):
        start = ind * block_len
        end = start + block_len + 1
        segment = text[:, start:end]
        
        data, label = segment[:, :-1], segment[:, 1:]
        out = model(data)
        # out = out.transpose(1,2)
        # print(out.shape, label.shape)
        loss = F.cross_entropy(out, label)
        total_loss = total_loss + (loss / segments)
    optim.zero_grad()
    total_loss.backward()
    # print(model.model[9].state.grad)
    optim.step()
    print(f"epoch: {epoch}, loss: {total_loss.item()}")
    
    

for i in range(10):
    train(model, train_loader, val_loader, optim, BLOCK_LEN, i)



        


    



