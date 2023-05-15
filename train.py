from pathlib import Path
import numpy as np
import torch
from data.dataset import PG19Dataset
from torch.utils.data import DataLoader
from block_recurrent_transformer.model import BlockRecurrentTransformer
from tqdm import trange
from torch.optim import Adam
from transformers import Adafactor
from train_utils import lr_rsqrt_decay
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


## hyper-parameters start here
TOTAL_STEP = int(32*5e5)
BATCH_SIZE = 1
SEQ_LEN = 4096
device = 'cuda:2'
TEST_EVERY = 5000
GRAD_EVERY = 32
BLOCK_LEN = 512
WINDOW_WIDTH = 512
STATE_NUM = 512
## hyper-parameters end here



train_paths = [str(x) for x in Path('/home/archen/pg19/train_processed').glob("**/*.txt")]
test_paths = [str(x) for x in Path('/home/archen/pg19/test_processed').glob("**/*.txt")]

model = BlockRecurrentTransformer(
    seq_size = SEQ_LEN,
    block_size=BLOCK_LEN, 
    window_size=WINDOW_WIDTH, 
    d_input = 2048,
    d_ff = 4096,
    head_num = 8,
    head_dim = 256,
    state_num = STATE_NUM,
    dropout = 0.1,
    transformer_num = 10,
    # word_num = 50625,
    word_num = 32000,
    device = device
)

optim = Adafactor(model.model.parameters(), lr = 1, beta1=0.9,relative_step=False)
scheduler = lr_rsqrt_decay(optim)
criterion = nn.CrossEntropyLoss()
tb_writer = SummaryWriter('/home/archen/block_recurrent_transformer/log')

train_data_id = 0
train_dataset = None
with open(train_paths[train_data_id], 'r') as f:
    train_dataset = eval(f.read())
train_dataset = torch.Tensor(train_dataset)
train_dataset = PG19Dataset(train_dataset, SEQ_LEN, device)
train_loader = iter(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False))
for i in range(len(model.model)-1):
    model.model[i].clear()
total_loss = 0.0
optim.zero_grad()

for step in trange(1,TOTAL_STEP+1):
    model.train()
    #读取PG19数据
    try:
        train_data, train_label = train_loader.__next__()
        
    except StopIteration:
        print('new dataset!')
        train_data_id += 1
        for i in range(len(model.model)-1):
            model.model[i].clear()
        with open(train_paths[train_data_id], 'r') as f:
            train_dataset = eval(f.read())
        train_dataset = torch.Tensor(train_dataset)
        train_dataset = PG19Dataset(train_dataset, SEQ_LEN, device)
        train_loader = iter(DataLoader(train_dataset, batch_size=1, shuffle=False))
        train_data, train_label = train_loader.__next__()
    
    seq_len = train_data.shape[1]
    segments = seq_len // BLOCK_LEN
    
    #将sequence分割为block分别计算loss
    
    # if model.model[-3].state_c is not None:
    #     print(model.model[-3].state_c.grad)
    #     print(model.model[-3].state_c.requires_grad)
    #     print(model.model[-3].state_c)
    for ind in range(segments):
        start = ind * BLOCK_LEN
        end = start + BLOCK_LEN
        data, label = train_data[:, start:end], train_label[:, start:end]
        out = model(data)
        loss = criterion(out, label)
        loss /= (segments*GRAD_EVERY)
        loss.backward()
        total_loss += loss.item()
    
        
    if step%GRAD_EVERY == 0:
        # total_loss /= GRAD_EVERY
        print(f'Step: {step}, Loss: {total_loss}')
        tb_writer.add_scalar('loss/training_loss', total_loss, step//GRAD_EVERY)
        optim.step()
        optim.zero_grad()
        scheduler.step()
        total_loss = 0.0

    if step%TEST_EVERY == 0:
        torch.save(model.state_dict(), f'/home/archen/model/brt_{step}.pth')
    

    
        
        
    

