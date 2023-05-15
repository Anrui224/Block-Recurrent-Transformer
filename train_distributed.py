from pathlib import Path
import numpy as np
import torch
from data.dataset import TextSamplerDataset
from torch.utils.data import DataLoader
from block_recurrent_transformer.model import BlockRecurrentTransformer
import tqdm
from transformers import Adafactor, get_cosine_schedule_with_warmup
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
# print(torch.cuda.nccl.version())
accelerator = Accelerator(split_batches=True)

## hyper-parameters start here
NUM_BATCHES = int(1e4)
BATCH_SIZE = 8
SEQ_LEN = 2048
device = accelerator.device
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 500
BLOCK_LEN = 512
WINDOW_WIDTH = 512
STATE_NUM = 64
## hyper-parameters end here


def cycle(loader):
    while True:
        for data in loader:
            yield data


paths = [str(x) for x in Path('/home/archen/pg19/train_processed').glob("**/*.txt")]
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
    word_num = 30000,
    device = device
)
model = model.to(device)
optim = Adafactor(model.parameters(), lr = 1e-3, relative_step=False,warmup_init=False)
scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=1000, num_training_steps=NUM_BATCHES)


# training


def train(model, train_loader, optim, block_len, epoch):
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
        loss = F.cross_entropy(out, label)
        total_loss = total_loss + (loss / segments)
    optim.zero_grad()
    accelerator.backward(total_loss)
    # print(model.model[9].state.grad)
    optim.step()
    print(f"epoch: {epoch}, train_loss: {total_loss.item()}")
    return total_loss.item()

def val(model, val_loader, block_len, epoch):
    model.eval()
    text = next(val_loader)
    seq_len = text.shape[1]
    segments = seq_len // block_len
    with torch.no_grad():
        total_loss = 0.0
        for i in range(len(model.model)-1):
            model.model[i].clear()
        for ind in range(segments):
            start = ind * block_len
            end = start + block_len + 1
            segment = text[:, start:end]
        
            data, label = segment[:, :-1], segment[:, 1:]
            out = model(data)
            loss = F.cross_entropy(out, label)
            total_loss = total_loss + (loss / segments)
    
    print(f"epoch: {epoch}, val_loss: {total_loss.item()}")
    return total_loss.item()

def main(model, train_loader, val_loader, optim, block_len = BLOCK_LEN):
    model, optim, train_loader, val_loader = accelerator.prepare(model, optim, train_loader, val_loader)
    tb_writer = SummaryWriter('/home/archen/block_recurrent_transformer/log')
    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
        train_loss = train(model, train_loader, optim, scheduler, block_len, i)
        tb_writer.add_scalar('loss/train_loss', train_loss, i)
        if i % VALIDATE_EVERY == 0:
            val_loss = val(model, val_loader, block_len, i)
            tb_writer.add_scalar('loss/val_loss', val_loss, i)
            torch.save(model.state_dict(), f'/home/archen/block_recurrent_transformer/model/brt_{i}.pth')
    tb_writer.close()


if __name__ == "__main__":
    main(model, train_loader, val_loader, optim)





        


    




