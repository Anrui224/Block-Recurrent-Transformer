from torch.utils.data import Dataset
import torch

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, device):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.device = device

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(self.device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

class PG19Dataset(Dataset):
    def __init__(self, data, seq_len, device):
        super().__init__()
        self.data = data.tolist()
        self.seq_len = seq_len
        if len(self.data)%(self.seq_len + 1) != 0:
            self.data = self.data + [0]*(self.seq_len + 1 - len(self.data)%(self.seq_len + 1))
        self.data = torch.tensor(self.data)
        self.device = device
    
    def __getitem__(self, index):
        # rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        # full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        index = index*(self.seq_len + 1)
        full_seq = self.data[index : index + self.seq_len + 1].long()
        return full_seq[:-1].to(self.device), full_seq[1:].to(self.device)
    
    def __len__(self):
        return len(self.data)//self.seq_len

class PG19TestDataset(Dataset):
    def __init__(self, data, seq_len, device):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.device = device
    
    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        # index = index*(self.seq_len + 1)
        # full_seq = self.data[index : index + self.seq_len + 1].long()
        return full_seq.to(self.device)
    
    def __len__(self):
        return len(self.data)//self.seq_len





    
