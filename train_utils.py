from torch.optim.lr_scheduler import _LRScheduler
import math


class lr_rsqrt_decay(_LRScheduler):
    def __init__(self, optimizer, max_steps=500000, warmup_steps=1000, max_lr=0.05, min_lr=0.001, last_epoch = -1, verbose = False):
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [base_lr*min(self.min_lr/math.sqrt(max(self.last_epoch, self.warmup_steps)/float(self.max_steps)), self.max_lr) for base_lr in self.base_lrs]
    
