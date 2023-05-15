import torch
import torch.nn as nn
from block_recurrent_transformer.components import *
from random import random
import block_recurrent_transformer.utils as utils
import torch.nn.functional as F
def divisible_by(numer, denom):
    return (numer % denom) == 0

class BlockRecurrentTransformer(nn.Module):
    def __init__(self,
                 seq_size,#必须为block_size整倍数
                 block_size,
                 window_size,
                 d_input,
                 d_ff,
                 head_num,
                 head_dim,
                 state_num,
                 dropout,
                 transformer_num,
                 word_num,
                 device):
        super().__init__()
        rotary_pos_emb,xpos_scale = utils.RotaryEmbedding(head_dim*head_num)(block_size*2)

        TransformerLayer_kwargs = dict(
            rotary_pos_emb=rotary_pos_emb,
            xpos_scale=xpos_scale,
            d_input=d_input,
            d_ff=d_ff,
            head_num=head_num,
            head_dim=head_dim,
            block_size=block_size,
            window_size=window_size,
            dropout=dropout,
            device=device
        )
        BlockRecurrentTransformerLayer_kwargs = dict(
            rotary_pos_emb=rotary_pos_emb,
            xpos_scale=xpos_scale,
            state_num=state_num,
            d_input=d_input,
            head_num=head_num,
            head_dim=head_dim,
            block_size=block_size,
            window_size=window_size,
            dropout=dropout,
            device=device
        )
        self.device = device
        self.model = nn.Sequential(*([TransformerLayer(**TransformerLayer_kwargs) for _ in range(transformer_num)]
                                   + [BlockRecurrentTransformerLayer(**BlockRecurrentTransformerLayer_kwargs)]
                                     + [TransformerLayer(**TransformerLayer_kwargs)]
                                     + [nn.Linear(d_input, word_num)])).to(device)
        self.embed = nn.Embedding(word_num, d_input).to(device)

    def forward(self, input):
        input = input.to(self.device)
        input_feature = self.embed(input)
        output = self.model(input_feature)
        output = output.transpose(1,2)
        return output

