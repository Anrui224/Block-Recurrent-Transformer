import torch
import torch.nn as nn
import math
import block_recurrent_transformer.utils as utils
from einops import repeat

count = 1
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return nn.functional.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class LSTMGate(nn.Module):
    def __init__(self, input_size, hidden_size, state_length, device=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.state_length = state_length
        self.device = device
        self.sigmoid = nn.Sigmoid().to(self.device)
        self.tanh = nn.Tanh().to(self.device)
        # i_t
        self.W_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(state_length, hidden_size))
        # f_t
        self.W_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(state_length, hidden_size))
        # z_t
        self.W_z = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_z = nn.Parameter(torch.Tensor(state_length, hidden_size))
        # o_t
        self.W_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(state_length, hidden_size))
        self.ones = nn.Parameter(torch.ones(state_length, hidden_size), requires_grad=False)
        # initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, h_t, c_t):
        # print(h_t.shape, self.W_i.shape, self.b_i.shape)
        i_t = self.sigmoid(h_t @ self.W_i + self.b_i - self.ones)
        f_t = self.sigmoid(h_t @ self.W_f + self.b_f + self.ones)
        z_t = self.tanh(h_t @ self.W_z + self.b_z)
        # print(f_t.shape)
        # print(c_t.shape)
        # print(i_t.shape)
        # print(z_t.shape)
        c_t = f_t * c_t + i_t * z_t
        return c_t


class Attention(nn.Module):
    # TODO normalization & pos_embed
    def __init__(self, d_head, dropout, device=None):
        super().__init__()
        self.d_head = d_head
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.device = device

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_head)
        if mask is not None:
            # print(scores.shape, mask.shape)
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = self.softmax(scores)
        p_attn = self.dropout(p_attn)
        
        return torch.matmul(p_attn, v), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head_num, head_dim, dropout=0.1, device=None):
        # TODO d_head
        super().__init__()
        self.d_model = d_model
        self.head_num = head_num
        self.head_dim = head_dim
        self.attn = Attention(self.head_dim, dropout).to(device)
        self.linears = utils.clones(nn.Linear(d_model, d_model), 4).to(device)

    def forward(self, q, k, v, mask=None):
        # print(q.shape)
        # print(self.head_dim)
        batch_size = q.size(0)
        q, k, v = [l(x).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2) for l, x in
                   zip(self.linears, (q, k, v))]
        x, attn = self.attn(q, k, v, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.linears[-1](x), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_input, d_ff, device=None):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            LayerNorm(d_model).to(device),
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_input, bias=False)
        ).to(device)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        output = self.fc(inputs)
        return output  # [batch_size, seq_len, d_model]


class TransformerLayer(nn.Module):
    def __init__(self,
                 d_input,
                 d_ff,
                 head_num,
                 head_dim,
                 block_size,
                 window_size,
                 rotary_pos_emb,
                 xpos_scale,
                 dropout=0.1,
                 device=None, ):
        super().__init__()
        self.norm = LayerNorm(d_input)
        self.rotary_pos_emb = rotary_pos_emb.to(device)
        self.xpos_scale = xpos_scale.to(device)
        self.d_input = d_input
        self.window_size = window_size
        self.block_size = block_size
        self.device = device
        self.d_model = head_num * head_dim
        self.qnorm = LayerNorm(self.d_model)
        self.knorm = LayerNorm(self.d_model)
        self.attn = MultiHeadAttention(d_model=self.d_model, head_num=head_num, head_dim=head_dim, dropout=dropout,
                                       device=device).to(device)
        self.pos_ffn = PoswiseFeedForwardNet(self.d_model, self.d_input, d_ff, device).to(device)
        self.mask = torch.from_numpy(utils.sliding_window_mask(block_size, block_size + window_size, window_size)).to(
            device)
        self.key_cache = None
        self.value_cache = None
        self.K = nn.Linear(d_input, self.d_model, False)
        self.V = nn.Linear(d_input, self.d_model, False)
        self.Q = nn.Linear(d_input, self.d_model, False)

    # TODO clear()
    def clear(self):
        self.key_cache = None
        self.value_cache = None

    def forward(self, input):
        # TODO LayerNorm
        # global count
        # print(count)
        batch_size = input.size(0)
        if self.key_cache is None or self.value_cache is None:
            self.key_cache = torch.zeros((batch_size, self.window_size, self.d_model)).requires_grad_(False).to(self.device)
            self.value_cache = torch.zeros((batch_size, self.window_size, self.d_model)).requires_grad_(False).to(self.device)

        input = self.norm(input)

        k, v, q = self.K(input), self.V(input), self.Q(input)  # batch*block_size*d_model
        k, q = self.knorm(k), self.qnorm(q)
        v = torch.cat([self.value_cache, v], dim=1)
        k = torch.cat([self.key_cache, k], dim=1)
        if input.shape[-2] == self.block_size:
            self.key_cache = k[:, -self.window_size:, :]
            self.value_cache = v[:, -self.window_size:, :]
            self.key_cache = self.key_cache.detach()
            self.value_cache = self.value_cache.detach()
        # print(k.shape, q.shape, v.shape)
        # rotary emb
        q_r = utils.apply_rotary_pos_emb(q, self.rotary_pos_emb, self.xpos_scale, self.device)
        k_r = utils.apply_rotary_pos_emb(k, self.rotary_pos_emb, self.xpos_scale ** -1, self.device)

        output, attn = self.attn(q_r, k_r, v, self.mask)
        output = self.pos_ffn(output)
        # count+=1
        # print(count)
        return output


class BlockRecurrentTransformerLayer(nn.Module):
    def __init__(self,
                 d_input,
                 state_num,
                 head_num,
                 head_dim,
                 block_size,
                 window_size,
                 rotary_pos_emb,
                 xpos_scale,
                 dropout=0.1,
                 device=None):
        super().__init__()
        self.norm = LayerNorm(d_input)
        self.state_norm = LayerNorm(d_input)
        self.rotary_pos_emb = rotary_pos_emb
        self.xpos_scale = xpos_scale
        self.d_model = head_num * head_dim
        self.qnorm = LayerNorm(self.d_model)
        self.knorm = LayerNorm(self.d_model)
        self.block_size = block_size
        self.window_size = window_size
        self.state_num = state_num
        self.attn = MultiHeadAttention(d_model=self.d_model, head_num=head_num, head_dim=head_dim, dropout=dropout)
        self.K_e = nn.Linear(d_input, self.d_model)
        self.V_e = nn.Linear(d_input, self.d_model)
        self.Q_ev = nn.Linear(d_input, self.d_model)
        self.Q_sv = nn.Linear(d_input, self.d_model)
        self.K_s = nn.Linear(d_input, self.d_model)
        self.V_s = nn.Linear(d_input, self.d_model)
        self.Q_sh = nn.Linear(d_input, self.d_model)
        self.Q_eh = nn.Linear(d_input, self.d_model)
        self.linear_v = nn.Linear(2 * self.d_model, d_input)
        self.linear_h = nn.Linear(2 * self.d_model, d_input)
        self.mlp_v = PoswiseFeedForwardNet(d_input, d_input, d_input * 2)
        self.mlp_h = PoswiseFeedForwardNet(d_input, d_input, d_input * 2)
        self.state = nn.Parameter(torch.FloatTensor(state_num, d_input))
        self.state.data.normal_(0, 0.1)
        self.state_c = None
        self.state_pos_ids = nn.Parameter(torch.randn(state_num, d_input))
        self.state_pos_ids_c = None
        self.gate1 = LSTMGate(d_input, d_input, state_num, device)
        self.gate2 = LSTMGate(d_input, d_input, state_num, device)
        self.key_cache = None
        self.value_cache = None
        self.device = device

    def clear(self):
        self.key_cache = None
        self.value_cache = None
        self.state_c = None

    def vertical(self, input):
        # self-attention
        # global count
        # print(count)
        batch_size = input.size(0)
        if self.key_cache is None or self.value_cache is None:
            self.key_cache = torch.zeros((batch_size, self.window_size, self.d_model)).requires_grad_(False).to(self.device)
            self.value_cache = torch.zeros((batch_size, self.window_size, self.d_model)).requires_grad_(False).to(self.device)
        if self.state_c is None:
            self.state_c = self.state.clone().detach()
            self.state_c = repeat(self.state_c, "n d -> b n d", b=batch_size).detach().to(self.device)
            
        if self.state_pos_ids_c is None:
            self.state_pos_ids_c = self.state_pos_ids.clone()
            self.state_pos_ids_c = repeat(self.state_pos_ids_c, "n d -> b n d", b=batch_size).to(self.device)
        input = self.norm(input)
        # print(self.state_c)
        input = input.to(self.device)

        k_self = self.K_e(input)
        v_self = self.V_e(input)
        q_self = self.Q_ev(input)
        
        k_self, q_self = self.knorm(k_self), self.qnorm(q_self)
        
        v_self = torch.cat([self.value_cache, v_self], dim=1)
        k_self = torch.cat([self.key_cache, k_self], dim=1)
        
        if input.shape[-2] == self.block_size:
            self.key_cache = k_self[:, -self.window_size:, :]
            self.value_cache = v_self[:, -self.window_size:, :]
            self.key_cache = self.key_cache.detach()
            self.value_cache = self.value_cache.detach()
        mask_self = torch.from_numpy(
            utils.sliding_window_mask(self.block_size, self.block_size + self.window_size, self.window_size)).to(
            self.device)
        
        # rotary emb
        q_r = utils.apply_rotary_pos_emb(q_self, self.rotary_pos_emb, self.xpos_scale, self.device)
        k_r = utils.apply_rotary_pos_emb(k_self, self.rotary_pos_emb, self.xpos_scale ** -1, self.device)
        
        out_self, attn_self = self.attn(q_r, k_r, v_self, mask_self)
        
        # cross_attention (state IDs need to be implemented)
        # TODO clear() & batch_size*state_vector
        state_c_n = self.state_norm(self.state_c)
        
        k_cross = self.K_s(state_c_n + self.state_pos_ids_c)  # state_num*d_model
        v_cross = self.V_s(state_c_n + self.state_pos_ids_c)
        q_cross = self.Q_sv(input)  # batch_size*block_size*d_model
        
        mask_cross = None
        out_cross, attn_cross = self.attn(q_cross, k_cross, v_cross, mask_cross)
        
        output = torch.cat([out_cross, out_self], dim=-1).to(self.device)
        
        output = self.linear_v(output) + input
        
        output = self.mlp_v(output) + output
        # count+=1
        # print(count)
        return output

    def horizental(self, input):
        input = self.norm(input)
        batch_size = input.size(0)
        # cross_attention (state IDs need to be implemented)
        k_cross = self.K_e(input)  # 4*1024*512
        v_cross = self.V_e(input)  # 4*1024*512
        
        # TODO clear() & batch_size*state_vector & layer_norm
        q_cross = self.Q_eh(self.state_c + self.state_pos_ids_c)  # 512*512
        
        mask = None
        
        out_cross, attn_cross = self.attn(q_cross, k_cross, v_cross, mask)
        
        
        # self-attention (state IDs need to be implemented)
        state_c_n = self.state_norm(self.state_c)
        
        k_self = self.K_s(state_c_n + self.state_pos_ids_c)
        v_self = self.V_s(state_c_n + self.state_pos_ids_c)
        q_self = self.Q_sh(state_c_n + self.state_pos_ids_c)
        
        mask = None
        out_self, attn_self = self.attn(q_self, k_self, v_self, mask)
        

        output = torch.cat([out_self, out_cross], dim=-1).to(self.device)
        
        output = self.linear_h(output)
        
        new_state_c = self.gate1(output, self.state_c)
        new_state_c = self.gate2(self.mlp_h(new_state_c), new_state_c)
        self.state_c = new_state_c
        self.state_c = self.state_c.detach()
        return new_state_c

    def forward(self, input):
        output = self.vertical(input)
        
        if input.shape[-2] == self.block_size:
            self.horizental(input)
        return output














