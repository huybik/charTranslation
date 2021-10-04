import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)
import math

class TransformerConfig:
    vocab_size = 1000
    sequence_len = 128
    n_block = 8
    n_head = 8
    embed_dim = 100
    attn_pdrop = 0.1
    resid_pdrop = 0.1
    embed_pdrop = 0.1
    causal = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def default(val, default_val):
    return val if val is not None else default_val

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
#         embed_dim embed_dim
        assert config.embed_dim % config.n_head == 0
        embed_dim = config.embed_dim
        

        # self attention is followed by layernorm so bias is not required
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        # regularization in the form of dropout
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # concatenate n_head into one output and project
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.sequence_len, config.sequence_len))
                                     .view(1, 1, config.sequence_len, config.sequence_len))
        self.n_head = config.n_head
        self.causal = config.causal

    def forward(self, q, k, v, pad_mask=None, causal=None):
        '''
        split input dimension into n_head for query, key, value. calculate (q.Tk/sqrt(dk)).v

        '''

        batch_size, sequence_len, embed_dim = v.shape 
        n_head = self.n_head
        # last dimension after splitting
        dk = embed_dim//n_head
        
        q = self.query(q).view(batch_size, sequence_len, n_head, dk).transpose(1,2) # B,nh,T,dk
        k = self.key(k).view(batch_size, sequence_len, n_head, dk).transpose(1,2)
        v = self.value(v).view(batch_size, sequence_len, n_head, dk).transpose(1,2)
        
        # scale q and k
        
        att = (q @ k.transpose(-2,-1))/(dk**0.5) # B,nh,T,dk x B,nh,dk,T -> B,nh,T,T
        if pad_mask is not None:
            att = att.masked_fill(pad_mask==0, float('-1e10'))
        if causal is not None:
            att = att.masked_fill(self.causal_mask == 0, float('-1e10'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # B,nh,T,T x B,Nh,T,dk -> B,nh,T,dk
        # swap n_head back to last dimension then re-assemble side by side to embed_dim
        y = y.transpose(1,2).contiguous().view(batch_size, sequence_len, n_head*dk) 
        # output projection
        y = self.resid_drop(self.proj(y)) # output is batch_size, sequence_len, embed_dim
        
        return y

# LinformerSelfattention
class LinformerSelfAttention(nn.Module):
    def __init__(self, config, k = 64, dim_head = None, one_kv_head = False, share_kv = True):
        super().__init__()

        heads = config.n_head
        dropout = config.attn_pdrop
        seq_len = config.sequence_len
        dim = config.embed_dim

        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, querry, key, value, pad_mask=None, **kwargs):
        x = querry
        context = key
        pad_mask = None

        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        if pad_mask is not None:
            dots = dots.masked_fill(pad_mask==0, float('-1e10')) # bn -> bhnk
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim, n_head = config.embed_dim, config.n_head
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.attention = LinformerSelfAttention(config)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
                            nn.Linear(embed_dim,4*embed_dim),
                            nn.GELU(),
                            nn.Linear(4*embed_dim,embed_dim),
                            nn.Dropout(config.resid_pdrop),
                            )
        
    def forward(self, x, pad_mask=None, causal=None):
        x_norm = self.layernorm1(x)
        x = x + self.attention(x_norm, x_norm, x_norm, pad_mask=pad_mask, causal=causal)
        x_norm = self.layernorm2(x)
        x = x + self.feedforward(x_norm)
        
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.drop = nn.Dropout(config.embed_pdrop)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.sequence_len, config.embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(config) for i in range(config.n_block)])
        self.layernorm = nn.LayerNorm(config.embed_dim)
        self.linear = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.embed_dim = config.embed_dim
        
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        self.device = config.device
        self.causal = config.causal


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, idx, targets=None):
        batch_size, sequence_len = idx.shape 
        embed_dim = self.embed_dim

        pad_mask_x = (idx>0).view(batch_size, 1, 1, sequence_len) # 0 is padding idx
        embed = self.embed(idx) # each token map to learnable vector
        position_embedding = self.position_embedding[:,:sequence_len,:]

        x = self.drop(embed + position_embedding)
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask_x, causal=self.causal)

        x = self.layernorm(x)
        
        logits = self.linear(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets[:,1:].reshape(-1))

        return logits, loss

    # def generate_output(self, sample, dataset, temperature=1, top_k=None, steps = 1000):
    #     '''
    #     Generate n samples characters given x prompt
    #     '''
    #     import numpy as np
    #     top_k = 1
    #     if sample is None: sample = 'hôm nay trông bạn thật đẹp!'


    #     self.eval()
    #     with torch.no_grad():
            
    #         device = self.device
    #         vocab_size = dataset.vocab_size

    #         idx = [dataset.ch2i[k] for k in sample]
    #         if len(idx) > dataset.sequence_len:
    #             idx = idx[:dataset.sequence_len]
    #         else:
    #             idx = [0]*(dataset.sequence_len-len(idx)) + idx


    #         x = torch.tensor(idx).to('cuda').unsqueeze(0).long()
           
    #         for i in range(steps - len(x)):
    #             logits, loss = self.forward(x)
    #             logits = logits[:,-1,:]
    #             logits = logits / temperature
    #             v, ix = logits.topk(k=top_k, dim=-1)
    #             logits[logits < v[:,-1]] = -float('inf')
    #             probs = torch.softmax(logits, dim=-1).view(-1)

    #             next_index = torch.multinomial(probs, num_samples=1)

    #             idx += [next_index.item()]
    #             next_index = next_index.unsqueeze(0).long()
                
    #             x = torch.cat((x, next_index), dim=1)
    #             x = x[:,1:]
                
    #         out = ''.join([dataset.i2ch[i] for i in idx])
        
    #     self.train()
        
    #     return out

    def generate_output(self, sample, dataset, temperature=1, top_k=None, steps = 1000):
        '''
        Generate n samples characters given x prompt
        '''
        import numpy as np
        top_k = 1
        if sample is None: sample = 'hôm nay trông bạn thật đẹp!'


        self.eval()
        with torch.no_grad():
            
            device = self.device
            vocab_size = dataset.vocab_size

            idx = [dataset.ch2i[k] for k in sample]
            if len(idx) > dataset.sequence_len:
                idx = idx[:dataset.sequence_len]
            else:
                idx = [0]*(dataset.sequence_len-len(idx)) + idx


            x = torch.tensor(idx).to('cuda').unsqueeze(0).long()
           
            logits, loss = self.forward(x)
            idx = torch.argmax(logits, dim=-1).squeeze().detach().cpu().tolist()
            print(idx)
            out = ''.join([dataset.i2ch[i] for i in idx])
        
        self.train()
        
        return out

