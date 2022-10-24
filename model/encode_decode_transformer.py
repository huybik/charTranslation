import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
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
    device = 'cuda'

    
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class SelfAttention(nn.Module):
    '''
    self attention mechanism with querry, key, value analogy. 
    '''
    def __init__(self, config):
        super().__init__()
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
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.sequence_len, config.sequence_len)).bool()
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
        if pad_mask is not None and causal is not None:
            full_mask = pad_mask & self.causal_mask # b,1,1,T & 1,1,T,T -> b,1,T,T
            att = att.masked_fill(full_mask == 0, -1e9)
        elif pad_mask is not None:
            att = att.masked_fill(pad_mask == 0, -1e9)
        elif causal is not None:
            att = att.masked_fill(self.causal_mask == 0, -1e9)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # B,nh,T,T x B,Nh,T,dk -> B,nh,T,dk
        # swap n_head back to last dimension then re-assemble side by side to embed_dim
        y = y.transpose(1,2).contiguous().view(batch_size, sequence_len, n_head*dk) 
        # output projection
        y = self.resid_drop(self.proj(y)) # output is batch_size, sequence_len, embed_dim
        
        return y

class EncoderBlock(nn.Module):
    '''
    combining layer norm, self attention, feed forward and skip connection (+) in between
    '''
    def __init__(self, config):
        super().__init__()
        embed_dim, n_head = config.embed_dim, config.n_head
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(config)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
                            nn.Linear(embed_dim,4*embed_dim),
                            nn.GELU(),
                            nn.Linear(4*embed_dim,embed_dim),
                            nn.Dropout(config.resid_pdrop),
                            )
        
    def forward(self, x, pad_mask=None):
        x_norm = self.layernorm1(x) # layer norm
        # self attention + skip connection
        x = x + self.attention(x_norm, x_norm, x_norm, pad_mask)
        x_norm = self.layernorm2(x)
        # feed forward + skip connection
        x = x + self.feedforward(x_norm)
        
        return x

class DecoderBlock(nn.Module):
    '''
    Raw target is add as input for self attention, which is called teacher-forcing. This
    is masked by causal mask to prevent network from looking ahead. On next attention block,
    encoder input is feed in as key and value, and output of last attention block as querry.
    '''
    def __init__(self, config):
        super().__init__()
        embed_dim, n_head = config.embed_dim, config.n_head
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.attention1 = SelfAttention(config)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.attention2 = SelfAttention(config)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.layernorm4 = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(
                            nn.Linear(embed_dim,4*embed_dim),
                            nn.GELU(),
                            nn.Linear(4*embed_dim,embed_dim),
                            nn.Dropout(config.resid_pdrop),
                            )
        
    def forward(self, x, enc_x, pad_mask=None, causal=True):
        x_norm = self.layernorm1(x)
        x = x + self.attention1(x_norm, x_norm, x_norm, pad_mask, causal)
        x_norm = self.layernorm2(x)
        x_enc_norm = self.layernorm3(enc_x)
        x = x + self.attention2(x_norm, x_enc_norm, x_enc_norm)
        x_norm = self.layernorm4(x)
        x = x + self.feedforward(x_norm)
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(config.sequence_len, config.embed_dim)
        self.drop = nn.Dropout(config.embed_pdrop)

        self.encode_blocks = nn.ModuleList([EncoderBlock(config) for i in range(config.n_block)])
        self.decode_blocks = nn.ModuleList([DecoderBlock(config) for i in range(config.n_block)])

        self.layernorm = nn.LayerNorm(config.embed_dim)
        self.linear = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.embed_dim = config.embed_dim
        self.apply(self._init_weights)
        self.causal = config.causal
        self.device = config.device

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        # TODO: Same or separate embed? how about drop? position embed?
    def get_sequence_len(self):
        return self.sequence_len

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, target=None):
        x = self.encode(idx)
        logits = self.decode(target[:,:-1], x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target[:,1:].reshape(-1), ignore_index=0)
        return logits, loss

    def encode(self, idx):
        batch_size, sequence_len = idx.shape
        pad_mask = (idx!=0).unsqueeze(1).unsqueeze(2) # 0 is padding idx, b,1,1,T
        pos = torch.arange(0, sequence_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        position_embedding = self.position_embedding(pos)
       
        embed = self.embed(idx) # each token map to learnable vector
        x = self.drop(embed + position_embedding)

        for block in self.encode_blocks:
            x = block(x, pad_mask)

        return x

    def decode(self, idx, x_encode):
        batch_size, sequence_len = idx.shape
        pad_mask = (idx!=0).unsqueeze(1).unsqueeze(2) # 0 is padding idx, b,1,1,T
        pos = torch.arange(0, sequence_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        position_embedding = self.position_embedding(pos)

        embed = self.embed(idx)
        y = self.drop(embed + position_embedding)

        for block in self.decode_blocks:
            y = block(y, x_encode, pad_mask)
        y = self.layernorm(y)
        logits = self.linear(y)
        

        return logits


    def generate_output(self, samples, dataset, temperature=1, top_k=None, steps = 1000, print_process=None):
        '''
        Generate target translations given samples. Here beam search is added to find top-k
        sequence with lowest score.
        '''
        import numpy as np
        assert samples is not None, "Test samples not given!"
            
        self.eval()
        with torch.no_grad():
            out = list()
            if print_process is not None:
                samples = tqdm(samples)
            for sample in samples:
                if dataset.encoder is None:
                    indx = dataset.padding([dataset.ch2i[ch] for ch in sample] + [dataset.ch2i['<eos>']])
                else:
                    indx = dataset.padding(dataset.encoder.encode(sample).ids + [dataset.ch2i['<eos>']])
                indx = torch.tensor(indx, dtype=torch.long).unsqueeze(0).to(self.device)
                x = self.encode(indx)
                beam = BeamSearch(self, x, top_k, dataset.ch2i['<sos>'], dataset.ch2i['<eos>'])
                
                for i in range(0,dataset.sequence_len):

                    status = beam.advance()
                    if status is None: # cant advance anymore
                        break
                # calcuate new score by dividing seq len
                all_candidates = [(seq, score/len(seq)) for seq, score in beam.output]
                sequences = sorted(all_candidates, key = lambda tup: tup[1])[:1]
                for seq, score in sequences:
                    if dataset.encoder is None:
                        out.append(''.join([dataset.i2ch[idx] for idx in seq if (idx not in 
                        [dataset.ch2i['<sos>'],dataset.ch2i['<eos>'],dataset.ch2i['<pad>']])]))
                    else:
                        out.append(''.join(dataset.encoder.decode(seq, skip_special_tokens=True)))

        self.train()
        
        return out

class BeamSearch:
    def __init__(self, model, x_encode, beam_width, BOS, EOS):
        '''
        My implementation of beam search algorithm. 
        At each step there will be vocab_size candidates that the network output as 
        probability score. 
        All candidates is added to current sequence, along with probability of output 
        that accumulate into score. 
        All sequences then being ranked, and top n sequences are select for keeping, 
        discard the rest.
        '''
        self.model = model
        self.k = beam_width
        self.current_location = 0
        self.sequence_len = x_encode.shape[1]
        self.EOS = EOS

        self.x = x_encode
        self.output = list()
        
        self.sequences = [[[BOS], 0.0]]

    def advance(self):
        num_seq = len(self.sequences)
        x = self.x.clone().expand(num_seq,-1,-1)

        # stack sequences and paste to zeros tensor, ready as input to network
        temp = torch.stack([torch.tensor(seq) for seq,score in self.sequences], dim=0) # get the sequences

        padded = torch.zeros((num_seq, self.sequence_len), dtype=torch.long, device=self.model.device)
        padded[:, :self.current_location+1] = temp # [k,l]

        y = self.model.decode(padded, x)[:,self.current_location,:] # [k,h]

        # add n output of netowrk to current sequences, and re-rank them
        all_candidates = list()
        for i in range(num_seq):
            softmax = torch.softmax(y[i],dim=-1)
            v , idx = torch.topk(softmax,self.k)
            seq, score = self.sequences[i]
            for j in range(len(v)):
                candidate = [seq + [idx[j].item()], score - v[j].item()] # higher probability means lower score
                all_candidates.append(candidate)

        ordered = sorted(all_candidates, key = lambda tup: tup[1])
        self.sequences = ordered[:self.k] # get k best sequence base on lowest score

        # if a sequence stop at EOS char, move that sequence to output
        i = 0 
        while i < len(self.sequences):
            seq, score = self.sequences[i]            
            if seq[-1] == self.EOS:
                self.output.append(self.sequences[i])
                self.sequences.pop(i)
            else: i+=1
        self.current_location += 1

        # if all the sequences are moved to output, stop searching
        return None if len(self.sequences) == 0 else True
            
