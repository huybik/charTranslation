from tqdm import tqdm
import math
import logging

import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils.utils import pickle

logger = logging.getLogger(__name__)


class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    ckpt_n_print_iter = None # print every n iter
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
#     final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    # device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, config, test_dataset, collate):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.collate = collate
        # take over whatever gpus are on the system
        self.device = config.device
        self.min_loss = 1e10

        if self.device=='cuda':
#             self.device = torch.cuda.current_device()
            self.model.to(self.device)
#             self.model = torch.nn.DataParallel(self.model).to(self.device)
        

    def train(self):
        model, config = self.model, self.config

        # optimizer = model.configure_optimizers(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas)
        

        losses = []
        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                batch_size=config.batch_size, 
                collate_fn=self.collate,
                drop_last=True,
                num_workers=config.num_workers)

            final_token = len(loader)*config.batch_size
            loss_smooth = 0
            
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x,y) in pbar:
                x,y = x.to(self.device), y.to(self.device)
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x,y)
                    loss = loss.mean()
                    losses.append(loss.item())
                    
                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    
                    if config.lr_decay:

                        self.tokens += config.batch_size # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, final_token - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        # print(lr_mult, self.tokens, config.warmup_tokens)
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    loss_smooth = 0.9*loss_smooth + 0.1*loss.item()
#                     accuracy = 0.9*accuracy + 0.1*torch.sum(torch.argmax(b,axis=-1)==Y_batch)/len(A)
                    pbar.set_description(f"epoch: {epoch+1} | train loss: {loss_smooth:.5f}  | lr: {lr:e}") # | Accuracy:{accuracy:.5f}
                
                if not is_train:
                    test_loss = float(np.mean(losses))
                    logger.info("test loss: %f", test_loss)
                    return test_loss

                if config.ckpt_n_print_iter is not None:
                    if it % config.ckpt_n_print_iter == 0:
                        print(model.generate_output(None, data, top_k=3, temperature=0.5))
                        if config.ckpt_path is not None:
                            if loss.item() < self.min_loss:
                                self.min_loss = loss.item()
                                pickle(config.ckpt_path, model.state_dict()) # save


                        
        best_loss = float('inf')
        self.tokens = 0
        for epoch in range(config.max_epochs):
            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
