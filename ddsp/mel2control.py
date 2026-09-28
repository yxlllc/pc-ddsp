import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from .lynxnet2 import LYNXNet2Block


def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))


class Mel2Control(nn.Module):
    def __init__(
            self,
            n_mels,
            block_size,
            output_splits):
        super().__init__()
        self.output_splits = output_splits        
        self.mel_emb = nn.Linear(n_mels, 256)      
        self.stack = nn.Sequential(
                weight_norm(nn.Conv1d(2 * block_size, 512, 3, 1, 1)),
                nn.PReLU(num_parameters=512),
                weight_norm(nn.Conv1d(512, 256, 3, 1, 1)))
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2Block(
                    dim=256,
                    expansion_factor=1,
                    kernel_size=31,
                    glu_type='softsign_glu'
                )
                for i in range(3)
            ]
        )
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(nn.Linear(256, self.n_out))

    def forward(self, mel, source, noise):
        
        '''
        input: 
            B x n_frames x n_mels
        return: 
            dict of B x n_frames x feat
        '''
        exciter = torch.cat((source, noise), dim=-1).transpose(1,2)
        x = self.mel_emb(mel) + self.stack(exciter).transpose(1,2)
        for layer in self.residual_layers:
            x = layer(x)
        x = F.rms_norm(x, (x.size(-1), ))
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
    
        return controls 

