import torch
import torch.nn as nn
import torch.nn.functional as F


class XBM(nn.Module):
    def __init__(self, size, embed_dim):
        super(XBM, self).__init__()
        self.size = size

        # Create memory bank, modified from 
        # https://github.com/facebookresearch/moco/blob/master/moco/builder.py
        self.register_buffer("embed_queue", torch.randn(size, embed_dim))
        self.embed_queue = F.normalize(self.embed_queue, dim=0)
        self.register_buffer("label_queue", torch.zeros(size, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.is_full = False
    
    @torch.no_grad()
    def update(self, embeddings, labels):
        batch_size = embeddings.shape[0]
        ptr = int(self.queue_ptr)

        assert self.size % batch_size == 0

        # Enqueue
        self.embed_queue[ptr:ptr+batch_size, :] = embeddings
        self.label_queue[ptr:ptr+batch_size] = labels
        if ptr + batch_size >= self.size:
            self.is_full = True
        ptr = (ptr + batch_size) % self.size

        self.queue_ptr[0] = ptr

    def get(self):
        if self.is_full:
            return self.embed_queue, self.label_queue
        else:
            return self.embed_queue[:self.queue_ptr], self.label_queue[:self.queue_ptr]
