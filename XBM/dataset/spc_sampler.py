import random
import torch
import numpy as np
import collections


class SPC_Sampler(torch.utils.data.sampler.Sampler):
    def __init__(self, labels, batch_size, samples_per_class):
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class

        self.length = len(labels) // batch_size
        self.classes_to_indices = get_classes_to_indices(labels)
        self.classes = list(self.classes_to_indices.keys())
    
    def __iter__(self):
        for _ in range(self.length):
            batch = []
            for _ in range(self.batch_size // self.samples_per_class):
                # Sample random class
                sample_class = random.choice(self.classes)

                # Sample random samples from specific class
                batch.extend(random.choices(self.classes_to_indices[sample_class], k=self.samples_per_class))
            yield batch

    def __len__(self):
        return self.length

def get_classes_to_indices(labels):
    classes_to_indices = collections.defaultdict(list)
    for i, label in enumerate(labels):
        classes_to_indices[label].append(i)
    return classes_to_indices
