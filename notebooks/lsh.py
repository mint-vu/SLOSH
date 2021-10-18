import os
import numpy as np
from collections import Counter, defaultdict
import torch


class HashTable:
    def __init__(self, k, feature_dim):
        self.hash_size = k
        self.feature_dim = feature_dim
        self.projections = torch.randn(k, feature_dim)
  
    def generate_hash(self, inputs):
        digits = (self.projections @ inputs  > 0).long()
        return ''.join([str(i.item()) for i in digits])
    
    
class LSH:
    def __init__(self, k, feature_dim, N):
        self.hash_size = k
        self.feature_dim = feature_dim
        self.num_hash_table = N
        self.hash_tables = [HashTable(k, feature_dim) for _ in range(N)]
        self.hashcode_label_dicts = []
        self.hashcode_data_dicts = []

    def train(self, train_x, train_y):
        for hash_table in self.hash_tables:
            hashcodes_label_dict = defaultdict(list)
            hashcodes_sample_dict = defaultdict(list)
            for i in range(train_x.shape[0]):
                label = train_y[i]
                hashcode = hash_table.generate_hash(train_x[i, :].unsqueeze(1))
                hashcodes_label_dict[hashcode].append(label)
                hashcodes_sample_dict[hashcode].append(train_x[i, :].unsqueeze(1))
            self.hashcode_label_dicts.append(hashcodes_label_dict)
            self.hashcode_data_dicts.append(hashcodes_sample_dict)

    def predict(self, sample_x):
        preds = []
        for i, hash_table in enumerate(self.hash_tables):
            sample_hashcode = hash_table.generate_hash(sample_x)
            cand_mat = torch.cat(self.hashcode_data_dicts[i][sample_hashcode], dim=1)
            dist_mat = ((cand_mat - sample_x) ** 2).mean(0)
            nn_idx = torch.argmin(dist_mat).item()
            preds.append(self.hashcode_label_dicts[i][sample_hashcode][nn_idx])
        return Counter(preds).most_common(1)[0][0]