import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import torch
import pandas as pd
import pickle
from .poolings import SWE, FSP, GeM, Cov
import os
import faiss
from scipy import stats
from tqdm import tqdm
import time


class Experiment():
    def __init__(self, dataset, pooling, ann, k=9, ref_size=None, code_length=None, random_state=0, **kwargs):
        self.dataset_name = dataset
        self.pooling_name = pooling
        self.ann_name = ann
        self.k = k
        self.code_length = code_length
        self.state = random_state
        self.exp_report = {'dataset': dataset,
                           'pooling': pooling,
                           'ann': ann,
                           'code_length': code_length}
        
        self.data = self.load_dataset()
        self.ref_size = ref_size
        self.pooling = self.init_pooling(**kwargs)
        self.embedding = self.load_embedding('base')
        self.ann = self.train_ann()

    def get_exp_report(self):
        return self.exp_report

    def test(self):
        test_emb, emb_time = self.compute_embedding('query')
        self.exp_report['emb_time_per_sample'] = emb_time
        ann = self.ann

        start = time.time()
        if self.ann_name == 'faiss-lsh':
            D, I = ann.search(test_emb, self.k)

            labels = self.data['y_train']
            knn_labels = labels[I]

        passed = time.time() - start
        self.exp_report['inf_time_per_sample'] = passed / test_emb.shape[0]

        preds = stats.mode(knn_labels.T)[0].squeeze()
        gts = self.data['y_test']
        precision_k = (knn_labels == gts[:, None]).mean()
        acc = accuracy_score(gts, preds)

        self.exp_report['acc'] = acc
        self.exp_report['precision_k'] = precision_k

        return acc

    def train_ann(self):
        assert self.ann_name in ['faiss-lsh', 'annoy']
        if self.ann_name == 'faiss-lsh':
            emb = self.embedding
            d = emb.shape[1]
            n_bits = self.code_length * 2
            ann = faiss.IndexLSH(d, n_bits)
            ann.train(emb)
            ann.add(emb)
            
        return ann
        
    def load_dataset(self):
        assert self.dataset_name in ['point_mnist', 'modelnet40', 'oxford'], f'unknown dataset {self.dataset_name}'
        print('loading dataset...')
        if self.dataset_name == 'point_mnist':
            df_train = pd.read_csv("../dataset/pointcloud_mnist_2d/train.csv")
            df_test = pd.read_csv("../dataset/pointcloud_mnist_2d/test.csv")
            
            X_train = df_train[df_train.columns[1:]].to_numpy()
            y_train = df_train[df_train.columns[0]].to_numpy()
            X_train = X_train.reshape(X_train.shape[0], -1, 3)
            
            X_test = df_test[df_test.columns[1:]].to_numpy()
            y_test = df_test[df_test.columns[0]].to_numpy()
            X_test = X_test.reshape(X_test.shape[0], -1, 3)

        elif self.dataset_name == 'oxford':
            with open('../dataset/oxford/train_test.pkl', 'rb') as f:
                data = pickle.load(f)

            X_train, y_train, X_test, y_test, classnames = data
            y_train = np.array(y_train)
            y_test = np.array(y_test)

        return {'x_train': X_train, 'y_train': y_train, 'x_test': X_test, 'y_test': y_test}

    def init_pooling(self, **kwargs):
        assert self.pooling_name in ['swe', 'fs', 'cov', 'gem'], f'unknown pooling {self.pooling_name}'
        if self.pooling_name == 'swe':
            assert 'num_slices' in kwargs.keys(), 'keyword argument num_slices should be provided'
            ref = self.init_reference(num_slices=kwargs['num_slices'])
            pooling = SWE(ref, kwargs['num_slices'])
        elif self.pooling_name == 'fs':
            ref = self.init_reference()
            pooling = FSP(ref, self.ref_size)
        elif self.pooling_name == 'gem':
            assert 'power' in kwargs.keys(), 'keyword argument power should be provided'
            self.power = kwargs['power']
            self.exp_report['power'] = kwargs['power']
            pooling = GeM(kwargs['power'])
        elif self.pooling_name == 'cov':
            pooling = Cov()
            
        return pooling
    
    def init_reference(self, **kwargs):
        if self.pooling_name == 'swe':
            if self.dataset_name in ['point_mnist', 'oxford']:
                print('preprocess samples...')
                processed_samples_lst = self.preprocess_samples('base')
                processed_samples = np.concatenate(processed_samples_lst, axis=0)
                np.random.seed(self.state)
                ind = np.random.permutation(processed_samples.shape[0])[:10000]

                print('compute reference...')
                n_clusters = self.ref_size // kwargs['num_slices']
                kmeans = KMeans(
                    n_clusters=n_clusters, random_state=self.state
                ).fit(processed_samples[ind])
                ref = torch.from_numpy(kmeans.cluster_centers_).to(torch.float)

        elif self.pooling_name == 'fs':
            if self.dataset_name == 'point_mnist':
                ref = torch.ones(self.ref_size // 2, 2).to(torch.float)
            elif self.dataset_name == 'oxford':
                ref = torch.ones(self.ref_size // 512, 512).to(torch.float)
        
        return ref
        
    def load_embedding(self, target):
        if self.pooling_name == 'gem':
            emb_dir = f'results/cached_emb/{self.dataset_name}_{self.ann_name}_{self.pooling_name}_{self.power}.npy'
        else:
            emb_dir = f'results/cached_emb/{self.dataset_name}_{self.ann_name}_{self.pooling_name}_{self.ref_size}.npy'
        if not os.path.exists(emb_dir):
            out_dir = os.path.dirname(emb_dir)
            os.makedirs(out_dir, exist_ok=True)
            emb, time_passed = self.compute_embedding(target)
            # np.save(emb_dir, emb)
        else:
            print('loading cached base embedding...')
            emb = np.load(emb_dir)
            
        return emb

    def compute_embedding(self, target):
        pooling = self.pooling

        print(f'compute {target} embedding...')
        start = time.time()
        if self.pooling_name in ['swe', 'fs', 'gem', 'cov']:
            samples = [torch.from_numpy(sample).to(torch.float) for sample in self.preprocess_samples(target)]
            embs = []
            for sample in tqdm(samples):
                v = pooling.embedd(sample)
                embs.append(v)

            embs = torch.stack(embs, dim=0)
            embs = embs.reshape(embs.shape[0], -1).numpy()

        passed = time.time() - start
        return embs, passed / len(samples)

    def preprocess_samples(self, target):
        if target == 'base':
            X = self.data['x_train']
        elif target == 'query':
            X = self.data['x_test']
        
        if self.dataset_name == 'point_mnist':
            samples = []
            for i in range(X.shape[0]):
                sample = X[i, :, :]
                sample = sample[sample[:, 2] > 0][:, :2]
                samples.append(sample)
        elif self.dataset_name == 'oxford':
            samples = X
        
        return samples