import numpy as np
from sklearn.metrics import accuracy_score
import torch
import pandas as pd
import pickle
from .poolings import SWE, FSP, GeM, Cov
import os
import faiss
from scipy import stats
from tqdm import tqdm
import time
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints


class Experiment():
    def __init__(self, dataset, pooling, ann,
                 k=9, project=False, projector='pca', n_components=2, ref_size=None, code_length=None, random_state=0, mode='test', **kwargs):
        self.dim_dict = {'point_mnist': 2, 'oxford': 512, 'modelnet40': 3}
        self.projector_dict = {'pca': PCA(n_components=n_components),
                               'kernel_pca': KernelPCA(n_components=n_components, kernel='cosine')}
        self.dataset_name = dataset
        self.pooling_name = pooling
        self.ann_name = ann
        self.k = k
        self.code_length = code_length
        self.state = random_state
        self.exp_report = {'dataset': dataset,
                           'pooling': pooling,
                           'ann': ann,
                           'k': k,
                           'code_length': code_length}

        self.mode = mode
        self.data = self.load_dataset()
        self.ref_size = ref_size
        self.n_components = n_components if project else self.dim_dict[self.dataset_name]
        self.project = project
        self.projector = self.projector_dict[projector]
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
        if self.ann_name in ['faiss-lsh', 'faiss-exact']:
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
        assert self.ann_name in ['faiss-lsh', 'faiss-exact']
        if self.ann_name == 'faiss-lsh':
            emb = self.embedding
            d = emb.shape[1]
            n_bits = self.code_length * 2
            ann = faiss.IndexLSH(d, n_bits)
            ann.add(emb)
        elif self.ann_name == 'faiss-exact':
            emb = self.embedding
            d = emb.shape[1]
            ann = faiss.IndexFlatL2(d)
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
            
        elif self.dataset_name == 'modelnet40':
            if os.path.exists('../modelnet40/train_test.pkl'):
                with open('../modelnet40/train_test.pkl', 'rb') as f:
                    processed = pickle.load(f)
                X_train, X_test, y_train, y_test = processed['x_train'], processed['y_train'], processed['x_test'], processed['y_test']

            else:
                transforms = Compose([SamplePoints(1024),RandomRotate((-45,45), axis=2)])
                data_train_ = ModelNet(root='../modelnet', name='40', train=True, transform=transforms)
                data_test_ = ModelNet(root='../modelnet', name='40', train=False, transform=transforms)

                mean = 512
                sigma = 64

                np.random.seed(self.state)
                train_num_points = np.floor(sigma * np.random.randn(len(data_train_)) + mean).astype(int)
                test_num_points = np.floor(sigma * np.random.randn(len(data_test_)) + mean).astype(int)

                X_train_ = [data_train_[i].pos.numpy()[:train_num_points[i]] for i in range(len(data_train_))]
                y_train = np.array([data_train_[i].y.numpy() for i in range(len(data_train_))]).squeeze()
                X_test_ = [data_test_[i].pos.numpy()[:test_num_points[i]] for i in range(len(data_test_))]
                y_test = np.array([data_test_[i].y.numpy() for i in range(len(data_test_))]).squeeze()

                def normalize(data):
                    normalized = []
                    for sample in data:
                        sample_min = sample.min(0)
                        sample_max = sample.max()
                        normalized.append((sample - sample_min) / sample_max)
                    return normalized

                X_train = normalize(X_train_)
                X_test = normalize(X_test_)

                with open('../modelnet40/train_test.pkl', 'wb') as f:
                    processed = {'x_train': X_train, 'y_train': y_train, 'x_test': X_test, 'y_test': y_test}
                    pickle.dump(processed, f)

        if self.mode == 'validation':
            print('validation mode...')
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=self.state)
            print(X_train.shape, X_test.shape)

        return {'x_train': X_train, 'y_train': y_train, 'x_test': X_test, 'y_test': y_test}

    def init_pooling(self, **kwargs):
        assert self.pooling_name in ['swe', 'fs', 'cov', 'gem'], f'unknown pooling {self.pooling_name}'
        if self.pooling_name == 'swe':
            assert 'num_slices' in kwargs.keys(), 'keyword argument num_slices should be provided'
            ref = self.init_reference()
            pooling = SWE(ref, kwargs['num_slices'], random_state=self.state)
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
    
    def init_reference(self):
        if self.pooling_name == 'swe':
            ref = torch.ones(self.ref_size, self.n_components).to(torch.float)

        elif self.pooling_name == 'fs':
            ref = torch.ones(self.ref_size, self.n_components).to(torch.float)
        
        return ref

    def fit_projector(self):
        if self.dataset_name == 'oxford':
            print('preprocess samples...')
            processed_samples_lst = self.preprocess_samples('base')

            np.random.seed(self.state)
            sample_ind = np.random.permutation(len(processed_samples_lst))[:5000]
            selected_samples_lst = [processed_samples_lst[i] for i in sample_ind]

            np.random.seed(self.state)
            downsampled_samples = []
            for sample in tqdm(selected_samples_lst):
                ind = np.random.permutation(sample.shape[0])[:1]
                downsampled_samples.append(sample[ind, :])
            downsampled_samples = np.concatenate(downsampled_samples, axis=0)
            self.projector.fit(downsampled_samples)
            # print(self.projector.explained_variance_ratio_)
        
    def load_embedding(self, target):
        if self.pooling_name == 'gem':
            emb_dir = f'results/cached_emb/{self.dataset_name}_{self.ann_name}_{self.pooling_name}_{self.power}.npy'
        elif self.pooling_name == 'cov':
            emb_dir = f'results/cached_emb/{self.dataset_name}_{self.ann_name}_{self.pooling_name}.npy'
        elif self.pooling_name == 'fs':
            emb_dir = f'results/cached_emb/{self.dataset_name}_{self.ann_name}_{self.pooling_name}_{self.ref_size}.npy'
        elif self.pooling_name == 'swe':
            emb_dir = f'results/cached_emb/{self.dataset_name}_{self.ann_name}_{self.pooling_name}_{self.ref_size}.npy'

        if not os.path.exists(emb_dir) or self.mode != 'test':
            out_dir = os.path.dirname(emb_dir)
            os.makedirs(out_dir, exist_ok=True)
            emb, time_passed = self.compute_embedding(target)
            np.save(emb_dir, emb)
        else:
            print('loading cached base embedding...')
            emb = np.load(emb_dir)
            
        return emb

    def compute_embedding(self, target):
        if target == 'base' and self.project:
            self.fit_projector()
        pooling = self.pooling

        print(f'compute {target} embedding...')
        start = time.time()
        if self.pooling_name in ['swe', 'fs', 'gem', 'cov']:
            preprocess_samples = self.preprocess_samples(target)
            if self.dataset_name == 'oxford' and self.project:
                samples = []
                for sample in tqdm(preprocess_samples):
                    projected = self.projector.transform(sample)
                    samples.append(torch.from_numpy(projected).to(torch.float))
            else:
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
        elif self.dataset_name in ['oxford', 'modelnet40']:
            samples = X
        
        return samples