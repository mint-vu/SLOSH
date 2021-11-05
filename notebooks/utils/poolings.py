import numpy as np
import torch
from .torchinterp import Interp1d


class SWE():
    def __init__(self, ref, L, random_state=0):
        self.M, self.dim = ref.shape
        self.ref = ref
        self.num_slices = L          
        self.state = random_state
        self.theta = self.generate_theta(self.dim, L)
        self.sliced_ref = self.slicer(self.ref)
        self.sliced_ref_sorted, self.sliced_ref_sort_ind = torch.sort(self.sliced_ref, 0)
        self.sliced_ref_cdf = torch.cumsum(torch.ones_like(self.sliced_ref), 0)/self.M

    def generate_theta(self, d, L):
        torch.manual_seed(self.state)
        theta = [th / torch.sqrt(torch.sum((th**2))) for th in torch.randn(L, d)]

        return torch.stack(theta, dim=0) 

    def slicer(self, X):
        if len(self.theta.shape) == 1:
            return torch.matmul(X, self.theta)
        else:
            return torch.matmul(X, self.theta.T)

    def embedd(self, x):
        N,d=x.shape
        assert d==self.dim
        sliced_data=self.slicer(x)
        sliced_data_sorted, sliced_data_index = torch.sort(sliced_data, dim=0)
        sliced_data_cdf = torch.cumsum(torch.ones_like(sliced_data),0)/N
        mongeMap = Interp1d()(sliced_data_cdf.T, sliced_data_sorted.T, self.sliced_ref_cdf.T).T
        for l in range(self.num_slices):
            mongeMap[self.sliced_ref_sort_ind[:,l],l]=torch.clone(mongeMap[:,l])

        embedd=(mongeMap-self.sliced_ref)/self.M
        return embedd
    
    
class FSP():
    def __init__(self, ref, ref_size, reduce=False):
        self.ref = ref
        self.ref_size = ref_size
        self.reduce = reduce
        
    def embedd(self, x):
        ref_size = self.ref_size // x.shape[1]
        ref_domain = torch.linspace(0, 1, ref_size).repeat(x.shape[1], 1)
        x_sorted, _ = torch.sort(x, dim=0, descending=True)

        size = x_sorted.shape[0]
        x_sorted = x_sorted.to(torch.float)

        x_domain = torch.linspace(0, 1, size).repeat(x_sorted.shape[1], 1)
        x_interp = Interp1d()(x_domain, x_sorted.T, ref_domain).T

        dot = x_interp * self.ref

        if self.reduce:
            return dot.mean(0)

        return dot


class GeM():
    def __init__(self, power):
        self.p = power
        
    def embedd(self, x):
        power = self.p
        p_avg = torch.cat([torch.pow(torch.pow(x, p).mean(0), 1/p) for p in range(1, power + 1)])
        return p_avg


class Cov():
    def __init__(self, lam=0.0001):
        self.lam = lam

    def embedd(self, x):
        d = x.shape[1]
        cov = torch.cov(x.T)
        reg = torch.trace(cov) * torch.eye(d)

        cov = (cov + self.lam * reg).reshape(d * d, -1).squeeze()
        return cov

    