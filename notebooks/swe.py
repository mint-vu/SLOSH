import torch
from torchinterp1d import Interp1d

class SWE:
    def __call__(self, dataset, reference, L, slicer_fn='linear', random_state=0):
        '''Inputs:
                dataset: input set with N elements
                reference: reference set (empirical distribution) of M samples
                slicer_fn: slicer for the generalized swd
                L: number of slices
        '''
        assert dataset.shape[1]==reference.shape[1]
        self.dim = dataset.shape[1]
        
        self.N = dataset.shape[0]
        self.M = reference.shape[0]
        self.num_slices = L
        if slicer_fn=='linear':
            self.slicer = self.linear_slicer
        else: raise Exception('Slicer function not defined')
        self.state = random_state
        
        return self.sw_embedding(dataset, reference)
    
    def generate_theta(self, d, L):
        '''This function generates a set of L parameters sampled uniformly from
           dim-dimensional unit hypersphere.
           The shape of \theta is L*dim.
        '''
        torch.manual_seed(self.state)
        theta = [th_ / torch.sqrt(torch.sum((th_**2))) for th_ in torch.randn(L, d)]

        return torch.stack(theta, dim=0)
    
    def get_slices(self, X):
        '''This function uses the given slicer and theta to get the slices of the 
           dataset or reference set.
           The shape of slices is N*L.
        '''
        theta = self.generate_theta(self.dim, self.num_slices)
        
        return self.slicer(X, theta)
    
    def interpolation(self, data_sorted, permutation):
        '''In general N != M, in which case the Monge coupling is obtained via linear interpolation*.
           *https://github.com/aliutkus/torchinterp1d
           Inputs:
                data_sorted: the sorted data slices (shape: N*L)
                permutation: the ordering that permutes the sorted set back based on the sorting of 
                             the reference set (shape: M*L)
        '''
        #calculate the pdf of sorted dataset.
        fx = 1 / self.N * torch.ones(self.N, self.num_slices)
        #calculate the cdf of sorted dataset
        Fx = torch.cumsum(fx, dim=0)
        
        #cdf of reference set
        z = 1/self.M*(torch.add(permutation,1))
        #evaluate the inverse of cdf of data_sorted at z, shape of interp is L*M
        interp = Interp1d()(Fx.T, data_sorted.T, z.T)
        
        # return the monge copuling with shape M*L
        return interp.T

    def monge_coupling(self, dataset, reference):
        '''This function calculates the monge coupling of the dataset and the reference set by slicing, sorting and coupling. The dimensionality of the Monge coupling is M*L.
        '''
        sliced_data = self.get_slices(dataset)
        sliced_ref = self.get_slices(reference)
        
        data_sorted,_ = torch.sort(sliced_data, dim=0)
        ref_ind = torch.argsort(sliced_ref, dim=0)
        permutation = torch.argsort(ref_ind, dim=0)
        
        #in case the dataset and reference set have the same cardinality
        if dataset.shape[0] == reference.shape[0]:
            coupling = data_sorted.T[torch.arange(self.num_slices).unsqueeze(1).repeat((1, dataset.shape[0])).flatten(), permutation.T.flatten()].view(self.num_slices, dataset.shape[0]).T
        else: coupling = self.interpolation(data_sorted, permutation)
        
        return coupling
        
    def sw_embedding(self, dataset, reference):
        
        M = torch.tensor(reference.shape[0])
        
        #return the final embedding
        return 1/torch.sqrt(M)*(self.monge_coupling(dataset, reference)-self.get_slices(reference))
    
    def linear_slicer(self, X, theta):
        #g_\theta(x)=\theta^T*x
        if len(theta.shape)==1:
            return torch.matmul(X,theta)
        else:
            return torch.matmul(X,theta.T)