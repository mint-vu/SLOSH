# demo_for _pointmnist
[demo_for_pointmnist.ipynb](https://github.com/mint-vu/SLOSH/blob/main/notebooks/demo_for_pointmnist.ipynb): This notebook is a demo of the retrieval results of the SLOSH against the baselines on the pointcloud mnist 2d dataset. For all methods, we use a hash code length of 1024, and report the results for $k=4,8,16$. For SLOSH and FSPool, the size of the reference set is chosen to be the median of sizes of the input sets.

## dataset used: 
* [pointcloud mnist 2d](https://www.kaggle.com/cristiangarcia/pointcloudmnist2d): put downloaded data in ```/dataset/pointcloud_mnist_2d```

## structure:
* SLOSH: Set Locality Sensitive Hashing via Sliced Wasserstein Embedding. The number of slices is selected to be 16 based on cross validation. The reference is a set of random numbers from a uniform distribution on the interval $\[0,1\)$.
* Baselines:
    - FSPool: Featurewise Sort Pooling.
    - Cov: Covariance Pooling.
    - GeM-1: Generalized-Mean Pooling for power=1(average pooling). 
    - GeM-2: Generalized-Mean Pooling for power=2.  
    - GeM-4: Generalized-Mean Pooling for power=4.  



