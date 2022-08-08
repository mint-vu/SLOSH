# SLOSH

### Set Locality Sensitive Hashing via Sliced Wasserstein Embedding

## Datasets: 
- [Point Cloud MNIST 2d](https://www.kaggle.com/cristiangarcia/pointcloudmnist2d): put downloaded data in ```/dataset/pointcloud_mnist_2d```
- [ModelNet40](https://modelnet.cs.princeton.edu/): put downloaded data in ```/dataset/modelnet```
- [Oxford 5K](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/): put the extracted 8-dimensional features ```train_test_AE8.pkl``` in ```/dataset/oxford/```

## Baselines:
  - WE: Wasserstein Embedding
  - FSPool: Featurewise Sort Pooling.
  - Cov: Covariance Pooling.
  - GeM-1: Generalized-Mean Pooling for power=1(average pooling). 
  - GeM-2: Generalized-Mean Pooling for power=2.  
  - GeM-4: Generalized-Mean Pooling for power=4.  

## Code:
- [```experiments.ipynb```](./notebooks/experiments.ipynb): notebook to reproduce results in table 1 and wall-clock analysis in our paper.
- [```sensitivity_to_code_length.py```](./notebooks/sensitivity_to_code_length.py): scripts to perform sensitivity analysis on hash code length.
- [```sensitivity_to_num_slices.py```](./notebooks/sensitivity_to_num_slices.py): scripts to perform sensitivity analysis on the number of slices used in SLOSH.
