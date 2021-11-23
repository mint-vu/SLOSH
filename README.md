# SLOSH
Set Locality Sensitive Hashing via Sliced Wasserstein Embedding

## datasets used: 
- [pointcloud mnist 2d](https://www.kaggle.com/cristiangarcia/pointcloudmnist2d): put downloaded data in ```/dataset/pointcloud_mnist_2d```
- [ModelNet40](https://modelnet.cs.princeton.edu/): put downloaded data in ```/modelnet```
- [oxford 5K](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/): the extracted 8-dimensional features are put in ```/dataset/oxford/train_test_AE8_v2.pkl```

## notebooks:
-  ```demo_for_pointmnist.ipynb```: demonstrates the retrieval results of the SLOSH and the baselines on the pointcloud mnist 2d dataset.
-  ```experiments_modelnet40.ipynb```: demonstrates the retrieval results of the SLOSH and the baselines on the ModelNet40 dataset.
-  ```modelnet40_normalization.ipynb```: visualizes examples of the normalized data from the ModelNet40 dataset.
-  ```VGG16_Feature_Extractor.ipynb```: contains the pretrained VGG16 as the feature extractor for oxford 5K dataset.
-  ```sensitivity_to_code_length.ipynb```: shows the sensitivity to code length experiment on the pointcloud mnist 2d dataset.

