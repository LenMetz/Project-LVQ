
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from kglvq import Glvq
import numpy as np

from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs

n_samples = 60
n_class = 2 #how many clusters to make , aka class numbers

samples, labels = make_blobs(n_samples=n_samples,# X = returned generated samples, y = The integer labels for each generated sample.
                       centers=n_class,
                       cluster_std=0.5, #how tight the cliuster is 
                       random_state=0) 
#print(repr(samples))
#print(repr(labels)) #repr output array with commas 


input_data = samples.copy() #make it static
data_label = labels.copy()

prototype_per_class = 2
epochs = 100#iterations
learning_rate = 0.01 #也叫升级公式中的gain factor, 影响prototype 的动量, paper 中是0.001

glvq = Glvq()

#inputsample_cut1, inputsample_cut2, sample_cut1_labels, sample_cut2_labels = train_test_split(input_data,
#                                                    data_label,
#                                                    test_size=0.3,
#                                                    random_state=0)
#从数据中抽样


glvq.fit(input_data, data_label, n_class, prototype_per_class ,learning_rate, epochs)


