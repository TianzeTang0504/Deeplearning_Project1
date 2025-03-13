import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

test_file = "./data/deep-learning-spring-2025-project-1/cifar-10-python/cifar-10-batches-py/test_batch"
test = unpickle(test_file)

images = test[b'data'].reshape(-1, 3, 32, 32)

np.save("test_images.npy", images)
np.save("test_labels.npy", test[b'labels'])
