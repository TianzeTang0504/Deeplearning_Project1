import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

test_file = "./data/deep-learning-spring-2025-project-1/cifar-10-python/cifar-10-batches-py/test_batch"
test = unpickle(test_file)

# 重新调整数据形状: (50000, 3072) → (50000, 32, 32, 3)
images = test[b'data'].reshape(-1, 3, 32, 32)

# 保存转换后的数据
np.save("test_images.npy", images)   # 保存图像数据
np.save("test_labels.npy", test[b'labels'])  # 保存标签