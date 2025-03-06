import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

# 读取所有 batch 文件
batch_files = ["./data/deep-learning-spring-2025-project-1/cifar-10-python/cifar-10-batches-py/data_batch_1", 
               "./data/deep-learning-spring-2025-project-1/cifar-10-python/cifar-10-batches-py/data_batch_2", 
               "./data/deep-learning-spring-2025-project-1/cifar-10-python/cifar-10-batches-py/data_batch_3", 
               "./data/deep-learning-spring-2025-project-1/cifar-10-python/cifar-10-batches-py/data_batch_4", 
               "./data/deep-learning-spring-2025-project-1/cifar-10-python/cifar-10-batches-py/data_batch_5"
]
# 初始化合并的数据
merged_data = {
    b'data': [],
    b'labels': []
}

# 读取并合并所有 batch
for file in batch_files:
    batch = unpickle(file)
    merged_data[b'data'].append(batch[b'data'])
    merged_data[b'labels'].extend(batch[b'labels'])

# 合并数据
merged_data[b'data'] = np.vstack(merged_data[b'data'])  # (50000, 3072)
merged_data[b'labels'] = np.array(merged_data[b'labels'])  # (50000,)

# 重新调整数据形状: (50000, 3072) → (50000, 32, 32, 3)
images = merged_data[b'data'].reshape(-1, 3, 32, 32)

# 保存转换后的数据
np.save("images.npy", images)   # 保存图像数据
np.save("labels.npy", merged_data[b'labels'])  # 保存标签

print("数据转换完成并保存！")
print("Images shape:", images.shape)  # (50000, 32, 32, 3)
print("Labels shape:", merged_data[b'labels'].shape)  # (50000,)
