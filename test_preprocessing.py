import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

test_file = "./data/deep-learning-spring-2025-project-1/cifar_test_nolabel.pkl"
test = unpickle(test_file)

test_images = test[b'data']
test_images = test_images.transpose(0, 3, 1, 2)
test_ids = test[b'ids']  # 确保获取 id 信息
np.savez("test_data.npz", images=test_images, ids=test_ids)


# 加载.npy文件
file_path = "test_data.npz"  # 替换为你的.npy文件路径
test_data = np.load(file_path)  # 形状为 (N, 32, 32, 3)
images = test_data["images"]
ids = test_data["ids"]

# 随机选择一张图片
random_index = np.random.randint(0, images.shape[0])  # 生成随机索引
random_image = np.transpose(images[random_index], (1, 2, 0))  # 获取随机图片

# 显示图片
plt.imshow(random_image.astype(np.uint8))  # 确保图片数据类型为 uint8
plt.axis("off")  # 隐藏坐标轴
plt.title(f"Random Image Index: {random_index}")  # 显示索引
plt.show()