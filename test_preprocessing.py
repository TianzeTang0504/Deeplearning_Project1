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
test_ids = test[b'ids']
np.savez("test_data.npz", images=test_images, ids=test_ids)


file_path = "test_data.npz"
test_data = np.load(file_path)
images = test_data["images"]
ids = test_data["ids"]

random_index = np.random.randint(0, images.shape[0])
random_image = np.transpose(images[random_index], (1, 2, 0))

plt.imshow(random_image.astype(np.uint8))
plt.axis("off")
plt.title(f"Random Image Index: {random_index}")
plt.show()
