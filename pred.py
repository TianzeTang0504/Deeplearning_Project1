import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from model import unet_resnet
from model_resnet import resnet18_custom
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CIFARDataset(Dataset):
    def __init__(self, images, ids, transform=None):
        # images shape: (N, H, W, C)
        self.images = torch.tensor(images, dtype=torch.float32) / 255.0
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.transform = transform
        self.to_pil = ToPILImage()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        id = self.ids[idx]
        if self.transform is not None:
            img = self.to_pil(img)
            img = self.transform(img)
        return img, id


test_data = np.load("test_data.npz")
test_images = test_data["images"]
test_ids = test_data["ids"]

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = CIFARDataset(test_images, test_ids, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

model = resnet18_custom().to(device)
model.load_state_dict(torch.load("res-1.pth", map_location=device))
model.eval()

i = 0
predictions = []
with torch.no_grad():
    for images, ids in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        predictions.append([ids.item(), predicted.item()])
        i += 1


df = pd.DataFrame(predictions, columns=["ID", "Labels"])
df.to_csv("submission.csv", index=False)

print("Saved submission.csv!")


for i in range(1):
    random_index = np.random.randint(0, test_images.shape[0])
    random_index = 18
    random_image = test_images[random_index]
    random_image_plt = random_image
    random_image = random_image.astype(np.float32) / 255.0
    with torch.no_grad():
        image = torch.tensor(random_image).unsqueeze(0).to(device)
        output_class = model(image)

    labels_map = [b'airplane', b'automobile', b'bird', b'cat', b'deer', 
                b'dog', b'frog', b'horse', b'ship', b'truck']

    predicted_label = output_class.argmax(dim=1).item()
    predicted_text = labels_map[predicted_label].decode("utf-8")

    random_image_plt = np.transpose(random_image_plt, (1, 2, 0))
    plt.imshow(random_image_plt.astype(np.uint8))
    plt.axis("off")

    plt.text(random_image_plt.shape[1], random_image_plt.shape[0] // 2, 
            f'Pred: {predicted_text}', fontsize=12, color='red', 
            verticalalignment='center')

    plt.title(f"Random Image Index: {random_index}")
    plt.show()
