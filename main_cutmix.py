import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision.transforms as transforms
from deep_resnet import resnet18_custom
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
from tqdm import tqdm

images = np.load("wholeimages.npy")
labels = np.load("wholelabels.npy")
test_images = np.load("test_images.npy")
test_labels = np.load("test_labels.npy")

class CIFARDataset(Dataset):
    def __init__(self, images, labels, transform=None, cutmix_prob=0.5, beta=1.0):
        self.images = torch.tensor(images, dtype=torch.float32) / 255.0
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        self.to_pil = ToPILImage()
        self.cutmix_prob = cutmix_prob
        self.beta = beta

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # CutMix
        if np.random.rand() < self.cutmix_prob:
            lam = np.random.beta(self.beta, self.beta)
            rand_idx = random.randint(0, len(self.labels) - 1)
            img2, label2 = self.images[rand_idx], self.labels[rand_idx]

            if self.transform is not None:
                img = self.to_pil(img)
                img2 = self.to_pil(img2)
                img = self.transform(img)
                img2 = self.transform(img2)

            bbx1, bby1, bbx2, bby2 = self._rand_bbox(img.shape, lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.shape[1] * img.shape[2]))

            return img, label, label2, lam

        if self.transform is not None:
            img = self.to_pil(img)
            img = self.transform(img)

        return img, label, label, 1.0

    def _rand_bbox(self, size, lam):
        W, H = size[1], size[2]
        cut_w = int(W * np.sqrt(1 - lam))
        cut_h = int(H * np.sqrt(1 - lam))
        cx, cy = np.random.randint(W), np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

train_size = int(0.9 * len(images))
test_size = len(images) - train_size

indices = np.arange(len(images))
np.random.shuffle(indices)
train_indices = indices[:train_size]
test_indices = indices[train_size:train_size+test_size]

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

full_dataset_train = CIFARDataset(images, labels, transform=train_transform, cutmix_prob=0.5)
full_dataset_test = CIFARDataset(images, labels, transform=test_transform, cutmix_prob=0.0)

train_dataset = Subset(full_dataset_train, train_indices)
test_dataset = Subset(full_dataset_test, test_indices)
val_dataset = CIFARDataset(test_images, test_labels, transform=test_transform, cutmix_prob=0.0)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18_custom().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=0.001)

train_losses = []
test_accuracies = []

def train(model, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

    for images, labels1, labels2, lam in loop:
        images, labels1, labels2 = images.to(device), labels1.to(device), labels2.to(device)
        lam = lam.to(device) if isinstance(lam, torch.Tensor) else torch.tensor(lam, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = lam * criterion(outputs, labels1) + (1 - lam) * criterion(outputs, labels2)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels1.size(0)
        batch_correct = (lam * predicted.eq(labels1).float() + (1 - lam) * predicted.eq(labels2).float()).sum().item()
        correct += batch_correct

        loop.set_postfix(loss=total_loss / (total / labels1.size(0)), acc=100. * correct / total)

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {100. * correct / total:.2f}%")


def test(model, test_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels, labels, lam in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_acc = 100. * correct / total
    test_accuracies.append(avg_acc)
    print(f"Test Loss: {total_loss/len(test_loader):.4f}, Test Accuracy: {avg_acc:.2f}%")
    return avg_acc

def val(model, val_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels, labels, lam in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_acc = 100. * correct / total
    print(f"Val Loss: {total_loss/len(test_loader):.4f}, Val Accuracy: {avg_acc:.2f}%")
    return avg_acc

best_acc = 0
for epoch in range(500): 
    train(model, train_loader, criterion, optimizer, scheduler, epoch)
    acc = test(model, test_loader, criterion)
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved Best Model!")

val(model, val_loader, criterion)

print(f"Best Test Accuracy: {best_acc:.2f}%")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='s', linestyle='-', color='g')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy Over Epochs')

plt.savefig("training_progress.png")
plt.show()

print("Training progress saved as 'training_progress.png'")
