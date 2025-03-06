import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision.transforms as transforms
from model import unet_resnet
from model_resnet import resnet18_custom
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage

# 读取数据
images = np.load("images.npy")
labels = np.load("labels.npy")
test_images = np.load("test_images.npy")
test_labels = np.load("test_labels.npy")

class CIFARDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # images shape: (N, H, W, C)
        self.images = torch.tensor(images, dtype=torch.float32) / 255.0  # 归一化到 [0,1]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        self.to_pil = ToPILImage()  # 用于将 tensor 转换为 PIL 图片

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        # 如果提供了 transform，则先将 tensor 转换为 PIL Image，再应用 transform
        if self.transform is not None:
            img = self.to_pil(img)
            img = self.transform(img)
        return img, label

# 划分训练集 & 测试集
train_size = int(0.8 * len(images))  # 80% 训练
test_size = len(images) - train_size  # 20% 测试

indices = np.arange(len(images))
np.random.shuffle(indices)
train_indices = indices[:train_size]
test_indices = indices[train_size:train_size+test_size]

train_transform = transforms.Compose([
    # 数据增强：随机裁剪（32x32的图片，添加4像素的边缘，再随机裁剪回32x32）
    transforms.RandomCrop(32, padding=4),
    
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    
    # 随机旋转，范围 (-15, 15) 度
    transforms.RandomRotation(15),
    
    # 颜色抖动（随机改变亮度、对比度、饱和度和色调）
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    
    # 随机仿射变换（平移、缩放、旋转等）
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),

    # 转换为Tensor
    transforms.ToTensor(),
    
    # Cutout（随机擦除部分图像，模仿 occlusion 影响）
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# 重新应用 transform
full_dataset_train = CIFARDataset(images, labels, transform=train_transform)
full_dataset_test = CIFARDataset(images, labels, transform=test_transform)

train_dataset = Subset(full_dataset_train, train_indices)
test_dataset = Subset(full_dataset_test, test_indices)
val_dataset = CIFARDataset(test_images, test_labels, transform=test_transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = resnet18_custom().to(device)

# 损失函数 & 优化器

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

# 记录训练过程
train_losses = []
test_accuracies = []

# 训练函数
def train(model, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {100. * correct / total:.2f}%")

# 测试函数
def test(model, test_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
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
        for images, labels in val_loader:
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

# 训练与测试
best_acc = 0
for epoch in range(100): 
    train(model, train_loader, criterion, optimizer, scheduler, epoch)
    acc = test(model, test_loader, criterion)
    
    # 保存最优模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved Best Model!")

val(model, val_loader, criterion)

print(f"Best Test Accuracy: {best_acc:.2f}%")

# 绘制 Loss 和 Accuracy 变化曲线
plt.figure(figsize=(10, 4))

# 训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Over Epochs')

# 测试精度曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='s', linestyle='-', color='g')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy Over Epochs')

# 保存图像
plt.savefig("training_progress.png")
plt.show()

print("Training progress saved as 'training_progress.png'")
