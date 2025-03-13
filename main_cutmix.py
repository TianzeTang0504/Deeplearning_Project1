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
from model import unet_resnet
from deep_resnet import resnet18_custom
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
from tqdm import tqdm

# 读取数据
images = np.load("wholeimages.npy")
labels = np.load("wholelabels.npy")
test_images = np.load("test_images.npy")
test_labels = np.load("test_labels.npy")

class CIFARDataset(Dataset):
    def __init__(self, images, labels, transform=None, cutmix_prob=0.5, beta=1.0):
        """
        :param images: CIFAR-10 图像数据 (N, H, W, C)
        :param labels: 图像对应的类别标签 (N,)
        :param transform: 数据增强方法
        :param cutmix_prob: 进行 CutMix 的概率
        :param beta: 控制 CutMix 混合比例（Beta 分布）
        """
        self.images = torch.tensor(images, dtype=torch.float32) / 255.0  # 归一化到 [0,1]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        self.to_pil = ToPILImage()  # 用于转换为 PIL 图片
        self.cutmix_prob = cutmix_prob
        self.beta = beta

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # **CutMix 数据增强**
        if np.random.rand() < self.cutmix_prob:
            lam = np.random.beta(self.beta, self.beta)  # 生成混合比例 lambda
            rand_idx = random.randint(0, len(self.labels) - 1)  # 选择随机图像进行 CutMix
            img2, label2 = self.images[rand_idx], self.labels[rand_idx]

            # 转换为 PIL 并应用 transform
            if self.transform is not None:
                img = self.to_pil(img)
                img2 = self.to_pil(img2)
                img = self.transform(img)
                img2 = self.transform(img2)

            # 生成 CutMix 矩形区域
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(img.shape, lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]  # 叠加部分区域
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.shape[1] * img.shape[2]))  # 重新计算混合比例

            return img, label, label2, lam  # 返回两个标签和混合比例

        # **普通数据增强**
        if self.transform is not None:
            img = self.to_pil(img)
            img = self.transform(img)

        return img, label, label, 1.0  # 仅返回原始标签

    def _rand_bbox(self, size, lam):
        """ 生成 CutMix 需要的 bbox 坐标 """
        W, H = size[1], size[2]  # 通道数 x H x W
        cut_w = int(W * np.sqrt(1 - lam))
        cut_h = int(H * np.sqrt(1 - lam))
        cx, cy = np.random.randint(W), np.random.randint(H)  # 随机选取中心点
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

# 划分训练集 & 测试集
train_size = int(0.9 * len(images))
test_size = len(images) - train_size

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

    # 转换为Tensor、
    transforms.ToTensor(),
    
    # Cutout（随机擦除部分图像，模仿 occlusion 影响）
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# 重新应用 transform
full_dataset_train = CIFARDataset(images, labels, transform=train_transform, cutmix_prob=0.5)
full_dataset_test = CIFARDataset(images, labels, transform=test_transform, cutmix_prob=0.0)

train_dataset = Subset(full_dataset_train, train_indices)
test_dataset = Subset(full_dataset_test, test_indices)
val_dataset = CIFARDataset(test_images, test_labels, transform=test_transform, cutmix_prob=0.0)

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
optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=0.001)

# 记录训练过程
train_losses = []
test_accuracies = []

# 训练函数
def train(model, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    # 使用 tqdm 添加进度条
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

        # 更新 tqdm 进度条上的信息
        loop.set_postfix(loss=total_loss / (total / labels1.size(0)), acc=100. * correct / total)

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {100. * correct / total:.2f}%")


# 测试函数
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

# 训练与测试
best_acc = 0
for epoch in range(500): 
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
