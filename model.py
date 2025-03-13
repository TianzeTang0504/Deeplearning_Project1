import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class UNetResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(UNetResNet, self).__init__()
        self.encoder1 = ConvBlock(3, 32)  # (N, 32, 32, 32)
        self.pool1 = nn.MaxPool2d(2)      # (N, 32, 16, 16)

        self.encoder2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)      # (N, 64, 8, 8)

        self.encoder3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)      # (N, 128, 4, 4)

        self.bottleneck = ConvBlock(128, 256)  # (N, 256, 4, 4)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(128 + 128, 128)  # Skip Connection

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(64 + 64, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(32 + 32, 32)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)  # Skip Connection
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)

        # 全局池化 + 全连接
        out = self.global_avg_pool(d1)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def unet_resnet():
    return UNetResNet()

if __name__ == "__main__":
    model = unet_resnet()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {param_count / 1e6:.2f}M parameters")
