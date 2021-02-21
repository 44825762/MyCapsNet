
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from capsnet import CapsNet, CapsuleLoss
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

def squash(x, dim=-1):
    # squshing 使得最终的输出向量长度在 0-1，该函数将小向量压缩为0，将大向量压缩为单位向量。
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True);
    scale = squared_norm / (1 + squared_norm);
    return scale * x / (squared_norm.sqrt() + 1e-8);

class PrimaryCaps(nn.Module):
    # 卷积层后 接 PrimaryCaps 层， 后接 DigitCaps 层
    # 该层是军A级之后得到特征图，然后将特征图展开成一维，对应位置组合。一共得到1152个8维向量神经元，也就是胶囊。
    # 然后通过动态路由算法得到 DigitCaps 层。DigitCaps 层的向量的模长大小就是预测的结果。
    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__();
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * num_conv_units,
            kernel_size=kernel_size,
            stride=stride
        );
        self.out_channels = out_channels;

    def forward(self,x):
        out = self.conv(x);
        batch_size = out.shape[0];
        return squash(out.contiguous().view(batch_size, -1, self.out_channels), dim=-1);


class DigitCaps(nn.Module):
    def __init__(self, in_dim, in_caps, out_caps, out_dim, num_routing):
        super(DigitCaps, self).__init__();
        self.in_dim = in_dim;
        self.in_caps = in_caps;
        self.out_caps = out_caps;
        self.out_dim = out_dim;
        self.num_routing = num_routing;
        self.device = torch.device('cuda');
        self.W = nn.Parameter(0.01 * torch.randn(1, out_caps, in_caps, out_dim, in_dim),
                              requires_grad=True);

    def forward(self,x):
        # 动态路由算法
        batch_size = x.size(0);
        # W @ x =
        # (1, out_caps, in_caps, out_dim, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, out_caps, in_caps, out_dims, 1)
        x = x.unsqueeze(1).unsqueeze(4);
        u_hat = torch.matmul(self.W, x);
        u_hat = u_hat.squeeze(-1);
        temp_u_hat = u_hat.detach();
        b = torch.zeros(batch_size, self.out_caps, self.in_caps, 1).to('cuda');
        for route_iter in range(self.num_routing - 1):
            # (batch_size, out_caps, in_caps, 1) -> Softmax along out_caps
            c = b.softmax(dim=1);
            s = (c * temp_u_hat).sum(dim=2);
            # apply "squashing" non-linearity along out_dim
            v = squash(s);
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1));
            b += uv;

        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1);
        s = (c * u_hat).sum(dim=2);
        # apply "squashing" non-linearity along out_dim
        v = squash(s);
        return v;

class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__();

        # 卷积层
        self.conv = nn.Conv2d(1,256,9);
        self.relu = nn.ReLU(inplace=True);

        # PrimaryCaps
        self.primarycaps = PrimaryCaps(num_conv_units=32,
                                        in_channels=256,
                                        out_channels=8,
                                        kernel_size=9,
                                        stride=2);

        # DigitCaps
        self.digitcaps = DigitCaps(in_dim=8,
                                    in_caps=32 * 6 * 6,
                                    out_caps=10,
                                    out_dim=16,
                                    num_routing=3);

        # Reconstruction 根据学习到的特征，重构图像
        self.decoder = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        );

    def forward(self,x):
        out = self.relu(self.conv(x));
        out = self.primarycaps(out);
        out = self.digitcaps(out);

        # Shape of logits: (batch_size, out_capsules)
        logits = torch.norm(out, dim=-1);
        pred = torch.eye(10).to('cuda').index_select(dim=0, index=torch.argmax(logits, dim=1));

        # Reconstruction
        batch_size = out.shape[0];
        reconstruction = self.decoder((out * pred.unsqueeze(2)).contiguous().view(batch_size, -1));

        return logits, reconstruction;

class CapsuleLoss(nn.Module):
    # margin loss
    # reconstruction loss

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__();
        self.upper = upper_bound;
        self.lower = lower_bound;
        self.lmda = lmda;
        self.reconstruction_loss_scalar = 5e-4;
        self.mse = nn.MSELoss(reduction='sum');

    def forward(self, images, labels, logits, reconstructions):
        # Shape of left / right / labels: (batch_size, num_classes)
        left = (self.upper - logits).relu() ** 2;  # True negative
        right = (logits - self.lower).relu() ** 2;  # False positive
        margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right);

        # Reconstruction loss
        reconstruction_loss = self.mse(reconstructions.contiguous().view(images.shape), images);

        # Combine two losses
        return margin_loss + self.reconstruction_loss_scalar * reconstruction_loss;



























