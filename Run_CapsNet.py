import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from MyCapsNet import CapsNet, CapsuleLoss

# Reference: https://arxiv.org/abs/1710.09829
def train():
    PATH = "./MyCapsNet.pt";

    # Load Model
    model = CapsNet().to('cuda');
    criterion = CapsuleLoss();
    optimizer = optim.Adam(model.parameters(),lr=1e-3);
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.96);

    # Load Data
    transform = transforms.Compose([
        transforms.RandomCrop(28,padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ]);
    DATA_PATH = './data';
    BATCH_SIZE = 128;
    train_loader = DataLoader(
        dataset = MNIST(root=DATA_PATH,train=True,transform=transform),
        batch_size = BATCH_SIZE,
        num_workers = 4,
        shuffle = True
    );

    test_loader = DataLoader(
      dataset = MNIST(root=DATA_PATH,train=False,transform=transform),
      batch_size = BATCH_SIZE,
      num_workers = 4,
        shuffle = True
    );

    # Train
    EPOCHES = 40;
    for ep in range(EPOCHES):
        batch_id = 1;
        correct, total, total_loss = 0., 0., 0.;
        for images, labels in train_loader:
            optimizer.zero_grad();
            images = images.to('cuda');
            labels = torch.eye(10).index_select(dim =0, index =labels).to('cuda');
            logits, reconstruction = model(images);

            # 计算损失和准确率
            loss = criterion(images,labels,logits,reconstruction);
            correct += torch.sum(torch.argmax(logits,dim=1) == torch.argmax(labels,dim=1)).item();
            total += len(labels);
            accuracy = correct/total;
            total_loss += loss;
            loss.backward();
            optimizer.step();
            print('Epoch {}, batch {}, loss: {}, accuracy: {}'.format(ep + 1,
                                                                      batch_id,
                                                                      total_loss / batch_id,
                                                                      accuracy));
            batch_id += 1;
        scheduler.step(ep);
        print('Total loss for epoch {}: {}'.format(ep + 1, total_loss));



    # Eval
    model.eval();
    correct, total = 0, 0;
    for images, labels in test_loader:
        # Add channels = 1
        images = images.to('cuda');
        # Categogrical encoding
        labels = torch.eye(10).index_select(dim=0, index=labels).to('cuda');
        logits, reconstructions = model(images);
        pred_labels = torch.argmax(logits, dim=1);
        correct += torch.sum(pred_labels == torch.argmax(labels, dim=1)).item();
        total += len(labels);
    print('Accuracy: {}'.format(correct / total));

    # Save model
    # torch.save(model.state_dict(), './model/capsnet_ep{}_acc{}.pt'.format(EPOCHES, correct / total))
    torch.save(model.state_dict(), PATH);








train();







