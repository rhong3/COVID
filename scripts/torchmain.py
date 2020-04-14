"""
pytorch main to use pretrained models

Created on 04/13/2020

@author: RH
"""

import sys
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import Sample_prep
import matplotlib
matplotlib.use('Agg')


modeldict = {'resnet18': models.resnet18(pretrained=True).cuda(),
             'alexnet': models.alexnet(pretrained=True).cuda(),
             'squeezenet': models.squeezenet1_0(pretrained=True).cuda(),
             'vgg16': models.vgg16(pretrained=True).cuda(),
             'densenet': models.densenet161(pretrained=True).cuda(),
             'inception': models.inception_v3(pretrained=True).cuda(),
             'googlenet': models.googlenet(pretrained=True).cuda(),
             'shufflenet': models.shufflenet_v2_x1_0(pretrained=True).cuda(),
             'mobilenet': models.mobilenet_v2(pretrained=True).cuda(),
             'resnext50_32x4d': models.resnext50_32x4d(pretrained=True).cuda(),
             'wide_resnet50_2': models.wide_resnet50_2(pretrained=True).cuda(),
             'mnasnet': models.mnasnet1_0(pretrained=True).cuda()
             }

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transformer = transforms.Compose([
    transforms.ColorJitter(brightness=0.35, contrast=0.5, saturation=0.5, hue=0.35),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


class DataSet(Dataset):
    def __init__(self, datadir, transform=None):
        self.data_dir = datadir
        self.transform = transform
        self.imglist = pd.read_csv(self.data_dir, header=0).values.tolist()

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imglist[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.imglist[idx][1])}
        return sample


dirr = sys.argv[1]  # output directory
bs = sys.argv[2]    # batch size
bs = int(bs)
md = sys.argv[3]    # model to use

try:
    ep = sys.argv[4]  # epochs to train
    ep = int(ep)
except IndexError:
    ep = 100

# paths to directories
LOG_DIR = "../Results/{}".format(dirr)
METAGRAPH_DIR = "../Results/{}".format(dirr)
data_dir = "../Results/{}/data".format(dirr)
out_dir = "../Results/{}/out".format(dirr)

# make directories if not exist
for DIR in (LOG_DIR, METAGRAPH_DIR, data_dir, out_dir):
    try:
        os.mkdir(DIR)
    except FileExistsError:
        pass


if __name__ == '__main__':
    try:
        trs = DataSet(str(data_dir + '/tr_sample.csv'), transform=train_transformer)
        tes = DataSet(str(data_dir + '/te_sample.csv'), transform=val_transformer)
        vas = DataSet(str(data_dir + '/va_sample.csv'), transform=val_transformer)
    except FileNotFoundError:
        _, _, _ = Sample_prep.set_sep(path=data_dir)
        trs = DataSet(str(data_dir + '/tr_sample.csv'), transform=train_transformer)
        tes = DataSet(str(data_dir + '/te_sample.csv'), transform=val_transformer)
        vas = DataSet(str(data_dir + '/va_sample.csv'), transform=val_transformer)

    train_loader = DataLoader(trs, batch_size=bs, drop_last=False, shuffle=True)
    val_loader = DataLoader(vas, batch_size=bs, drop_last=False, shuffle=False)
    test_loader = DataLoader(tes, batch_size=bs, drop_last=False, shuffle=False)

    model = modeldict[md]
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # train
    for epoch in range(ep):
        train_loss = 0
        train_correct = 0
        for batch_index, batch_samples in enumerate(train_loader):
            data, target = batch_samples['img'].to('cuda'), batch_samples['label'].to('cuda')
            data = data[:, 0, :, :]
            data = data[:, None, :, :]
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss(output, target.long())
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.long().view_as(pred)).sum().item()
            # Display progress and write to tensorboard
            if batch_index % bs == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                    epoch, batch_index, len(train_loader),
                    100.0 * batch_index / len(train_loader), loss.item() / bs), flush=True)
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
            100.0 * train_correct / len(train_loader.dataset)), flush=True)

        # validation
        val_loss = 0
        correct = 0
        with torch.no_grad():
            # Predict
            for batch_index, batch_samples in enumerate(val_loader):
                data, target = batch_samples['img'].to('cuda'), batch_samples['label'].to('cuda')
                data = data[:, 0, :, :]
                data = data[:, None, :, :]
                output = model(data)

                loss = nn.CrossEntropyLoss(output, target.long())
                val_loss += loss
                score = F.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.long().view_as(pred)).sum().item()
            print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                val_loss / len(val_loader.dataset), correct, len(val_loader.dataset),
                100.0 * correct / len(val_loader.dataset)), flush=True)

    # test
    test_loss = 0
    correct = 0
    with torch.no_grad():
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].to('cuda'), batch_samples['label'].to('cuda')
            data = data[:, 0, :, :]
            data = data[:, None, :, :]
            output = model(data)

            loss = nn.CrossEntropyLoss(output, target.long())
            test_loss += loss
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss / len(test_loader.dataset), correct, len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset)), flush=True)
