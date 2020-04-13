"""
pytorch main to use pretrained models

Created on 04/13/2020

@author: RH
"""

import sys
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from imageio import imread, imsave
from torch.nn import functional as F
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import Sample_prep
import matplotlib
matplotlib.use('Agg')


dirr = sys.argv[1]  # output directory
bs = sys.argv[2]    # batch size
bs = int(bs)
md = sys.argv[3]    # architecture to use

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

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True)

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

alpha = None
device = 'cuda'


def train(optimizer, epoch):
    model.train()

    train_loss = 0
    train_correct = 0

    for batch_index, batch_samples in enumerate(train_loader):

        # move data to device
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
        data = data[:, 0, :, :]
        data = data[:, None, :, :]
        #         data, targets_a, targets_b, lam = mixup_data(data, target.long(), alpha, use_cuda=True)

        optimizer.zero_grad()
        output = model(data)

        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())
        #         loss = mixup_criterion(criteria, output, targets_a, targets_b, lam)
        train_loss += criteria(output, target.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()

        # Display progress and write to tensorboard
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item() / bs))

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    f = open('model_result/{}.txt'.format(modelname), 'a+')
    f.write('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    f.write('\n')
    f.close()


def val(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    results = []

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():
        tpr_list = []
        fpr_list = []

        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            data = data[:, 0, :, :]
            data = data[:, None, :, :]
            output = model(data)

            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            #             print('target',target.long()[:, 2].view_as(pred))
            correct += pred.eq(target.long().view_as(pred)).sum().item()

            #             print(output[:,1].cpu().numpy())
            #             print((output[:,1]+output[:,0]).cpu().numpy())
            #             predcpu=(output[:,1].cpu().numpy())/((output[:,1]+output[:,0]).cpu().numpy())
            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)

    return targetlist, scorelist, predlist


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    results = []

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():
        tpr_list = []
        fpr_list = []

        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            data = data[:, 0, :, :]
            data = data[:, None, :, :]
            #             print(target)
            output = model(data)

            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            #             print('target',target.long()[:, 2].view_as(pred))
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            #             TP += ((pred == 1) & (target.long()[:, 2].view_as(pred).data == 1)).cpu().sum()
            #             TN += ((pred == 0) & (target.long()[:, 2].view_as(pred) == 0)).cpu().sum()
            # #             # FN    predict 0 label 1
            #             FN += ((pred == 0) & (target.long()[:, 2].view_as(pred) == 1)).cpu().sum()
            # #             # FP    predict 1 label 0
            #             FP += ((pred == 1) & (target.long()[:, 2].view_as(pred) == 0)).cpu().sum()
            #             print(TP,TN,FN,FP)

            #             print(output[:,1].cpu().numpy())
            #             print((output[:,1]+output[:,0]).cpu().numpy())
            #             predcpu=(output[:,1].cpu().numpy())/((output[:,1]+output[:,0]).cpu().numpy())
            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)

    return targetlist, scorelist, predlist



