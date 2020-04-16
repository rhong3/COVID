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
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
import Sample_prep
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pt_modeldict = {'resnet18': models.resnet18(pretrained=True).cuda(),
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

modeldict = {'resnet18': models.resnet18(pretrained=False).cuda(),
             'alexnet': models.alexnet(pretrained=False).cuda(),
             'squeezenet': models.squeezenet1_0(pretrained=False).cuda(),
             'vgg16': models.vgg16(pretrained=False).cuda(),
             'densenet': models.densenet161(pretrained=False).cuda(),
             'inception': models.inception_v3(pretrained=False).cuda(),
             'googlenet': models.googlenet(pretrained=False).cuda(),
             'shufflenet': models.shufflenet_v2_x1_0(pretrained=False).cuda(),
             'mobilenet': models.mobilenet_v2(pretrained=False).cuda(),
             'resnext50_32x4d': models.resnext50_32x4d(pretrained=False).cuda(),
             'wide_resnet50_2': models.wide_resnet50_2(pretrained=False).cuda(),
             'mnasnet': models.mnasnet1_0(pretrained=False).cuda()
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
                  'label': int(self.imglist[idx][1]),
                  'patient': str(self.imglist[idx][2]),
                  'path': str(self.imglist[idx][0])}
        return sample


dirr = sys.argv[1]  # output directory
bs = sys.argv[2]    # batch size
bs = int(bs)
md = sys.argv[3]    # model to use
ptr = sys.argv[4]   # pretrained?

if ptr:
    print("loading pretrained model...")
else:
    modeldict = pt_modeldict

try:
    ep = sys.argv[5]  # epochs to train
    ep = int(ep)
except IndexError:
    ep = 100

# paths to directories
METAGRAPH_DIR = "../Results/{}".format(dirr)
data_dir = "../Results/{}/data".format(dirr)
out_dir = "../Results/{}/out".format(dirr)

# make directories if not exist
for DIR in (METAGRAPH_DIR, data_dir, out_dir):
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
        Sample_prep.set_sep(path=data_dir)
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
    best_epoch = -1
    model.train()
    for epoch in range(ep):
        train_loss = 0
        train_correct = 0
        for batch_index, batch_samples in enumerate(train_loader):
            data, target = batch_samples['img'].to('cuda'), batch_samples['label'].to('cuda')
            optimizer.zero_grad()
            output = model(data)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, target.long())
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.long().view_as(pred)).sum().item()
        print('\nEpoch: {} \nTrain set: Average loss: {:.4f}, Image Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
            100.0 * train_correct / len(train_loader.dataset)), flush=True)

        # validation
        val_loss = 0
        correct = 0
        model.eval()
        predlist = []
        scorelist = []
        targetlist = []
        patientlist = []
        pathlist = []
        losslist = []
        with torch.no_grad():
            # Predict
            for batch_index, batch_samples in enumerate(val_loader):
                data, target, patient, impath = batch_samples['img'].to('cuda'), batch_samples['label'].to('cuda'), \
                                                batch_samples['patient'], batch_samples['path']
                output = model(data)

                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(output, target.long())
                val_loss += loss
                score = F.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.long().view_as(pred)).sum().item()
                targetcpu = target.long().cpu().numpy()
                predlist = np.append(predlist, pred.cpu().numpy())
                scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
                targetlist = np.append(targetlist, targetcpu)
                patientlist = np.append(patientlist, patient)
                pathlist = np.append(pathlist, impath)
            ave_val_loss = val_loss.cpu().numpy() / len(val_loader.dataset)
            losslist = np.append(losslist, ave_val_loss)

            if epoch != 0 and ave_val_loss == min(losslist):
                best_epoch = epoch
                print('Temporary best model found @ epoch {}! Saving...'.format(epoch))
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, '{}/model'.format(METAGRAPH_DIR))

                best_joined = pd.DataFrame({
                    'prediction': predlist,
                    'target': targetlist,
                    'score': scorelist,
                    'patient': patientlist,
                    'path': pathlist
                })
                best_joined.to_csv('{}/best_validation_image.csv'.format(out_dir), index=False)

            print('\nValidation set: Average loss: {:.4f}, Image Accuracy: {}/{} ({:.0f}%)\n'.format(
                ave_val_loss, correct, len(val_loader.dataset),
                100.0 * correct / len(val_loader.dataset)), flush=True)
            TP = ((predlist == 1) & (targetlist == 1)).sum()
            TN = ((predlist == 0) & (targetlist == 0)).sum()
            FN = ((predlist == 0) & (targetlist == 1)).sum()
            FP = ((predlist == 1) & (targetlist == 0)).sum()
            print("\nPer image metrics: ")
            print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
            print('TP+FP=', TP + FP)
            if (TP + FP) != 0:
                p = TP / (TP + FP)
                print('precision=', p)
                p = TP / (TP + FP)
                r = TP / (TP + FN)
                print('recall=', r)
                F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
            print('F1=', F1)
            print('acc=', acc)
            AUC = roc_auc_score(targetlist, scorelist)
            print('AUC=', AUC)

            joined = pd.DataFrame({
                'prediction': predlist,
                'target': targetlist,
                'score': scorelist,
                'patient': patientlist
            })

            joined = joined.groupby(['patient']).mean()
            joined = joined.round({'prediction': 0, 'target': 0})
            if best_epoch == epoch:
                joined.to_csv('{}/best_validation_patient.csv'.format(out_dir), index=False)

            print("\nPer patient metrics: ")
            TP = joined.loc[(joined['prediction'] == 1) & (joined['target'] == 1)].shape[0]
            TN = joined.loc[(joined['prediction'] == 0) & (joined['target'] == 0)].shape[0]
            FN = joined.loc[(joined['prediction'] == 0) & (joined['target'] == 1)].shape[0]
            FP = joined.loc[(joined['prediction'] == 1) & (joined['target'] == 0)].shape[0]
            print("Per image metrics: ")
            print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
            print('TP+FP=', TP + FP)
            if (TP+FP) != 0:
                p = TP / (TP + FP)
                print('precision=', p)
                p = TP / (TP + FP)
                r = TP / (TP + FN)
                print('recall=', r)
                F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
            print('F1=', F1)
            print('acc=', acc)
            AUC = roc_auc_score(joined['target'].tolist(), joined['score'].tolist())
            print('AUC=', AUC)

    print('\nBest model @ epoch: ', best_epoch)

    # test
    test_loss = 0
    correct = 0
    model = modeldict[md]
    model.load_state_dict(torch.load('{}/model'.format(METAGRAPH_DIR)))
    model.eval()
    predlist = []
    scorelist = []
    targetlist = []
    patientlist = []
    pathlist = []
    with torch.no_grad():
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target, patient, impath = batch_samples['img'].to('cuda'), batch_samples['label'].to('cuda'), \
                                            batch_samples['patient'], batch_samples['path']
            output = model(data)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, target.long())
            test_loss += loss
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)
            patientlist = np.append(patientlist, patient)
            pathlist = np.append(pathlist, impath)
        ave_test_loss = test_loss.cpu().numpy() / len(val_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Image Accuracy: {}/{} ({:.0f}%)\n'.format(
            ave_test_loss, correct, len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset)), flush=True)

        image_joined = pd.DataFrame({
            'prediction': predlist,
            'target': targetlist,
            'score': scorelist,
            'patient': patientlist,
            'path': pathlist
        })
        image_joined.to_csv('{}/test_image.csv'.format(out_dir), index=False)

        TP = ((predlist == 1) & (targetlist == 1)).sum()
        TN = ((predlist == 0) & (targetlist == 0)).sum()
        FN = ((predlist == 0) & (targetlist == 1)).sum()
        FP = ((predlist == 1) & (targetlist == 0)).sum()
        print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
        print('TP+FP', TP + FP)
        if (TP + FP) != 0:
            p = TP / (TP + FP)
            print('precision', p)
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            print('recall', r)
            F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('F1', F1)
        print('acc', acc)
        AUC = roc_auc_score(targetlist, scorelist)
        print('AUC', AUC)

        fpr, tpr, _ = roc_curve(targetlist, scorelist)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.5f)' % AUC)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of COVID')
        plt.legend(loc="lower right")
        plt.savefig("../Results/{}/out/ROC_image.png".format(dirr))

        average_precision = average_precision_score(targetlist, scorelist)
        print('Average precision-recall score: {0:0.5f}'.format(average_precision))
        plt.figure()
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        precision, recall, _ = precision_recall_curve(targetlist, scorelist)
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('COVID PRC: AP={:0.5f}; Accu={}'.format(average_precision, acc))
        plt.savefig("../Results/{}/out/PRC_image.png".format(dirr))

        joined = pd.DataFrame({
            'prediction': predlist,
            'target': targetlist,
            'score': scorelist,
            'patient': patientlist
        })

        joined = joined.groupby(['patient']).mean()
        joined = joined.round({'prediction': 0, 'target': 0})
        joined.to_csv('{}/test_patient.csv'.format(out_dir), index=False)

        print("Per patient metrics: ")
        TP = joined.loc[(joined['prediction'] == 1) & (joined['target'] == 1)].shape[0]
        TN = joined.loc[(joined['prediction'] == 0) & (joined['target'] == 0)].shape[0]
        FN = joined.loc[(joined['prediction'] == 0) & (joined['target'] == 1)].shape[0]
        FP = joined.loc[(joined['prediction'] == 1) & (joined['target'] == 0)].shape[0]
        print("Per image metrics: ")
        print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
        print('TP+FP=', TP + FP)
        if (TP + FP) != 0:
            p = TP / (TP + FP)
            print('precision=', p)
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            print('recall=', r)
            F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('F1=', F1)
        print('acc=', acc)
        AUC = roc_auc_score(joined['target'].tolist(), joined['score'].tolist())
        print('AUC=', AUC)

        fpr, tpr, _ = roc_curve(joined['target'].tolist(), joined['score'].tolist())
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.5f)' % AUC)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of COVID')
        plt.legend(loc="lower right")
        plt.savefig("../Results/{}/out/ROC_patient.png".format(dirr))

        average_precision = average_precision_score(joined['target'].tolist(), joined['score'].tolist())
        print('Average precision-recall score: {0:0.5f}'.format(average_precision))
        plt.figure()
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        precision, recall, _ = precision_recall_curve(targetlist, scorelist)
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('COVID PRC: AP={:0.5f}; Accu={}'.format(average_precision, acc))
        plt.savefig("../Results/{}/out/PRC_patient.png".format(dirr))
