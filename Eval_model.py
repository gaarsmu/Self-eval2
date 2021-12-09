import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import clip
from PIL import Image
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import metrics
import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import glob

class CLIPclassifier(nn.Module):
    
    def __init__(self, dims):
        super(CLIPclassifier, self).__init__()
        
        self.fc1 = nn.Linear(dims,512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        self.lrelu = nn.LeakyReLU(0.05)
        
    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.fc3(x)
        return x

    
class DatasetCM(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dp = self.data[idx]
        img_path = dp['img_local_path']
        caption1 = dp['caption 1']
        caption2 = dp['caption 2']
        
        img = preprocess(Image.open(img_path))
        cap1 = clip.tokenize(caption1, truncate = True).squeeze(0)
        cap2 = clip.tokenize(caption2, truncate = True).squeeze(0)
        label = dp['label']

        return img, cap1, cap2, label


def main(args):
    dataset_true = torch.load('datapoints_from_COSMOS2_real.pth')
    dataset_false = torch.load('datapoints_from_COSMOS2_fake.pth')

    dataset_true = np.array(dataset_true)
    dataset_false = np.array(dataset_false)

    train_indexes, test_indexes = train_test_split(np.arange(len(dataset_true)), test_size=3000, random_state=239)
    train_indexes, val_indexes = train_test_split(train_indexes, test_size=3000, random_state=239)

    train_data_fake = dataset_false[train_indexes]
    val_data_fake = dataset_false[val_indexes]
    test_data_fake = dataset_false[test_indexes]

    train_data_true = dataset_true[train_indexes]
    val_data_true = dataset_true[val_indexes]
    test_data_true = dataset_true[test_indexes]
    
    if args.mode == 'img_cap1':
        dims = 512*2
    elif args.mode == 'img_cap1_cap2':
        dims = 512*3
    classifier = CLIPclassifier(dims=dims)
    classifier.half()
    classifier.to(device)
    
    model_state_dict, classifier_state_dict = torch.load(args.model_path)

    model.load_state_dict(model_state_dict)
    classifier.load_state_dict(classifier_state_dict)
    
    sigm = nn.Sigmoid()
    
    batch_size = 128
    test_dataset = DatasetCM(np.concatenate([test_data_fake, test_data_true]))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers = 8)
    
    preds = []
    labels = []
    for batch in test_dataloader:
        img_batch, cap1_batch, cap2_batch, label_batch = batch

        img_batch = img_batch.to(device)
        text_batch1 = cap1_batch.to(device)
        text_batch2 = cap2_batch.to(device)

        with torch.no_grad():
            img_feat = model.encode_image(img_batch)
            text_feat1 = model.encode_text(text_batch1)
            text_feat2 = model.encode_text(text_batch2)
            
            if args.mode == 'img_cap1':
                cat_batch = torch.cat([img_feat, text_feat1], dim=1)
            elif args.mode == 'img_cap1_cap2':
                cat_batch = torch.cat([img_feat, text_feat1, text_feat2], dim=1)
            output = classifier(cat_batch)
        preds.append(sigm(output).cpu().numpy())
        labels.append(label_batch.numpy())
        
    print('Accuracy: {}'.format(accuracy_score(np.concatenate(labels), np.concatenate(preds).reshape(-1,) >=0.5)))
    print('ROC AUC score {}'.format(roc_auc_score(np.concatenate(labels), np.concatenate(preds).reshape(-1,) )))
    fpr, tpr, thresholds = metrics.roc_curve(np.concatenate(labels), np.concatenate(preds).reshape(-1,))
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.savefig('result_figure'+str(len(glob.glob('result_figure*'))+1)+'.jpg')

    
if __name__=='__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
    parser = argparse.ArgumentParser(description='Evaluation setting.')
    parser.add_argument("-m", "--mode", default='img_cap1_cap2', type = str, choices=['img_cap1', 'img_cap1_cap2'],
                        help="Choose the mode")
    parser.add_argument("--model_path", default='models/img_cap1_cap2_paired_best.pth', type=str,
                        help='Path to model pytorch save file')
    args = parser.parse_args()
    main(args)