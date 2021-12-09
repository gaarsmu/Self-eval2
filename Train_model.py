import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import clip
from PIL import Image
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
import argparse
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

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

class DatasetIMC1C2(Dataset):
    def __init__(self, correct_data, fake_data):
        self.correct_data = correct_data
        self.fake_data = fake_data

    def __len__(self):
        return min(len(self.correct_data), len(self.fake_data))
    
    def shuffle(self):
        self.correct_mapping = np.random.permutation(range(len(self.correct_data)))
        self.fake_mapping = np.random.permutation(range(len(self.fake_data)))

    def __getitem__(self, idx):
        correct_dp = self.correct_data[idx]
        fake_dp = self.fake_data[idx]
        
        img_path_corr = correct_dp['img_local_path']
        img_path_fake = fake_dp['img_local_path']
        
        caption1_corr = correct_dp['caption 1']
        caption1_fake = fake_dp['caption 1']
        
        caption2_corr = correct_dp['caption 2']
        caption2_fake = fake_dp['caption 2']
        
        
        img_correct = preprocess(Image.open(img_path_corr))
        cap1_correct = clip.tokenize(caption1_corr, truncate = True).squeeze(0)
        cap2_correct = clip.tokenize(caption2_corr, truncate = True).squeeze(0)
        
        img_fake = preprocess(Image.open(img_path_fake))
        cap1_fake = clip.tokenize(caption1_fake, truncate = True).squeeze(0)
        cap2_fake = clip.tokenize(caption2_fake, truncate = True).squeeze(0)
        
        return img_correct, cap1_correct, cap2_correct, img_fake, cap1_fake, cap2_fake    
    
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

    batch_size = 32
    train_dataset = DatasetIMC1C2(train_data_true, train_data_fake)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers = 8, shuffle=True)
    
    if args.mode == 'img_cap1':
        dims = 512*2
    elif args.mode == 'img_cap1_cap2':
        dims = 512*3
    classifier = CLIPclassifier(dims=dims)
    classifier.half()
    classifier.to(device)

    criterion = nn.BCEWithLogitsLoss()
    sigm = nn.Sigmoid()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4, eps=1e-4)
    
    for batch in tqdm(train_dataloader):
        img_corr_batch, cap1_corr_batch, cap2_corr_batch,\
            img_fake_batch, cap1_fake_batch, cap2_fake_batch = batch

        img_batch = torch.cat([img_corr_batch, img_fake_batch]).to(device)
        text_batch1 = torch.cat([cap1_corr_batch, cap1_fake_batch]).to(device)
        text_batch2 = torch.cat([cap2_corr_batch, cap2_fake_batch]).to(device)

        img_feat = model.encode_image(img_batch)
        text_feat1 = model.encode_text(text_batch1)
        text_feat2 = model.encode_text(text_batch2)

        if args.mode == 'img_cap1':
            cat_batch = torch.cat([img_feat, text_feat1], dim=1)
        elif args.mode == 'img_cap1_cap2':
            cat_batch = torch.cat([img_feat, text_feat1, text_feat2], dim=1)
        output = classifier(cat_batch)

        target = torch.cat([torch.zeros((output.shape[0]//2, 1)),
                            torch.ones((output.shape[0]//2,1))]).to(device)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    model_name = args.model_name

    optimizer_clip = torch.optim.AdamW(model.parameters(), lr=1e-6, eps=1e-4)

    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_clip, 'max', patience=2)

    torch.save((model.state_dict(), classifier.state_dict()) ,'models/'+model_name+'_best.pth')
    torch.save((model.state_dict(), classifier.state_dict()) ,'models/'+model_name+'_last.pth')

    val_acc_best = 0
    
    for epoch in range(1,1+args.num_epochs):
        print('Starting epoch ', epoch)
        losses = []

        for batch in tqdm(train_dataloader):
            img_corr_batch, cap1_corr_batch, cap2_corr_batch,\
                img_fake_batch, cap1_fake_batch, cap2_fake_batch = batch

            img_batch = torch.cat([img_corr_batch, img_fake_batch]).to(device)
            text_batch1 = torch.cat([cap1_corr_batch, cap1_fake_batch]).to(device)
            text_batch2 = torch.cat([cap2_corr_batch, cap2_fake_batch]).to(device)

            img_feat = model.encode_image(img_batch)
            text_feat1 = model.encode_text(text_batch1)
            text_feat2 = model.encode_text(text_batch2)

            if args.mode == 'img_cap1':
                cat_batch = torch.cat([img_feat, text_feat1], dim=1)
            elif args.mode == 'img_cap1_cap2':
                cat_batch = torch.cat([img_feat, text_feat1, text_feat2], dim=1)
            output = classifier(cat_batch)

            target = torch.cat([torch.zeros((output.shape[0]//2, 1)),
                                torch.ones((output.shape[0]//2,1))]).to(device)

            loss = criterion(output, target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer_clip.step()
            optimizer.zero_grad()
            optimizer_clip.zero_grad()
        torch.save((model.state_dict(), classifier.state_dict()) ,'models/'+model_name+'_last.pth')
        print('Train losses: ', np.mean(losses))
        val_loss, val_accuracy = evaluate_dataset(val_dataloader)
        print(val_loss, val_accuracy)
        scheduler1.step(val_accuracy)
        scheduler2.step(val_accuracy)
        if val_acc_best < val_accuracy:
            print('Validation losses are better, saving model as beest model')
            torch.save((model.state_dict(), classifier.state_dict()) ,'models/'+model_name+'_best.pth')
            val_acc_best = val_accuracy
        else:
            print('Validation losses are not better')
        print('Current learning rate:', optimizer_clip.param_groups[0]['lr'])


if __name__=='__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
    parser = argparse.ArgumentParser(description='Evaluation setting.')
    parser.add_argument("-m", "--mode", default='img_cap1_cap2', type = str, choices=['img_cap1', 'img_cap1_cap2'],
                        help="Choose the mode")
    parser.add_argument("--model_name", default='img_cap1_cap2_paired', type=str,
                        help='Path to model pytorch save file')
    parser.add_argument("--num_epochs", default=20, type=int,
                        help='Number of epochs to train the model')
    args = parser.parse_args()    
    main(args)