"""
Script to train adversary classifier
"""

import argparse
import numpy as np
import os
import pickle
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data_utils import *
from model import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, loader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0.0
    for i, (audio, label) in enumerate(loader):
        audio = audio.cuda(device)
        label = label.cuda(device)
        optimizer.zero_grad()
        output = model(audio)
        loss = criterion(output, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for i, (audio, label) in enumerate(loader):
            audio = audio.cuda(device)
            label = label.cuda(device)
            output = model(audio)
            pred = output.data.max(1, keepdim=True)[1]
            matchings = pred.eq(label.data.view_as(pred).type(torch.cuda.LongTensor))
            correct = correct + matchings.sum()
            total_samples = total_samples + audio.size()[0]
            loss = criterion(output, label)
            epoch_loss += loss.item()
    acc = float(correct)/total_samples
    return epoch_loss/len(loader), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', default='', type=str, help='checkpoint path to load')
    parser.add_argument('--cuda', default=0, type=int, help='cuda device')
    parser.add_argument('-w', '--num-workers', default=1, type=int, help='number of workers')
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--expt', default='exp-ls', type=str, help='experiment id')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--batch-size-2', default=4, type=int, help='batch size')
    parser.add_argument('--clip', default=0.25, type=float, help='max norm of the gradients')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--spkr-seed', default=-1, type=int, help='speaker seed')
    parser.add_argument('--mode', default='default', type=str, help='type of experiment')
    parser.add_argument('--num-spkr', default=1, type=int, help='# of train spkrs per gender')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_dir = 'runs/%s' % args.expt
    writer = SummaryWriter(log_dir)
    print('tensorboard log in %s' % log_dir)
    
    load_w2v2_cp = True
    ckpt_path_to_load = None
    ckpt_path = '%s/model.pt' % log_dir
    if os.path.exists(args.load):
        load_w2v2_cp = False
        ckpt_path_to_load = args.load
    elif os.path.exists(ckpt_path):
        load_w2v2_cp = False
        ckpt_path_to_load = ckpt_path

    num_classes = 2
    
    model = VGGVoxClf()
    model = model.cuda(args.cuda)

    if ckpt_path_to_load is not None:
        print('loading model')
        model.load_state_dict(torch.load(ckpt_path_to_load))
    
    print('The model has %d trainable parameters' % count_parameters(model))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    train_dataset = librispeech_dataset('train', mode=args.mode, num_spkr=args.num_spkr, spkr_seed=args.spkr_seed)
    val_dataset = librispeech_dataset('test', mode=args.mode, num_spkr=args.num_spkr)
    assert len(set(train_dataset.speaker_ids).intersection(set(val_dataset.speaker_ids))) == 0
    train_loader = DataLoader(train_dataset, collate_fn=collate_1d_float, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, collate_fn=collate_1d_float, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('created dataloaders')

    start_time = time.time()
    valid_loss, valid_acc = evaluate(model, val_loader, criterion, args.cuda)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: 0 | Time: {epoch_mins}m {epoch_secs}s | Acc: {valid_acc:.3f}')

    best_valid_loss = valid_loss
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, args.clip, args.cuda)
        valid_loss, valid_acc = evaluate(model, val_loader, criterion, args.cuda)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), ckpt_path)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Acc: {valid_acc:.3f}')
        test_acc = 0.0
        writer.add_scalars('scalar/accs', {'val': valid_acc, 'test': test_acc}, epoch)
        test_loss = 0.0
        writer.add_scalars('scalar/losses', {'train': train_loss, 'val': valid_loss, 'test': test_loss}, epoch)


if __name__ == "__main__":
    main()