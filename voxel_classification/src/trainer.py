#!/usr/bin/env python

import datetime
import math
import os
import os.path as osp
import shutil
import numpy as np
import pytz
import torch
import tqdm
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score

class Trainer(object):

    def __init__(self, device, model, optimizer,criterion,
                 train_loader, val_loader, out, max_iter,
                 size_average=False, interval_validate=None):
        self.device = device

        self.model = model
        self.optim = optimizer
        self.criterion = criterion

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Australia/Brisbane'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'valid/loss',
            'valid/precision',
            'valid/recall',
            'valid/f1_score',
            'elapsed_time',
        ]
        with open(osp.join(self.out, 'log.csv'), 'w') as f:
            f.write(','.join(self.log_headers) + '\n')
        self.writer = SummaryWriter(osp.join(self.out,'tensorboard'))
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_val_loss = 2
        self.best_f1_score = 0

    def validate(self):
        training = self.model.training
        self.model.eval()

        y_target = []
        y_pred = []
        val_loss = 0
        label_trues, label_preds = [], []
        for batch_idx, sample in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):

            inputs = Variable(torch.FloatTensor(sample['inputs']))
            coords = Variable(torch.LongTensor(sample['coords']))
            labels = Variable(torch.LongTensor(sample['labels']))

            inputs, coords, labels = inputs.to(self.device), coords.to(self.device), labels.to(self.device)

            with torch.no_grad():
                score = self.model(inputs, coords)

            loss = self.criterion(score, labels)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / (labels.data.shape[0])

            _, index = score.max(dim=1)

            y_pred.append(index.cpu().data.numpy())
            y_target.append(labels.cpu().data.numpy())

        y_pred = np.concatenate(y_pred)
        y_target = np.concatenate(y_target)

        val_loss /= len(self.val_loader)
        self.writer.add_scalar('Val/Loss', val_loss, self.iteration)

        # Compute performance scores
        precision = precision_score(y_target, y_pred)
        recall = recall_score(y_target, y_pred)
        f1 = f1_score(y_target, y_pred)

        self.writer.add_scalar('Val/Precision', precision, self.iteration)
        self.writer.add_scalar('Val/Recall', recall, self.iteration)
        self.writer.add_scalar('Val/F1-score', f1, self.iteration)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Australia/Brisbane')) -
                    self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [val_loss] + [precision] + [recall] + [f1] + [elapsed_time]

            log = map("{:5.5f}".format,log)
            f.write(','.join(log) + '\n')

        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': val_loss,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        is_best = f1 >= self.best_f1_score
        if is_best:
            self.best_f1_score = f1
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'best.pth.tar'))
        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        for batch_idx, sample in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()

            assert self.model.training

            inputs = Variable(torch.FloatTensor(sample['inputs']))
            coords = Variable(torch.LongTensor(sample['coords']))
            labels = Variable(torch.LongTensor(sample['labels']))

            inputs, coords, labels = inputs.to(self.device), coords.to(self.device), labels.to(self.device)

            self.optim.zero_grad()
            score = self.model(inputs, coords)

            loss = self.criterion(score, labels)

            loss_data = loss.data.item() / (labels.data.shape[0])

            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            self.writer.add_scalar('Train/Loss', loss_data, self.iteration)

            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
        self.writer.close()
