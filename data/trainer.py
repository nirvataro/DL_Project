import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import dataset
from torch.optim import Adam

def batch_loop():


def train(dataset, model, name, max_epochs=1000):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # lr decrease
    for epoch in range(max_epochs):
        for state in ['train', 'validation']:
            if state == 'train':
                iter = dataset.train_iter
                model.train()
                optimizer.zero_grad()
            else:
                iter = dataset.valid_iter
                model.eval()
            context = model.init_context()
            for batch in iter:
                y_pred = model(batch.comments, context, batch.features)
                loss = criterion(y_pred, batch.labels)
                if state == 'train':
                    loss.backward()
                    optimizer.step()
            if state == 'train':
                scheduler.step()
