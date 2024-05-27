# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.optim as optim
import data_loader
import model
from torchmetrics import Accuracy
import torch


net = model.Net()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters, lr = 0.01)
dataloader_train = data_loader.dataloader_train

for eopch in range(1000):
    for features, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = net(features)
        loss = criterion(
            outputs, labels.view(-1, 1)
            )
        loss.backward()
        optimizer.step()
        


# Set up binary accuracy metric
acc = Accuracy(task="binary")

net.eval()

with torch.no_grad():
    for features, labels in data_loader.dataloader_test:
        # Get predicted probabilities for test data batch
        outputs = net(features)
        preds = (outputs >= 0.5).float()
        acc(preds, labels.view(-1, 1))

# Compute total test accuracy
test_accuracy = acc.compute()
print(f"Test accuracy: {test_accuracy}")