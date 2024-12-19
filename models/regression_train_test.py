#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from math import sqrt as msqrt
from torch import square, sqrt, mean

# from torch.nn.utils import clip_grad_norm_
def train(model, optimizer,train_loader, device):
    model.train()
    # train_labels = 0
    # train_predictions = 0
    total_loss = total_examples = 0
    # regularization_strength = 0.001
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        # out = model(data)
        y = data.y.view([-1])
        out1 = out.view([-1])
        # print("train : ", y.shape)
        loss = F.mse_loss(out1, y)
        # l2_reg = sum(p.pow(2).sum() for p in model.parameters())
        # combined_loss = loss + regularization_strength * l2_reg
        # combined_loss.backward()
        loss.backward()
        # max_grad_norm = 1.0  # You can adjust this value
        # clip_grad_norm_(model.parameters(), max_grad_norm)
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
#         train_labels += train_labels + y
#         train_predictions += train_predictions + out1
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    return total_loss,msqrt(total_loss / total_examples)

@torch.no_grad()
def test(loader, model, device):
    # mse = []
    model.eval()
    # total_preds = torch.Tensor()
    # total_labels = torch.Tensor()
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        # out = model(data)
        # mse.append(F.mse_loss(out, data.y, reduction='none').cpu())
        # return float(torch.cat(mse, dim=0).mean().sqrt())
        y = data.y.view([-1])
        out1 = out.view([-1])
        # print("test : ", y.shape)
        test_loss = F.mse_loss(out1, y)
        # print("no of graphs: ", data.num_graphs)
        total_loss += float(test_loss) * data.num_graphs
        total_examples += data.num_graphs
#         total_preds = torch.cat((total_preds, out1.cpu()), 0)
#         total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
        # mse.append(test_loss).cpu()
    # return test_loss,float(torch.cat(mse, dim=0).mean().sqrt())
    return total_loss,msqrt(total_loss / total_examples) #,total_labels.numpy().flatten(),total_preds.numpy().flatten()


@torch.no_grad()
def predicting(loader, model, device):
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_loss = torch.FloatTensor()
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        y = data.y.view([-1])
        out1 = out.view([-1])
        # print("test : ", y.shape)
        test_loss = F.mse_loss(out1, y)
        # print("test_loss: ", test_loss.item())
        total_loss += float(test_loss) * data.num_graphs
        # print("total_loss: ", total_loss)
        total_examples += data.num_graphs
        total_preds = torch.cat((total_preds, out.view(-1, 1).cpu()), 0)
        total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    # print("total_loss: ", total_loss)
    return total_loss,msqrt(total_loss / total_examples),total_labels.numpy().flatten(),total_preds.numpy().flatten()


@torch.no_grad()
def predictingSingle(loader, model, device):
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_loss = torch.FloatTensor()
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
       # y = data.y.view([-1])
        out1 = out.view([-1])
        # print("test : ", y.shape)
   #     test_loss = F.mse_loss(out1, y)
        # print("test_loss: ", test_loss.item())
        # total_loss += float(test_loss) * data.num_graphs
        # print("total_loss: ", total_loss)
        total_examples += data.num_graphs
        total_preds = torch.cat((total_preds, out.view(-1, 1).cpu()), 0)
       # total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    # print("total_loss: ", total_loss)
    return total_preds.numpy().flatten()

