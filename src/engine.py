import torch
import torch.nn as nn 
import math
import sys 
from sklearn.metrics import log_loss

class Engine :
    def __init__(self , model , optimizer , criterion ,device ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion



    @staticmethod
    def multi_acc(outputs , targets):
        output_softmax = torch.log_softmax(outputs ,dim=1)
        _ , pred_class = torch.max(output_softmax , dim=1)
        correct_pred = (pred_class ==targets).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = torch.round(acc*100)
        return acc
    def train_fn(self , dataloader):
        self.model.train()
        train_epoch_loss = 0
        train_epoch_acc = 0
        for data in dataloader:
            inputs = data["x"].to(self.device)
            targets = data["y"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs , torch.flatten(targets) )
            acc = self.multi_acc(outputs , torch.flatten(targets))
            loss.backward()
            self.optimizer.step()
            train_epoch_loss +=loss.item()
            train_epoch_acc += acc.item()
        return train_epoch_loss / len(dataloader) , train_epoch_acc / len(dataloader)
 

    def eval_fn(self , dataloader):
        self.model.eval()
        valid_epoch_loss = 0
        valid_epoch_acc = 0
        with torch.no_grad():
            for data in dataloader:
                inputs = data["x"].to(self.device)
                targets = data["y"].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs , torch.flatten(targets) )
                acc = self.multi_acc(outputs , torch.flatten(targets))
                valid_epoch_loss +=loss.item()
                valid_epoch_acc += acc.item()
            return valid_epoch_loss / len(dataloader), valid_epoch_acc / len(dataloader)
