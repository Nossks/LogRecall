from utils import create_collate_fn
from src.components.triplet_dataset import TripletDataset
from src.components.model_builder import Model
from dataclasses import dataclass
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split

@dataclass
class TrainerConfig:
    model_path = "artifacts/"

class Trainer:
    def __init__(self,epochs=10,lr=0.1):
        self.model = Model()
        self.epochs = epochs
        self.lr = lr
    
    def InitiateTraining(self):
        X=df['masked_log']
        y=df['hash_id']
        X_train,X_val,y_train,y_val =  train_test_split(X,y,test_size=.20,random_state=33)
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        train_dataset = TripletDataset(X_train,y_train)
        val_dataset = TripletDataset(X_val,y_val)

        loss_function = nn.TripletMarginWithDistanceLoss()
        optimizer = torch.optim.SGD(self.parameters(),lr=self.lr)

        for epoch in range(self.epochs):
            total_loss=0
            for batch_f,batch_l in 