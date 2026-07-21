from src.utils import create_collate_fn
from src.components.Model.triplet_dataset import TripletDataset
from src.components.Model.model_builder import Model
from dataclasses import dataclass
import pandas as pd
import torch.nn as nn
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
torch.autograd.set_detect_anomaly(True) 

@dataclass
class TrainerConfig:
    model_path = "artifacts/"
    df_path = "artifacts/stage1_triplet_data.parquet"
    batch_size=32
    max_len=128

class Trainer:
    def __init__(self,epochs=5,lr=2e-5):
        self.config = TrainerConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.model = Model().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        self.epochs = epochs
        self.lr = lr
    
    def eval(self,val_dataloader):
        total_loss=0
        self.model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                
                anchor_inputs = {k: v.to(self.device) for k, v in batch["anchor"].items()}
                pos_inputs = {k: v.to(self.device) for k, v in batch["positive"].items()}
                neg_inputs = {k: v.to(self.device) for k, v in batch["negative"].items()}
                
                vec_anchor = self.model(anchor_inputs)
                vec_pos = self.model(pos_inputs)
                vec_neg = self.model(neg_inputs)

                # dist_pos = 1.0 - (vec_anchor * vec_pos).sum(dim=1)
                # dist_neg = 1.0 - (vec_anchor * vec_neg).sum(dim=1)                
                # loss = torch.nn.functional.relu(dist_pos - dist_neg + 0.3).mean()
                loss = self.loss_function(vec_anchor,vec_pos,vec_neg)

                total_loss += loss.item()

        return total_loss/len(val_dataloader)

    def InitiateTraining(self):
        df = pd.read_parquet(self.config.df_path)
        X=df['masked_log']
        y=df['hash_id']
        X_train,X_val,y_train,y_val =  train_test_split(X,y,test_size=.20,random_state=33)
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        train_dataset = TripletDataset(X_train,y_train)
        val_dataset = TripletDataset(X_val,y_val)

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, num_workers=3,
            collate_fn=create_collate_fn(self.tokenizer,self.config.max_len)
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, num_workers=3,
            collate_fn=create_collate_fn(self.tokenizer,self.config.max_len)
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.lr)
        self.loss_function = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - (x * y).sum(dim=1),
            margin=0.3
        )

        for epoch in range(self.epochs):
            self.model.train()
            total_loss=0
            for batch in train_dataloader:
                anchor_inputs = {k: v.to(self.device) for k, v in batch["anchor"].items()}
                pos_inputs = {k: v.to(self.device) for k, v in batch["positive"].items()}
                neg_inputs = {k: v.to(self.device) for k, v in batch["negative"].items()}
                
                vec_anchor = self.model(anchor_inputs)
                vec_pos = self.model(pos_inputs)
                vec_neg = self.model(neg_inputs)

                # dist_pos = 1.0 - (vec_anchor * vec_pos).sum(dim=1)
                # dist_neg = 1.0 - (vec_anchor * vec_neg).sum(dim=1)                
                # loss = torch.nn.functional.relu(dist_pos - dist_neg + 0.3).mean()

                loss = self.loss_function(vec_anchor,vec_pos,vec_neg)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
            
            train_loss = total_loss/len(train_dataloader)
            val_loss = self.eval(val_dataloader)

            print(f"for {epoch+1} epoch train_loss-> {train_loss} and val_loss-> {val_loss}")
            if val_loss<0.001:
                break
        
        torch.save(self.model.state_dict(), self.config.model_path + "stage1_encoder.pth")


# if __name__ == "__main__":
#     obj = Trainer(3)
#     obj.InitiateTraining()