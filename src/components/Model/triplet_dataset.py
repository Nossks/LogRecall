import pandas as pd
from torch.utils.data import DataLoader,Dataset
import random
from sklearn.model_selection import train_test_split

class TripletDataset(Dataset):
    def __init__(self,X,y):
        self.text = X
        self.hash = y

        self.hash_to_ind = self.hash.groupby(self.hash).groups
        self.all_hash = list(self.hash.unique())

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        anchor_text = self.text.iloc[index]
        anchor_hash = self.hash.iloc[index]

        pos_text = anchor_text

        neg_hash = random.choice(self.all_hash)
        while neg_hash==anchor_hash:
            neg_hash = random.choice(self.all_hash)
        
        neg_idx = random.choice(self.hash_to_ind[neg_hash])
        neg_text = self.text.iloc[neg_idx]

        return anchor_text,pos_text,neg_text