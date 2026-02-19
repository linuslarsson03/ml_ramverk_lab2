import torch
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
        def __init__(self, data):
            super().__init__()
            self.data = pd.read_csv("data/trainLabel.csv")



