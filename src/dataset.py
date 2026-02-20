import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from src.utils import TRAIN_DATA_PATH
from src.utils import TRAINLABELS_CSV_PATH


class CustomDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv(TRAINLABELS_CSV_PATH)
        self.image_path = TRAIN_DATA_PATH

        self.unique_labels = sorted(self.data["label"].unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}


   
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return 
    


