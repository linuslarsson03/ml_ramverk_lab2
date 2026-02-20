import pandas as pd
import os 
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
from src.utils import TRAIN_DATA_PATH, TRAINLABELS_CSV_PATH



class CustomDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.data = pd.read_csv(TRAINLABELS_CSV_PATH)
        self.image_path = TRAIN_DATA_PATH
        self.transform = transform

        self.unique_labels = sorted(self.data["label"].unique())
        self.label_map = {label: idx for idx, label in enumerate(self.unique_labels)}


   
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_id = row["id"]
        label_str = row["label"]
        label = self.label_map[label_str]
        
        image_path = os.path.join(self.image_path, f"{image_id}.png")
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, label

    def makee_dataloaders(cfg):
        transform = transforms.Compose([
            transforms.ToTensor()])
        
        dataset = CustomDataset(transform=transform)

        validation_size = int(len(dataset) * cfg.validation_split)
        train_size = len(dataset) - validation_size

        train_ds, validation_ds = random_split(dataset, [train_size, validation_size])

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_ds, batch_size=cfg.batch_size, shuffle=False)

        return train_loader, validation_loader