import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class OCRDataset(Dataset):
    def __init__(self, labels_dir, image_dir, transform=None):
        self.transform = transform
        self.labels_dir = labels_dir
        self.image_dir = image_dir
        self.labels = self.load_labels(labels_dir)
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_filename = list(self.labels.keys())[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.labels[img_filename]

    def load_labels(self, labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        return labels
