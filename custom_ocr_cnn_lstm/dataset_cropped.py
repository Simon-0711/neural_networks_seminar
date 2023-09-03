import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class OCRDatasetCropped(Dataset):
    def __init__(self, labels_dir, image_dir, transform=None):
        self.transform = transform
        self.labels_dir = labels_dir
        self.image_dir = image_dir
        self.labels = self.load_labels(labels_dir)
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_filename = list(self.labels.keys())[idx]
        crop_info = self.labels[img_filename]['html']['cells'][0]['bbox']  # Assuming the first bounding box contains crop info
        crop_info = "_".join(map(str, crop_info))  # Convert the crop info to a string
        img_filename = img_filename.split(".")[0]  # Remove the file extension
        img_filename = f"{img_filename}_bbox_{crop_info}.png"  # Construct the new img_filename
        img_path = os.path.join(self.base_path, self.labels[img_filename]['split'], img_filename)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[img_filename]

        return label, image

    def load_labels(self, labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        return labels
