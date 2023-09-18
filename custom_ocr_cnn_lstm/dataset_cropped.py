import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json

from utils import remove_xml_tags


class OCRDatasetCropped(Dataset):
    def __init__(self, labels_dir, image_dir, transform=None):
        self.transform = transform
        self.labels_dir = labels_dir
        self.image_dir = image_dir
        self.labels = self.load_labels(labels_dir)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get all filenames form list of labels
        labels = list(self.labels)
        # Get filename from list of labels
        img_filename = labels[idx]["filename"]
        # img_filename = list(self.labels["filename"])[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        # print("Image size: ", image.size)
        if self.transform:
            image = self.transform(image)
        labels = self.labels[idx]["tokens"]
        labels = remove_xml_tags("".join(labels))
        item = {"idx": idx, "label": labels, "image": image}
        return item

    def load_labels(self, labels_file):
        with open(labels_file, "r") as f:
            labels = json.load(f)
        return labels
