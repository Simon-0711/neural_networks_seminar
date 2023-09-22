import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from torchvision import datasets

class EMNISTDataset(Dataset):
    def __init__(self, number_of_sequences, digits_per_sequence):
        self.emnist_dataset = datasets.EMNIST('./EMNIST', split="digits", train=True, download=True)
        self.number_of_sequences = number_of_sequences
        self.digits_per_sequence = digits_per_sequence
        self.transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.2, 0.15), scale=(0.8, 1.1)),
            transforms.ToTensor()
        ])
        self.data, self.labels = self.prepare_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def prepare_data(self):
        dataset_sequences = []
        dataset_labels = []

        for i in range(self.number_of_sequences):
            random_indices = np.random.randint(len(self.emnist_dataset.data), size=(self.digits_per_sequence,))
            random_digits_images = self.emnist_dataset.data[random_indices]
            transformed_random_digits_images = []

            for img in random_digits_images:
                img = transforms.ToPILImage()(img)
                img = TF.rotate(img, -90, fill=0)
                img = TF.hflip(img)
                img = self.transform(img).numpy()
                transformed_random_digits_images.append(img)

            random_digits_images = np.array(transformed_random_digits_images)
            random_digits_labels = self.emnist_dataset.targets[random_indices]
            random_sequence = np.hstack(random_digits_images.reshape((self.digits_per_sequence, 28, 28)))
            random_labels = np.hstack(random_digits_labels.reshape(self.digits_per_sequence, 1))
            dataset_sequences.append(random_sequence / 255)
            dataset_labels.append(random_labels)

        dataset_data = torch.Tensor(np.array(dataset_sequences))
        dataset_labels = torch.IntTensor(np.array(dataset_labels))

        return dataset_data, dataset_labels


# Create the EMNIST dataset
from torch.utils.data import DataLoader, random_split
import hyperparameter as hp

def get_emnist_data_loaders():
    emnist_dataset = EMNISTDataset(hp.NUMBER_OF_SEQUENCES, hp.DIGITS_PER_SEQUENCE)

    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(emnist_dataset))
    val_size = len(emnist_dataset) - train_size
    train_set, val_set = random_split(emnist_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True)
    return train_loader, val_loader