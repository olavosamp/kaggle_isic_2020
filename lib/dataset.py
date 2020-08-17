import numpy as np
import skimage.io
import torch
import torch.utils.data
import csv
import os
import lib.defines as defs

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, metadata, target, transform, balance):
        self.image_path = image_path
        self.metadata = metadata
        self.target = target
        self.balance = balance
        self.transform = transform
        # If the dataset will be balanced, change length to the double of the
        # largest class (the negatives, in this case).
        if self.balance:
            self.length = 2 * (1 - self.target).sum()
        else:
            self.length = len(self.target)

    def imbalance_ratio(self):
        # Compute the ratio between the negative and positive classes.
        # If the dataset is artificially balanced, this ratio is 1.
        if self.balance:
            return 1.0
        else:
            return (1 - self.target).sum() / self.target.sum()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Balance the dataset by using the positive samples multiple times.
        # Indices between the original size and 2 times the number of negative
        # samples are reassigned to positive sample indices.
        if self.balance:
            if idx < 0:
                idx = self.length + idx
            if idx >= len(self.target):
                idx = (idx - len(self.target)) % self.target.sum()
                idx = (self.target != 0).nonzero()[0][idx].item()
        # Read and transform the image.
        image = skimage.io.imread(self.image_path[idx])
        if self.transform is not None:
            image = self.transform(image)
        # Convert metadata and target to torch tensors.
        # metadata = torch.from_numpy(self.metadata[idx], dtype=torch.float)
        metadata = torch.as_tensor(self.metadata[idx], dtype=torch.float)
        target   = torch.tensor(self.target[idx])
        return image, metadata, target

def create_dataset(data_path, csv_path, mean_age=48.9, std_age=14.4,
        transform=None, balance=False, sample=1.0):
    image_path = []
    metadata = []
    target = []
    anatom_category = defs.metadata_anatom_categories
    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        # Read path and metadata from the CSV file.
        for row in reader:
            image_filename = row["image_name"] + ".jpg"
            image_path.append(os.path.join(data_path, image_filename))
            metadata.append([0.0] * (3 + len(anatom_category)))

            # One-hot encode sex. Encoded as zeros if it is missing.
            if row["sex"] == "female":
                metadata[-1][0] = 1.0
            elif row["sex"] == "male":
                metadata[-1][1] = 1.0

            # If age is missing, assume it is the mean.
            if row["age_approx"] == "":
                metadata[-1][2] = mean_age
            else:
                metadata[-1][2] = float(row["age_approx"])

            # One-hot encode anatom site. Encodead as zeros if it is missing.
            for i, category in enumerate(anatom_category):
                if row["anatom_site_general_challenge"] == category:
                    metadata[-1][3 + i] = 1.0
            target.append(int(row["target"]))
    metadata = np.array(metadata)
    # Normalize age.
    metadata[:, 2] -= mean_age
    metadata[:, 2] /= std_age
    target = np.array(target)
    
    if sample < 1.0:
        # Sample only a fraction of the dataset (non-randomized)
        sample_index = round(sample*len(image_path))
        image_path  = image_path[:sample_index]
        metadata    = metadata[:sample_index, :]
        target      = target[:sample_index]

    # Create dataset.
    dataset = Dataset(image_path, metadata, target, transform, balance)
    return dataset
