import os
import torchio as tio
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
import torchio as tio  
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

class ScanDataset2D(Dataset):
    def __init__(self, cloud_file, splits_file, is_training, augment):
        self.data = pd.read_csv(splits_file)
        self.cloud_data = pd.read_csv(cloud_file)  # external file with image_path, mask_path, label
        self.resize = Resize((256, 256))
        self.is_training = is_training
        self.augment = augment

        # Keep only relevant columns
        self.data = self.data[['patient_id', f'split']]
        if self.is_training:
            self.data = self.data[self.data[f'split'] == 'train']
        else:
            self.data = self.data[self.data[f'split'] == 'val']

        # Mapping from patient_id to (image_path, mask_path, label)
        self.label_map = {
            row.patient_id: (row.image_path, row.mask_path, row.label)
            for row in self.cloud_data.itertuples()
        }

        # Augmentation
        self.augmentations = tio.OneOf({
            tio.RandomFlip(axes=0, p=1.0): 0.33,
            tio.RandomFlip(axes=1, p=1.0): 0.33,
            tio.Compose([]): 0.34
        })

        self.slices = []
        self.patient_info = []
        self.patient_slice_map = {}

        print("Pre-loading dataset...")
        for idx, row in enumerate(tqdm(self.data.itertuples(), total=len(self.data))):
            patient_id = row.patient_id

            # Use paths from label_file if available
            if patient_id not in self.label_map:
                continue  # skip if label info is missing

            img_path, mask_path, label = self.label_map[patient_id]

            img = nib.load(img_path).get_fdata()
            mask = nib.load(mask_path).get_fdata()

            start_idx = len(self.slices)
            for slice_idx in range(img.shape[2]):
                img_slice = img[:, :, slice_idx]
                mask_slice = mask[:, :, slice_idx]

                # Normalize and resize
                img_slice = self.normalize_slice(img_slice)
                img_slice = self.resize(torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0))
                mask_slice = torch.tensor(mask_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mask_slice = F.interpolate(mask_slice, size=[256, 256], mode='nearest').squeeze(0)

                # Apply augmentation
                if self.is_training and self.augment:
                    subject = tio.Subject(
                        image=tio.ScalarImage(tensor=img_slice.unsqueeze(0)),
                        mask=tio.LabelMap(tensor=mask_slice.unsqueeze(0))
                    )
                    subject = self.augmentations(subject)
                    img_slice = subject.image.tensor.squeeze(0)
                    mask_slice = subject.mask.tensor.squeeze(0)

                # Store the data
                self.slices.append((img_slice, mask_slice, label))
                self.patient_info.append((patient_id, slice_idx))
            self.patient_slice_map[patient_id] = (start_idx, len(self.slices))

    def normalize_slice(self, slice_data):
        min_val = slice_data.min()
        max_val = slice_data.max()
        return (slice_data - min_val) / (max_val - min_val + 1e-8)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img, mask, label = self.slices[idx]
        patient_id, slice_idx = self.patient_info[idx]
        return img, mask, patient_id, slice_idx, label

# splits_file = rf"glapalh/splits.csv"
# cloud_file = rf"glapalh/cloud.csv" 

#dataset = ScanDataset2D( cloud_file=cloud_file, splits_file=splits_file, is_training=True, augment=False)

# sample = dataset[0]
# img, mask, patient_id, slice_idx, label = sample

# print("Image shape:", img.shape)
# print("Mask shape:", mask.shape)
# print("Patient ID:", patient_id)
# print("Slice Index:", slice_idx)
# print("Label:", label)
