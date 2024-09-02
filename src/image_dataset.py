import h5py
from PIL import Image
from io import BytesIO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
        self,
        df,
        file_hdf,
        mode,
        img_size,
    ):
        self.mode = mode
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.df_positive = df[df["target"] == 1].reset_index(drop=True)
        self.df_negative = df[df["target"] == 0].reset_index(drop=True)

        if mode == "train":
            self.df = pd.concat(
                [
                    self.df_positive,
                    self.df_negative.sample(len(self.df_positive), random_state=0),
                ]
            ).sort_values("isic_id")

            self.transforms = A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.RandomRotate90(p=0.5),
                    A.Flip(p=0.5),
                    A.Downscale(p=0.25),
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.15,
                        rotate_limit=60,
                        p=0.5,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.5
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.1, 0.1),
                        contrast_limit=(-0.1, 0.1),
                        p=0.5
                    ),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0
                    ),
                    ToTensorV2(),
                ],
                p=1.
            )
        elif mode == "valid":
            self.df = pd.concat(
                [
                    self.df_positive,
                    self.df_negative.sample(len(self.df_positive) * 20, random_state=42),
                ]
            ).sort_values("isic_id")
            self.transforms = A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0
                    ),
                    ToTensorV2()
                ],
                p=1.,
            )
        elif mode == "eval":
            self.df = df
            self.transforms = A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0
                    ),
                    ToTensorV2()
                ],
                p=1.,
            )
        else:
            raise Exception(f"unknown mode: {mode}")

        self.isic_ids = self.df['isic_id'].values
        self.targets = self.df['target'].values

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))
        target = self.targets[index]

        img = self.transforms(image=img)["image"]

        return {
            'image': img,
            'target': target,
        }

    def resample(self, random_state):
        self.df = pd.concat(
            [
                self.df_positive,
                self.df_negative.sample(len(self.df_positive), random_state=random_state),
            ]
        )
        self.isic_ids = self.df['isic_id'].values
        self.targets = self.df['target'].values
