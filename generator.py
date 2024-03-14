import torch
import numpy as np
import pandas as pd
import rasterio
import os

BASE_PATH = 'D:\\'
IMAGES_PATH = BASE_PATH + 'data/train_img'
MASKS_PATH = BASE_PATH + 'data/train_mask'

TRAIN_IMAGE = 'D:\data/train_img'
TRAIN_MASK =  'D:\data/train_mask'
TEST_IMAGE = 'D:\data/test_img'
TRAIN_CSV = 'D:\data/train_meta.csv'
TEST_CSV = 'D:\data/test_meta.csv'


class CustomDataGenerator(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None):
        # read file names from csv files
        df = pd.read_csv(BASE_PATH + csv_path)    
        img, mask = df.columns.tolist()
        
        self.images = [os.path.join(TRAIN_IMAGE, x) for x in df[img].tolist()]
        self.masks = [os.path.join(TRAIN_MASK, x) for x in df[mask].tolist()]

        self.MAX_PIXEL_VALUE = 65535                                            # defined in the original code
        self.transform = transform                                              # image transforms for data augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = rasterio.open(self.images[idx]).read((7,6,2))               # only extract 3 channels  
        img = np.float32(img.transpose((1, 2, 0))) / self.MAX_PIXEL_VALUE

        mask = rasterio.open(self.masks[idx]).read().transpose((1, 2, 0))
        sample = {'image': img, 'mask': mask}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

class CustomDataGeneratorTest(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None):
        # read file names from csv files
        df = pd.read_csv(BASE_PATH + csv_path)    
        img, _ = df.columns.tolist()
        
        self.images = [os.path.join(TEST_IMAGE, x) for x in df[img].tolist()]

        self.MAX_PIXEL_VALUE = 65535                                            # defined in the original code
        self.transform = transform                                              # image transforms for data augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = rasterio.open(self.images[idx]).read((7,6,2))               # only extract 3 channels  
        img = np.float32(img.transpose((1, 2, 0))) / self.MAX_PIXEL_VALUE

        sample = {'image': img, 'name': self.images[idx].split('\\')[-1]}
        if self.transform:
            sample = self.transform(sample)

        return sample


from matplotlib import pyplot as plt
if __name__ == '__main__':
    # train_dataset = CustomDataGenerator('data/train_meta.csv')
    train_dataset = CustomDataGeneratorTest('data/test_meta.csv')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    for i, sample in enumerate(train_loader):
        print(i, sample['image'].shape, sample['name'])
        # plt.imshow(sample['image'][0])
        # plt.show()
        if i == 0:
            break