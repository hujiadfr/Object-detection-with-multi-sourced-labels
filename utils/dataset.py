# %%
import json
import os
import random
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from pycocotools.mask import encode, decode, area, toBbox




# %% [markdown]
# This is a sample data loader

# %%
def load_image(path, classes):
    with open(path, 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
    imgs = []
    for sample_data in json_data['smaple_data']:
        img = {}
        img['height'] = sample_data['height']
        img['width'] = sample_data['width']
        for i in range(len(sample_data['filename'])):
                    if sample_data['filename'][i] == 'n' and sample_data['filename'][i+1] == '0':
                        break
        sample_path = sample_data['filename'][i:]
        if(classes == 0):
            img['label'] = 0 #car
            img['path'] = 'nuimages/train/car/' + sample_path
        else:
            img['label'] = 1 #human
            img['path'] = 'nuimages/train/human/' + sample_path
        img['annotations'] = []
        for annotation in json_data['annotations']:
            if(annotation['sample_data_token'] == sample_data['token']):
                img['annotations'].append(annotation)
        imgs.append(img)
    return imgs


# %% [markdown]
# ## Custom Dataset

# %%
def Myloader(path):
    return Image.open(path).convert('RGB')

# %%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, loader=Myloader):
        self.data = data #data is image list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, idx): #data is imgs[index]
        img_path = self.data[idx]['path']
        img = self.loader(img_path)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

    def get_bbox(self, idx):
        img = self.data[idx]
        bbox_list = []
        for obj in img['annotations']:
            bbox_list.append(obj['bbox'])
        return bbox_list


