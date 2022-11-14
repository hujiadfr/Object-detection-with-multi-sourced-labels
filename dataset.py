# %%
import json
import os
import random
import numpy as np
import torch
import torch.utils.data
from PIL import Image


# %%
carpath = "nuimages/train/car.json"
humanpath = "nuimages/train/human.json"

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
imgs = []
imgs = load_image(carpath, 0)

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
        obj_ids = self.data[idx]['annotations']
        boxes = []
        labels = []
        # read bbox and lable
        for obj in obj_ids: #bbox <int> [4] -- Annotated amodal bounding box. Given as [xmin, ymin, xmax, ymax].
            boxes.append(obj['bbox'])
            labels.append(img['label'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        area = (boxes[:, 3]-boxes[:,1]) * (boxes[:, 2]-boxes[:, 0])
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["lables"] = labels
        target["image_id"] = image_id
        target["area"] = area

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


