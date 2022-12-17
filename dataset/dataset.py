# %%
import json
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms as tvtsf
from skimage import transform as sktsf
import torch as t
from dataset.util import flip_bbox, resize_bbox, random_flip, read_image



def inverse_normalize(img):
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()

def preprocess(img, min_size=600, max_size=1000):
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    normalize = pytorch_normalze
    return normalize(img)

# %% [markdown]
# This is a sample data loader

# %%
# def load_image(path, classes):
#     with open(path, 'r', encoding='utf8') as fp:
#             json_data = json.load(fp)
#     imgs = []
#     for sample_data in json_data['smaple_data']:
#         img = {}
#         img['height'] = sample_data['height']
#         img['width'] = sample_data['width']
#         for i in range(len(sample_data['filename'])):
#                     if sample_data['filename'][i] == 'n' and sample_data['filename'][i+1] == '0':
#                         break
#         sample_path = sample_data['filename'][i:]
#         if(classes == 0):
#             img['label'] = 0 #car
#             img['path'] = 'nuimages/train/car/' + sample_path
#         else:
#             img['label'] = 1 #human
#             img['path'] = 'nuimages/train/human/' + sample_path
#         img['annotations'] = []
#         for annotation in json_data['annotations']:
#             if(annotation['sample_data_token'] == sample_data['token']):
#                 img['annotations'].append(annotation)
#         imgs.append(img)
#     return imgs



def load_valimage(path):
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
        img['path'] = 'nuimages/val/human&car/' + sample_path
        img['annotations'] = []
        img['label'] = []
        for annotation in json_data['annotations']:
            if(annotation['sample_data_token'] == sample_data['token']):
                img['annotations'].append(annotation)
                if (annotation['category_token'] == 'fd69059b62a3469fbaef25340c0eab7f'):
                    img['label'].append(0)
                else: img['label'].append(1)
        imgs.append(img)
    return imgs


# %% [markdown]
# ## Custom Dataset

# %%
def Myloader(path):
    return Image.open(path).convert('RGB')

class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img_path = in_data['path']
        img = read_image(img_path)
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        H = in_data['height']
        W = in_data['width']
        scale = o_H / H
        bbox_list = []
        for obj in in_data['annotations']:
            bbox_list.append(obj['bbox'])
        bbox_list = np.array(bbox_list)
        #bbox = torch.from_numpy(bbox_list)
        bbox = resize_bbox(bbox_list, (H, W), (o_H, o_W))
        label = in_data['label']
        label = np.array(label)
        #label = torch.from_numpy(label)
        # horizontally flip
        img, params = random_flip(
            img, x_random=True, return_param=True)
        bbox = flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])
        return img, bbox, label, scale

# %%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data #data is image list
        self.tsf = Transform()

    def __getitem__(self, idx): #data is imgs[index]
        img, bbox, label, scale = self.tsf(self.data.data[idx])
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return_val = (img.copy(), bbox.copy(), label.copy(), scale)
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.data.data)


