import os
import json
import random
import xml.etree.ElementTree as ET

import numpy as np

from utils.utils import get_classes

def load_image(path, set):
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
        if(set == 0):           #全是car的trainset
            img['path'] = 'nuimages/train/car/' + sample_path
        elif(set == 1):         #全是people的trainset
            img['path'] = 'nuimages/train/human/' + sample_path
        else:                   #有car&people的trainset
            img['path'] = 'nuimages/val/human&car/' + sample_path
        img['annotations'] = []
        for annotation in json_data['annotations']:
            if(annotation['sample_data_token'] == sample_data['token']):
                # img['annotations'].append(annotation)
                object={}
                #注意category从0开始，例如20类就是0~19
                if (annotation['category_token'] == 'fd69059b62a3469fbaef25340c0eab7f'):#car
                    category = 0
                elif (annotation['category_token'] == '1fa93b757fc74fb197cdd60001ad8abf'):#human
                    category = 1
                object['bbox']= annotation['bbox']
                object['category'] = category
                img['annotations'].append(object)
        imgs.append(img)
    return imgs

def generate_txt(imgs,filename):
    list_file = open('%s.txt'%(filename), 'w', encoding='utf-8')
    for imag in imgs:
        #imag filename
        list_file.write(imag['path'])
        #bbox & category
        for ann in imag['annotations']:
            bbox = ann['bbox']
            list_file.write(" " + ",".join([str(a) for a in bbox]) + ',' + str(ann['category']))
        list_file.write('\n')
    list_file.close()
    print('generate %s.txt'%(filename))

carpath = "nuimages/train/car.json"
car_imgs = []
car_imgs = load_image(carpath, 0)
generate_txt(car_imgs,'car_train')

humanpath = "nuimages/train/human.json"
human_imgs = []
human_imgs = load_image(humanpath, 1)
generate_txt(human_imgs,'human_train')

valpath = "nuimages/val/human&car.json"
val_imgs = []
val_imgs = load_image(valpath,2)
generate_txt(val_imgs,'val')