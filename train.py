from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from dataset.dataset import Dataset, inverse_normalize, load_image
from Net.fast_rcnn import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
# import resource
#
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))
#
# matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for imgs, sizes, gt_bboxes_, gt_labels_ in dataloader:
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    carpath = "nuimages/train/car.json"
    humanpath = "nuimages/train/human.json"
    car_imgs = []
    car_imgs = load_image(carpath, 0)
    car_train = Dataset(car_imgs)
    dataset = Dataset(car_train)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True)
    # testset = TestDataset(opt)
    # test_dataloader = data_.DataLoader(testset,
    #                                    batch_size=1,
    #                                    num_workers=opt.test_num_workers,
    #                                    shuffle=False, \
    #                                    pin_memory=True
    #                                    )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    # trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr = 1e-3
    for epoch in range(30):
        trainer.reset_meters()
        for img, bbox_, label_, scale in dataloader:
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

        eval_result = eval(dataloader, faster_rcnn)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        # log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
        #                                           str(eval_result['map']),
        #                                           str(trainer.get_meter_data()))
        # trainer.vis.log(log_info)
        #
        # if eval_result['map'] > best_map:
        #     best_map = eval_result['map']
        #     best_path = trainer.save(best_map=best_map)
        # if epoch == 9:
        #     trainer.load(best_path)
        #     trainer.faster_rcnn.scale_lr(opt.lr_decay)
        #     lr_ = lr_ * opt.lr_decay
        #
        # if epoch == 13:
        #     break


if __name__ == '__main__':
    # import fire
    #
    # fire.Fire()
    train()