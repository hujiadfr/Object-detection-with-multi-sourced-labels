from __future__ import  absolute_import
from dataset.dataset import Dataset, TestDataset,inverse_normalize, load_image, load_testimage
from Net.fast_rcnn import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc


def eval(dataloader, faster_rcnn, test_num=10):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    count = 0
    for imgs, sizes, gt_bboxes_, gt_labels_ in dataloader:
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        # gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if count == test_num: break
        count += 1

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults=None,
        use_07_metric=True)
    return result


def train(**kwargs):
    carpath = "nuimages/train/car.json"
    humanpath = "nuimages/train/human.json"
    testpath = "nuimages/val/human&car.json"
    car_imgs = []
    car_imgs = load_image(carpath, 0)
    human_imgs = []
    human_imgs = load_image(humanpath, 1)
    human_train = Dataset(human_imgs)
    car_train = Dataset(car_imgs)

    print('load data')
    car_dataloader = data_.DataLoader(car_train, \
                                  batch_size=1, \
                                  shuffle=True)
    human_dataloader = data_.DataLoader(human_train, \
                                  batch_size=1, \
                                  shuffle=True)
    test_imgs = load_testimage(testpath)
    testset = TestDataset(test_imgs)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       shuffle=False)
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    # trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr = 1e-3
    lr_decay = 0.1
    best_map = 0
    for epoch in range(100):
        trainer.reset_meters()
        for img, bbox_, label_, scale in car_dataloader:
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)
        for img, bbox_, label_, scale in human_dataloader:
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

        eval_result = eval(test_dataloader, faster_rcnn)
        #trainer.vis.plot('test_map', eval_result['map'])
        print(eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        print(log_info)
        #
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 50:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(lr_decay)
            lr_ = lr_ * lr_decay



if __name__ == '__main__':
    # import fire
    #
    # fire.Fire()
    train()