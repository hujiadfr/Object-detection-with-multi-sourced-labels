
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import array_tool as at
from torchvision.ops import RoIPool
class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier #vgg16中的classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        nn.init.normal_(self.cls_loc.weight, 0, 0.001)
        nn.init.normal_(self.score.weight, 0, 0.01) #全连接层权重初始化

        self.n_class = n_class #加上背景3类
        self.roi_size = roi_size #7
        self.spatial_scale = spatial_scale # 1/16
        #将大小不同的roi变成大小一致，得到pooling后的特征，大小为[300, 512, 7, 7]。.利用Cupy实现在线编译的
        #使用的是from torchvision.ops import RoIPool
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)
    def forward(self, x, rois, roi_indices):
        """Forward the chain."""
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float() #ndarray->tensor
        rois = at.totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous() #把tensor变成在内存中连续分布的形式

        pool = self.roi(x, indices_and_rois) #ROIPOOLING
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool) #decom_vgg16（）得到的calssifier,得到4096
        roi_cls_locs = self.cls_loc(fc7) #（4096->84） 84=21*4
        roi_scores = self.score(fc7) #（4096->21）
        return roi_cls_locs, roi_scores
