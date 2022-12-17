import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision.ops import nms
from chainer import cuda
from Net.utils.bbox_tools import loc2bbox


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):          # 利用base anchor生成所有对应feature map的anchor
    # Enumerate all shifted anchors:                                             # anchor_base :(9,4) 坐标，这里 A=9
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    shift_y = xp.arange(0, height * feat_stride, feat_stride)           # 纵向偏移量（0，16，32，...）
    shift_x = xp.arange(0, width * feat_stride, feat_stride)            # 横向偏移量（0，16，32，...）
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]    # 9
    K = shift.shape[0]          # K = hh*ww  ，K约为20000
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor     # 返回（K，4），所有anchor的坐标



class ProposalCreator:
    def __init__(self, parent_model, nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000,
                 n_test_pre_nms=6000, n_test_post_nms=300, min_size=16):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        # Different numbers of candidate boxes are used in the training phase and reasoning phase
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        # The actual information of anchor is obtained according to the offset
        roi = loc2bbox(anchor, loc)
        # Limit the width and height of the prediction box to the preset range
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]
        # Sort to get the candidate box of the high confidence part
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]
        # The nms process is not expanded in detail here. pytorch1.2 + can be directly used by importing nms through from torchvision.ops import
        keep = nms(torch.from_numpy(roi).cuda(), torch.from_numpy(score).cuda(), self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        # Returns the generated candidate box
        return roi


def generate_base_anchor(base_size=16, ratios=None, anchor_scale=None):
    """
    It is assumed that the size of the obtained characteristic graph is w×h，A total of 9 are generated at each location anchor，So there are anchor
    Number is w×h×9. In the original paper anchor The ratio is 1:2,2:1 And 1:1，Scales 128, 256, and 512 (relative to
    In terms of the original drawing). Therefore, the actual scales on the feature map sampled at 16 times are 8, 16 and 32.
    """
    # Ratio and scale of anchor
    if anchor_scale is None:
        anchor_scale = [8, 16, 32]
    if ratios is None:
        ratios = [0.5, 1, 2]
    # The position of the upper left corner of the feature map is mapped back to the position of the original map
    py = base_size / 2
    px = base_size / 2
    # Initialization variable (9,4). Here, take the top left corner vertex of the feature graph as an example to generate anchor
    base_anchor = np.zeros((len(ratios) * len(anchor_scale), 4), dtype=np.float32)
    # The loop generates 9 anchor s
    for i in range(len(ratios)):
        for j in range(len(anchor_scale)):
            # Generate height and width (relative to the original)
            # Take i=0, j=0 as an example, h=16 × eight × (0.5)^1/2,w=16 × eight × 1 / 0.5, then h × w=128^2
            h = base_size * anchor_scale[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scale[j] * np.sqrt(1. / ratios[i])
            # Index of currently generated anchor (0 ~ 8)
            index = i * len(anchor_scale) + j
            # Calculate the upper left and lower right coordinates of anchor
            base_anchor[index, 0] = py - h / 2
            base_anchor[index, 1] = px - w / 2
            base_anchor[index, 2] = py + h / 2
            base_anchor[index, 3] = px + w / 2
    # Anchor relative to the original size (x_min, y_min, x_max, y_max)
    return base_anchor

def generate_all_base_anchor(base_anchor, feat_stride, height, width):
    """
    height*feat_stride/width*feat_stride Equivalent to the height of the original drawing/Width, equivalent to starting from 0,
    every other feat_stride=16 Sampling a position, which is equivalent to gradually sampling on the feature map sampled at 16 times. this
    A process is used to identify each group anchor The center point of the.
    """
    # Longitudinal offset [0,16,32,...]
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    # Lateral offset [0,16,32,...]
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    # np.meshgrid is used to change two one-dimensional vectors into two two-dimensional matrices. Where, the first 2D returned
    # The row vector of the matrix is the first parameter and the number of repetitions is the length of the second parameter; Column vector of the second two-dimensional matrix
    # Is the second parameter and the number of repetitions is the length of the first parameter. The resulting shift_x and shift_y as follows:
    # shift_x = [[0,16,32,...],
    #            [0,16,32,...],
    #            [0,16,32,...],
    #            ...]
    # shift_y = [[0, 0, 0,... ],
    #            [16,16,16,...],
    #            [32,32,32,...],
    #            ...]
    # Notice the shift_x and shift_y is equal to the scale of the feature graph, and the of each position corresponds to the scale of the feature graph
    # The combination of the values of the two matrices corresponds to the point on the characteristic graph mapped back to the upper left corner coordinate of the original graph.
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # NP. Travel () expands the matrix into a one-dimensional vector, shift_x and shift_ The expanded forms of Y are:
    # [0,16,32,...,0,16,32,..,0,16,32,...]，(1,w*h)
    # [0,0,0,...,16,16,16,...,32,32,32,...]，(1,w*h)
    # axis=0 is equivalent to stacking by line, and the shape obtained is (4,w*h);
    # axis=1 is equivalent to stacking by column, and the resulting shape is (w*h,4). The value of shift obtained by this statement is:
    # [[0,  0, 0,  0],
    #  [16, 0, 16, 0],
    #  [32, 0, 32, 0],
    #  ...]
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    # Number of anchor s per location
    num_anchor_per_loc = base_anchor.shape[0]
    # Gets the total number of positions on the feature graph
    num_loc = shift.shape[0]
    # Use generate_ base_ The anchor at the upper left corner generated by anchor can be obtained by adding the offset
    # The following anchor information (here only refers to the change of anchor center point position, not the change of anchor)
    # Width and height). We first define the shape of the final anchor. We know that it should be w*h*9, then all
    # The stored variables of anchor are (w*h*9,4). First, the anchor shape generated by the first position is changed to
    # (1, 9, 4), and then change the shape of the shift to (1,w*h,4). And change it through the transfer function
    # The shape of the shift is (w*h,1,4), and then the broadcast mechanism is used to add the two, that is, the shape of the two is divided
    # Do not be (1,num_anchor_per_loc,4)+(num_loc,1,4), and finally add the result
    # The shape is (num_loc,num_anchor_per_loc,4). Here, the first item added is:
    # [[[x_min_0,y_min_0,x_max_0,y_max_0],
    #   [x_min_1,y_min_1,x_max_1,y_max_1],
    #   ...,
    #   [x_min_8,y_min_8,x_max_8,y_max_8]]]
    # The second item added is:
    # [[[0,  0, 0,  0]],
    #  [[0, 16, 0, 16]],
    #  [[0, 32, 0, 32]],
    #  ...]
    # In the process of addition, we first expand the two addends into the target shape. Specifically, the first one can
    # Expand to:
    # [[[x_min_0,y_min_0,x_max_0,y_max_0],
    #   [x_min_1,y_min_1,x_max_1,y_max_1],
    #   ...,
    #   [x_min_8,y_min_8,x_max_8,y_max_8]],
    #  [[x_min_0,y_min_0,x_max_0,y_max_0],
    #   [x_min_1,y_min_1,x_max_1,y_max_1],
    #   ...,
    #   [x_min_8,y_min_8,x_max_8,y_max_8]],
    #  [[x_min_0,y_min_0,x_max_0,y_max_0],
    #   [x_min_1,y_min_1,x_max_1,y_max_1],
    #   ...,
    #   [x_min_8,y_min_8,x_max_8,y_max_8]],
    #   ...]
    # The second can be expanded to:
    # [[[0,  0, 0,  0],
    #   [0,  0, 0,  0],
    #   ...],
    #  [[0, 16, 0, 16],
    #   [0, 16, 0, 16],
    #   ...],
    #  [[0, 32, 0, 32],
    #  [0, 32, 0, 32],
    #  ...],
    #  ...]
    # Now the two dimensions are consistent and can be added directly. The resulting shape is:
    # (num_loc,num_anchor_per_loc,4)
    anchor = base_anchor.reshape((1, num_anchor_per_loc, 4)) + \
             shift.reshape((1, num_loc, 4)).transpose((1, 0, 2))
    # reshape the anchor shape to the final shape (num_loc*num_anchor_per_loc,4).
    anchor = anchor.reshape((num_loc * num_anchor_per_loc, 4)).astype(np.float32)
    return anchor


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32], feat_stride=16,
                 proposal_creator_params=dict(), ):
        super(RegionProposalNetwork, self).__init__()
        # anchor corresponding to the top left corner of the feature graph
        self.anchor_base = generate_base_anchor(anchor_scale=anchor_scales, ratios=ratios)
        # Down sampling multiple
        self.feat_stride = feat_stride
        # Generate candidate boxes for Fast RCNN
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        # Weight initialization
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.score.weight, 0, 0.01)
        nn.init.normal_(self.loc.weight, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, hh, ww = x.shape
        # Generate all anchor s
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, hh, ww)
        n_anchor = anchor.shape[0] // (hh * ww)
        # Regression branch of rpn part
        h = F.relu(self.conv1(x))
        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # Classification branch of rpn part
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)
        # Generate rois part
        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)
        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor