# Project 6
1. 数据集预处理
2. 将其输入到一个预训练好的神经网络中（ResNeXt等）获得对应的feature map；
3. 对这个feature map中的每一点设定预定个的ROI，从而获得多个候选ROI；
4. 将这些候选的ROI送入RPN网络进行二值分类（前景或背景）和BB回归，过滤掉一部分候选的ROI；
5. 对这些剩下的ROI进行ROIAlign操作（即先将原图和feature map的pixel对应起来，然后将feature map和固定的feature对应起来）；
6. 对这些ROI进行分类（N类别分类）、BB回归和MASK生成（在每一个ROI里面进行FCN操作）