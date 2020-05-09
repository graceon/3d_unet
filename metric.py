import cv2
import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio

from matplotlib import pyplot
import torch.nn as nn

# class MeanShift(nn.Conv2d):
#     def __init__():

#         super(MeanShift, self).__init__(3, 3, kernel_size=1)
#         std = torch.Tensor(rgb_std)
#         self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
#         self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
#         for p in self.parameters():
#             p.requires_grad = False
class weakness(nn.Module):
    def __init__(self,gt_1_max_threshold,gt_0_min_threshold):
        super(weakness, self).__init__()
        self.max_threshold=gt_1_max_threshold
        self.min_threshold=gt_0_min_threshold
    def forward(self,x,mask):



        

        weakness_gt_1=(x<self.max_threshold)*mask*x
        ceil=(x>self.max_threshold)*mask

        not_mask=-mask+1
        
        weakness_gt_0=(x>self.min_threshold)*not_mask*x
        #floor=(x<self.min_threshold)*not_mask-1
        



        res=weakness_gt_1+weakness_gt_0+ceil

        if 0:
            fig=pyplot.figure()
            fig.add_subplot(1, 5, 1)
            pyplot.imshow(x.clone().detach().view(512,512).cpu(),cmap='Greys_r')

            fig.add_subplot(1, 5, 2)
            pyplot.imshow(mask.view(512,512).cpu(),cmap='Greys_r')

            fig.add_subplot(1, 5, 3)
            pyplot.imshow(weakness_gt_0.clone().detach().view(512,512).cpu(),cmap='Greys_r')

            fig.add_subplot(1, 5, 4)
            pyplot.imshow(weakness_gt_1.clone().detach().view(512,512).cpu(),cmap='Greys_r')


            fig.add_subplot(1, 5, 5)
            pyplot.imshow(ceil.clone().detach().view(512,512).cpu(),cmap='Greys_r')

            pyplot.show()
        return res

def iou(predict,mask,threshold):
    # image_mask = cv2.imread(mask_name,0)
    # if np.all(image_mask == None):
    #     image_mask = imageio.mimread(mask_name)
    #     image_mask = np.array(image_mask)[0]
    #     image_mask = cv2.resize(image_mask,(576,576))
    #image_mask = mask
    # print(image.shape)
    # height = predict.shape[0]
    # weight = predict.shape[1]
    # print(height*weight)
    mask=(mask==1).float()
    predict=(predict>=threshold).float()

    interArea = predict*mask
    tem = predict + mask
    unionArea = tem - interArea
    inter = interArea.sum(dim=(-1,-2))

    union = unionArea.sum(dim=(-1,-2))
    


    if 0:
        fig=pyplot.figure()
        fig.add_subplot(1, 5, 1)
        pyplot.imshow(predict.clone().detach().view(512,512).cpu(),cmap='Greys_r')

        fig.add_subplot(1, 5, 2)
        pyplot.imshow(mask.view(512,512).cpu(),cmap='Greys_r')

        fig.add_subplot(1, 5, 3)
        pyplot.imshow(interArea.clone().detach().view(512,512).cpu(),cmap='Greys_r')

        fig.add_subplot(1, 5, 4)
        pyplot.imshow(unionArea.clone().detach().view(512,512).cpu(),cmap='Greys_r')


        fig.add_subplot(1, 5, 5)
        pyplot.imshow(tem.clone().detach().view(512,512).cpu(),cmap='Greys_r')

        pyplot.show()
    
    # iou_tem = torch.div(inter,union)
    iou_tem = inter/union

    # fig=pyplot.figure()
    # fig.add_subplot(1, 2, 1)
    # pyplot.imshow(mask.view(512,512).cpu(),cmap='Greys_r')
    # fig.add_subplot(1, 2, 2)
    # pyplot.imshow(predict.view(512,512).cpu(),cmap='Greys_r')
    # pyplot.show()
    #print(iou_tem.shape)
    # Iou = IOUMetric(2)  #2表示类别，肝脏类+背景类
    # Iou.add_batch(predict, image_mask)
    # a, b, c, d, e= Iou.evaluate()
    # print(iou_tem)
    return iou_tem

def dice(mask,predict,threshold):
    # image_mask = cv2.imread(mask_name, 0)
    # if np.all(image_mask == None):
    #     image_mask = imageio.mimread(mask_name)
    #     image_mask = np.array(image_mask)[0]
    #     image_mask = cv2.resize(image_mask,(576,576))
    # height = predict.shape[0]
    # weight = predict.shape[1]
    # o = 0
    # for row in range(height):
    #     for col in range(weight):
    #         if predict[row, col] < 0.5:  # 由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
    #             predict[row, col] = 0
    #         else:
    #             predict[row, col] = 1
    #         if predict[row, col] == 0 or predict[row, col] == 1:
    #             o += 1
    # height_mask = image_mask.shape[0]
    # weight_mask = image_mask.shape[1]
    # for row in range(height_mask):
    #     for col in range(weight_mask):
    #         if image_mask[row, col] < 125:  # 由于mask图是黑白的灰度图，所以少于125的可以看作是黑色
    #             image_mask[row, col] = 0
    #         else:
    #             image_mask[row, col] = 1
    #         if image_mask[row, col] == 0 or image_mask[row, col] == 1:
    #             o += 1

    predict = predict > threshold
    intersection = (predict*mask).sum()
    dice = (2. *intersection) /(predict.sum()+image_mask.sum())
    return dice

def hausdorff(mask,predict):
    # image_mask = cv2.imread(mask_name, 0)
    # # print(mask_name)
    # # print(image_mask)
    # if np.all(image_mask == None):
    #     image_mask = imageio.mimread(mask_name)
    #     image_mask = np.array(image_mask)[0]
    #     image_mask = cv2.resize(image_mask,(576,576))
    # #image_mask = mask
    # height = predict.shape[0]
    # weight = predict.shape[1]
    # o = 0
    # for row in range(height):
    #     for col in range(weight):
    #         if predict[row, col] < 0.5:  # 由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
    #             predict[row, col] = 0
    #         else:
    #             predict[row, col] = 1
    #         if predict[row, col] == 0 or predict[row, col] == 1:
    #             o += 1
    # height_mask = image_mask.shape[0]
    # weight_mask = image_mask.shape[1]
    # for row in range(height_mask):
    #     for col in range(weight_mask):
    #         if image_mask[row, col] < 125:  # 由于mask图是黑白的灰度图，所以少于125的可以看作是黑色
    #             image_mask[row, col] = 0
    #         else:
    #             image_mask[row, col] = 1
    #         if image_mask[row, col] == 0 or image_mask[row, col] == 1:
    #             o += 1

    hd1 = directed_hausdorff(image_mask, predict)[0]
    hd2 = directed_hausdorff(predict, image_mask)[0]
    res = None
    if hd1>hd2 or hd1 == hd2:
        res=hd1
        return res
    else:
        res=hd2
        return res



# def show(predict):
#     height = predict.shape[0]
#     weight = predict.shape[1]
#     for row in range(height):
#         for col in range(weight):
#             predict[row, col] *= 255
#     plt.imshow(predict)
#     plt.show()