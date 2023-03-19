"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/AL-SSL/


from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.augmentations import jaccard_numpy

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
# VOC_ROOT = osp.join(HOME, "tmp/VOC0712/")
VOC_ROOT = '/usr/wiss/elezi/data/VOC0712'

class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    def __init__(self, root, supervised_indices=None,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712', pseudo_labels={}, bounding_box_dict={}):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.supervised_indices = supervised_indices
        self.pseudo_labels = pseudo_labels
        self.pseudo_labels_indices = self.get_pseudo_label_indices()
        self.bounding_box_dict = bounding_box_dict
        self.bounding_box_indices = self.get_bounding_box_indices()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
        self.class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))

    def __getitem__(self, index):
        im, gt, h, w, semi = self.pull_item(index)
        return im, gt, semi

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            if index in self.bounding_box_indices:
                target_real = self.target_transform(target, width, height)
                target = self.target_transform_bounding_box(self.bounding_box_dict, index, target_real, width, height)

            else:
                target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])

            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        if self.supervised_indices != None:
            if index in self.supervised_indices:
                semi = np.array([1])
            elif index in self.pseudo_labels_indices or index in self.bounding_box_indices:
                semi = np.array([2])
            # elif index in self.bounding_box_indices:
            #    semi = np.array([2])
            #    print(semi)
            else:
                semi = np.array([0])
                target = np.zeros([1,5])
        else:
            # it does not matter
            semi = np.array([0])

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width, semi

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_pseudo_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        return self.pseudo_labels[index]

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def get_pseudo_label_indices(self):
        pseudo_label_indices = []
        if self.pseudo_labels is not None:
            for key in self.pseudo_labels:
                pseudo_label_indices.append(key)
            return pseudo_label_indices
        else:
            return None

    def get_bounding_box_indices(self):
        bounding_box_indices = []
        for key in self.bounding_box_dict:
            bounding_box_indices.append(key)
        return bounding_box_indices

    def target_transform_pseudo_label(self, pseudo_labels, index):
        # height, width = pseudo_labels[index][0][3], pseudo_labels[index][0][4]
        height, width = pseudo_labels[index][0][4], pseudo_labels[index][0][3]
        # height, width = 300, 300
        res = []
        for i in range(len(pseudo_labels[index])):
            pts = [pseudo_labels[index][i][5], pseudo_labels[index][i][6], pseudo_labels[index][i][7], pseudo_labels[index][i][8]]
            name = pseudo_labels[index][i][2]
            pts[0] /= height
            pts[2] /= height
            pts[1] /= width
            pts[3] /= width
            for i in range(len(pts)):
                if pts[i] < 0:
                    pts[i] = 0
            if pts[0] > height:
                pts[0] = height
            if pts[2] > height:
                pts[2] = height
            label_idx = float(self.class_to_ind[name])
            pts.append(label_idx)
            res.append(pts)
        return res

    def target_transform_bounding_box(self, bounding_box_dict, index, target_real, width, height):
        predictions = np.array(bounding_box_dict[index])
        predictions = predictions[:, :-1]
        target_real_numpy = np.array(target_real)
        target_real_numpy = target_real_numpy[:, :-1]
        iou_all = np.zeros((predictions.shape[0], len(target_real)))
        for i in range(len(target_real)):
            iou_all[:, i] = jaccard_numpy(predictions, target_real_numpy[i])
        max_val = 2
        print()
        print(iou_all)
        print()

        targets_intersected = []

        # otherwise get the value
        while max_val > 0:
            max_val = np.max(iou_all)
            if max_val > 0:
                argmax_val = np.where(iou_all == np.amax(iou_all))
                iou_all[argmax_val[0], :] = np.zeros((1, iou_all.shape[1]))
                iou_all[:, argmax_val[1]] = np.zeros((iou_all.shape[0], 1))
                # get the GT from target_real
                targets_intersected.append(target_real[int(argmax_val[1])])

            else:
                # get a random one from target real
                targets_intersected.append(target_real[0])

        return targets_intersected

