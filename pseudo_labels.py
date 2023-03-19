# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/AL-SSL/


import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import random
import pickle
from torch.autograd import Variable
from ssd import build_ssd
import torch.nn as nn
from data import *
from data import VOC_CLASSES as labels1
from data import COCO_CLASSES as labels_2
from collections import defaultdict
import random
random.seed(314)
torch.manual_seed(314)


def predict_pseudo_labels(unlabeled_set, net_name, threshold=0.5, root='../tmp/VOC0712/', voc=1, num_classes=21):
    labels = labels1 if voc else labels_2
    if voc:
        testset = VOCDetection(root=root, image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                            supervised_indices=None,
                            transform=None)
    else:
        testset = COCODetection(root=root, image_set='train2014',
                            supervised_indices=None,
                            transform=None)
 
    print("Doing PL")
    net = build_ssd('test', 300, num_classes)  
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(net_name))
    boxes = get_pseudo_labels(testset, net, labels, unlabeled_set=unlabeled_set, threshold=threshold, voc=voc)
    return boxes


def get_pseudo_labels(testset, net, labels, unlabeled_set=None, threshold=0.99, voc=1):

    boxes = defaultdict(list)
    for ii, img_id in enumerate(unlabeled_set):
        print(ii)
        image = testset.pull_image(img_id)
        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)

        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)

        xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = net(xx)

        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor(image.shape[1::-1]).repeat(2)

        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= threshold:
                score = detections[0, i, j, 0]
                if voc == 1:
                    label_name = labels[i - 1]
                else:
                    label_name = labels[i - 1]
                pt = (detections[0, i, j, 1:5] * scale).cpu().numpy() 
                j += 1
                # sore as [prediction_confidence, label_id, label_name, height, width, bbox_coordiates in range [0, inf]
                boxes[img_id].append([score.cpu().detach().item(), (i-1), label_name, image.shape[0], image.shape[1], int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3])])

    return boxes







