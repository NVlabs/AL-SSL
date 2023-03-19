# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/AL-SSL/


import random
import torch.utils.data as data

from data import *
from pseudo_labels import predict_pseudo_labels
from subset_sequential_sampler import SubsetSequentialSampler, BalancedSampler
from utils.augmentations import SSDAugmentation

random.seed(314)


def create_loaders(args):
    indices_1 = list(range(args.num_total_images))
    random.shuffle(indices_1)
    labeled_set = indices_1[:args.num_initial_labeled_set]
    indices = list(range(args.num_total_images))
    unlabeled_set = set(indices) - set(labeled_set)
    labeled_set = list(labeled_set)
    unlabeled_set = list(unlabeled_set)
    random.shuffle(indices)


    print(len(indices))

    if args.dataset_name == 'voc':
        supervised_dataset = VOCDetection(root=args.dataset_root, image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                                          supervised_indices=labeled_set,
                                          transform=SSDAugmentation(args.cfg['min_dim'], MEANS))

        unsupervised_dataset = VOCDetection(args.dataset_root, supervised_indices=None,
                                            image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                                            transform=BaseTransform(300, MEANS),
                                            target_transform=VOCAnnotationTransform())

    else:
        supervised_dataset = COCODetection(root=args.dataset_root, supervised_indices=labeled_set, image_set='train2014', 
                                           transform=SSDAugmentation(args.cfg['min_dim']),
                                           target_transform=COCOAnnotationTransform(), 
                                           dataset_name='MS COCO')

        unsupervised_dataset = COCODetection(root=args.dataset_root, supervised_indices=None, image_set='val2014',
                                            transform=SSDAugmentation(args.cfg['min_dim']),
                                            target_transform=COCOAnnotationTransform(), 
                                            dataset_name='MS COCO')
    

    supervised_data_loader = data.DataLoader(supervised_dataset, batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             sampler=BalancedSampler(indices, labeled_set, unlabeled_set, ratio=1),
                                             collate_fn=detection_collate,
                                             pin_memory=True)

    unsupervised_data_loader = data.DataLoader(unsupervised_dataset, batch_size=1,
                                               sampler=SubsetSequentialSampler(unlabeled_set),
                                               num_workers=args.num_workers,
                                               collate_fn=detection_collate,
                                               pin_memory=True)

    return supervised_dataset, supervised_data_loader, unsupervised_dataset, unsupervised_data_loader, indices, labeled_set, unlabeled_set


def change_loaders(args, supervised_dataset, unsupervised_dataset, labeled_set, 
                   unlabeled_set, indices, net, pseudo=True):
    print("Labeled set size: " + str(len(labeled_set)))
    unsupervised_data_loader = data.DataLoader(unsupervised_dataset, batch_size=1,
                                               sampler=SubsetSequentialSampler(unlabeled_set),
                                               num_workers=args.num_workers,
                                               collate_fn=detection_collate,
                                               pin_memory=True)

    if pseudo:
        if args.dataset_name == 'voc':
            voc = True
            num_classes = 21
        else:
            voc = False
            num_classes = 81
        pseudo_labels = predict_pseudo_labels(unlabeled_set=unlabeled_set, net_name=net,
                                                  threshold=args.pseudo_threshold, root=args.dataset_root, 
                                                  voc=voc, num_classes=num_classes)
    else:
        pseudo_labels = {}

    if args.dataset_name == 'voc':
        supervised_dataset = VOCDetection(root=args.dataset_root,
                                          image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                                          supervised_indices=labeled_set,
                                          transform=SSDAugmentation(args.cfg['min_dim'], MEANS),
                                          pseudo_labels=pseudo_labels)
    else:
        supervised_dataset = COCODetection(root=args.dataset_root, supervised_indices=labeled_set, image_set='train2014', 
                                           transform=SSDAugmentation(args.cfg['min_dim']),
                                           target_transform=COCOAnnotationTransform(), 
                                           dataset_name='MS COCO',
                                           pseudo_labels=pseudo_labels)

    print("Changing the loaders")

    supervised_data_loader = data.DataLoader(supervised_dataset, batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             sampler=BalancedSampler(indices, labeled_set, unlabeled_set, ratio=1),
                                             collate_fn=detection_collate,
                                             pin_memory=True)

    return supervised_data_loader, unsupervised_data_loader
