from .config import HOME
import cv2
import numpy as np
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


COCO_ROOT = '/usr/wiss/elezi/data/coco'
IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_API = 'PythonAPI'
INSTANCES_SET = 'instances_{}.json'
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map(osp.join(COCO_ROOT, 'coco_labels.txt'))

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, supervised_indices=None, image_set='train2014', transform=None,
                 target_transform=COCOAnnotationTransform(), dataset_name='MS COCO', pseudo_labels={}):
        sys.path.append(osp.join(root, COCO_API))
        self.supervised_indices = supervised_indices
        from pycocotools.coco import COCO
        self.root = osp.join(root, IMAGES, image_set)
        self.coco = COCO(osp.join(root, ANNOTATIONS,
                                  INSTANCES_SET.format(image_set)))
        self.ids = list(self.coco.imgToAnns.keys())
        self.supervised_indices = supervised_indices
        self.pseudo_labels = pseudo_labels
        self.pseudo_labels_indices = self.get_pseudo_label_indices()
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.class_to_ind = dict(zip(COCO_CLASSES, range(len(COCO_CLASSES))))
        self.class_to_ind_voc = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.dict_coco_to_voc = {0: 0, 1: 1, 2: 6, 3: 13, 4: 0, 5: 5, 6: 18, 8: 3, 15: 2, 16: 7, 17: 11, 18: 12, 19: 16,
                            20: 9, 40: 4, 57: 8, 58: 17, 59: 15, 61: 10, 63: 19}

    def contain_voc(self):
        return self.coco_contain_voc

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, h, w, semi = self.pull_item(index)
        return im, gt, semi

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        target = self.coco.imgToAnns[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        target = self.coco.loadAnns(ann_ids)
        path = osp.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        split_path = path.split("/")
        split_path = os.path.join(split_path[-2], split_path[-1])
        # if split_path in self.coco_for_voc_valid:
        #     print("Inside")
        #     self.coco_contain_voc.append(index)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        # img = cv2.imread(osp.join(self.root, path))
        img = cv2.imread(path)

        height, width, _ = img.shape

        if self.target_transform is not None:
            if index not in self.pseudo_labels_indices:
                target = self.target_transform(target, width, height)
            else:
                target = self.target_transform_pseudo_label(self.pseudo_labels, index)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4],
                                                target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        if self.supervised_indices != None:
            if index in self.supervised_indices:
                semi = np.array([1])
            elif index in self.pseudo_labels_indices:
                semi = np.array([2])
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
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

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
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

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

    def get_pseudo_label_indices(self):
        pseudo_label_indices = []
        for key in self.pseudo_labels:
            pseudo_label_indices.append(key)
        return pseudo_label_indices

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
            # label_idx = float(self.class_to_ind_voc[name])
            # transform to voc standard
            # label_idx = self.class_to_ind[name]
            # label_idx = float(self.dict_coco_to_voc[label_idx])
            pts.append(label_idx)
            res.append(pts)
        return res

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
