import numpy as np
import torch

from collections import defaultdict
from copy import deepcopy

from layers.box_utils import decode, nms


def class_consistency_loss_al(conf, conf_flip):
    conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()

    conf_class = conf.clone()
    conf_class_flip = conf_flip.clone()

    consistency_conf_loss_a = conf_consistency_criterion(conf_class.log(),
                                                         conf_class_flip.detach()).sum(-1)
    consistency_conf_loss_b = conf_consistency_criterion(conf_class_flip.log(),
                                                         conf_class.detach()).sum(-1)
    return (consistency_conf_loss_a + consistency_conf_loss_b) / 2


def active_learning_inconsistency(args, batch_iterator, labeled_set, unlabeled_set, net, num_classes,
                            criterion_select='consistency_class', loader=None):
    criterion_UC = np.zeros(len(batch_iterator))
    batch_iterator = iter(loader)
    thresh = args.thresh

    for j in range(len(batch_iterator)):  # 3000
        print(j)
        images, lab, _ = next(batch_iterator)
        images = images.cuda()

        out, conf, conf_flip, loc, loc_flip, _ = net(images)
        loc, _, priors = out

        num = loc.size(0)  # batch size
        num_priors = priors.size(0)
        output = torch.zeros(num, num_classes, 200, 6)
        conf_preds = conf.view(num, num_priors, num_classes).transpose(2, 1)
        conf_preds_flip = conf_flip.view(num, num_priors, num_classes).transpose(2, 1)
        variance = [0.1, 0.2]

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc[i], priors, variance)
            decoded_boxes_flip = decode(loc_flip[i], priors, variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            conf_scores_flip = conf_preds_flip[i].clone()

            if criterion_select == 'consistency_class' or criterion_select == 'consistency':
                H = class_consistency_loss_al(conf, conf_flip).squeeze()
            else:
                H = conf_scores * (
                            torch.log(conf_scores) / torch.log(torch.tensor(num_classes).type(torch.FloatTensor)))
                H = H.sum(dim=0) * (-1.0)

            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(0.01)  # confidence threshold
                scores = conf_scores[cl][c_mask]
                Entropy = H[c_mask]
                if scores.size(0) == 0:
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)

                ids, count = nms(boxes.detach(), scores.detach(), 0.5, 200)
                output[i, cl, :count] = torch.cat(
                    (scores[ids[:count]].unsqueeze(1), boxes[ids[:count]], Entropy[ids[:count]].unsqueeze(1)), 1)

        count_num = 0
        UC_max = 0
        for p in range(output.size(1)):  # [1, 21, 200, 9]
            q = 0
            while output[0, p, q, 0] >= thresh:  # filtering using threshold, To do: Increasing acoording to iteration?
                count_num += 1
                score = output[0, p, q, 0]
                entropy = output[0, p, q, 5:6]
                UC_max_temp = entropy.item()
                if (UC_max < UC_max_temp):
                    UC_max = UC_max_temp
                q += 1

        if count_num == 0:  
            criterion_UC[j] = 0
        else:
            criterion_UC[j] = UC_max

    if args.criterion_select == 'combined':
        return criterion_UC

    sorted_indices = np.argsort(criterion_UC)[::-1]
    labeled_set += list(np.array(unlabeled_set)[sorted_indices[:args.acquisition_budget]])
    unlabeled_set = list(np.array(unlabeled_set)[sorted_indices[args.acquisition_budget:]])

    # assert that sizes of lists are correct and that there are no elements that are in both lists
    assert len(list(set(labeled_set) | set(unlabeled_set))) == args.num_total_images
    assert len(list(set(labeled_set) & set(unlabeled_set))) == 0

    # save the labeled set
    return labeled_set, unlabeled_set


def active_learning_entropy(args, batch_iterator, labeled_set, unlabeled_set, net, num_classes, criterion_select, loader=None):
    criterion_UC = np.zeros(len(batch_iterator))
    batch_iterator = iter(loader)
    thresh = args.thresh

    for j in range(len(batch_iterator)):  # 3000
        print(j)
        images, lab, _ = next(batch_iterator)
        images = images.cuda()

        out, _, _, _, _, _ = net(images)
        loc, conf, priors = out
        conf = torch.softmax(conf.detach(), dim=2)

        num = loc.size(0)  # batch size
        num_priors = priors.size(0)
        output = torch.zeros(num, num_classes, 200, 6)
        conf_preds = conf.view(num, num_priors, num_classes).transpose(2, 1)
        variance = [0.1, 0.2]
        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc[i], priors, variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            H = conf_scores * (torch.log(conf_scores) / torch.log(torch.tensor(num_classes).type(torch.FloatTensor)))
            H = H.sum(dim=0) * (-1.0)

            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(0.01)  # confidence threshold
                scores = conf_scores[cl][c_mask]
                Entropy = H[c_mask]  # jwchoi
                if scores.size(0) == 0:
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)

                ids, count = nms(boxes.detach(), scores.detach(), 0.5, 200)
                output[i, cl, :count] = torch.cat(
                    (scores[ids[:count]].unsqueeze(1), boxes[ids[:count]], Entropy[ids[:count]].unsqueeze(1)), 1)

        if criterion_select == 'random':
            val = np.random.normal(0, 1, 1)
            criterion_UC[j] = val[0]


        elif criterion_select == 'Max_aver':
            count_num = 0
            # UC_sum = 0
            UC_max = 0
            for p in range(output.size(1)):  # [1, 21, 200, 9]
                q = 0
                while output[
                    0, p, q, 0] >= thresh:  # filtering using threshold, To do: Increasing acoording to iteration?
                    count_num += 1
                    score = output[0, p, q, 0]
                    entropy = output[0, p, q, 5:6]
                    UC_max += entropy.item()
                    q += 1

            if count_num == 0:  
                criterion_UC[j] = 0
            else:
                criterion_UC[j] = UC_max / count_num  

        else:
            count_num = 0
            # UC_sum = 0
            UC_max = 0
            for p in range(output.size(1)):  # [1, 21, 200, 9]
                q = 0
                while output[
                    0, p, q, 0] >= thresh:  # filtering using threshold, To do: Increasing acoording to iteration?
                    count_num += 1
                    score = output[0, p, q, 0]
                    entropy = output[0, p, q, 5:6]
                    UC_max_temp = entropy.item()
                    if (UC_max < UC_max_temp):
                        UC_max = UC_max_temp
                    q += 1

            if count_num == 0:  # UC_sum == 0:
                # if the net cannot detect anything, give it a very high inconsistency score, that image is hard
                criterion_UC[j] = 0
            else:
                criterion_UC[j] = UC_max

    if args.criterion_select == 'combined':
        return criterion_UC
    sorted_indices = np.argsort(criterion_UC)[::-1]

    labeled_set += list(np.array(unlabeled_set)[sorted_indices[:args.acquisition_budget]])
    unlabeled_set = list(np.array(unlabeled_set)[sorted_indices[args.acquisition_budget:]])

    # assert that sizes of lists are correct and that there are no elements that are in both lists
    assert len(list(set(labeled_set) | set(unlabeled_set))) == args.num_total_images
    assert len(list(set(labeled_set) & set(unlabeled_set))) == 0

    # save the labeled set
    return labeled_set, unlabeled_set


def combined_score(args, batch_iterator, labeled_set, unlabeled_set, net, unsupervised_data_loader):
    entropy_score = active_learning_entropy(args, batch_iterator, labeled_set, unlabeled_set, net, 
                                            args.cfg['num_classes'], 'entropy', 
                                            loader=unsupervised_data_loader)
    consistency_score = active_learning_inconsistency(args, batch_iterator, labeled_set, unlabeled_set, net,
                                                      args.cfg['num_classes'], 'consistency_class',
                                                      loader=unsupervised_data_loader)

    len_arr = len(entropy_score)
    ind = np.argpartition(entropy_score, -args.filter_entropy_num)[:len_arr - args.filter_entropy_num]

    consistency_score[ind] = 0.

    sorted_indices = np.argsort(consistency_score)[::-1]

    labeled_set += list(np.array(unlabeled_set)[sorted_indices[:args.acquisition_budget]])
    unlabeled_set = list(np.array(unlabeled_set)[sorted_indices[args.acquisition_budget:]])

    # assert that sizes of lists are correct and that there are no elements that are in both lists
    assert len(list(set(labeled_set) | set(unlabeled_set))) == args.num_total_images
    assert len(list(set(labeled_set) & set(unlabeled_set))) == 0

    return labeled_set, unlabeled_set