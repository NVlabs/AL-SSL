import warnings
import argparse
import math
import numpy as np
import random
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init

from active_learning import combined_score, active_learning_inconsistency, active_learning_entropy
from csd import build_ssd_con
from data import *
from layers.modules import MultiBoxLoss
from loaders import create_loaders, change_loaders

random.seed(314)
torch.manual_seed(314)

warnings.filterwarnings("ignore")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class get_al_hyperparams():
    def __init__(self, dataset_name='voc'):
        self.dataset_name = dataset_name
        self.dataset_path = {'voc': '/usr/wiss/elezi/data/VOC0712',
                             'coco': '/usr/wiss/elezi/data/coco'}

        self.num_ims = {'voc': 16551, 'coco': 82081}
        self.num_init = {'voc': 2011, 'coco': 5000}
        self.pseudo_threshold = {'voc': 0.99, 'coco': 0.75}
        self.config = {'voc': voc300, 'coco': coco}
        self.batch_size = {'voc': 16, 'coco': 32}

    def get_dataset_path(self):
        return self.dataset_path[self.dataset_name]

    def get_num_ims(self):
        return self.num_ims[self.dataset_name]

    def get_num_init(self):
        return self.num_init[self.dataset_name]

    def get_pseudo_threshold(self):
        return self.pseudo_threshold[self.dataset_name]

    def get_config(self):
        return self.config[self.dataset_name]

    def get_dataset_name(self):
        return self.dataset_name

    def get_batch_size(self):
        return self.batch_size[self.dataset_name]



parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
al_hyperparams = get_al_hyperparams('voc') # 'voc' for voc, 'coco' for coco 
parser.add_argument('--dataset_name', default=al_hyperparams.get_dataset_name(), type=str,
                    help='Dataset name')
parser.add_argument('--cfg', default=al_hyperparams.get_config(), type=dict,
                    help='configurer for the specific dataset')
parser.add_argument('--dataset', default='VOC300', choices=['VOC300', 'VOC512'],
                    type=str, help='VOC300 or VOC512')
parser.add_argument('--dataset_root', default=al_hyperparams.get_dataset_path(), type=str,
                    help='Dataset root directory path')
parser.add_argument('--num_total_images', default=al_hyperparams.get_num_ims(), type=int,
                    help='Number of images in the dataset')
parser.add_argument('--num_initial_labeled_set', default=al_hyperparams.get_num_init(), type=int,
                    help='Number of initially labeled images')
parser.add_argument('--acquisition_budget', default=1000, type=int,
                    help='Active labeling cycle budget')
parser.add_argument('--num_cycles', default=5, type=int,
                    help='Number of active learning cycles')
parser.add_argument('--criterion_select', default='combined',
                    choices=['random', 'entropy', 'consistency', 'combined'],
                    help='Active learning acquisition score')
parser.add_argument('--filter_entropy_num', default=3000, type=int,
                    help='How many samples to pre-filer with entropy')
parser.add_argument('--id', default=2, type=int,
                    help='the id of the experiment')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=al_hyperparams.get_batch_size(), type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='../al_ssl/weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--net_name', default=None, type=str,
                    help='the net checkpoint we need to load')
parser.add_argument('--is_apex', default=0, type=int,
                    help='if 1 use apex to do mixed precision training')
parser.add_argument('--is_cluster', default=1, type=int,
                    help='if 1 use GPU cluster, otherwise do the computations on the local PC')
parser.add_argument('--do_PL', default=1, type=int,
                    help='if 1 use pseudo-labels, otherwise do not use them')
parser.add_argument('--pseudo_threshold', default=al_hyperparams.get_pseudo_threshold(), type=float,
                    help='pseudo label confidence threshold for voc dataset')
parser.add_argument('--thresh', default=0.5, type=float,
                    help='we define an object if the probability of one class is above thresh')
parser.add_argument('--do_AL', default=1, type=int, help='if 0 skip AL')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
    cudnn.benchmark = True
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def load_net_optimizer_multi(cfg):
    net = build_ssd_con('train', cfg['min_dim'], cfg['num_classes'])
    vgg_weights = torch.load(args.save_folder + args.basenet)
    print('Loading the backbone pretrained in Imagenet...')
    net.vgg.load_state_dict(vgg_weights)
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    if args.is_cluster:
        net = nn.DataParallel(net)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    return net, optimizer


def compute_consistency_loss(conf, loc, conf_flip, loc_flip, conf_consistency_criterion):
    conf_class = conf[:, :, 1:].clone()
    background_score = conf[:, :, 0].clone()
    each_val, each_index = torch.max(conf_class, dim=2)
    mask_val = each_val > background_score
    mask_val = mask_val.data

    mask_conf_index = mask_val.unsqueeze(2).expand_as(conf)
    mask_loc_index = mask_val.unsqueeze(2).expand_as(loc)

    conf_mask_sample = conf.clone()
    loc_mask_sample = loc.clone()
    conf_sampled = conf_mask_sample[mask_conf_index].view(-1, args.cfg['num_classes'])
    loc_sampled = loc_mask_sample[mask_loc_index].view(-1, 4)

    conf_mask_sample_flip = conf_flip.clone()
    loc_mask_sample_flip = loc_flip.clone()
    conf_sampled_flip = conf_mask_sample_flip[mask_conf_index].view(-1, args.cfg['num_classes'])
    loc_sampled_flip = loc_mask_sample_flip[mask_loc_index].view(-1, 4)

    if (mask_val.sum() > 0):
        # Compute Jenson-Shannon divergence (symmetric KL actually)
        conf_sampled_flip = conf_sampled_flip + 1e-7
        conf_sampled = conf_sampled + 1e-7
        consistency_conf_loss_a = conf_consistency_criterion(conf_sampled.log(),
                                                             conf_sampled_flip.detach()).sum(-1).mean()
        consistency_conf_loss_b = conf_consistency_criterion(conf_sampled_flip.log(),
                                                             conf_sampled.detach()).sum(-1).mean()
        consistency_conf_loss = consistency_conf_loss_a + consistency_conf_loss_b

        # Compute location consistency loss
        consistency_loc_loss_x = torch.mean(torch.pow(loc_sampled[:, 0] + loc_sampled_flip[:, 0], exponent=2))
        consistency_loc_loss_y = torch.mean(torch.pow(loc_sampled[:, 1] - loc_sampled_flip[:, 1], exponent=2))
        consistency_loc_loss_w = torch.mean(torch.pow(loc_sampled[:, 2] - loc_sampled_flip[:, 2], exponent=2))
        consistency_loc_loss_h = torch.mean(torch.pow(loc_sampled[:, 3] - loc_sampled_flip[:, 3], exponent=2))

        consistency_loc_loss = torch.div(
            consistency_loc_loss_x + consistency_loc_loss_y + consistency_loc_loss_w + consistency_loc_loss_h,
            4)

    else:
        consistency_conf_loss = torch.cuda.FloatTensor([0])
        consistency_loc_loss = torch.cuda.FloatTensor([0])

    consistency_loss = torch.div(consistency_conf_loss, 2) + consistency_loc_loss
    return consistency_loss


def rampweight(iteration):
    ramp_up_end = 32000
    ramp_down_start = 100000

    if (iteration < ramp_up_end):
        ramp_weight = math.exp(-5 * math.pow((1 - iteration / ramp_up_end), 2))
    elif (iteration > ramp_down_start):
        ramp_weight = math.exp(-12.5 * math.pow((1 - (120000 - iteration) / 20000), 2))
    else:
        ramp_weight = 1

    if (iteration == 0):
        ramp_weight = 0

    return ramp_weight


def train(dataset, data_loader, cfg, labeled_set, supervised_dataset, indices):
    # net, optimizer = load_net_optimizer_multi(cfg)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()

    # loss counters
    print('Loading the dataset...')
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')

    step_index = 0

    # create batch iterator
    batch_iterator = iter(data_loader)

    finish_flag = True

    while finish_flag:
        net, optimizer = load_net_optimizer_multi(cfg)
        net.train()
        for iteration in range(cfg['max_iter']):
            print(iteration)

            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)
            try:
                images, targets, semis = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(data_loader)
                images, targets, semis = next(batch_iterator)

            images = images.cuda()
            targets = [ann.cuda() for ann in targets]

            # forward
            t0 = time.time()
            out, conf, conf_flip, loc, loc_flip, _ = net(images)
            sup_image_binary_index = np.zeros([len(semis), 1])

            semis_index, new_semis = [], []
            for iii, super_image in enumerate(range(len(semis))):
                new_semis.append(int(semis[super_image]))
                if (int(semis[super_image]) > 0):
                    sup_image_binary_index[super_image] = 1
                    semis_index.append(super_image)
                else:
                    sup_image_binary_index[super_image] = 0

                if (int(semis[len(semis) - 1 - super_image]) == 0):
                    del targets[len(semis) - 1 - super_image]

            sup_image_index = np.where(sup_image_binary_index == 1)[0]
            loc_data, conf_data, priors = out

            if (len(sup_image_index) != 0):
                loc_data = loc_data[sup_image_index, :, :]
                conf_data = conf_data[sup_image_index, :, :]
                output = (loc_data, conf_data, priors)

            consistency_loss = compute_consistency_loss(conf, loc, conf_flip, loc_flip, conf_consistency_criterion)
            ramp_weight = rampweight(iteration)
            consistency_loss = torch.mul(consistency_loss, ramp_weight)

            if (len(sup_image_index) == 0):
                loss_l, loss_c = torch.cuda.FloatTensor([0]), torch.cuda.FloatTensor([0])
            else:
                loss_l, loss_c = criterion(output, targets, np.array(new_semis)[semis_index])
            loss = loss_l + loss_c + consistency_loss
            print(loss)

            if (loss.data > 0):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                print("Loss is 0")

            if (float(loss) > 100 or torch.isnan(loss)):
                # if the net diverges, go back to point 0 and train from scratch
                break
            t1 = time.time()

            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(
                    iteration) + ': loss: %.4f , loss_c: %.4f , loss_l: %.4f , loss_con: %.4f, lr : %.4f, super_len : %d\n' % (
                          loss.data, loss_c.data, loss_l.data, consistency_loss.data,
                          float(optimizer.param_groups[0]['lr']),
                          len(sup_image_index)))

            if iteration != 0 and (iteration + 1) % 120000 == 0:
                print('Saving state, iter:', iteration)
                net_name = 'weights/' + repr(iteration + 1) + args.criterion_select + '_id_' + str(args.id)  + \
                           '_pl_threshold_' + str(args.pseudo_threshold) + '_labeled_set_' + str(len(labeled_set)) + '_.pth'
                torch.save(net.state_dict(), net_name)

            if iteration >= 119000:
                finish_flag = False
    return net, net_name


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def main():
    print(os.path.abspath(os.getcwd()))
    if args.cuda: cudnn.benchmark = True
    supervised_dataset, supervised_data_loader, unsupervised_dataset, unsupervised_data_loader, indices, labeled_set, unlabeled_set = create_loaders(args)
    net, net_name = train(supervised_dataset, supervised_data_loader, args.cfg, labeled_set, supervised_dataset, indices)

    net, _ = load_net_optimizer_multi(args.cfg)
    if not args.is_cluster:
        net = nn.DataParallel(net)

    # net_name = os.path.join('/usr/wiss/elezi/PycharmProjects/al_ssl/weights_good/120000combined_id_2_pl_threshold_0.99_labeled_set_3011_.pth')
    # net.load_state_dict(torch.load(net_name))

    # do active learning cycles
    for i in range(args.num_cycles):
        net.eval()

        if args.do_AL:
            if args.criterion_select in ['Max_aver', 'entropy', 'random']:
                batch_iterator = iter(unsupervised_data_loader)
                labeled_set, unlabeled_set = active_learning_entropy(args, batch_iterator, labeled_set, unlabeled_set, net,
                                                                   args.cfg['num_classes'],
                                                                   args.criterion_select,
                                                                   loader=unsupervised_data_loader)
            elif args.criterion_select == 'consistency':
                batch_iterator = iter(unsupervised_data_loader)
                labeled_set, unlabeled_set = active_learning_inconsistency(args, batch_iterator, labeled_set, unlabeled_set, net,
                                                                     args.cfg['num_classes'], 
                                                                     args.criterion_select,
                                                                     loader=unsupervised_data_loader)
            elif args.criterion_select == 'combined':
                print("Combined")
                batch_iterator = iter(unsupervised_data_loader)
                labeled_set, unlabeled_set = combined_score(args, batch_iterator, labeled_set, unlabeled_set, net,
                                                            unsupervised_data_loader)

        supervised_data_loader, unsupervised_data_loader = change_loaders(args, supervised_dataset, 
            unsupervised_dataset, labeled_set, unlabeled_set, indices, net_name, pseudo=args.do_PL)
        net, net_name = train(supervised_dataset, supervised_data_loader, args.cfg, labeled_set, supervised_dataset,
                              indices)


if __name__ == '__main__':
    main()
