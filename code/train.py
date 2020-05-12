#!/usr/bin/env python3
"""
    Import packages
"""

import numpy as np
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from net import ResBase, ResClassifier, RelativeRotationClassifier
from data_loader import DatasetGeneratorMultimodal, MyTransformer
from utils import OptimizerManager, EvaluationManager, IteratorWrapper, weights_init, default_paths, map_to_device, \
    add_base_args, entropy_loss
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser()

add_base_args(parser)
parser.add_argument('--test_batches', default=100, type=int)
args = parser.parse_args()

# Load default paths if needed
default_paths(args)

# Run name
hp_list = [args.task, args.net, args.epoch, args.lr, args.lr_mult, args.batch_size, args.weight_rot, args.weight_ent]
hp_list = [str(hp) for hp in hp_list]
hp_string = '_'.join(hp_list) + args.suffix
print("Run: " + hp_string)
# Tensorboard summary
writer = SummaryWriter(log_dir=os.path.join(args.tensorboard, hp_string), flush_secs=5)

# Device
device = torch.device('cuda:{}'.format(args.gpu))

# Center crop, no random flip
test_transform = MyTransformer([int((256 - 224) / 2), int((256 - 224) / 2)], False)

"""
    Prepare datasets
"""

# Source: training set
train_set_source = DatasetGeneratorMultimodal(args.data_root_source, args.train_file_source, do_rot=False)
# Source: test set
test_set_source = DatasetGeneratorMultimodal(args.data_root_source, args.test_file_source, do_rot=False,
                                             transform=test_transform)
# Target: training set (for entropy)
train_set_target = DatasetGeneratorMultimodal(args.data_root_target, args.train_file_target, ds_name=args.target,
                                              do_rot=False)
# Source: training set (for relative rotation)
rot_set_source = DatasetGeneratorMultimodal(args.data_root_source, args.train_file_source, do_rot=True)
# Source: test set (for relative rotation)
rot_test_set_source = DatasetGeneratorMultimodal(args.data_root_source, args.test_file_source, do_rot=True)
# Target: training and test set (for relative rotation)
rot_set_target = DatasetGeneratorMultimodal(args.data_root_target, args.train_file_target, ds_name=args.target,
                                            do_rot=True)

"""
    Prepare data loaders
"""

# Source training recognition
train_loader_source = DataLoader(train_set_source,
                                 shuffle=True,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers)

# Source test recognition
test_loader_source = DataLoader(test_set_source,
                                shuffle=True,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

# Target train
train_loader_target = DataLoader(train_set_target,
                                 shuffle=True,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers)

# Target test
test_loader_target = DataLoader(train_set_target,
                                shuffle=True,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

# Source rot
rot_source_loader = DataLoader(rot_set_source,
                               shuffle=True,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers)

rot_test_source_loader = DataLoader(rot_test_set_source,
                                    shuffle=True,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)

# Target rot

rot_target_loader = DataLoader(rot_set_target,
                               shuffle=True,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers)

rot_test_target_loader = DataLoader(rot_set_target,
                                    shuffle=True,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)

"""
    Set up network & optimizer
"""
input_dim_F = 2048 if args.net == 'resnet50' else 512
netG_rgb = ResBase(architecture=args.net)
netG_depth = ResBase(architecture=args.net)
netF = ResClassifier(input_dim=input_dim_F * 2, class_num=args.class_num, dropout_p=args.dropout_p, extract=False)
netF_rot = RelativeRotationClassifier(input_dim=input_dim_F * 2, class_num=4)
netF_rot.apply(weights_init)
netF.apply(weights_init)

net_list = [netG_rgb, netG_depth, netF, netF_rot]
net_list = map_to_device(device, net_list)

ce_loss = nn.CrossEntropyLoss()

opt_g_rgb = optim.SGD(netG_rgb.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
opt_g_depth = optim.SGD(netG_depth.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
opt_f = optim.SGD(netF.parameters(), lr=args.lr * args.lr_mult, momentum=0.9, weight_decay=args.weight_decay)
opt_f_rot = optim.SGD(netF_rot.parameters(), lr=args.lr * args.lr_mult, momentum=0.9, weight_decay=args.weight_decay)

optims_list = [opt_g_rgb, opt_g_depth, opt_f, opt_f_rot]

for epoch in range(1, args.epoch + 1):
    print("Epoch {} / {}".format(epoch, args.epoch))
    # ========================= TRAINING =========================

    # Train source (recognition)
    train_loader_source_rec_iter = train_loader_source
    # Train target (entropy)
    train_target_loader_iter = IteratorWrapper(train_loader_target)
    # Test target
    test_target_loader_iter = IteratorWrapper(test_loader_target)

    # Source (rotation)
    rot_source_loader_iter = IteratorWrapper(rot_source_loader)
    # Target (rotation)
    rot_target_loader_iter = IteratorWrapper(rot_target_loader)

    with tqdm(total=len(train_loader_source), desc="Train") as pb:
        for batch_num, (img_rgb, img_depth, img_label_source) in enumerate(train_loader_source_rec_iter):
            if img_rgb.size(0) != args.batch_size:
                break

            # The optimization step is performed by OptimizerManager
            with OptimizerManager(optims_list):
                # Compute source features
                img_rgb, img_depth, img_label_source = map_to_device(device, (img_rgb, img_depth, img_label_source))
                feat_rgb, _ = netG_rgb(img_rgb)
                feat_depth, _ = netG_depth(img_depth)
                features_source = torch.cat((feat_rgb, feat_depth), 1)
                logits = netF(features_source)

                # Classification los
                loss_rec = ce_loss(logits, img_label_source)

                # Entropy loss
                if args.weight_ent > 0.:
                    # Compute target features
                    img_rgb, img_depth, _ = train_target_loader_iter.get_next()
                    img_rgb, img_depth = map_to_device(device, (img_rgb, img_depth))
                    feat_rgb, _ = netG_rgb(img_rgb)
                    feat_depth, _ = netG_depth(img_depth)
                    features_target = torch.cat((feat_rgb, feat_depth), 1)
                    logits = netF(features_target)

                    loss_ent = entropy_loss(logits)
                else:
                    loss_ent = 0

                # Backpropagate
                loss = loss_rec + args.weight_ent * loss_ent
                loss.backward()

                del img_rgb, img_depth, img_label_source, feat_rgb, feat_depth, logits

                # Relative Rotation
                if args.weight_rot > 0.0:
                    # Load batch: rotation, source
                    img_rgb, img_depth, _, rot_label = rot_source_loader_iter.get_next()
                    img_rgb, img_depth, rot_label = map_to_device(device, (img_rgb, img_depth, rot_label))

                    # Compute features (without pooling!)
                    _, pooled_rgb = netG_rgb(img_rgb)
                    _, pooled_depth = netG_depth(img_depth)
                    # Prediction
                    logits_rot = netF_rot(torch.cat((pooled_rgb, pooled_depth), 1))

                    # Classification loss for the rleative rotation task
                    loss_rot = ce_loss(logits_rot, rot_label)
                    loss = args.weight_rot * loss_rot
                    # Backpropagate
                    loss.backward()

                    loss_rot = loss_rot.item()

                    del img_rgb, img_depth, rot_label, pooled_rgb, pooled_depth, logits_rot, loss

                    # Load batch: rotation, target
                    img_rgb, img_depth, _, rot_label = rot_target_loader_iter.get_next()
                    img_rgb, img_depth, rot_label = map_to_device(device, (img_rgb, img_depth, rot_label))

                    # Compute features (without pooling!)
                    _, pooled_rgb = netG_rgb(img_rgb)
                    _, pooled_depth = netG_depth(img_depth)
                    # Prediction
                    logits_rot = netF_rot(torch.cat((pooled_rgb, pooled_depth), 1))

                    # Classification loss for the rleative rotation task
                    loss = args.weight_rot * ce_loss(logits_rot, rot_label)
                    # Backpropagate
                    loss.backward()

                    del img_rgb, img_depth, rot_label, pooled_rgb, pooled_depth, logits_rot, loss

                pb.update(1)

    # ========================= VALIDATION =========================

    # Recognition - source
    actual_test_batches = min(len(test_loader_source), args.test_batches)
    with EvaluationManager(net_list), tqdm(total=actual_test_batches, desc="TestRecS") as pb:
        test_source_loader_iter = iter(test_loader_source)
        correct = 0.0
        num_predictions = 0.0
        val_loss = 0.0
        for num_batch, (img_rgb, img_depth, img_label_source) in enumerate(test_source_loader_iter):
            # By default validate only on 100 batches
            if num_batch >= args.test_batches:
                break

            # Compute source features
            img_rgb, img_depth, img_label_source = map_to_device(device, (img_rgb, img_depth, img_label_source))
            feat_rgb, _ = netG_rgb(img_rgb)
            feat_depth, _ = netG_depth(img_depth)
            features_source = torch.cat((feat_rgb, feat_depth), 1)

            # Compute predictions
            preds = netF(features_source)

            val_loss += ce_loss(preds, img_label_source).item()
            correct += (torch.argmax(preds, dim=1) == img_label_source).sum().item()
            num_predictions += preds.shape[0]

            pb.update(1)

        val_acc = correct / num_predictions
        val_loss = val_loss / args.test_batches
        print("Epoch: {} - Validation source accuracy (recognition): {}".format(epoch, val_acc))

    del img_rgb, img_depth, img_label_source, feat_rgb, feat_depth, preds

    writer.add_scalar("Loss/train", loss_rec.item(), epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    # Relative Rotation
    if args.weight_rot > 0.0:

        # Rotation - source
        cf_matrix = np.zeros([4, 4])
        actual_test_batches = min(len(rot_test_source_loader), args.test_batches)
        with EvaluationManager(net_list), tqdm(total=actual_test_batches, desc="TestRotS") as pb:
            rot_test_source_loader_iter = iter(rot_test_source_loader)
            correct = 0.0
            num_predictions = 0.0
            for num_val_batch, (img_rgb, img_depth, _, rot_label) in enumerate(rot_test_source_loader_iter):
                if num_val_batch > args.test_batches:
                    break
                img_rgb, img_depth, rot_label = map_to_device(device, (img_rgb, img_depth, rot_label))

                # Compute features (without pooling)
                _, pooled_rgb = netG_rgb(img_rgb)
                _, pooled_depth = netG_depth(img_depth)
                # Compute predictions
                preds = netF_rot(torch.cat((pooled_rgb, pooled_depth), 1))

                val_loss_rot = ce_loss(preds, rot_label).item()
                correct += (torch.argmax(preds, dim=1) == rot_label).sum().item()
                num_predictions += preds.shape[0]

                pb.update(1)

            del img_rgb, img_depth, rot_label, preds

            rot_val_acc = correct / num_predictions
            print("Epoch: {} - Validation source rotation accuracy: {}".format(epoch, rot_val_acc))

        # Rotation - target
        actual_test_batches = min(len(rot_test_target_loader), args.test_batches)
        with EvaluationManager(net_list), tqdm(total=actual_test_batches, desc="TestRotT") as pb:
            rot_test_target_loader_iter = iter(rot_test_target_loader)
            correct = 0.0
            val_loss_rot = 0.0
            num_predictions = 0.0
            for num_val_batch, (img_rgb, img_depth, _, rot_label) in enumerate(rot_test_target_loader_iter):
                if num_val_batch > args.test_batches:
                    break

                img_rgb, img_depth, rot_label = map_to_device(device, (img_rgb, img_depth, rot_label))

                # Compute features (without pooling)
                _, pooled_rgb = netG_rgb(img_rgb)
                _, pooled_depth = netG_depth(img_depth)
                # Compute predictions
                preds = netF_rot(torch.cat((pooled_rgb, pooled_depth), 1))

                val_loss_rot += ce_loss(preds, rot_label).item()
                correct += (torch.argmax(preds, dim=1) == rot_label).sum().item()
                num_predictions += preds.shape[0]

                pb.update(1)

            rot_val_acc = correct / num_predictions
            val_loss_rot = val_loss_rot / args.test_batches
            print("Epoch: {} - Validation target rotation accuracy: {}".format(epoch, rot_val_acc))

        del img_rgb, img_depth, rot_label, preds

        writer.add_scalar("Loss/rot", loss_rot, epoch)
        writer.add_scalar("Loss/rot_val", val_loss_rot, epoch)
        writer.add_scalar("Accuracy/rot_val", rot_val_acc, epoch)

    # Save models
    if epoch % 5 == 0:
        if not os.path.exists(args.snapshot):
            os.mkdir(args.snapshot)

        torch.save(netG_rgb.state_dict(),
                   os.path.join(args.snapshot, hp_string + "_netG_rgb_epoch" + str(epoch) + ".pth"))
        torch.save(netG_depth.state_dict(),
                   os.path.join(args.snapshot, hp_string + "_netG_depth_epoch" + str(epoch) + ".pth"))
        torch.save(netF.state_dict(), os.path.join(args.snapshot, hp_string + "_netF_rgbd_epoch" + str(epoch) + ".pth"))
        torch.save(netF_rot.state_dict(),
                   os.path.join(args.snapshot, hp_string + "_netF_rot_rgbd_epoch" + str(epoch) + ".pth"))
