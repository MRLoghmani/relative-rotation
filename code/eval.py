#!/usr/bin/env python3

# Import packages
import argparse
import os

from tqdm import tqdm
import torch
import torch.nn.functional as F

from net import ResBase, ResClassifier
from data_loader import DatasetGeneratorMultimodal, MyTransformer
from utils import make_paths, add_base_args, default_paths, map_to_device

# Prepare default dataset paths
data_root_source, data_root_target, split_source_train, split_source_test, split_target = make_paths()

# Parse arguments
parser = argparse.ArgumentParser()
add_base_args(parser)
parser.add_argument('--output_path', default='eval_result')
args = parser.parse_args()
default_paths(args)
args.data_root = args.data_root_target
args.test_file = args.test_file_target

device = torch.device('cuda:{}'.format(args.gpu))

# Data loader (center crop, no random flip)
test_transform = MyTransformer([int((256 - 224) / 2), int((256 - 224) / 2)], False)
test_set = DatasetGeneratorMultimodal(args.data_root, args.test_file, ds_name=args.target, do_rot=False)
test_loader = torch.utils.data.DataLoader(test_set,
                                          shuffle=True,
                                          batch_size=args.batch_size,
                                          num_workers=args.num_workers)

# Setup network
input_dim_F = 2048 if args.net == 'resnet50' else 512
netG_rgb = ResBase(architecture=args.net).to(device)
netG_depth = ResBase(architecture=args.net).to(device)
netF = ResClassifier(input_dim=input_dim_F * 2, class_num=args.class_num, extract=False,
                     dropout_p=args.dropout_p).to(device)
netG_rgb.eval()
netG_depth.eval()
netF.eval()

# Run name
hp_list = [args.task, args.net, args.epoch, args.lr, args.lr_mult, args.batch_size, args.weight_rot, args.weight_ent]
hp_string = '_'.join(map(str, hp_list)) + args.suffix

print("Run: {}".format(hp_string))

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

# Create a CSV file
with open(os.path.join(args.output_path, hp_string + '.csv'), 'w') as fp:
    for epoch in range(5, args.epoch + 1, 5):
        # Load network weights
        netG_rgb.load_state_dict(
            torch.load(os.path.join(args.snapshot, hp_string + "_netG_rgb_epoch" + str(epoch) + ".pth"),
                       map_location=device))
        netG_depth.load_state_dict(
            torch.load(os.path.join(args.snapshot, hp_string + "_netG_depth_epoch" + str(epoch) + ".pth"),
                       map_location=device))
        netF.load_state_dict(
            torch.load(os.path.join(args.snapshot, hp_string + "_netF_rgbd_epoch" + str(epoch) + ".pth"),
                       map_location=device))

        correct = 0
        total = 0
        with tqdm(total=len(test_loader), desc="TestEpoch{}".format(epoch)) as pb:
            # Load batches
            for (imgs_rgb, imgs_depth, labels) in test_loader:
                # Move tensors to GPU
                imgs_rgb, imgs_depth, labels = map_to_device(device, (imgs_rgb, imgs_depth, labels))
                # Compute features
                feat_rgb, _ = netG_rgb(imgs_rgb)
                feat_depth, _ = netG_depth(imgs_depth)
                # Compute predictions
                pred = netF(torch.cat((feat_rgb, feat_depth), 1))
                pred = F.softmax(pred, dim=1)
                correct += (torch.argmax(pred, dim=1) == labels).sum().item()
                total += labels.shape[0]

                pb.update(1)

        accuracy = correct / total
        # stdout
        print("{} @{} epochs: overall accuracy = {}".format(hp_string, epoch, accuracy))
        # CSV
        fp.write("{},{},{}\n".format(hp_string, epoch, accuracy))
        fp.flush()
