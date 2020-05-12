import os
import random

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_image(path):
    return Image.open(path).convert('RGB')


def make_sync_dataset(root, label, ds_name='synROD'):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(' ')
        if not is_image_file(data[0]):
            continue
        path = os.path.join(root, data[0])

        if ds_name == 'synROD' or ds_name in ['synHB', 'valHB']:
            path_rgb = path.replace('***', 'rgb')
            path_depth = path.replace('***', 'depth')
        elif ds_name == 'ROD':
            path_rgb = path.replace('***', 'crop')
            path_rgb = path_rgb.replace('???', 'rgb')
            path_depth = path.replace('***', 'depthcrop')
            path_depth = path_depth.replace('???', 'surfnorm')
        else:
            raise ValueError('Unknown dataset {}. Known datasets are synROD, synHB, ROD, valHB'.format(ds_name))
        gt = int(data[1])
        item = (path_rgb, path_depth, gt)
        images.append(item)
    return images


def get_relative_rotation(rgb_rot, depth_rot):
    rel_rot = rgb_rot - depth_rot
    if rel_rot < 0:
        rel_rot += 4
    assert rel_rot in range(4)
    return rel_rot


class MyTransformer(object):

    def __init__(self, crop, flip):
        super(MyTransformer, self).__init__()
        self.crop = crop
        self.flip = flip
        self.angles = [0, 90, 180, 270]

    def __call__(self, img, rot=None):
        img = TF.resize(img, (256, 256))
        img = TF.crop(img, self.crop[0], self.crop[1], 224, 224)
        if self.flip:
            img = TF.hflip(img)
        if rot is not None:
            img = TF.rotate(img, self.angles[rot])
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img


class DatasetGeneratorMultimodal(Dataset):
    def __init__(self, root, label, ds_name='synROD', do_rot=False, transform=None):
        imgs = make_sync_dataset(root, label, ds_name=ds_name)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.do_rot = do_rot

    def __getitem__(self, index):
        path_rgb, path_depth, target = self.imgs[index]
        img_rgb = load_image(path_rgb)
        img_depth = load_image(path_depth)
        rot_rgb = None
        rot_depth = None

        # If a custom transform is specified apply that transform
        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            img_depth = self.transform(img_depth)
        else:  # Otherwise define a random one (random cropping, random horizontal flip)
            top = random.randint(0, 256 - 224)
            left = random.randint(0, 256 - 224)
            flip = random.choice([True, False])
            if self.do_rot:
                rot_rgb = random.choice([0, 1, 2, 3])
                rot_depth = random.choice([0, 1, 2, 3])

            transform = MyTransformer([top, left], flip)
            # Apply the same transform to both modalities, rotating them if required
            img_rgb = transform(img_rgb, rot_rgb)
            img_depth = transform(img_depth, rot_depth)

        if self.do_rot and (self.transform is None):
            return img_rgb, img_depth, target, get_relative_rotation(rot_rgb, rot_depth)
        return img_rgb, img_depth, target

    def __len__(self):
        return len(self.imgs)
