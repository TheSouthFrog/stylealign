###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

#import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def get_list(dir, label_file):
    labels = []
    images = []
    fh = open(label_file)
    for line in fh.readlines():
        item = line.split()
        path = os.path.join(dir, item.pop(-1))
        images.append(path)
        labels.append(tuple([float(v) for v in item]))
    return labels, images


def get_path(dir, img_list):
    images = []
    fh = open(img_list)
    for line in fh.readlines():
        item = line.split()
        assert(len(item)==1)
        path = os.path.join(dir, item[0])
        images.append(path)
    return images


def default_loader(path):
    return Image.open(path).convert('RGB')

