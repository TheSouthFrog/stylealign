import PIL.Image
from multiprocessing.pool import ThreadPool
import numpy as np
import pickle
import os
import os.path
import cv2
import math
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from util.curve import points_to_heatmap,points_to_heatmap_68points,points_to_heatmap_98pt,points_to_landmark_map
from Data.image_folder import make_dataset, get_list
from PIL import Image
from transfroms.affine_transforms import AffineCompose
import sys



class BufferedWrapper(object):
    """Fetch next batch asynchronuously to avoid bottleneck during GPU
    training."""
    def __init__(self, gen):
        self.gen = gen
        self.n = gen.n
        self.pool = ThreadPool(1)
        self._async_next()


    def _async_next(self):
        self.buffer_ = self.pool.apply_async(next, (self.gen,))


    def __next__(self):
        result = self.buffer_.get()
        self._async_next()
        return result


def load_img(path, target_size):
    """Load image. target_size is specified as (height, width, channels)
    where channels == 1 means grayscale. uint8 image returned."""
    img = PIL.Image.open(path)
    grayscale = target_size[2] == 1
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    wh_tuple = (target_size[1], target_size[0])
    if img.size != wh_tuple:
        img = img.resize(wh_tuple, resample = PIL.Image.BILINEAR)

    x = np.asarray(img, dtype = "uint8")
    if len(x.shape) == 2:
        x = np.expand_dims(x, -1)

    return x


def save_image(X, name):
    """Save image as png."""
    fname = os.path.join(out_dir, name + ".png")
    PIL.Image.fromarray(X).save(fname)


def preprocess(x):
    """From uint8 image to [-1,1]."""
    return np.cast[np.float32](x / 127.5 - 1.0)


def postprocess(x):
    """[-1,1] to uint8."""
    x = (x + 1.0) / 2.0
    x = np.clip(255 * x, 0, 255)
    x = np.cast[np.uint8](x)
    return x


def tile(X, rows, cols):
    """Tile images for display."""
    tiling = np.zeros((rows * X.shape[1], cols * X.shape[2], X.shape[3]), dtype = X.dtype)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < X.shape[0]:
                img = X[idx,...]
                tiling[
                        i*X.shape[1]:(i+1)*X.shape[1],
                        j*X.shape[2]:(j+1)*X.shape[2],
                        :] = img
    return tiling


def plot_batch(X, out_path):
    """Save batch of images tiled."""
    n_channels = X.shape[3]
    if n_channels > 3:
        X = X[:,:,:,np.random.choice(n_channels, size = 3)]
    X = postprocess(X)
    rc = math.sqrt(X.shape[0])
    rows = cols = math.ceil(rc)
    canvas = tile(X, rows, cols)
    canvas = np.squeeze(canvas)
    PIL.Image.fromarray(canvas).save(out_path)

def plot_batch_joint(X, out_path):
    n_channels = X.shape[3]
#    if n_channels > 3:
#        X = X[:,:,:,np.random.choice(n_channels, size = 3)]
    v_max = np.max(X)
    v_min = np.min(X)
    X = ((X - v_min) / (v_max - v_min)) * 255.0
    X = np.cast[np.uint8](X)
    rc = math.sqrt(X.shape[0])
    rows = cols = math.ceil(rc)
    canvas = tile(X, rows, cols)
    canvas = np.squeeze(canvas)
    PIL.Image.fromarray(canvas).save(out_path)


def TensorCenterCrop(img,cropx,cropy):
    _,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:,starty:starty+cropy,startx:startx+cropx]

def TensorRandomCrop(img, width, height):
    assert img.shape[2] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[:,y:y+height, x:x+width]
    return img


class FaceDataset(data.Dataset):
    def __init__(self,img_shape,train,sigma, root_dir,img_list,input_dim_b,zoom_min,zoom_max,color = True):
        super(FaceDataset, self).__init__()
        self.img_shape = img_shape
        self.train = train
        self.root_dir = root_dir
        self.img_list = img_list
        self.labels, self.paths = get_list(self.root_dir, self.img_list)
        self.sigma = sigma
        self.input_dim_b = input_dim_b
        corr_list = [0,32,1,31,2,30,3,29,4,28,5,27,6,26,7,25,8,24,9,23,10,22,11,21,12,20,13,19,14,18,15,17,33,46,34,45,35,44,36,43,37,42,38,50,39,49,40,48,41,47,55,59,56,58,60,72,61,71,62,70,63,69,64,68,65,75,66,74,67,73,76,82,77,81,78,80,88,92,89,91,95,93,87,83,86,84,96,97]
        #self.transform = AffineCompose(rotation_range=5, translation_range = 10, zoom_range=[0.97,1.03],output_img_width = 256, output_img_height = 256, mirror = True, corr_list = corr_list)
        self.transform = AffineCompose(rotation_range=0, translation_range = 0, zoom_range=[zoom_min,zoom_max],output_img_width = self.img_shape[0], output_img_height = self.img_shape[1], mirror = True, corr_list = corr_list)

    def __getitem__(self,index):
        inputs = []
        inputs_transform = []
        path = self.paths[index]
        img = load_img(path, target_size = [384,384,3])
        label = self.labels[index]
        label = np.asarray(label).reshape(-1,2)
        inputs = []
        inputs.append(img)
        inputs.append(label)
        inputs_transform = self.transform(*inputs)
        img = inputs_transform[0].transpose((2,0,1))
        #print(img.shape)
        img = preprocess(img)
        #print(img.shape)
        landmark = inputs_transform[1]

        if self.input_dim_b == 45:
            struc_map = points_to_heatmap_98pt(landmark.astype(np.float32), 15, self.img_shape[0], self.img_shape[0], self.sigma)
        elif self.input_dim_b == 3:
            struc_map = points_to_landmark_map(landmark.astype(np.float32), 1, self.img_shape[0],self.img_shape[0], self.sigma)
        else:
            sys.exit("Not Implement.")

        struc_map = np.tile(struc_map, (1, 1, 3))
        struc_map = struc_map.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        struc_map = torch.from_numpy(struc_map).float()
        return img, struc_map

    def __len__(self):
        return len(self.paths)
