from __future__ import print_function
from PIL import Image
import inspect, re
import collections
import cv2
import PIL.Image
import numpy as np
import os
import os.path
import math


def load_img(path, target_size):
    """Load image. target_size is specified as (height, width, channels)"""
    img = PIL.Image.open(path)
    grayscale = target_size[2] == 1
    # where channels == 1 means grayscale. uint8 image returned.
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

    v_max = np.max(X)
    v_min = np.min(X)
    X = ((X - v_min) / (v_max - v_min)) * 255.0
    X = np.cast[np.uint8](X)
    rc = math.sqrt(X.shape[0])
    rows = cols = math.ceil(rc)
    canvas = tile(X, rows, cols)
    canvas = np.squeeze(canvas)
    PIL.Image.fromarray(canvas).save(out_path)


def crop_center(img,cropx,cropy):
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def randomCrop(img, width, height):
    assert img.shape[2] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[:,y:y+height, x:x+width,:]
    return img



# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, img_type='regular', imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    if img_type == 'regular':
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    elif img_type == 'heatmap':
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        v_max = np.max(image_numpy)
        v_min = np.min(image_numpy)
        image_numpy = ((image_numpy - v_min) / (v_max - v_min)) * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def affine2d(x, matrix, output_img_width, output_img_height, center=True, is_landmarks=False):
    """
    2D Affine image transform on numpy array
    
    """
    assert(matrix.ndim == 2)
    matrix = matrix[:2,:]
    transform_matrix = matrix

    if is_landmarks:
        src = x

        dst = np.empty((src.shape[0],2), dtype=np.float32)
        for i in range(src.shape[0]):
            dst[i,:] = AffinePoint(np.expand_dims(src[i,:], axis=0), transform_matrix)

    else:

        src = x.astype(np.uint8)
        # cols, rows, channels = src.shape
        dst = cv2.warpAffine(src,transform_matrix,dsize=(output_img_width,output_img_height),flags =  cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue=(127,127,127))

        # for gray image
#        if len(dst.shape) == 2:
#            dst = np.expand_dims(np.asarray(dst), axis=2)

#        dst = dst.transpose((2, 0, 1))
        dst = dst.astype(np.float)

    return dst


def normalize_image(input_tf, normalisation_type):
    img_normalise = np.empty((input_tf.shape[0], input_tf.shape[1], input_tf.shape[2]), dtype=np.float32)
    num_channels = input_tf.shape[0]
    img = input_tf.numpy()
    
    if normalisation_type == 'channel-wise':
        for i in xrange(num_channels):
            mean = np.mean(img[i])
            std = np.std(img[i])
            if std < 1E-6:
                std = 1.0
            img_normalise[i] = (img[i] - mean) / std
    elif normalisation_type == 'regular':
        for i in xrange(num_channels):
            mean = 0.5
            std = 0.5
            img_normalise[i] = (img[i]/255. - mean) / std
    elif normalisation_type == 'heatmap':
        assert(num_channels==1)
        for i in xrange(num_channels):
            img_normalise[i] = img[i]/255.

    dst = torch.from_numpy(img_normalise).float()

    return dst


def AffinePoint(point, affine_mat):
    """
    Affine 2d point
    """
    assert(affine_mat.shape[0] == 2)
    assert(affine_mat.shape[1] == 3)
    assert(point.shape[1] == 2)

    point_x = point[0,0]
    point_y = point[0,1]
    result = np.empty((1,2), dtype=np.float32)
    result[0,0] = affine_mat[0,0] * point_x + \
                affine_mat[0,1] * point_y + \
                affine_mat[0,2]
    result[0,1] = affine_mat[1,0] * point_x + \
                affine_mat[1,1] * point_y + \
                affine_mat[1,2]
    
    return result


def exchange_landmarks(input_tf, corr_list):
    """
    Exchange value of pair of landmarks
    """
    for i in xrange(corr_list.shape[0]):
        temp = input_tf[0,corr_list[i][0],:].clone()
        input_tf[0,corr_list[i][0],:] = input_tf[0,corr_list[i][1],:]
        input_tf[0,corr_list[i][1],:] = temp

    return input_tf


def de_normalise(batch):
    # de normalise for regular normalisation
    batch = (batch + 1.0) / 2.0 * 255.0
    return batch

def normalise_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.div(batch, 255.0)
    batch -= Variable(mean)
    batch = batch / Variable(std)
    return batch

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)