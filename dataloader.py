import numpy as np
from util.curve import points_to_landmark_map
from util.image_folder import get_list
from util.util import load_img, preprocess
from util.buffer import BufferedWrapper
from transfroms.affine_transforms import AffineCompose


class LandmarkLoader(object):
    def __init__(self, shape, train, sigma, data_dir, img_list, fill_batches = True, shuffle = False,
            return_keys = ["imgs", "joints", "norm_imgs", "norm_joints"]):
        self.shape = shape
        self.batch_size = self.shape[0]
        self.img_shape = self.shape[1:]
        self.data_dir = data_dir
        self.img_list = img_list
        self.labels, self.paths = get_list(self.data_dir, self.img_list)
        corr_list=[0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9,17,26,18,25,19,24,20,2321,22,36,45,37,44,38,43,39,42,41,46,40,47,31,35,32,34,48,54,49,53,50,52,60,64,61,63,67,65,59,55,58,56]
        self.transform = AffineCompose(rotation_range=0, translation_range = 0, zoom_range=[1.0,1.0],output_img_width = 256, output_img_height = 256, mirror = False, corr_list = corr_list)
        self.train = train
        self.fill_batches = fill_batches
        self.shuffle_ = shuffle
        self.return_keys = return_keys
        self.sigma = sigma
        self.indices = np.array([i for i in range(len(self.labels))])
        self.n = self.indices.shape[0]
        self.shuffle()


    def __next__(self):
        batch = dict()
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        batch_indices = self.indices[batch_start:batch_end]
        if self.fill_batches and batch_indices.shape[0] != self.batch_size:
            n_missing = self.batch_size - batch_indices.shape[0]
            batch_indices = np.concatenate([batch_indices, self.indices[:n_missing]], axis = 0)
            assert batch_indices.shape[0] == self.batch_size
        batch_indices = np.array(batch_indices)
        batch["indices"] = batch_indices
        if batch_end >= self.n:
            self.shuffle()
        else:
            self.batch_start = batch_end
        # load images

        batch["imgs"] = list()
        batch["joints"] = list()
        for i in batch_indices:
            path = self.paths[i]
            img = load_img(path, target_size = [256,256,3])
            label = self.labels[i]
            label = np.asarray(label).reshape(-1,2)
            # inputs = []
            # inputs.append(img)
            # inputs.append(label)
            # inputs_transform = self.transform(*inputs)
            # batch['imgs'].append(inputs_transform[0])
            batch['imgs'].append(img)
            # label = inputs_transform[1]
            label = label * 2 / 3
            boundary = points_to_landmark_map(label.astype(np.float32),1,256,256,self.sigma)
            boundary = np.tile(boundary, (1, 1, 3))
            batch["joints"].append(boundary)
        batch["imgs"] = np.stack(batch["imgs"])
        batch["imgs"] = preprocess(batch["imgs"])
        batch["joints"] = np.stack(batch["joints"])
        batch["norm_imgs"] = batch["imgs"]
        batch["norm_joints"] = batch["joints"]

        batch_list = [batch[k] for k in self.return_keys]
        return batch_list


    def shuffle(self):
        self.batch_start = 0
        if self.shuffle_:
            np.random.shuffle(self.indices)


def landmarks_loader(shape, train, sigma, data_dir, img_list, fill_batches = True, shuffle = False,
        return_keys = ["imgs", "joints", "norm_imgs", "norm_joints"]):
    loader = LandmarkLoader(shape, train, sigma,data_dir, img_list, fill_batches, shuffle, return_keys)
    return BufferedWrapper(loader)
