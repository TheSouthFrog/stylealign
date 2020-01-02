import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset, get_list
from PIL import Image
import numpy as np
import math
import cv2
from transforms.affine_transforms import AffineCompose
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from util.curve import points_to_heatmap,points_to_heatmap_68points
import time

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def norm_angle(angle):
    norm_angle = sigmoid(10 * (abs(angle) / 45.0 - 1))
    return norm_angle

maps_angle = {
    '110': 90,
    '120': 75,
    '090': 60,
    '080': 45,
    '130': 30,
    '140': 15,
    '051': 0,
    '050': 15,
    '041': 30,
    '190': 45,
    '200': 60,
    '010': 75,
    '240': 90
}



class BasicFaceLandmarksDataset(BaseDataset):

    def initialize(self, txt_file, img_dir, transform=None):
        self.landmarks_frame = pd.read_table(txt_file,sep = ' ',header = None)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir,
            self.landmarks_frame.iloc[idx, -1])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 0:-1].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)
        return sample



class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt

        self.A_paths = []
        self.B_paths = []
        self.I_paths = []
        self.labels = []
        for root_dir_item in opt.root_dir:
            dir_A_tmp = os.path.join(root_dir_item, opt.phase + '_A')
            dir_B_tmp = os.path.join(root_dir_item, opt.phase + '_B')
            dir_I_tmp = os.path.join(root_dir_item, opt.phase + '_I')

            img_list = os.path.join(root_dir_item, opt.name_img_list)

            A_labels_tmp, A_paths_tmp = get_list(dir_A_tmp, img_list)
            B_labels_tmp, B_paths_tmp = get_list(dir_B_tmp, img_list)
            I_labels_tmp, I_paths_tmp = get_list(dir_I_tmp, img_list)

            self.A_paths += A_paths_tmp
            self.B_paths += B_paths_tmp
            self.I_paths += I_paths_tmp
            self.labels += A_labels_tmp

        # self.A_paths = make_dataset(self.dir_A)
        # self.B_paths = make_dataset(self.dir_B)
        # self.I_paths = make_dataset(self.dir_I)

        # self.A_paths = sorted(self.A_paths)
        # self.B_paths = sorted(self.B_paths)
        # self.I_paths = sorted(self.I_paths)

        assert(len(self.A_paths) == len(self.B_paths))
        assert(len(self.A_paths) == len(self.I_paths))
        assert(len(self.A_paths) == len(self.labels))
        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

        self.crop_type = opt.crop_type


    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        assert(A_path[A_path.index(self.opt.phase+'_A')+len(self.opt.phase+'_A')+1:] == B_path[B_path.index(self.opt.phase+'_B')+len(self.opt.phase+'_B')+1:])
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_img = A_img.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B_img = B_img.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        w = A_img.size(2)
        h = A_img.size(1)

        if self.crop_type == 'center_crop':
            w_offset = (w - self.opt.fineSize - 1) / 2
            h_offset = (h - self.opt.fineSize - 1) / 2
        else:
            w_offset = random.randint(((w - self.opt.fineSize - 1) / 2) - self.opt.translate_range, ((w - self.opt.fineSize - 1) / 2) + self.opt.translate_range)
            h_offset = random.randint(((h - self.opt.fineSize - 1) / 2) - self.opt.translate_range, ((h - self.opt.fineSize - 1) / 2) + self.opt.translate_range)

        A_img = A_img[:, h_offset:h_offset + self.opt.fineSize,
                w_offset:w_offset + self.opt.fineSize]
        B_img = B_img[:, h_offset:h_offset + self.opt.fineSize,
                w_offset:w_offset + self.opt.fineSize]

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # if (not self.opt.no_flip) and random.random() < 0.5:
        #     idx = [i for i in range(A.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     A = A.index_select(2, idx)
        #     B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A_img[0, ...] * 0.299 + A_img[1, ...] * 0.587 + A_img[2, ...] * 0.114
            A_img = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B_img[0, ...] * 0.299 + B_img[1, ...] * 0.587 + B_img[2, ...] * 0.114
            B_img = tmp.unsqueeze(0)

        if self.opt.illumination_prior:
            I_path = self.I_paths[index]
            assert(A_path[A_path.index(self.opt.phase+'_A')+len(self.opt.phase+'_A')+1:] == I_path[I_path.index(self.opt.phase+'_I')+len(self.opt.phase+'_I')+1:])
            I_img = Image.open(I_path).convert('RGB')
            I_img = I_img.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            I_img = self.transform(I_img)
            I_img = I_img[:, h_offset:h_offset + self.opt.fineSize,
                    w_offset:w_offset + self.opt.fineSize]
            if input_nc == 1:  # RGB to gray
                tmp = I_img[0, ...] * 0.299 + I_img[1, ...] * 0.587 + I_img[2, ...] * 0.114
                I_img = tmp.unsqueeze(0)
            A_img = torch.cat((A_img, I_img), 0)

        if self.opt.use_weight:
            label = self.labels[index]
            assert(len(label) >= 1)
            label = np.asarray(label)
            label = torch.from_numpy(label).float()

            return {'A': A_img, 'B': B_img,
                    'A_paths': A_path, 'B_paths': B_path,
                    'labels': label}

        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'


class AlignedDatasetHeatmap2Face(BaseDataset):
    def initialize(self, opt):
        self.opt = opt

        self.A_paths = []
        self.B_paths = []
        for root_dir_item in opt.root_dir:
            dir_A_tmp = os.path.join(root_dir_item, opt.phase + '_A')
            dir_B_tmp = os.path.join(root_dir_item, opt.phase + '_B')

            img_list = os.path.join(root_dir_item, opt.name_img_list)

            A_labels_tmp, A_paths_tmp = get_list(dir_A_tmp, img_list)
            B_labels_tmp, B_paths_tmp = get_list(dir_B_tmp, img_list)

            self.A_paths += A_paths_tmp
            self.B_paths += B_paths_tmp

        assert(len(self.A_paths) == len(self.B_paths))
        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

        self.crop_type = opt.crop_type


    def __getitem__(self, index):
        B_path = self.B_paths[index]
        B_img = Image.open(B_path).convert('RGB')

        B_img = B_img.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B_img = self.transform(B_img)

        w = B_img.size(2)
        h = B_img.size(1)

        if self.crop_type == 'center_crop':
            w_offset = (w - self.opt.fineSize - 1) / 2
            h_offset = (h - self.opt.fineSize - 1) / 2
        else:
            w_offset = random.randint(((w - self.opt.fineSize - 1) / 2) - self.opt.translate_range, ((w - self.opt.fineSize - 1) / 2) + self.opt.translate_range)
            h_offset = random.randint(((h - self.opt.fineSize - 1) / 2) - self.opt.translate_range, ((h - self.opt.fineSize - 1) / 2) + self.opt.translate_range)

        B_img = B_img[:, h_offset:h_offset + self.opt.fineSize,
                w_offset:w_offset + self.opt.fineSize]

        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc

        for i in xrange(15):
            A_path = self.A_paths[index][:-4]+'_'+str(i)+'.jpg'
            A_img_split = Image.open(A_path).convert('RGB')
            A_img_split = A_img_split.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            A_img_split = self.transform(A_img_split)
            A_img_split = A_img_split[:, h_offset:h_offset + self.opt.fineSize,
                    w_offset:w_offset + self.opt.fineSize]
            tmp = A_img_split[0, ...] * 0.299 + A_img_split[1, ...] * 0.587 + A_img_split[2, ...] * 0.114
            A_img_split = tmp.unsqueeze(0)
            if i == 0:
                A_img = A_img_split
            elif self.opt.input_sum:
                A_img += A_img_split
            else:
                A_img = torch.cat((A_img, A_img_split), 0)



        if output_nc == 1:  # RGB to gray
            tmp = B_img[0, ...] * 0.299 + B_img[1, ...] * 0.587 + B_img[2, ...] * 0.114
            B_img = tmp.unsqueeze(0)


        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDatasetHeatmap2Face'


class AlignedBoundaryDetectionHeatmapIn(BaseDataset):

    def initialize(self, opt):
        self.opt = opt

        self.A_paths = []
        self.B_paths = []
        for root_dir_item in opt.root_dir:
            dir_A_tmp = os.path.join(root_dir_item, opt.phase + '_A')
            dir_B_tmp = os.path.join(root_dir_item, opt.phase + '_B')

            img_list = os.path.join(root_dir_item, opt.name_img_list)

            A_labels_tmp, A_paths_tmp = get_list(dir_A_tmp, img_list)
            B_labels_tmp, B_paths_tmp = get_list(dir_B_tmp, img_list)

            self.A_paths += A_paths_tmp
            self.B_paths += B_paths_tmp

        self.color_A = opt.color_A
        self.color_B = opt.color_B
        self.heatmap_size = opt.fineSize_B
        self.transform = AffineCompose(rotation_range=opt.rotate_range,
                                      translation_range=opt.translate_range,
                                      zoom_range=opt.zoom_range,
                                      output_img_width=opt.fineSize_A,
                                      output_img_height=opt.fineSize_A,
                                      mirror=opt.mirror,
                                      corr_list=None,
                                      normalise=opt.normalise,
                                      normalisation_type=opt.normalisation_type)

    def __getitem__(self, index):
        inputs = []
        inputs_transform = []

        A_path = self.A_paths[index]
        A_img = cv2.imread(A_path, 1)
        # convert BGR to RGB
        b,g,r = cv2.split(A_img)
        A_img = cv2.merge([r,g,b])
        if not self.color_A:
            A_img = cv2.cvtColor(A_img, cv2.COLOR_BGR2GRAY)
            A_img = np.expand_dims(np.asarray(A_img), axis=2)
        A_img = np.asarray(A_img)
        A_img = A_img.transpose((2, 0, 1))
        A_img = torch.from_numpy(A_img).float()
        inputs.append(A_img)

        for i in xrange(self.opt.output_nc):
            B_path = self.B_paths[index][:-4]+'_'+str(i)+'.jpg'
            B_img = cv2.imread(B_path, 1)
            if not self.color_B:
                B_img = B_img[:,:,0]
                B_img = np.expand_dims(np.asarray(B_img), axis=2)
            B_img = np.asarray(B_img)
            B_img = B_img.transpose((2, 0, 1))
            B_img = torch.from_numpy(B_img).float()
            inputs.append(B_img)

        inputs_transform = self.transform(*inputs)


        for i in xrange(1,len(inputs_transform)):
            # heatmap_tmp.shape = (256,256,1)
            heatmap_tmp = inputs_transform[i].numpy().transpose((1, 2, 0)).astype(np.float32)
            # heatmap_resize.shape = (64,64)
            heatmap_resize = cv2.resize(heatmap_tmp, (self.heatmap_size,self.heatmap_size), 0, 0, cv2.INTER_CUBIC)
            heatmap_resize = np.expand_dims(np.asarray(heatmap_resize), axis=2)
            heatmap_resize = heatmap_resize.transpose((2, 0, 1))
            heatmap_resize = torch.from_numpy(heatmap_resize).float()
            if i == 1:
                heatmaps = heatmap_resize
            else:
                heatmaps = torch.cat((heatmaps, heatmap_resize), 0)

        # debug
        assert(inputs_transform[0].shape == (3,self.opt.fineSize_A,self.opt.fineSize_A))
        assert(heatmaps.shape == (self.opt.output_nc,self.opt.fineSize_B,self.opt.fineSize_B))

        # vis_dir = '/mnt/lustre/wuwenyan/workspace_1/eccv_2018_2/exp_model/PyTorch_CycleGAN_and_Pix2Pix/checkpoints/exp_2034/vis/'

        # plt.figure()
        # image_tensor = inputs_transform[0]
        # real_A = image_tensor.cpu().float().numpy()
        # if real_A.shape[0] == 1:
        #     real_A = np.tile(real_A, (3, 1, 1))
        # real_A = (np.transpose(real_A, (1, 2, 0)) + 1) / 2.0 * 255.0
        # plt.imshow(real_A.astype(np.uint8))
        # plt.savefig(vis_dir+"real_A_"+"batch_"+str(index)+".png")


        # for j in xrange(15):
        #     image_tensor = heatmaps[j,:,:].unsqueeze(0)
        #     real_B = image_tensor.cpu().float().numpy()
        #     if real_B.shape[0] == 1:
        #         real_B = np.tile(real_B, (3, 1, 1))
        #     real_B = (np.transpose(real_B, (1, 2, 0))) * 255.0
        #     plt.imshow(real_B.astype(np.uint8))
        #     plt.savefig(vis_dir+"real_B_"+"batch_"+str(index)+"_part_"+str(j)+".png")

        return {'A': inputs_transform[0], 'B': heatmaps}


    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedBoundaryDetectionHeatmapIn'


class AlignedFaceDataset(BaseDataset):

    def initialize(self, opt):
        self.opt = opt

        self.A_paths = []
        self.A_labels = []
        for root_dir_item in opt.root_dir:
            dir_A_tmp = os.path.join(root_dir_item, 'Image')
            img_list = os.path.join(root_dir_item, opt.name_landmarks_list)
            A_labels_tmp, A_paths_tmp = get_list(dir_A_tmp, img_list)
	    self.A_labels += A_labels_tmp
            self.A_paths += A_paths_tmp

        self.color_A = opt.color_A
        self.label_size = opt.fineSize
	self.output_size = opt.fineSize_A
        self.heatmap_size = opt.fineSize_B
        self.heatmap_num = opt.output_nc
        self.sigma = opt.sigma
        self.label_num = opt.label_num
        self.transform = AffineCompose(rotation_range=opt.rotate_range,
                                      translation_range=opt.translate_range,
                                      zoom_range=opt.zoom_range,
                                      output_img_width=self.label_size,
                                      output_img_height=self.label_size,
                                      mirror=opt.mirror,
                                      corr_list=opt.corr_list,
                                      normalise=opt.normalise,
                                      normalisation_type=opt.normalisation_type)

    def __getitem__(self, index):
        inputs = []
        inputs_transform = []

        A_path = self.A_paths[index]
        A_img = cv2.imread(A_path, 1)
        B_index = np.random.randint(4000)
        # convert BGR to RGB
        b,g,r = cv2.split(A_img)
        A_img = cv2.merge([r,g,b])
        if not self.color_A:
            A_img = cv2.cvtColor(A_img, cv2.COLOR_BGR2GRAY)
            A_img = np.expand_dims(np.asarray(A_img), axis=2)
        A_img = np.asarray(A_img)
        A_img = A_img.transpose((2, 0, 1))
        A_img = torch.from_numpy(A_img).float()
	inputs.append(A_img)
        A_yaw = norm_angle(maps_angle[A_path[128:131]])

        landmarks = self.A_labels[index]
        assert(len(landmarks) == self.label_num)
        landmarks = np.asarray(landmarks)
        landmarks = landmarks.reshape(-1, 2)
        landmarks = torch.from_numpy(landmarks).float()
        inputs.append(landmarks)

        inputs_transform = self.transform(*inputs)

        A_img = inputs_transform[0]

        inputs = []
        inputs_transform = []

	B_path = self.A_paths[B_index]
	B_img = cv2.imread(B_path, 1)
        b,g,r = cv2.split(B_img)
        B_img = cv2.merge([r,g,b])
        if not self.color_A:
            B_img = cv2.cvtColor(B_img, cv2.COLOR_BGR2GRAY)
            B_img = np.expand_dims(np.asarray(B_img), axis=2)
        B_img = np.asarray(B_img)
        B_img = B_img.transpose((2, 0, 1))
        B_img = torch.from_numpy(B_img).float()
        B_yaw = norm_angle(maps_angle[B_path[128:131]])

        inputs.append(B_img)

        landmarks = self.A_labels[B_index]
        assert(len(landmarks) == self.label_num)
        landmarks = np.asarray(landmarks)
        landmarks = landmarks.reshape(-1, 2)
        landmarks = torch.from_numpy(landmarks).float()
        inputs.append(landmarks)

        # time_0 = time.time()
        inputs_transform = self.transform(*inputs)
        B = inputs_transform[0]
        B_BD = points_to_heatmap_68points(inputs_transform[1].squeeze().numpy().astype(np.float32),
            self.heatmap_num, self.heatmap_size, self.label_size, self.sigma)
        B_BD = B_BD.transpose((2, 0, 1))
        B_BD = torch.from_numpy(B_BD).float()
	    # print A_img.shape
	    # print (self.opt.input_nc,self.opt.fineSize_A,self.opt.fineSize_A)

        # debug
        assert(A_img.shape == (self.opt.input_nc,self.opt.fineSize,self.opt.fineSize))
        assert(B_BD.shape == (self.opt.output_nc,self.opt.fineSize_B,self.opt.fineSize_B))

        return {'A': A_img, 'B': B_BD, 'raw_B': B, 'A_yaw': A_yaw, 'B_yaw': B_yaw}


    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedFaceDataset'




class AlignedBoundaryDetection(BaseDataset):

    def initialize(self, opt):
        self.opt = opt

        self.A_paths = []
        self.A_labels = []
        for root_dir_item in opt.root_dir:
            dir_A_tmp = os.path.join(root_dir_item, 'Image')
            img_list = os.path.join(root_dir_item, opt.name_landmarks_list)
            A_labels_tmp, A_paths_tmp = get_list(dir_A_tmp, img_list)

            self.A_paths += A_paths_tmp
            self.A_labels += A_labels_tmp

        self.color_A = opt.color_A
        self.label_size = opt.fineSize_A
        self.heatmap_size = opt.fineSize_B
        self.heatmap_num = opt.output_nc
        self.sigma = opt.sigma
        self.label_num = opt.label_num
        self.transform = AffineCompose(rotation_range=opt.rotate_range,
                                      translation_range=opt.translate_range,
                                      zoom_range=opt.zoom_range,
                                      output_img_width=self.label_size,
                                      output_img_height=self.label_size,
                                      mirror=opt.mirror,
                                      corr_list=opt.corr_list,
                                      normalise=opt.normalise,
                                      normalisation_type=opt.normalisation_type)

    def __getitem__(self, index):
        inputs = []
        inputs_transform = []

        A_path = self.A_paths[index]
        A_img = cv2.imread(A_path, 1)
        # convert BGR to RGB
        b,g,r = cv2.split(A_img)
        A_img = cv2.merge([r,g,b])
        if not self.color_A:
            A_img = cv2.cvtColor(A_img, cv2.COLOR_BGR2GRAY)
            A_img = np.expand_dims(np.asarray(A_img), axis=2)
        A_img = np.asarray(A_img)
        A_img = A_img.transpose((2, 0, 1))
        A_img = torch.from_numpy(A_img).float()
        inputs.append(A_img)

        landmarks = self.A_labels[index]
        assert(len(landmarks) == self.label_num)
        landmarks = np.asarray(landmarks)
        landmarks = landmarks.reshape(-1, 2)
        landmarks = torch.from_numpy(landmarks).float()
        inputs.append(landmarks)

        # time_0 = time.time()
        # inputs_transform = self.transform(*inputs)
	inputs_transform = inputs
        # time_1 = time.time()
        # print('TIME_0_1:{}'.format(time_1-time_0))

        A = inputs_transform[0]
        B = points_to_heatmap_68points(inputs_transform[1].squeeze().numpy().astype(np.float32),
            self.heatmap_num, self.heatmap_size, self.label_size, self.sigma)
        # time_2 = time.time()
        # print('TIME_1_2:{}'.format(time_2-time_1))
        B = B.transpose((2, 0, 1))
        B = torch.from_numpy(B).float()

        # debug
        assert(A.shape == (self.opt.input_nc,self.opt.fineSize_A,self.opt.fineSize_A))
        assert(B.shape == (self.opt.output_nc,self.opt.fineSize_B,self.opt.fineSize_B))

        # vis_dir = '/mnt/lustre/wuwenyan/workspace_1/eccv_2018_2/exp_model/PyTorch_CycleGAN_and_Pix2Pix/checkpoints/exp_2039/vis/'

        # plt.figure()
        # image_tensor = A
        # real_A = image_tensor.cpu().float().numpy()
        # if real_A.shape[0] == 1:
        #     real_A = np.tile(real_A, (3, 1, 1))
        # real_A = (np.transpose(real_A, (1, 2, 0)) + 1) / 2.0 * 255.0
        # plt.imshow(real_A.astype(np.uint8))
        # plt.savefig(vis_dir+"real_A_"+"batch_"+str(index)+".png")


        # for j in xrange(15):
        #     image_tensor = B[j,:,:].unsqueeze(0)
        #     real_B = image_tensor.cpu().float().numpy()
        #     if real_B.shape[0] == 1:
        #         real_B = np.tile(real_B, (3, 1, 1))
        #     real_B = (np.transpose(real_B, (1, 2, 0))) * 255.0
        #     # real_B = (np.transpose(real_B, (1, 2, 0)))
        #     plt.imshow(real_B.astype(np.uint8))
        #     plt.savefig(vis_dir+"real_B_"+"batch_"+str(index)+"_part_"+str(j)+".png")

        return {'A': A, 'B': B}


    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedBoundaryDetection'


class AlignedBoundaryDetectionLandmark(BaseDataset):

    def initialize(self, opt):
        self.opt = opt

        self.A_paths = []
        self.A_labels = []
        for root_dir_item in opt.root_dir:
            dir_A_tmp = os.path.join(root_dir_item, 'Image')
            img_list = os.path.join(root_dir_item, opt.name_landmarks_list)
            A_labels_tmp, A_paths_tmp = get_list(dir_A_tmp, img_list)

            self.A_paths += A_paths_tmp
            self.A_labels += A_labels_tmp

        self.color_A = opt.color_A
        self.label_size = opt.fineSize_A
        self.heatmap_size = opt.fineSize_B
        self.heatmap_num = opt.output_nc
        self.sigma = opt.sigma
        self.label_num = opt.label_num
        self.transform = AffineCompose(rotation_range=opt.rotate_range,
                                      translation_range=opt.translate_range,
                                      zoom_range=opt.zoom_range,
                                      output_img_width=self.label_size,
                                      output_img_height=self.label_size,
                                      mirror=opt.mirror,
                                      corr_list=opt.corr_list,
                                      normalise=opt.normalise,
                                      normalisation_type=opt.normalisation_type)

    def __getitem__(self, index):
        inputs = []
        inputs_transform = []

        A_path = self.A_paths[index]
        A_img = cv2.imread(A_path, 1)
        # convert BGR to RGB
        b,g,r = cv2.split(A_img)
        A_img = cv2.merge([r,g,b])
        if not self.color_A:
            A_img = cv2.cvtColor(A_img, cv2.COLOR_BGR2GRAY)
            A_img = np.expand_dims(np.asarray(A_img), axis=2)
        A_img = np.asarray(A_img)
        A_img = A_img.transpose((2, 0, 1))
        A_img = torch.from_numpy(A_img).float()
        inputs.append(A_img)

        landmarks = self.A_labels[index]
        assert(len(landmarks) == self.label_num)
        landmarks = np.asarray(landmarks)
        landmarks = landmarks.reshape(-1, 2)
        landmarks = torch.from_numpy(landmarks).float()
        inputs.append(landmarks)

        # time_0 = time.time()
        inputs_transform = self.transform(*inputs)
        # time_1 = time.time()
        # print('TIME_0_1:{}'.format(time_1-time_0))

        A = inputs_transform[0]
        B = points_to_heatmap(inputs_transform[1].squeeze().numpy().astype(np.float32),
            self.heatmap_num, self.heatmap_size, self.label_size, self.sigma)
        # time_2 = time.time()
        # print('TIME_1_2:{}'.format(time_2-time_1))
        B = B.transpose((2, 0, 1))
        B = torch.from_numpy(B).float()

        # debug
        assert(A.shape == (self.opt.input_nc,self.opt.fineSize_A,self.opt.fineSize_A))
        assert(B.shape == (self.opt.output_nc,self.opt.fineSize_B,self.opt.fineSize_B))

        return {'A': A, 'B': B, 'C': landmarks}


    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedBoundaryDetectionLandmark'


class AlignedFace2Boudnary2Face(BaseDataset):

    def initialize(self, opt):
        self.opt = opt

        self.F1_paths = []
        self.F1_labels = []
        for root_dir_item in opt.root_dir:
            dir_F1_tmp = os.path.join(root_dir_item, 'train' + '_F1')
            img_list = os.path.join(root_dir_item, opt.name_landmarks_list)
            F1_labels_tmp, F1_paths_tmp = get_list(dir_F1_tmp, img_list)

            self.F1_paths += F1_paths_tmp
            self.F1_labels += F1_labels_tmp

        self.serial_batches = opt.serial_batches
        self.need_boundary_gt = opt.need_boundary_gt
        self.color_F1 = opt.color_F1
        self.fineSize_F1 = opt.fineSize_F1
        self.fineSize_Boundary = opt.fineSize_Boundary
        self.nc_Boundary = opt.nc_Boundary
        self.sigma = opt.sigma
        self.label_num = opt.label_num
        self.transform = AffineCompose(rotation_range=opt.rotate_range,
                                      translation_range=opt.translate_range,
                                      zoom_range=opt.zoom_range,
                                      output_img_width=self.fineSize_F1,
                                      output_img_height=self.fineSize_F1,
                                      mirror=opt.mirror,
                                      corr_list=opt.corr_list,
                                      normalise=opt.normalise,
                                      normalisation_type=opt.normalisation_type)

    def __getitem__(self, index):
        # if self.serial_batches:
        #     index = index % len(self.F1_paths)
        # else:
        #     index = random.randint(0, len(self.F1_paths) - 1)

        inputs = []
        inputs_transform = []

        F1_path = self.F1_paths[index]
        F1_img = cv2.imread(F1_path, 1)
        # convert BGR to RGB
        b,g,r = cv2.split(F1_img)
        F1_img = cv2.merge([r,g,b])
        if not self.color_F1:
            F1_img = cv2.cvtColor(F1_img, cv2.COLOR_BGR2GRAY)
            F1_img = np.expand_dims(np.asarray(F1_img), axis=2)
        F1_img = np.asarray(F1_img)
        F1_img = F1_img.transpose((2, 0, 1))
        F1_img = torch.from_numpy(F1_img).float()
        inputs.append(F1_img)

        landmarks = self.F1_labels[index]
        assert(len(landmarks) == self.label_num)
        landmarks = np.asarray(landmarks)
        landmarks = landmarks.reshape(-1, 2)
        landmarks = torch.from_numpy(landmarks).float()
        inputs.append(landmarks)

        inputs_transform = self.transform(*inputs)

        F1 = inputs_transform[0]

        if self.need_boundary_gt:
            Boundary = points_to_heatmap(inputs_transform[1].squeeze().numpy().astype(np.float32),
                self.nc_Boundary, self.fineSize_Boundary, self.fineSize_F1, self.sigma)
            Boundary = Boundary.transpose((2, 0, 1))
            Boundary = torch.from_numpy(Boundary).float()
        else:
            Boundary = torch.FloatTensor(self.opt.nc_Boundary,self.opt.fineSize_Boundary,self.opt.fineSize_Boundary).zero_()

        # debug
        assert(F1.shape == (self.opt.nc_F1,self.opt.fineSize_F1,self.opt.fineSize_F1))
        assert(Boundary.shape == (self.opt.nc_Boundary,self.opt.fineSize_Boundary,self.opt.fineSize_Boundary))
        # vis_dir = '/mnt/lustre/wuwenyan/workspace_1/eccv_2018_2/exp_model/PyTorch_CycleGAN_and_Pix2Pix/checkpoints/exp_2039/vis/'

        # plt.figure()
        # image_tensor = A
        # real_A = image_tensor.cpu().float().numpy()
        # if real_A.shape[0] == 1:
        #     real_A = np.tile(real_A, (3, 1, 1))
        # real_A = (np.transpose(real_A, (1, 2, 0)) + 1) / 2.0 * 255.0
        # plt.imshow(real_A.astype(np.uint8))
        # plt.savefig(vis_dir+"real_F1_"+"batch_"+str(index)+".png")


        # for j in xrange(15):
        #     image_tensor = B[j,:,:].unsqueeze(0)
        #     real_B = image_tensor.cpu().float().numpy()
        #     if real_B.shape[0] == 1:
        #         real_B = np.tile(real_B, (3, 1, 1))
        #     real_B = (np.transpose(real_B, (1, 2, 0))) * 255.0
        #     # real_B = (np.transpose(real_B, (1, 2, 0)))
        #     plt.imshow(real_B.astype(np.uint8))
        #     plt.savefig(vis_dir+"real_B_"+"batch_"+str(index)+"_part_"+str(j)+".png")

        return {'F1': F1, 'Boundary': Boundary, 'F2': F1, 'F1_path': F1_path}


    def __len__(self):
        return len(self.F1_paths)

    def name(self):
        return 'AlignedFace2Boudnary2Face'


class AlignedFace2Face(BaseDataset):

    def initialize(self, opt):
        self.opt = opt

        self.F1_paths = []
        self.F1_labels = []
        for root_dir_item in opt.root_dir:
            dir_F1_tmp = os.path.join(root_dir_item, 'train' + '_F1')
            img_list = os.path.join(root_dir_item, opt.name_landmarks_list)
            F1_labels_tmp, F1_paths_tmp = get_list(dir_F1_tmp, img_list)

            self.F1_paths += F1_paths_tmp
            self.F1_labels += F1_labels_tmp

        self.color_F1 = opt.color_F1
        self.fineSize_F1 = opt.fineSize_F1
        self.sigma = opt.sigma
        self.label_num = opt.label_num
        self.transform = AffineCompose(rotation_range=opt.rotate_range,
                                      translation_range=opt.translate_range,
                                      zoom_range=opt.zoom_range,
                                      output_img_width=self.fineSize_F1,
                                      output_img_height=self.fineSize_F1,
                                      mirror=opt.mirror,
                                      corr_list=opt.corr_list,
                                      normalise=opt.normalise,
                                      normalisation_type=opt.normalisation_type)

    def __getitem__(self, index):
        # if self.serial_batches:
        #     index = index % len(self.F1_paths)
        # else:
        #     index = random.randint(0, len(self.F1_paths) - 1)

        inputs = []
        inputs_transform = []

        F1_path = self.F1_paths[index]
        F1_img = cv2.imread(F1_path, 1)
        # convert BGR to RGB
        b,g,r = cv2.split(F1_img)
        F1_img = cv2.merge([r,g,b])
        if not self.color_F1:
            F1_img = cv2.cvtColor(F1_img, cv2.COLOR_BGR2GRAY)
            F1_img = np.expand_dims(np.asarray(F1_img), axis=2)
        F1_img = np.asarray(F1_img)
        F1_img = F1_img.transpose((2, 0, 1))
        F1_img = torch.from_numpy(F1_img).float()
        inputs.append(F1_img)

        landmarks = self.F1_labels[index]
        assert(len(landmarks) == self.label_num)
        landmarks = np.asarray(landmarks)
        landmarks = landmarks.reshape(-1, 2)
        landmarks = torch.from_numpy(landmarks).float()
        inputs.append(landmarks)

        inputs_transform = self.transform(*inputs)

        F1 = inputs_transform[0]

        # debug
        assert(F1.shape == (self.opt.nc_F1,self.opt.fineSize_F1,self.opt.fineSize_F1))

        return {'A': F1, 'B': F1, 'A_paths': F1_path}


    def __len__(self):
        return len(self.F1_paths)

    def name(self):
        return 'AlignedFace2Face'
