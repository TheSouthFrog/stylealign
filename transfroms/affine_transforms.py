"""
Affine transforms and
requiring only one interpolation
"""
import random
import cv2

from util.util import affine2d, normalize_image, exchange_landmarks
import numpy as np


class AffineCompose(object):

    def __init__(self,
                rotation_range,
                translation_range,
                zoom_range,
                output_img_width,
                output_img_height,
                mirror=False,
                corr_list=None,
                normalisation_type='regular',
                ):

        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.zoom_range = zoom_range
        self.output_img_width = output_img_width
        self.output_img_height = output_img_height
        self.mirror = mirror
        self.corr_list = corr_list
        self.normalisation_type = normalisation_type

    def __call__(self, *inputs):
        input_img_width = inputs[0].shape[0]
        input_img_height = inputs[0].shape[1]
        rotate = random.uniform(-self.rotation_range, self.rotation_range)
        trans_x = random.uniform(-self.translation_range, self.translation_range)
        trans_y = random.uniform(-self.translation_range, self.translation_range)
        if not isinstance(self.zoom_range, list) and not isinstance(self.zoom_range, tuple):
            raise ValueError('zoom_range must be tuple or list with 2 values')
        zoom = random.uniform(self.zoom_range[0], self.zoom_range[1])

        # rotate 
        transform_matrix = np.zeros((3,3))
        center = (inputs[0].shape[0]/2.-0.5, inputs[0].shape[1]/2-0.5)
        M = cv2.getRotationMatrix2D(center, rotate, 1)
        transform_matrix[:2,:] = M
        transform_matrix[2,:] = np.array([0, 0, 1])
        # translate 
        transform_matrix[0,2] += trans_x
        transform_matrix[1,2] += trans_y
        # zoom
        for i in range(3):
            transform_matrix[0,i] *= zoom
            transform_matrix[1,i] *= zoom
        transform_matrix[0,2] += (1.0 - zoom) * center[0]
        transform_matrix[1,2] += (1.0 - zoom) * center[1]
        # if needed, apply crop together with affine to accelerate
        transform_matrix[0,2] -= (input_img_width-self.output_img_width) / 2.0;
        transform_matrix[1,2] -= (input_img_height-self.output_img_height) / 2.0;

        # mirror about x axis in cropped image
        do_mirror = False
        if self.mirror:
            mirror_rng = random.uniform(0.,1.)
            if mirror_rng>0.5:
                do_mirror = True
        if do_mirror:
            transform_matrix[0,0] = -transform_matrix[0,0]
            transform_matrix[0,1] = -transform_matrix[0,1]
            transform_matrix[0,2] = float(self.output_img_width)-transform_matrix[0,2];


        outputs = []
        for idx, _input in enumerate(inputs):
            if _input.ndim == 3:
                is_landmarks = False
            else:
                is_landmarks = True
            input_tf = affine2d(_input,
                                   transform_matrix,
                                   output_img_width=self.output_img_width,
                                   output_img_height=self.output_img_height,
                                   is_landmarks=is_landmarks)
            if is_landmarks and do_mirror and isinstance(self.corr_list, np.ndarray):
                # input_tf.shape: (1L, nL, 2L)
                # print("mirror!")
                input_tf = exchange_landmarks(input_tf, self.corr_list)
            outputs.append(input_tf)
        return outputs if idx >= 1 else outputs[0]
