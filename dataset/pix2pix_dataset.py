import os
import cv2
import imageio
import numpy as np
from PIL import Image

import util.util as util
from dataset.base_dataset import BaseDataset, get_params, get_transform, normalize

import torchvision.utils as vutils
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

norm = normalize()

class Pix2pixDataset(BaseDataset):
    """ adapted from SPADE repo. """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.mask = opt.mask

        label_paths, image_paths, instance_paths, mask_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        util.natural_sort(mask_paths) # MASK
        if not opt.no_instance:
            util.natural_sort(instance_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]
        #mask_paths = mask_paths[:] will get random mask from there.


        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths
        self.mask_paths = mask_paths # MASK

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params, normalize=False)
        image_tensor = transform_image(image)

        # reference: https://github.com/knazeri/edge-connect/blob/master/src/dataset.py#L116-L151
        mask, mask_ix = self.load_mask(image_tensor, index)
        mask_tensor = self.edge_connect_to_tensor(mask)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'mask': mask_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def load_mask(self, img_tensor, index):
        imgh, imgw = img_tensor.shape[1:]

        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # # random block
        # if mask_type == 1:
        #     return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # # half
        # if mask_type == 2:
        #     # randomly choose right or left
        #     return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_ix = np.random.randint(0, len(self.mask_paths))
            mask = imageio.imread(self.mask_paths[mask_ix])
            mask = self.edge_connect_resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask, mask_ix

    def edge_connect_resize(self, img, height, width, centerCrop=True):
        if len(img.shape) == 2: # gray image
            imgh, imgw= img.shape
        else:
            imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            if len(img.shape) == 2: # gray image
                img = img[j:j + side, i:i + side]
            else:
                img = img[j:j + side, i:i + side, ...]

        img = cv2.resize(img, (height, width))

        return img

    def edge_connect_to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
