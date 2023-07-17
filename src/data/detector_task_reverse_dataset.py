import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad

def get_valid_box(boxes):
    valid_size_bbox_idx = np.logical_and(boxes[:, 3] > 0.01, boxes[:, 4] > 0.01)
    valid_xrange_bbox_idx = np.logical_and(boxes[:, 1] - boxes[:, 3]/2 > 0, boxes[:, 1] + boxes[:, 3]/2 < 0.99)
    valid_yrange_bbox_idx = np.logical_and(boxes[:, 2] - boxes[:, 4]/2 > 0, boxes[:, 2] + boxes[:, 4]/2 < 0.99)
    valid_bbox_idx = np.logical_and(valid_xrange_bbox_idx, valid_yrange_bbox_idx)
    valid_bbox_idx = np.logical_and(valid_bbox_idx, valid_size_bbox_idx)

    boxes = boxes[valid_bbox_idx]
    boxes = boxes.astype(dtype=np.float32)
    return boxes

class DetectorTaskReverseDataset(BaseDataset):
    """
    This dataset also load oneshot file and label to provide constrain in G(A)

    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prep are two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, 'trainA')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, 'trainB')  # create a path '/path/to/data/trainB'
        self.dir_labeled_A = os.path.join(opt.dataroot, 'labeledB')  # create a path '/path/to/data/labeledB'


        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.labeled_B_paths = sorted(make_dataset(self.dir_labeled_A, opt.max_dataset_size))    # load images from '/path/to/data/labeledB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.labeled_B_size = len(self.labeled_B_paths)  # get the size of dataset B

        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        if 'aug' not in opt.preprocess:
            self.transform_A = get_transform(self.opt, grayscale=(self.input_nc == 1))
            self.transform_B = get_transform(self.opt, grayscale=(self.output_nc == 1))
            self.transform_labeled_B = get_transform(self.opt, grayscale=(self.output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
            index_labeled_B = index % self.labeled_B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
            if self.labeled_B_size > 0:
                index_labeled_B = random.randint(0, self.labeled_B_size - 1)
            else:
                index_labeled_B = 0

        B_path = self.B_paths[index_B]
        labeled_B_path = self.labeled_B_paths[index_labeled_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        labeled_B_img = Image.open(labeled_B_path).convert('RGB')

        if 'aug' in self.opt.preprocess:
            self.transform_A, self.transform_B  = get_transform(self.opt, img_size=A_img.size, grayscale=(self.input_nc == 1))
            
            # Preprocess for A
            A_label_path = A_path.replace("train", "label").replace(".png", ".txt").replace(".jpg", ".txt")
            boxes = np.loadtxt(A_label_path).reshape(-1, 5)
            boxes = get_valid_box(boxes)
            boxes = torch.from_numpy(boxes)
            # print("boxes.dtype: ", boxes.dtype)
            A_img = np.asarray(A_img)
            class_labels = ['grapes' for i in range(len(boxes))]

            try:
                A_transformed = self.transform_A(image=A_img, bboxes=boxes[:, 1:], class_labels=class_labels)
            except ValueError as e:
                print("e: ", e)
                print("boxes: ", A_label_path, boxes)

            A = A_transformed["image"]
            A_transformed_bboxes = A_transformed['bboxes']

            if len(A_transformed_bboxes) > 0:
                A_targets = np.zeros((len(A_transformed_bboxes), 6))
                A_targets[:, 2:] = np.asarray(A_transformed_bboxes)
            else:
                A_targets = np.asarray([])

            A_img_detector = A
            A_targets = A_targets.astype(dtype=np.float32)
            A_targets = torch.from_numpy(A_targets)
            # print("A_targets.dtype: ", A_targets.shape)

            # Preprocess for B
            B_img = np.asarray(B_img)
            B_transformed = self.transform_B(image=B_img)
            B = B_transformed["image"]



            # Preprocess for labeled B
            B_label_path = labeled_B_path.replace(".png", ".txt").replace(".jpg", ".txt")
            boxes = np.loadtxt(B_label_path).reshape(-1, 5)
            boxes = get_valid_box(boxes)
            boxes = torch.from_numpy(boxes)
            class_labels = ['grapes' for i in range(len(boxes))]

            # print("boxes.dtype: ", boxes.dtype)

            labeled_B_img = np.asarray(labeled_B_img)

            try:
                labeled_B_transformed = self.transform_A(image=labeled_B_img, bboxes=boxes[:, 1:], class_labels=class_labels)
            except ValueError as e:
                print("e: ", e)
                print("boxes: ", A_label_path, boxes)

            labeled_B = labeled_B_transformed["image"]
            labeled_B_bboxes = labeled_B_transformed['bboxes']

            if len(labeled_B_bboxes) > 0:
                labeled_B_targets = np.zeros((len(labeled_B_bboxes), 6))
                labeled_B_targets[:, 2:] = np.asarray(labeled_B_bboxes)
            else:
                labeled_B_targets = np.asarray([])
            labeled_B_targets = labeled_B_targets.astype(dtype=np.float32)
            labeled_B_targets = torch.from_numpy(labeled_B_targets)


        else:
            if self.opt.random_view >= 1: # This become a dynamic random resizing
                self.transform_A = get_transform(self.opt, img_size=A_img.size, grayscale=(self.input_nc == 1))
                self.transform_B = get_transform(self.opt, img_size=B_img.size, grayscale=(self.output_nc == 1))
                self.transform_labeled_B = get_transform(self.opt, img_size=B_img.size, grayscale=(self.output_nc == 1))

            # apply image transformation
            A = self.transform_A(A_img)
            B = self.transform_B(B_img)
            labeled_B = self.transform_labeled_B(labeled_B_img)

            # --------------------------------
            # detector Task
            # --------------------------------
            # Preproess for detector input images
            # Make a copy of the img
            A_img_detector = A.detach().clone()
            _, h, w = A_img_detector.shape
            h_factor, w_factor = (h, w)
            # Pad to square resolution
            A_img_detector, pad = pad_to_square(A_img_detector, 0)
            _, padded_h, padded_w = A_img_detector.shape

            # Label for A
            A_label_path = A_path.replace("train", "label").replace(".png", ".txt").replace(".jpg", ".txt")

            if os.path.exists(A_label_path):
                boxes = torch.from_numpy(np.loadtxt(A_label_path).reshape(-1, 5))
                # Extract coordinates for unpadded + unscaled image
                x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
                y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
                x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
                y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
                # Adjust for added padding
                x1 += pad[0]
                y1 += pad[2]
                x2 += pad[1]
                y2 += pad[3]
                # Returns (x, y, w, h)
                boxes[:, 1] = ((x1 + x2) / 2) / padded_w
                boxes[:, 2] = ((y1 + y2) / 2) / padded_h
                boxes[:, 3] *= w_factor / padded_w
                boxes[:, 4] *= h_factor / padded_h

                A_targets = torch.zeros((len(boxes), 6))
                A_targets[:, 1:] = boxes

            # Label for B
            B_label_path = labeled_B_path.replace(".png", ".txt").replace(".jpg", ".txt")
            if os.path.exists(B_label_path):
                boxes = torch.from_numpy(np.loadtxt(B_label_path).reshape(-1, 5))
                labeled_B_targets = torch.zeros((len(boxes), 6))
                labeled_B_targets[:, 1:] = boxes

        A_img_detector = resize(A_img_detector, self.opt.detector_img_size)  

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 
                'A_label': A_targets, 'A_task': A_img_detector, 
                'labeled_B': labeled_B, 'labeled_B_label': labeled_B_targets}

    def collate_fn(self, batch_batch):
        A_batch = torch.stack([data['A'] for data in batch_batch])
        B_batch = torch.stack([data['B'] for data in batch_batch])
        labeled_B_batch = torch.stack([data['labeled_B'] for data in batch_batch])
        A_task_batch = torch.stack([data['A_task'] for data in batch_batch])
        A_path_batch = [data['A_paths'] for data in batch_batch]
        B_path_batch = [data['B_paths'] for data in batch_batch]
        labeled_B_path_batch = [data['labeled_B'] for data in batch_batch]

        A_targets_batch =  [data['A_label'] for data in batch_batch if data['A_label'] is not None]
        # Add sample index to targets
        for i, boxes in enumerate(A_targets_batch):
            if len(boxes) > 0:
                boxes[:, 0] = i
        A_targets_batch = torch.cat(A_targets_batch, 0)
        # return paths, imgs, targets

        labeled_B_targets_batch =  [data['labeled_B_label'] for data in batch_batch if data['labeled_B_label'] is not None]
        # Add sample index to targets
        for i, boxes in enumerate(labeled_B_targets_batch):
            if len(boxes) > 0:
                boxes[:, 0] = i
        labeled_B_targets_batch = torch.cat(labeled_B_targets_batch, 0)
        return {'A': A_batch, 'B': B_batch, 'labeled_B': labeled_B_batch, 'A_paths': A_path_batch, 
                'B_paths': B_path_batch, 'A_label': A_targets_batch, 'A_task': A_task_batch,
                'labeled_B_paths': labeled_B_path_batch,  'labeled_B_label': labeled_B_targets_batch}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
