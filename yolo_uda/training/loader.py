import random
import os

from torch.utils.data import DataLoader
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.utils import worker_seed_set

from datasets import UDAListDataset

K_VAL_MAP = {
    -1:24, # Use for test runs, 24 is the full test set
    1:1,
    4:1,
    8:1,
    12:2,
    16:3,
    24:6,
    32:8,
    40:10,
    98:28,
    58:15
}

def prepare_data(train_path, target_train_path, target_val_path, K=0, skip_preparation=False):
    if skip_preparation:
        print("Skipping file preparation")
        return

    # K_val is determined by k per CropGAN paper.   
    if K in K_VAL_MAP:
        K_val = K_VAL_MAP[K]
    else:
        # For Gemini if we use a different k value than in paper.
        K_val = max(1,int(0.25*K))

    # create list to store file paths
    paths = [target_train_path, target_val_path, train_path]
    train_paths = []
    target_train_paths = []
    target_val_paths = []

    # create a list to track whether a sample is target/source
    sample_loc_train = []
    sample_loc_target_train = []
    sample_loc_target_val = []

    # loop through the files in the directory
    for i in range(0,3):
        for filename in os.listdir(paths[i]):
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                file_path = os.path.join(paths[i],filename)
                if i == 0:
                    target_train_paths.append(file_path)
                    sample_loc_target_train.append(1)
                elif i == 1 and len(target_val_paths) < K_val:
                    target_val_paths.append(file_path)
                    sample_loc_target_val.append(1)
                else:
                    train_paths.append(file_path)
                    sample_loc_train.append(0)
    
    # add target examples if K > 0
    if K > 0:
        sample = random.sample(range(0, len(target_train_paths)), K)
        examples = [target_train_paths[i] for i in sample]
        train_paths += examples
        sample_loc_train += [1] * K

    # write to txt file if doesn't exist
    train_output = os.path.join(os.path.dirname(train_path), f'train_k_{K}.txt')
    target_train_output = os.path.join(os.path.dirname(target_train_path), 'target_train.txt')
    target_val_output = os.path.join(os.path.dirname(target_val_path), f'target_val_k_{K}.txt')
    
    for fname, sample_locs, paths in zip(
            [train_output, target_train_output, target_val_output],
            [sample_loc_train, sample_loc_target_train, sample_loc_target_val],
            [train_paths, target_train_paths, target_val_paths]):
        if not os.path.exists(fname):
            with open(fname, 'w') as file:
                for path, loc in zip(paths, sample_locs):
                    file.write(path + ' ' + str(loc) + '\n')
            print(f"File paths have been saved to {fname}")

        
def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = UDAListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    print(dataset.collate_fn)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader


def _create_validation_data_loader(img_path, batch_size, img_size, n_cpu):
    """
    Creates a DataLoader for validation.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = UDAListDataset(img_path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader

