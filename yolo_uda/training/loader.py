import random
import os

from torch.utils.data import DataLoader
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
from pytorchyolo.utils.utils import worker_seed_set

def prepare_data(train_path, val_path, K=0):

    # create list to store file paths
    paths = [val_path, train_path]
    train_paths = []
    val_paths = []
    gan_paths = [os.path.join(gan_path, path) for path in os.listdir(gan_path)]

    # loop through the files in the directory
    for i in range(0,2):
        for filename in os.listdir(paths[i]):
            if filename.endswith('.jpg'):
                file_path = os.path.join(paths[i],filename)
                if i == 0:
                    val_paths.append(file_path)
                else:
                    train_paths.append(file_path)
    
    # add target examples if K > 0
    if K > 0:
        sample = random.sample(range(0, len(val_paths)), K)
        examples = [val_paths[i] for i in sample]
        train_paths += examples

    # write to txt file
    train_output = os.path.join(os.path.dirname(train_path),'train.txt')
    val_output = os.path.join(os.path.dirname(val_path),'val.txt')
    paths_output = [val_output, train_output]

    for i in range(0,2):
        with open(paths_output[i], 'w') as file:
            if i == 0:
                for path in val_paths:
                    file.write(path + '\n')
            else:
                for path in train_paths:
                    file.write(path + '\n')
        print(f"File paths have been saved to {paths_output[i]}")
        
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
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader

