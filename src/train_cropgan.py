"""
trainning the sementic constrained cycle GAN (Crop GAN ).
Step 1: Initialize a YOLO model from real A.
    Step 2: Train a yolo constrained generator A (for several epoch). [A better generator!]
    Step 3: Generate fake B image dataset using G_A (label same as A).
    Step 4: Further train YOLO model using the data generate in step 3. [A better YOLO!]
    Return to step 2
"""

import os
import glob
import time
import wandb

from options.image_gen_options import ImageGenOptions
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.dataset_yolo import ListDataset
from util.util import save_image_tensor
from util.util import plot_analysis_double_task
from generate_cropgan_images import generate_images_from_source



if __name__ == '__main__':
    iou_thres=0.5
    conf_thres=0.5
    nms_thres = 0.5
    img_size = 416
    
    class_names = ['grapes']

    opt = ImageGenOptions().parse()   # get training options + ImageGen options to generate images
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print("opt: ", opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    
    # speed up life
    # (auto-detect checkpoint location based on the given alpha/lambda, and search
    #  within the respective folder for the latest `ckpt_last_*.pth` model file).
    if opt.yolo_b_weights == '':
        opt.yolo_b_weights = opt.yolo_a_weights
    if os.path.isdir(opt.yolo_a_weights):
        possible_weight_path = os.path.join(
            opt.yolo_a_weights, f"k-{opt.reverse_task_k}_alpha-{opt.grl_alpha}_lambda-{opt.grl_lambda}_lmmd-{opt.grl_lmmd}", "ckpt_best_map.pth")
        print(f"Loading ckpt_best_map.pth from {possible_weight_path}")
        weight_file = glob.glob(possible_weight_path)[0]

        # latest_weight, latest_time = "", datetime.datetime(2024, 1, 1)
        # for weight_file in weight_files:
        #     print(weight_file)
        #     weight_time = "_".join(os.path.splitext(os.path.basename(weight_file))[0].split("_")[-2:])
        #     weight_time = datetime.datetime.strptime(weight_time, '%Y-%m-%d_%H-%M-%S')
        #     if weight_time > latest_time:
        #         latest_weight, latest_time = weight_file, weight_time

        # if latest_weight == "":
        #     raise FileNotFoundError(f"No weights found at {possible_weight_path}")

        opt.yolo_a_weights = weight_file
        opt.yolo_b_weights = weight_file

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    # yolo_save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
    # intermediate_yolo_folder = os.path.join(opt.checkpoints_dir, opt.name) + "/intermediate/" # the folder to save intermediate yolo training data
    plot_save_dir = os.path.join(opt.checkpoints_dir, opt.name) + "/plots/" 
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)

    # Initialize logging
    run = wandb.init(project="cropgan-uda", name=opt.wandb_name, config=opt)

    # MHS: Don't need to resave the model since we aren't updating the weights.
    # model.save_yolo_networks("init")
    # save_intermediate_images(opt, intermediate_yolo_folder, save_real_A=True)

    # dual step cycle gan related variables
    # util_yolo.evaluate_yolo_net(model.netYoloB, opt.yolo_valid_path, iou_thres, conf_thres, nms_thres, img_size, class_names)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        wandb.log({
            'G_lr': model.optimizer_G.param_groups[0]['lr'],
            'D_lr': model.optimizer_D.param_groups[0]['lr'],
            'epoch': epoch,
        })

        for i, data in enumerate(dataset):  # inner loop within one epoch
            # print('Cycle GAN optimization start')
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            # print('Cycle GAN optimization end')

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                # print('Display start')
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                if opt.display_id > 0:
                    print("Visualizer display_current_results")
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                    print("Visualizer finish")

                save_path = plot_save_dir + "/epoch_%.3i_%.3i.jpg"%(epoch, total_iters)
                plot_analysis_double_task(model, data, conf_thres=0.1, reverse=True, title="epoch_%.3i_%.3i.jpg"%(epoch, total_iters), save_name=save_path)
                # print('Display end')

            losses = model.get_current_losses()
            for k, v in losses.items():
                wandb.log({k: v})

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                # print('print loss start')
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    print("Visualizer plot")
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                    print("Visualizer finish")

                # print('print loss end')

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                # print('saving the latest model  start')
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                # print('saving the latest model end')

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            # print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            # print('saving the model end')

    # Generate images - straight from generate
    print("Generating images from trained CropGAN model.")
    model.eval()

    generate_images_from_source(opt=opt, model=model, run=run)