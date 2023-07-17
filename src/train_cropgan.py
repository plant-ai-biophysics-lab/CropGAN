"""
trainning the sementic constrained cycle GAN (Crop GAN ).
Step 1: Initialize a detector model from real A.
    Step 2: Train a detector-constrained generator A (for several epoch). [A better generator!]
    Step 3: Generate fake B image dataset using G_A (label same as A).
    Step 4: Further train detector model using the data generate in step 3. [A better detector!]
    Return to step 2
"""
import time
import os
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import save_image_tensor
from util.util import plot_analysis_double_task
import  util.util_detector as util_detector


# def log_yolo(model):
#     # ----------------
#     #   Log progress
#     # ----------------
#     metrics = [
#         "grid_size",
#         "loss",
#         "x",
#         "y",
#         "w",
#         "h",
#         "conf",
#         "cls",
#         "cls_acc",
#         "recall50",
#         "recall75",
#         "precision",
#         "conf_obj",
#         "conf_noobj",
#     ]
#     log_str = "\n--------\n" 
#     # TODO update for yolo
#     metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.netYolo.yolo_layers))]]]

#     # Log metrics at each YOLO layer
#     for i, metric in enumerate(metrics):
#         formats = {m: "%.6f" for m in metrics}
#         formats["grid_size"] = "%2d"
#         formats["cls_acc"] = "%.2f%%"
#         row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.netYolo.yolo_layers]
#         metric_table += [[metric, *row_metrics]]

#         # Tensorboard logging
#         tensorboard_log = []
#         for j, yolo in enumerate(model.netYolo.yolo_layers):
#             for name, metric in yolo.metrics.items():
#                 if name != "grid_size":
#                     tensorboard_log += [(f"{name}_{j+1}", metric)]
#         tensorboard_log += [("loss", loss.item())]
#         # logger.list_of_scalars_summary(tensorboard_log, batches_done)

#     log_str += AsciiTable(metric_table).table
#     log_str += f"\nTotal loss {loss.item()}"

#     print(log_str)


# def save_intermediate_images(opt, intermediate_yolo_folder, save_real_A=False, save_one_shot=True):
#     # Create intermediate dataset
#     evaL_opt = copy.copy(opt)
#     evaL_opt.serial_batches = True
#     intermediate_dataset = create_dataset(evaL_opt)
#     image_folder = intermediate_yolo_folder + "/images/"
#     label_folder = intermediate_yolo_folder + "/labels/"
#     for folder in [intermediate_yolo_folder, image_folder, label_folder]:
#         if not os.path.exists(folder):
#             os.makedirs(folder)

#     for i, data in enumerate(tqdm.tqdm(intermediate_dataset)):
#         A_path = data['A_paths'][0]
#         A_label_path = A_path.replace("train", "label").replace(".png", ".txt").replace(".jpg", ".txt")
#         model.set_input(data)  # unpack data from data loader

#         if not save_real_A:
#             model.test()           # run inference
#             save_img_path = intermediate_yolo_folder + "/images/fake_B_" + os.path.basename(A_path)
#             save_label_path = intermediate_yolo_folder + "/labels/fake_B_" + os.path.basename(A_path).replace(".png", ".txt").replace(".jpg", ".txt")
#             save_image_tensor(model.fake_B, save_img_path)
#             shutil.copy(A_label_path, save_label_path)
#         else:
#             # copy real A data and labels
#             save_img_path = intermediate_yolo_folder + "/images/real_A_" + os.path.basename(A_path)
#             save_label_path = intermediate_yolo_folder + "/labels/real_A_" + os.path.basename(A_path).replace(".png", ".txt").replace(".jpg", ".txt")
#             save_image_tensor(model.real_A, save_img_path)
#             shutil.copy(A_label_path, save_label_path)

#         if i >= len(intermediate_dataset.dataset.A_paths) -1:
#             break

#     # include the one shot training sample if needed
#     if opt.yolo_one_shot_file is not 'None' and save_one_shot:
#         print("\n Save one shot yolo training data")
#         oneshot_img_path = opt.yolo_one_shot_file
#         oneshot_label_path = oneshot_img_path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
#         oneshot_img_save = intermediate_yolo_folder + "/images/oneshot_" + os.path.basename(oneshot_img_path)
#         oneshot_label_save = intermediate_yolo_folder + "/labels/oneshot_" + os.path.basename(oneshot_label_path)
#         shutil.copy(oneshot_img_path, oneshot_img_save)
#         shutil.copy(oneshot_label_path, oneshot_label_save)





if __name__ == '__main__':
    # TODO: Args to be added into option
    iou_thres=0.5
    conf_thres=0.5
    nms_thres = 0.5
    img_size = 416
    
    class_names = ['grapes']
    # End TODO

    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print("opt: ", opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    # detector_save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
    # intermediate_detector_folder = os.path.join(opt.checkpoints_dir, opt.name) + "/intermediate/" # the folder to save intermediate detector training data
    plot_save_dir = os.path.join(opt.checkpoints_dir, opt.name) + "/plots/" 
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)

    model.save_detector_networks("init")
    # save_intermediate_images(opt, intermediate_detector_folder, save_real_A=True)

    # dual step cycle gan related variables
    # TODO: Do we want this uncommented?
    # util_detector.evaluate_detector_net(model.netDetectorB, opt.detector_valid_path, iou_thres, conf_thres, nms_thres, img_size, class_names)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.

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

