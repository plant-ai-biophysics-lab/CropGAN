import os
from datetime import datetime

import torch
import tqdm
import wandb
import numpy as np
from torch import nn
from terminaltables import AsciiTable
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.utils.utils import to_cpu
from models import Upsample
from metrics import FeatureMapCosineSimilarity, FeatureMapEuclideanDistance
from evaluate import _evaluate
from metrics import TSNEVisualizer

binary_accuracy = BinaryAccuracy(threshold=0.5).to('cuda')
tsne = TSNEVisualizer(
    n_components=2,
    perplexity=30.0,
    init='pca'
)

def tsne_visualization(source_features: torch.Tensor, target_features: torch.Tensor, step: int):

    # preprocess source and target features
    source_global = tsne.preprocess(x = source_features)
    target_global = tsne.preprocess(x = target_features)

    # concatenate and create labels
    features = np.vstack([source_global, target_global])
    labels = np.array([0]*len(source_global) + [1]*len(target_global))

    # run tsne and plot
    features_reduced = tsne.run_tsne(x = features)
    tsne.plot_tsne(
        features = features_reduced, 
        labels = labels, 
        step = step
    )

def train(
    model: nn.Module,
    source_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    optimizer_global_classifier: torch.optim.Optimizer,
    optimizer_local_classifier: torch.optim.Optimizer,
    target_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    save_dir: str,
    run: wandb.run,
    visualize_tsne: bool,
    verbose: bool = False,
    epochs: int = 10,
    class_names: list = None,
    log_img_every_n_epochs: int = 50,
    log_img_count: int = 10,
):
    mini_batch_size = model.batch_size
    batches_done = 0

    best_map, map_ckpt_name = 0.0, ""
    best_precision, precision_ckpt_name = 0.0, ""
    best_recall, recall_ckpt_name = 0.0, ""
    best_f1, f1_ckpt_name = 0.0, ""
    for epoch in range(1, epochs+1):
        print("\n---- Training Model ----")
        wandb.log({'epoch': epoch}, step=batches_done)

        # set to training mode
        model.train() # set yolo model to training mode
        # Collect discriminator accuracy over training batches
        # Note: total is the sum of batch-level accuracy, not sample-level accuracy.
        # To get the average for the dataset, divide by the batch count.
        global_discriminator_acc = {"total": 0, "batch_count": 0, "batch_size": mini_batch_size*2}
        local_discriminator_acc = {"total": 0, "batch_count": 0, "batch_size": mini_batch_size*2}

        # Feature map similarity metrics
        # Cosine similarity metrics
        cosine_similarity_metrics_l15 = FeatureMapCosineSimilarity(layer="15")
        cosine_similarity_metrics_l22 = FeatureMapCosineSimilarity(layer="22")

        # Euclidean distance metrics
        euclidean_distance_metrics_l15 = FeatureMapEuclideanDistance(layer="15")
        euclidean_distance_metrics_l22 = FeatureMapEuclideanDistance(layer="22")

        # tracker
        updated_lr_this_epoch = False

        # for tsne visual
        if visualize_tsne:
            if epoch == 1 or epoch == epochs:
                all_source_features = []
                all_target_features = []

        for batch_i, contents in enumerate(
            tqdm.tqdm(zip(source_dataloader, target_dataloader), desc=f"Training Epoch {epoch}")
        ):
            # # Reset gradients
            optimizer.zero_grad()
            optimizer_global_classifier.zero_grad()
            optimizer_local_classifier.zero_grad()

            batches_done = len(target_dataloader) * (epoch-1) + batch_i
  
            loss, loss_components, batch_discriminator_acc, source_features, target_features = model(batch=contents)
            if visualize_tsne and source_features is not None and target_features is not None:
                if epoch == 1 or epoch == epochs:
                    all_source_features.append(source_features[1])
                    all_target_features.append(target_features[1])
            if loss is None:
                # catches incomplete training batches
                continue
            loss.backward()

            # run optimizer
            # if batches_done % model.yolo_model.hyperparams['subdivisions'] == 0:
            # adapt learning rate
            lr = model.yolo_model.hyperparams['learning_rate']
            if batches_done < model.yolo_model.hyperparams['burn_in']:
                lr *= (batches_done / model.yolo_model.hyperparams['burn_in'])
            else:
                # manually select which steps to decay the LR, and by what value
                if 'lr_steps' in model.yolo_model.hyperparams:
                    for threshold, value in model.yolo_model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value

                # decay every N steps or every N epochs
                elif 'lr_gamma' in model.yolo_model.hyperparams:
                    if 'lr_step' in model.yolo_model.hyperparams:
                        if batches_done % model.yolo_model.hyperparams['lr_step'] == 0:
                            print("Decaying learning rate ({}) by gamma {}".format(lr, model.yolo_model.hyperparams['lr_gamma'])
                                  + " to {}".format(lr * model.yolo_model.hyperparams['lr_gamma']))
                            lr = lr * model.yolo_model.hyperparams['lr_gamma']
                            model.yolo_model.hyperparams['learning_rate'] = lr
                    elif 'lr_epoch' in model.yolo_model.hyperparams:
                        if epoch % model.yolo_model.hyperparams['lr_epoch'] == 0:
                            if not updated_lr_this_epoch:
                                updated_lr_this_epoch = True
                                print("Decaying learning rate ({}) by gamma {}".format(lr, model.yolo_model.hyperparams['lr_gamma'])
                                      + " to {}".format(lr * model.yolo_model.hyperparams['lr_gamma']))
                                lr = lr * model.yolo_model.hyperparams['lr_gamma']
                                model.yolo_model.hyperparams['learning_rate'] = lr

            # log the learning rate
            wandb.log({"lr": lr}, step=batches_done)

            # set learning rate
            for g in optimizer.param_groups:
                g['lr'] = lr

            # Run optimizer
            optimizer.step()
            optimizer_global_classifier.step()
            optimizer_local_classifier.step()

            # Metrics
            # Track discriminator accuracy
            global_discriminator_acc["total"] += batch_discriminator_acc["global_discriminator_acc"]
            global_discriminator_acc["batch_count"] += 1
            local_discriminator_acc["total"] += batch_discriminator_acc["local_discriminator_acc"]
            local_discriminator_acc["batch_count"] += 1

            # Update cosine similarity metrics
            # *_features[0] and *_features[1] are the feature maps of different yolo layers.
            cosine_similarity_metrics_l15.update(source_features=source_features[0],target_features=target_features[0])
            cosine_similarity_metrics_l22.update(source_features=source_features[1],target_features=target_features[1])

            # Update euclidean distance metrics
            euclidean_distance_metrics_l15.update(source_features=source_features[0],target_features=target_features[0])
            euclidean_distance_metrics_l22.update(source_features=source_features[1],target_features=target_features[1])


            # log progress
            if verbose:
                print(AsciiTable(
                        [
                            ["Type", "Value"],
                            ["IoU loss", float(loss_components["iou_loss"])],
                            ["Object loss", float(loss_components["obj_loss"])],
                            ["Class loss", float(loss_components["cls_loss"])],
                            ["Loss", float(loss_components["yolo_loss"])],
                            ["Discriminator batch loss", loss_components["discriminator_loss"]],
                        ]).table)
            wandb.log(loss_components, step=batches_done)
            

        # perform tsne visualization
        if visualize_tsne:
            if epoch == 1 or epoch == epochs:
                try:
                    tsne_visualization(
                        source_features = all_source_features, 
                        target_features = all_target_features,
                        step = batches_done
                    )
                except Exception as e:
                    print(e)

        # Training epoch metrics
        # Discriminator accuracy
        wandb.log({"glob_dscm_acc": global_discriminator_acc["total"] / global_discriminator_acc["batch_count"]}, step=batches_done)
        wandb.log({"loc_dscm_acc": local_discriminator_acc["total"] / local_discriminator_acc["batch_count"]}, step=batches_done)

        # Average cosine similarity within source, within target, and across source-target
        # For both feature layers
        for metric in [cosine_similarity_metrics_l15, cosine_similarity_metrics_l22, euclidean_distance_metrics_l15, euclidean_distance_metrics_l22, model.mmd_metric]:
            wandb.log(metric.return_metrics(), step=batches_done)
            metric.reset()

        # evaluate
        print("\n---- Evaluating Model ----")
        # Evaluate the model on the validation set
        
        # Every 10th epoch, log 10 images
        log_img_every_n_epochs = 10
        num_imgs_to_log = 0
        if epoch % log_img_every_n_epochs == 0:
            num_imgs_to_log = log_img_count

        metrics_output = _evaluate(
            model,
            validation_dataloader,
            class_names,
            img_size=model.yolo_model.hyperparams['height'],
            verbose=verbose,
            step=batches_done,
            num_imgs_to_log=num_imgs_to_log,
        )

        if metrics_output is not None:
            precision, recall, AP, f1, ap_class = metrics_output
            wandb.log({
                "precision": precision.mean(),
                "recall": recall.mean(),
                "f1": f1.mean(),
                "mAP": AP.mean()
            }, step=batches_done)

            # Save the best checkpoint for each metric
            save_date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
            ckpt_name = "ckpt_best_{mt}.pth"
            if precision.mean() >= best_precision:
                best_precision = precision.mean()
                if precision_ckpt_name:
                    os.remove(os.path.join(save_dir, precision_ckpt_name))
                precision_ckpt_name = ckpt_name.format(
                    mt="precision", value=best_precision, date=save_date, epoch=epoch)
                torch.save(model.state_dict(),
                           os.path.join(save_dir, precision_ckpt_name))
            if recall.mean() >= best_recall:
                best_recall = recall.mean()
                if recall_ckpt_name:
                    os.remove(os.path.join(save_dir, recall_ckpt_name))
                recall_ckpt_name = ckpt_name.format(
                    mt="recall", value=best_recall, date=save_date, epoch=epoch)
                torch.save(model.state_dict(),
                           os.path.join(save_dir, recall_ckpt_name))
            if AP.mean() >= best_map:
                best_map = AP.mean()
                if map_ckpt_name:
                    os.remove(os.path.join(save_dir, map_ckpt_name))
                map_ckpt_name = ckpt_name.format(
                    mt="map", value=best_map, date=save_date, epoch=epoch)
                torch.save(model.state_dict(),
                           os.path.join(save_dir, map_ckpt_name))
            if f1.mean() >= best_f1:
                best_f1 = f1.mean()
                if f1_ckpt_name:
                    os.remove(os.path.join(save_dir, f1_ckpt_name))
                f1_ckpt_name = ckpt_name.format(
                    mt="f1", value=best_f1, date=save_date, epoch=epoch)
                torch.save(model.state_dict(),
                           os.path.join(save_dir, f1_ckpt_name))

    return model

