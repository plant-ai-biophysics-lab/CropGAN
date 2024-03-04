import argparse
import wandb
import os
import pathlib

import torch
import torch.optim as optim
from PIL import Image
from torchvision import transforms
# from pytorchyolo.test import _create_validation_data_loader

from loader import prepare_data, _create_data_loader, _create_validation_data_loader
from models import load_model, Discriminator, Upsample
from trainer import train
from validate import validate
from datetime import datetime

def create_save_dir(args):
    save_folder = f"k-{args.k}_alpha-{args.alpha}_lambda-{args.lambda_disc}_lmmd-{args.lambda_mmd}"
    save_dir = os.path.join(args.save, save_folder)
    return save_dir

def main(args, hyperparams, run):
    
    # initialize class names
    class_names = [str(i) for i in range(args.num_classes)]

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # prepare data
    prepare_data(args.train_path, args.target_train_path, args.target_val_path, args.k, args.skip_preparation)
    
    # load models
    model = load_model(args.config, args.pretrained_weights).to(device)
    wandb.config.update(model.hyperparams)
    discriminator = Discriminator(alpha=args.alpha).to(device)
    
    # create dataloaders
    # mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']
    mini_batch_size = hyperparams['batch_size']
    
    source_dataloader = _create_data_loader(
        os.path.dirname(args.train_path)+f"/train_k_{args.k}.txt",
        batch_size=hyperparams['batch_size'],
        img_size=hyperparams['img_size'],
        n_cpu=args.n_cpu,
        multiscale_training=False
    )
    target_dataloader = _create_data_loader(
        os.path.dirname(args.target_train_path)+"/target_train.txt",
        batch_size=hyperparams['batch_size'],
        img_size=hyperparams['img_size'],
        n_cpu=args.n_cpu,
        multiscale_training=False
    )
    
    validation_dataloader = _create_validation_data_loader(
        os.path.dirname(args.target_val_path)+f"/target_val_k_{args.k}.txt",
        batch_size=1,
        img_size=hyperparams['img_size'],
        n_cpu=args.n_cpu
    )
    
    # create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    params_classifier = [p for p in discriminator.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        params,
        lr=float(model.hyperparams['learning_rate']),
        weight_decay=float(model.hyperparams['decay'])
    )
    optimizer_classifier = optim.Adam(
        params_classifier,
        lr=float(hyperparams["learning_rate_disc"]),
        weight_decay=float(hyperparams["decay_disc"])
    )

    if args.eval_only:
        # validate
        model = validate(
            model = model,
            device = device,
            validation_dataloader = validation_dataloader,
            class_names = class_names,
            iou_thresh=hyperparams["iou_thresh"],
            conf_thresh=hyperparams["conf_thresh"],
            nms_thresh=hyperparams["nms_thresh"],
            run=run,
        )
        
    else:
        # train
        save_dir = create_save_dir(args)
        
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True) 
        
        model = train(
            model=model,
            discriminator=discriminator,
            source_dataloader=source_dataloader,
            device=device,
            optimizer=optimizer,
            optimizer_classifier=optimizer_classifier,
            mini_batch_size=mini_batch_size,
            target_dataloader=target_dataloader,
            validation_dataloader=validation_dataloader,
            lambda_discriminator=args.lambda_disc,
            lambda_mmd=args.lambda_mmd,
            verbose=args.verbose,
            epochs=args.epochs,
            save_dir=save_dir,
            class_names=class_names,
            iou_thresh=hyperparams["iou_thresh"],
            conf_thresh=hyperparams["conf_thresh"],
            nms_thresh=hyperparams["nms_thresh"],
            run=run,
        )
        # save model weights
        save_name = f"ckpt_last_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
        save_filepath = os.path.join(save_dir, save_name)
        torch.save(model.state_dict(), save_filepath)
        best_model = wandb.Artifact(args.name, type="model")
        best_model.add_file(save_filepath)
        # run.log_artifact(best_model)
        # run.link_artifact(best_model, "model-registry/yolo-uda")
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=0,
                    help="Number of target examples to add to training set")
    ap.add_argument("-a", "--alpha", type=float, default=0.0,
                    help="Constant for gradient reversal layer")
    ap.add_argument("-l", "--lambda-disc", type=float, default=0.5,
                    help="Weighting for discriminator loss, yolo weight is 1.0")
    ap.add_argument("--lambda-mmd", type=float, default=0.001,
                    help="Weighting for MMD loss, yolo weight is 1.0")
    ap.add_argument("--lr-disc", type=float, default=0.0001,
                    help="Learning rate for discriminator")
    ap.add_argument("--decay-disc", type=float, default=0.0001,
                    help="Weight decay for discriminator")
    ap.add_argument("-b", "--batch-size", type=int, default=2,
                    help="Number of samples per batch.")
    ap.add_argument("-t", "--train-path", required=True,
                    help="Path to file containing training images")
    ap.add_argument("-tt", "--target-train-path", required=True,
                    help="Path to file containing target training images")
    ap.add_argument("-tv", "--target-val-path", required=True,
                    help="Path to file containing target validation images")
    ap.add_argument("-c", "--config", required=True,
                    help="YOLOv3 configuration file")
    ap.add_argument("-p", "--pretrained_weights",
                    help="Path to pretrained weights", default=None)
    ap.add_argument("-e", "--epochs", type=int, default=300,
                    help="Number of training epochs")
    ap.add_argument("--n-cpu", type=int, default=6,
                    help="Number of cpu threads")
    ap.add_argument("--eval_interval", type=int, default=50,
                    help="Evaluate model every eval_interval epochs")
    ap.add_argument("--eval-only", action="store_true",
                    help="Only runs validation with the provided model weights.")
    ap.add_argument("--verbose", action="store_true",
                    help="Prints training progress and results")
    ap.add_argument("-n", "--name", type=str, default="DEFAULT-RUN-NAME-EMPTY",
                    help="Run name for wandb logging")
    ap.add_argument("-s", "--save", type=str, required=True,
                    help="Where to save model weights")
    ap.add_argument("--skip-preparation", action="store_true", default=False,
                    help="Whether to skip data preparation (use existing files)")
    ap.add_argument("--num-classes", type=int, default=1,
                    help="Number of classes in dataset")
    ap.add_argument("--ckpt-to-test", type=str, default="ckpt_best_map",
                    help="Which best checkpoint to use in test at end of training.")
    args = ap.parse_args()

    # hyperparams
    hyperparams = {
        "epochs": args.epochs,
        "iou_thresh": 0.5,
        "conf_thresh": 0.3,
        "nms_thresh": 0.5,
        "alpha": args.alpha,
        "lambda": args.lambda_disc,
        "lambda_mmd": args.lambda_mmd,
        "decay_disc": args.decay_disc,
        "k": args.k,
        "img_size": 416,
        "batch_size": args.batch_size,
        "learning_rate_disc": args.lr_disc,
    }

    # update the run name with the domain
    args.name = ("k-" + str(args.k) + "_a-" + str(args.alpha) + "_l-" + str(args.lambda_disc) + "_" + "lmmd-" + str(args.lambda_mmd)
                 + "_" + ("day" if "BordenDay" in args.train_path else "night" if "BordenNight" in args.train_path else "")
                 + "_" + args.name)
    if args.alpha == 0 and args.lambda_disc == 0:
        args.name = "BASELINE_" + args.name

    # initialize wandb
    run = wandb.init(project='yolo-uda', name=args.name)
    wandb.config.update(hyperparams)
    
    # start run
    main(args, hyperparams, run)

    # Test run
    if not args.eval_only:
        args.eval_only = True
        # Change to the new checkpoint
        save_dir = create_save_dir(args)
        args.pretrained_weights = os.path.join(save_dir,"ckpt_best_map.pth")
        args.k = -1
        # Use test set, not val set
        if args.target_val_path.split("/")[-2] == "valid":
            args.target_val_path = os.path.join(os.path.dirname(os.path.dirname(args.target_val_path)),"test/images")
        else:
            print(f"Running test on target_val_path: {args.target_val_path}")
        main(args, hyperparams, run)
