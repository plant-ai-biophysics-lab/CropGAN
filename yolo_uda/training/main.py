import argparse
import wandb
import os
import torch
import torch.optim as optim

from PIL import Image
from torchvision import transforms
# from pytorchyolo.test import _create_validation_data_loader
from loader import prepare_data, _create_data_loader, _create_validation_data_loader
from models import load_model, Discriminator, Upsample
from trainer import train
from datetime import datetime


def main(args, hyperparams, run):
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # prepare data
    prepare_data(args.train_path, args.val_path, args.k)
    
    # load models
    model = load_model(args.config, args.pretrained_weights).to(device)
    wandb.config.update(model.hyperparams)
    discriminator = Discriminator(alpha=args.alpha).to(device)
    
    # create dataloaders
    # mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']
    mini_batch_size = hyperparams['batch_size']
    
    source_dataloader = _create_data_loader(
        os.path.dirname(args.train_path)+"/train.txt",
        batch_size=hyperparams['batch_size'],
        img_size=hyperparams['img_size'],
        n_cpu=args.n_cpu,
        multiscale_training=False
    )
    target_dataloader = _create_data_loader(
        os.path.dirname(args.val_path)+"/val.txt",
        batch_size=hyperparams['batch_size'],
        img_size=hyperparams['img_size'],
        n_cpu=args.n_cpu,
        multiscale_training=False
    )
    
    validation_dataloader = _create_validation_data_loader(
        os.path.dirname(args.val_path)+"/val.txt",
        batch_size=1,
        img_size=model.hyperparams['height'],
        n_cpu=args.n_cpu
    )
    
    # create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    params_classifier = [p for p in discriminator.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        params,
        lr=model.hyperparams['learning_rate'],
        weight_decay=model.hyperparams['decay']
    )
    optimizer_classifier = optim.Adam(
        params_classifier,
        lr=hyperparams["learning_rate_disc"],
        weight_decay=hyperparams["decay_disc"]
    )

    

    # train
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
        verbose=args.verbose,
        epochs=args.epochs,
        evaluate_interval=args.eval_interval,
        class_names=["0"],
        iou_thresh=hyperparams["iou_thresh"],
        conf_thresh=hyperparams["conf_thresh"],
        nms_thresh=hyperparams["nms_thresh"]
    )

    # save model weights
    save_dir = os.path.join(args.save, f"{datetime.today().strftime('%Y-%m-%d')}_{run.id}.pth")
    torch.save(model.state_dict(), save_dir)
    best_model = wandb.Artifact(args.name, type="model")
    best_model.add_file(save_dir)
    # run.log_artifact(best_model)
    # run.link_artifact(best_model, "model-registry/yolo-uda")
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=0,
                    help="Number of target examples to add to training set")
    ap.add_argument("-a", "--alpha", type=float, required=True,
                    help="Constant for gradient reversal layer")
    ap.add_argument("-l", "--lambda-disc", type=float, default=0.5,
                    help="Weighting for discriminator loss, yolo weight is 1.0")
    ap.add_argument("--lr-disc", type=float, default=0.0001,
                    help="Learning rate for discriminator")
    ap.add_argument("--decay-disc", type=float, default=0.0001,
                    help="Weight decay for discriminator")
    ap.add_argument("-b", "--batch-size", type=int, default=2,
                    help="Number of samples per batch.")
    ap.add_argument("-t", "--train-path", required=True,
                    help="Path to file containing training images")
    ap.add_argument("-v", "--val-path", required=True,
                    help="Path to file containing validation images")
    ap.add_argument("-c", "--config", required=True,
                    help="YOLOv3 configuration file")
    ap.add_argument("-p", "--pretrained_weights",
                    help="Path to pretrained weights", default=None)
    ap.add_argument("-e", "--epochs", type=int, default=300,
                    help="Number of training epochs")
    ap.add_argument("--n-cpu", type=int, default=6,
                    help="Number of cpu threads")
    ap.add_argument("--eval_interval", type=int, default=1,
                    help="Evaluate model every eval_interval epochs")
    ap.add_argument("--verbose", action="store_true",
                    help="Prints training progress and results")
    ap.add_argument("-n", "--name", type=str, default="michael-test",
                    help="Run name for wandb logging")
    ap.add_argument("-s", "--save", type=str, required=True,
                    help="Where to save model weights")
    args = ap.parse_args()
    
    # hyperparams
    hyperparams = {
        "epochs": args.epochs,
        "iou_thresh": 0.5,
        "conf_thresh": 0.3,
        "nms_thresh": 0.5,
        "alpha": args.alpha,
        "lambda": args.lambda_disc,
        "decay_disc": args.decay_disc,
        "k": args.k,
        "img_size": 416,
        "batch_size": args.batch_size,
        "learning_rate_disc": args.lr_disc,
    }

    # initialize wandb
    run = wandb.init(project='yolo-uda', entity='paibl', name=args.name)
    wandb.config.update(hyperparams)
    
    # start run
    main(args, hyperparams, run)