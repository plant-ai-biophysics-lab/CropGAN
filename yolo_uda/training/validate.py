from typing import Optional, Callable, Union
import torch
import wandb

from torch import nn
from torch.utils.data import DataLoader
from evaluate import _evaluate


def validate(
    model: nn.Module,
    device: torch.device,
    validation_dataloader: DataLoader,
    run: wandb.run,
    class_names: list = ["0"],
    iou_thresh: float = 0.5,
    conf_thresh: float = 0.5,
    nms_thresh: float = 0.5,
    metrics_suffix: Optional[str] = "",
    # The rest are here so that the train() and validate() interfaces are the same
    global_discriminator: Optional[nn.Module] = None,
    local_discriminator: Optional[nn.Module] = None,
    source_dataloader: Optional[DataLoader] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    optimizer_global_classifier: Optional[torch.optim.Optimizer] = None,
    optimizer_local_classifier: Optional[torch.optim.Optimizer] = None,
    mini_batch_size: Optional[int] = 1,
    target_dataloader: Optional[DataLoader] = None,    
    lambda_discriminator: Optional[float] = 0.5,
    lambda_mmd: float = 0.001,
    verbose: Optional[bool] = False,
    epochs: Optional[int] = 10,
    discriminator_loss_function: Union[Callable, nn.Module] = None,
    log_img_every_n_epochs: int = 1,
    log_img_count: int = 10,  
):
    
    print("\n---- Evaluating Model ----")
    # Evaluate the model on the validation set
    metrics_output = _evaluate(
        model,
        global_discriminator,
        local_discriminator,
        discriminator_loss_function,
        validation_dataloader,
        class_names,
        img_size=model.hyperparams['height'],
        iou_thres=iou_thresh,
        conf_thres=conf_thresh,
        nms_thres=nms_thresh,
        verbose=verbose,
        step=0,
        num_imgs_to_log=log_img_count,
        device=device,
        mini_batch_size=mini_batch_size
    )
    
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        run.log({
            f"test_precision_{metrics_suffix}": precision.mean(),
            f"test_recall_{metrics_suffix}": recall.mean(),
            f"test_f1_{metrics_suffix}": f1.mean(),
            f"test_mAP_{metrics_suffix}": AP.mean()
        })
