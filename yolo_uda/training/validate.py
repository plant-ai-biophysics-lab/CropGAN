from typing import Optional, Callable, Union
import wandb

from torch import nn
from torch.utils.data import DataLoader
from evaluate import _evaluate


def validate(
    model: nn.Module,
    validation_dataloader: DataLoader,
    run: wandb.run,
    class_names: list = ["0"],
    metrics_suffix: Optional[str] = "",
    verbose: Optional[bool] = False,
    log_img_count: int = 10,  
):
    
    print("\n---- Evaluating Model ----")
    # Evaluate the model on the validation set
    metrics_output = _evaluate(
        model=model,
        dataloader=validation_dataloader,
        class_names=class_names,
        img_size=model.yolo_model.hyperparams['height'],
        verbose=verbose,
        step=0,
        num_imgs_to_log=log_img_count,
    )
    
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        run.log({
            f"test_precision_{metrics_suffix}": precision.mean(),
            f"test_recall_{metrics_suffix}": recall.mean(),
            f"test_f1_{metrics_suffix}": f1.mean(),
            f"test_mAP_{metrics_suffix}": AP.mean()
        })
