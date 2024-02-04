from __future__ import division

import torch
import os
import numpy as np
import torch.nn.functional as F

from itertools import chain
from typing import List, Tuple
from torch import nn
from pytorch_metric_learning.utils import common_functions as pml_cf
from pytorchyolo.utils.loss import compute_loss
# from pytorchyolo.utils.parse_config import parse_model_config
from pytorchyolo.utils.utils import weights_init_normal

import wandb

import sys
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from src.models.yolo_model import Darknet

def load_model(model_path, weights_path=None):
    """Loads the yolo model from file.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :return: Returns model
    :rtype: Darknet
    """
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Select device for inference
    model = GRLDarknet(model_path).to(device)

    model.apply(weights_init_normal)

    # If pretrained weights are specified, start from checkpoint or weight file
    if weights_path:
        if weights_path.endswith(".pth"):
            # Load checkpoint weights
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            # Load darknet weights
            model.load_darknet_weights(weights_path)
    return model



#####################
### Discriminator ###
#####################
class GradientReversal(torch.nn.Module):
    """
    Implementation of the gradient reversal layer described in
    [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818),
    which 'leaves the input unchanged during forward propagation
    and reverses the gradient by multiplying it
    by a negative scalar during backpropagation.'
    """

    def __init__(self, alpha: float = 1.0):
        """
        Arguments:
            weight: The gradients  will be multiplied by ```-alpha```
                during the backward pass.
        """
        super().__init__()
        self.register_buffer("alpha", torch.tensor([alpha]))
        pml_cf.add_to_recordable_attributes(self, "alpha")

    def update_weight(self, new_alpha):
        self.weight[0] = new_alpha

    def forward(self, x):
        """"""
        return _GradientReversal.apply(x, pml_cf.to_device(self.alpha, x))

    def extra_repr(self, delimiter="\n"):
        """"""
        return delimiter.join([f"{a}=str{getattr(self, a)}" for a in ["alpha"]])
        # return c_f.extra_repr(self, ["weight"])


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        wandb.log({
            "grl_forward_input_mean": input.mean().item(),
        }, commit=False)
        ctx.alpha = alpha
        return input

    @staticmethod
    def backward(ctx, grad_output):
        wandb.log({
            "grl_backward_grad_mean": grad_output.mean().item(),
        }, commit=False)
        return -ctx.alpha * grad_output, None
    
class Discriminator(nn.Module):
    """
    A 3-layer MLP + Gradient Reversal Layer for domain classification.
    """

    def __init__(self, in_size=255*13*13, h=2048, out_size=1, alpha=1.0):
        """
        Arguments:
            in_size: size of the input
            h: hidden layer size
            out_size: size of the output
            alpha: grl constant
        """

        super().__init__()

        self.net = nn.Sequential(
            GradientReversal(alpha=alpha),
            nn.Linear(in_size, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(torch.flatten(x,1)).squeeze(-1)

#####################
# YOLO architecture #
#####################
class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode: str = "nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
    
class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors: List[Tuple[int, int]], num_classes: int, new_coords: bool):
        """
        Create a YOLO layer

        :param anchors: List of anchors
        :param num_classes: Number of classes
        :param new_coords: Whether to use the new coordinate format from YOLO V7
        """
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.new_coords = new_coords
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        # Just to align with CropGAN's YOLOLayer
        self.metrics = {}
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x: torch.Tensor, img_size: int) -> torch.Tensor:
        """
        Forward pass of the YOLO layer

        :param x: Input tensor
        :param img_size: Size of the input image
        """
        stride = img_size // x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        
        if not self.training:  # inference
            pred = torch.clone(x)
            if self.grid.shape[2:4] != pred.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(pred.device)

            if self.new_coords:
                pred[..., 0:2] = (pred[..., 0:2] + self.grid) * stride  # xy
                pred[..., 2:4] = pred[..., 2:4] ** 2 * (4 * self.anchor_grid) # wh
            else:
                pred[..., 0:2] = (pred[..., 0:2].sigmoid() + self.grid) * stride  # xy
                pred[..., 2:4] = torch.exp(pred[..., 2:4]) * self.anchor_grid # wh
                pred[..., 4:] = pred[..., 4:].sigmoid() # conf, cls
            pred = pred.view(bs, -1, self.no)
        else:
            pred = None
        # We now return the original x during inference because CropGAN requires it.
        return x, pred 

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20) -> torch.Tensor:
        """
        Create a grid of (x, y) coordinates

        :param nx: Number of x coordinates
        :param ny: Number of y coordinates
        """
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class GRLDarknet(Darknet):
    """YOLOv3 object detection model"""

    def __init__(self, config_path: str, img_size: int =416, use_tiny: bool = None):
        # Need this for extracting feature_maps in forward()
        if use_tiny is None:
            use_tiny =  'tiny' in config_path
        self.use_tiny = use_tiny
        super(GRLDarknet, self).__init__(config_path=config_path,img_size=img_size)


    def forward(self, x, targets = None):
        feature_maps = [] # save feature maps for discriminator
        img_size = x.size(2)
        layer_outputs, yolo_outputs = [], []
        # Use different feature map layers if yolov3 vs. yolov3-tiny
        feature_map_layers = [15,22] if self.use_tiny else [81,93,105]
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:, group_size * group_id : group_size * (group_id + 1)] # Slice groupings used by yolo v4
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # x is now always the training yolo outputs, pred is the inference output
                x, pred = module[0](x, img_size)
                if self.training or targets is not None:
                    yolo_outputs.append(x)
                else:
                    yolo_outputs.append(pred)
            layer_outputs.append(x)
            if i in feature_map_layers:
                feature_maps.append(x)
        if self.training:
            # Training
            return [yolo_outputs, feature_maps]
        elif targets is not None:
            # CropGAN, need to calculate the loss but not inference metrics 
            if len(targets.shape) == 0:
                targets = torch.empty([0,6])
            loss, loss_components = compute_loss(yolo_outputs, targets,self)
            return loss, yolo_outputs
        else:
            # Inference
            return torch.cat(yolo_outputs, 1)

    def create_modules(self,module_defs: List[dict]) -> Tuple[dict, nn.ModuleList]:
        """
        Constructs module list of layer blocks from module configuration in module_defs

        :param module_defs: List of dictionaries with module definitions
        :return: Hyperparameters and pytorch module list
        """
        hyperparams = module_defs.pop(0)
        hyperparams.update({
            # 'batch': int(hyperparams['batch_size']),
            # 'subdivisions': int(hyperparams['subdivisions']),
            'width': int(hyperparams['width']),
            'height': int(hyperparams['height']),
            'channels': int(hyperparams['channels']),
            'optimizer': hyperparams.get('optimizer'),
            'momentum': float(hyperparams['momentum']),
            'decay': float(hyperparams['decay']),
            'learning_rate': float(hyperparams['learning_rate']),
            'burn_in': int(hyperparams['burn_in']),
            'max_batches': int(hyperparams['max_batches']),
            'policy': hyperparams['policy'],
        })

        # manually select which steps to decay the LR at (and by what value)
        if "steps" in hyperparams and "scales" in hyperparams:
            hyperparams.update({
                'lr_steps': list(zip(map(int,   hyperparams["steps"].split(",")),
                                    map(float, hyperparams["scales"].split(","))))
            })

        # decay by the value `lr_gamma` every N steps or every N epochs
        elif "lr_gamma" in hyperparams and "lr_step" in hyperparams:
            hyperparams.update({
                'lr_step': int(hyperparams["lr_step"]),
                'lr_gamma': float(hyperparams["lr_gamma"])
            })
        elif "lr_gamma" in hyperparams and "lr_epoch" in hyperparams:
            hyperparams.update({
                'lr_epoch': int(hyperparams["lr_epoch"]),
                'lr_gamma': float(hyperparams["lr_gamma"])
            })

        assert hyperparams["height"] == hyperparams["width"], \
            "Height and width should be equal! Non square images are padded with zeros."
        output_filters = [hyperparams["channels"]]
        module_list = nn.ModuleList()
        for module_i, module_def in enumerate(module_defs):
            modules = nn.Sequential()

            if module_def["type"] == "convolutional":
                bn = int(module_def["batch_normalize"])
                filters = int(module_def["filters"])
                kernel_size = int(module_def["size"])
                pad = (kernel_size - 1) // 2
                modules.add_module(
                    f"conv_{module_i}",
                    nn.Conv2d(
                        in_channels=output_filters[-1],
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=int(module_def["stride"]),
                        padding=pad,
                        bias=not bn,
                    ),
                )
                if bn:
                    modules.add_module(f"batch_norm_{module_i}",
                                    nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5))
                if module_def["activation"] == "leaky":
                    modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
                elif module_def["activation"] == "mish":
                    modules.add_module(f"mish_{module_i}", nn.Mish())
                elif module_def["activation"] == "logistic":
                    modules.add_module(f"sigmoid_{module_i}", nn.Sigmoid())
                elif module_def["activation"] == "swish":
                    modules.add_module(f"swish_{module_i}", nn.SiLU())

            elif module_def["type"] == "maxpool":
                kernel_size = int(module_def["size"])
                stride = int(module_def["stride"])
                if kernel_size == 2 and stride == 1:
                    modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
                maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                    padding=int((kernel_size - 1) // 2))
                modules.add_module(f"maxpool_{module_i}", maxpool)

            elif module_def["type"] == "upsample":
                upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
                modules.add_module(f"upsample_{module_i}", upsample)

            elif module_def["type"] == "route":
                layers = [int(x) for x in module_def["layers"].split(",")]
                filters = sum([output_filters[1:][i] for i in layers]) // int(module_def.get("groups", 1))
                modules.add_module(f"route_{module_i}", nn.Sequential())

            elif module_def["type"] == "shortcut":
                filters = output_filters[1:][int(module_def["from"])]
                modules.add_module(f"shortcut_{module_i}", nn.Sequential())

            elif module_def["type"] == "yolo":
                anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
                # Extract anchors
                anchors = [int(x) for x in module_def["anchors"].split(",")]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in anchor_idxs]
                num_classes = int(module_def["classes"])
                new_coords = bool(module_def.get("new_coords", False))
                # Define detection layer
                yolo_layer = YOLOLayer(anchors, num_classes, new_coords)
                modules.add_module(f"yolo_{module_i}", yolo_layer)
            # Register module list and number of output filters
            module_list.append(modules)
            output_filters.append(filters)

        return hyperparams, module_list