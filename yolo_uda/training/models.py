from __future__ import division

import sys
from itertools import chain
from typing import List, Tuple
import os

import numpy as np
from pytorch_metric_learning.utils import common_functions as pml_cf
from pytorchyolo.utils.loss import compute_loss
# from pytorchyolo.utils.parse_config import parse_model_config
from pytorchyolo.utils.utils import weights_init_normal, non_max_suppression
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy
import wandb

from metrics import FeatureMapCosineSimilarity, FeatureMapEuclideanDistance, MMDLoss

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from src.models.yolo_model import Darknet


def load_model(model_path, context=False):
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
    model = GRLDarknet(model_path, context=context).to(device)

    model.apply(weights_init_normal)

    return model

def load_yolo_weights(model, weights_path):
    """Loads the yolo model from file.

    :param model: Darknet model without weights loaded
    :type model: GRLDarknet
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :return: Returns model
    :rtype: Darknet
    """

    # If pretrained weights are specified, start from checkpoint or weight file
    if weights_path:
        if weights_path.endswith(".pth"):
            # Load checkpoint weights
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(weights_path, map_location=device),strict=False)
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


class GradientTracker(torch.nn.Module):
    """
    Tracks the gradient of the input during the forward pass.
    """

    def __init__(self):
        """
        Arguments:
            weight: The gradients  will be multiplied by ```-alpha```
                during the backward pass.
        """
        super().__init__()

    def forward(self, x):
        """"""
        return _GradientTracker.apply(x)


class _GradientTracker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        wandb.log({
            "context_forward_input_mean": input.mean().item(),
        }, commit=False)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        wandb.log({
            "context_backward_grad_mean": grad_output.mean().item(),
        }, commit=False)
        return grad_output, None


class GlobalDiscriminator(nn.Module):
    """
    A 3-layer MLP + Gradient Reversal Layer for domain classification.
    """

    def __init__(self, loss_func, in_size=255, out_size=1, alpha=1.0, context=False, use_tiny=True):
        """
        Arguments:
            in_size: size of the input
            out_size: size of the output
            alpha: grl constant
        """

        super().__init__()

        self.net = nn.Sequential(
            GradientReversal(alpha=alpha),
            nn.Conv2d(in_channels=in_size, out_channels=512, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        if use_tiny:
            self.net.append(nn.AvgPool2d(18)) # TODO: make flexible for cropped and non-cropped
        else:
            self.net.append(nn.AvgPool2d(36)) # TODO: Update this
        self.net.append(nn.Flatten())

        self.out = nn.Sequential(
            nn.Linear(128, out_size),
            nn.Sigmoid()
        )

        self.loss_func = loss_func
        self.context = context

    def forward(self, x):
        x = self.net(x)
        feat = x
        if self.context:
            return self.out(x).squeeze(1), feat
        return self.out(x).squeeze(1), torch.zeros_like(feat)


class LocalDiscriminator(nn.Module):
    """
    A 3-layer MLP + Gradient Reversal Layer for domain classification.
    """

    def __init__(self, in_size=256, alpha=1.0, loss_func=None, context=False):
        """
        Arguments:
            in_size: size of the input
            out_size: size of the output
            alpha: grl constant
        """

        super().__init__()

        self.net = nn.Sequential(
            GradientReversal(alpha=alpha),
            nn.Conv2d(in_channels=in_size, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
        if loss_func is None:
            loss_func = nn.MSELoss()
        self.loss_func = loss_func
        self.context = context

    def forward(self, x):
        x = self.net(x)
        feat = x
        if self.context:
            return self.out(x).squeeze(1), feat
        return self.out(x).squeeze(1), torch.zeros_like(feat)
        # return self.net(torch.flatten(x,1)).squeeze(-1)


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
                pred[..., 2:4] = pred[..., 2:4] ** 2 * (4 * self.anchor_grid)  # wh
            else:
                pred[..., 0:2] = (pred[..., 0:2].sigmoid() + self.grid) * stride  # xy
                pred[..., 2:4] = torch.exp(pred[..., 2:4]) * self.anchor_grid  # wh
                pred[..., 4:] = pred[..., 4:].sigmoid()  # conf, cls
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


class YOLOContextDownsample(nn.Module):
    """Downsamples the context + feature map output into the regular size"""

    def __init__(self, use_context = True):
        super(YOLOContextDownsample, self).__init__()

        self.context_downsample = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 255, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(255),
            nn.ReLU()
        )

        self.use_context = use_context

    def forward(self, x, global_context, local_context):
        if not self.use_context:
            return x

        global_context = global_context.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, local_context.shape[2], local_context.shape[3])
        context = global_context + local_context
        context = context.reshape(x.shape[0], -1, x.shape[2], x.shape[3])

        # log the context
        wandb.log({
            "global_context_mean": global_context.mean().item(),
            "local_context_mean": local_context.mean().item(),
        }, commit=False)

        context = self.context_downsample(context)
        return x + context


class GRLDarknet(Darknet):
    """YOLOv3 object detection model"""

    def __init__(self, config_path: str, img_size: int = 416, use_tiny: bool = None, context=False):
        # Need this for extracting feature_maps in forward()
        if use_tiny is None:
            use_tiny = 'tiny' in config_path
        self.use_tiny = use_tiny
        self.context = context

        super(GRLDarknet, self).__init__(config_path=config_path, img_size=img_size)

        self.context_fusion = YOLOContextDownsample(use_context=context)
        self.gradient_tracker = GradientTracker()

    def forward(self, x, targets = None):
        num_samples = x.shape[0]
        feature_maps = [] # save feature maps for discriminator
        img_size = x.size(2)
        layer_outputs, yolo_outputs = [], []
        # Use different feature map layers if yolov3 vs. yolov3-tiny
        feature_map_layers = [8,22] if self.use_tiny else [36, 105]
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
            if len(targets) < 1:
                loss = [0,0]
            else:
                loss, loss_components = compute_loss(yolo_outputs, targets,self)
            # Reshape the yolo outputs, as done in CropGAN
            yolo_outputs = torch.cat([yo.view(num_samples,-1,yo.shape[-1]) for yo in yolo_outputs],1)
            return loss[0], yolo_outputs
        else:
            # Inference
            return torch.cat(yolo_outputs, 1)

    def forward_features(self, x, targets=None, return_feature_maps=False):
        num_samples = x.shape[0]
        feature_maps = []  # save feature maps for discriminator
        img_size = x.size(2)
        layer_outputs, yolo_outputs = [], []
        # Use different feature map layers if yolov3 vs. yolov3-tiny
        feature_map_layers = [8, 22] if self.use_tiny else [36, 105]
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                combined_outputs = torch.cat(
                    [layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:,
                    group_size * group_id: group_size * (group_id + 1)]  # Slice groupings used by yolo v4
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # x is now always the training yolo outputs, pred is the inference output
                x, pred = module[0](x, img_size)
            layer_outputs.append(x)

            # get all of the feature maps
            if i in feature_map_layers:
                feature_maps.append(x)

        # return just the feature maps
        if self.training or return_feature_maps:
            # Training
            return feature_maps

        # specifically for inference mode
        elif targets is not None:
            # CropGAN, need to calculate the loss but not inference metrics
            if len(targets) < 1:
                loss = [0, 0]
            else:
                loss, loss_components = compute_loss(yolo_outputs, targets, self)
            # Reshape the yolo outputs, as done in CropGAN
            yolo_outputs = torch.cat([yo.view(num_samples, -1, yo.shape[-1]) for yo in yolo_outputs], 1)
            return loss[0], yolo_outputs
        else:
            # Inference
            return torch.cat(yolo_outputs, 1)

    def forward_with_context(self, x, global_context, local_context, targets=None):
        num_samples = x.shape[0]
        feature_maps = []
        img_size = x.size(2)
        layer_outputs, yolo_outputs = [], []
        # Use different feature map layers if yolov3 vs. yolov3-tiny
        feature_map_layers = [8, 22] if self.use_tiny else [36, 105]
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                combined_outputs = torch.cat(
                    [layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:,
                    group_size * group_id: group_size * (group_id + 1)]
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

            # if this is the last feature map, concatenate with the global & local context
            if i == feature_map_layers[-1]:
                x = self.context_fusion(x, global_context, local_context)
                x = self.gradient_tracker(x)

                wandb.log({
                    "post_context_mean": x.mean().item(),
                }, commit=False)

        if self.training:
            return yolo_outputs

        # specifically for inference mode
        elif targets is not None:
            # CropGAN, need to calculate the loss but not inference metrics
            if len(targets) < 1:
                loss = [0, 0]
            else:
                loss, loss_components = compute_loss(yolo_outputs, targets, self)
            # Reshape the yolo outputs, as done in CropGAN
            yolo_outputs = torch.cat([yo.view(num_samples, -1, yo.shape[-1]) for yo in yolo_outputs], 1)
            return loss[0], yolo_outputs
        else:
            # Inference
            return torch.cat(yolo_outputs, 1)

    @staticmethod
    def create_modules(module_defs: List[dict]) -> Tuple[dict, nn.ModuleList]:
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
                'lr_steps': list(zip(map(int, hyperparams["steps"].split(",")),
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
    


#####################
###   Full Model  ###
#####################

class YoloDA(torch.nn.Module):
    def __init__(self, 
                 yolo_model: GRLDarknet, 
                 global_discriminator: GlobalDiscriminator, 
                 local_discriminator: LocalDiscriminator,
                 lambda_discriminator: float,
                 device: str,
                 iou_thresh: float = 0.5,
                 conf_thresh: float = 0.5,
                 nms_thresh: float = 0.5,
                 lambda_mmd: float = 0,
                 batch_size: int = 4,
                ):
        super().__init__()

        self.yolo_model = yolo_model
        self.global_discriminator = global_discriminator
        self.local_discriminator = local_discriminator
        
        self.batch_size=batch_size
        self.device=device
        self.binary_accuracy = BinaryAccuracy(threshold=0.5).to('cuda')
        self.mmd_metric = MMDLoss()
        self.lambda_discriminator = lambda_discriminator
        self.lambda_mmd = lambda_mmd
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def forward(self, batch):
        if self.training:
            return self.forward_train(batch)
        else:
            return self.forward_eval(batch)

    def forward_eval(self, batch):
        # batch is a dict with keys: imgs and domain_labels
        
        source_features = self.yolo_model.forward_features(batch["imgs"], return_feature_maps=True)

        features, disc_labels = self.compose_discriminator_batch(
                source_features=source_features,
                labels_source=batch["domain_labels"],
                shuffle=True, #TODO: Should this be False?
            )

        # discriminator_step handles both global and local
        (global_discriminator_loss, local_discriminator_loss, batch_discriminator_acc,
            global_context, local_context) = self.discriminator_step(
            map_features=features,
            labels=disc_labels,
        )

        # duplicate along the first dimension for the global and local context
        global_context = global_context.repeat(2, 1)
        local_context = local_context.repeat(2, 1, 1, 1)

        # get the source outputs with the context
        outputs = self.yolo_model.forward_with_context(batch["imgs"], global_context, local_context)
        outputs = non_max_suppression(outputs, conf_thres=self.conf_thresh, iou_thres=self.nms_thresh)

        # # yolo loss if targets provided in batch (for CropGAN use)
        if "targets" in batch:
            yolo_loss, loss_components = compute_loss(outputs, batch["targets"], self.yolo_model)
            return yolo_loss, outputs

        return outputs

    def forward_train(self, batch):
        (data_source, data_target) = batch
        # get imgs from data
        _, imgs_s, targets, labels_source = data_source
        _, imgs_t, _, labels_target = data_target
        if len(imgs_s) < self.batch_size or len(imgs_t) < self.batch_size:
            return None, None, None, None, None
        
        source_imgs = imgs_s.to(self.device)
        target_imgs = imgs_t.to(self.device)
        targets = targets.to(self.device)
        
        # with context, we need to get the global/local features from the yolo model,
        # pass them through the discriminator to get the discriminator outputs and context
        # vectors, and then pass the context vectors back into the yolo model to get the
        # final yolo output -> this requires multiple steps

        # run source pass
        source_features = self.yolo_model.forward_features(source_imgs)
        # Run target pass to encode features for classifier
        target_features = self.yolo_model.forward_features(target_imgs)

        features, labels = self.compose_discriminator_batch(
            source_features=source_features,
            target_features=target_features,
            labels_source=labels_source,
            labels_target=labels_target,
        )

        # discriminator_step handles both global and local
        (global_discriminator_loss, local_discriminator_loss, batch_discriminator_acc,
            global_context, local_context) = self.discriminator_step(
            map_features=features,
            labels=labels,
        )

        # get the source outputs with the context
        source_outputs = self.yolo_model.forward_with_context(source_imgs, global_context, local_context)

        # yolo loss
        yolo_loss, loss_components = compute_loss(source_outputs, targets, self.yolo_model)

        # Calculate average MMD loss per batch
        mmd_loss = self.mmd_metric(source_features[1], target_features[1])

        # run backward propagation
        discriminator_loss = 0.05 * global_discriminator_loss + 0.95 * local_discriminator_loss
        loss = yolo_loss + self.lambda_discriminator * discriminator_loss + self.lambda_mmd * mmd_loss
        
        # Collect loss_components
        loss_dict = {k:float(v) for k,v in zip(["iou_loss","obj_loss","cls_loss","yolo_loss"],loss_components)}
        loss_dict["discriminator_loss"] = float(discriminator_loss)
        loss_dict["global_discriminator_loss"] = float(global_discriminator_loss)
        loss_dict["local_discriminator_loss"] = float(local_discriminator_loss)
        
        self.yolo_model.seen += imgs_s.size(0)

        return loss, loss_dict, batch_discriminator_acc, source_features, target_features

    def discriminator_step(
            self,
            map_features,
            labels,
        ):

        """
        Discriminator step performed between the source and targer domain.
        Input arguments:
        map_features: Tensor = feature map obtained from the feature extractor
        labels: Tensor = ground truth
        Return:
        Tensor = cross entropy loss between the prediction and the ground truth.
        """
        global_outputs, global_context = self.global_discriminator(map_features['global_features'])
        local_outputs, local_context = self.local_discriminator(map_features['local_features'])

        # calculate accuracy
        global_discriminator_acc = self.binary_accuracy(global_outputs, labels['global_labels'])
        local_discriminator_acc = self.binary_accuracy(local_outputs, labels['local_labels'])
        discriminator_acc = {"global_discriminator_acc": global_discriminator_acc, "local_discriminator_acc":local_discriminator_acc}

        # calculate loss
        global_discriminator_loss = self.global_discriminator.loss_func(global_outputs, labels['global_labels'].float())
        local_discriminator_loss = self.local_discriminator.loss_func(local_outputs, labels['local_labels'].float())

        return global_discriminator_loss, local_discriminator_loss, discriminator_acc, global_context, local_context

    def compose_discriminator_batch(
            self, 
            source_features: torch.Tensor, 
            labels_source: torch.Tensor, 
            target_features: torch.Tensor = None,
            labels_target: torch.Tensor = None,
            shuffle: bool = True
        ):
        
        # Create pixel-wise labels
        activation_dims = (source_features[0].shape[2], source_features[0].shape[3], 1)
        labels_source_pixelwise = labels_source.repeat(activation_dims).permute(2,0,1)
        
        if self.training:
            labels_target_pixelwise = labels_target.repeat(activation_dims).permute(2,0,1)

            # Combine source and target batches for discriminator
            features = {
                "global_features":torch.cat([source_features[1], target_features[1]],axis=0).to(self.device),
                "local_features":torch.cat([source_features[0], target_features[0]],axis=0).to(self.device)
                }
            labels = {
                "global_labels": torch.cat([labels_source, labels_target],axis=0).to(self.device),
                "local_labels": torch.cat([labels_source_pixelwise, labels_target_pixelwise],axis=0).to(self.device)
                }
        else:
            features = {
                "global_features": source_features[1].to(self.device),
                "local_features": source_features[0].to(self.device)
                }
            labels = {
                "global_labels": labels_source.to(self.device),
                "local_labels": labels_source_pixelwise.to(self.device)
                }

        if shuffle:
            # Shuffle batch
            idx = torch.randperm(features['global_features'].shape[0])
            features_shuffled = {key:value[idx] for key,value in features.items()}
            labels_shuffled = {key:value[idx] for key,value in labels.items()}
            return features_shuffled, labels_shuffled
        return features, labels

    @classmethod
    def create_from_config(
        cls, 
        config, 
        context: bool, 
        alpha: float, 
        use_tiny: bool, 
        device: str, 
        global_disc_loss_func, 
        lambda_discriminator: float,
        iou_thresh: float = 0.5,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.5,
        lambda_mmd: float = 0,         
        batch_size = 4, 
    ):
        
        yolo_model = load_model(config, context=context).to(device)
        global_discriminator = GlobalDiscriminator(alpha=alpha, context=context, loss_func=global_disc_loss_func, use_tiny=use_tiny).to(device)
        local_discriminator = LocalDiscriminator(alpha=alpha, context=context).to(device)

        # if pretrained_weights is not None:
        #     yolo_model = load_yolo_weights(yolo_model, pretrained_weights[0])
        #     global_discriminator.load_state_dict(torch.load(pretrained_weights[1]))
        #     local_discriminator.load_state_dict(torch.load(pretrained_weights[2]))

        return YoloDA(
            yolo_model=yolo_model, 
            global_discriminator=global_discriminator, 
            local_discriminator=local_discriminator,
            lambda_discriminator=lambda_discriminator,
            lambda_mmd=lambda_mmd,
            iou_thresh = iou_thresh,
            conf_thresh = conf_thresh,
            nms_thresh = nms_thresh,
            batch_size=batch_size,
            device=device,
        )
