import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy
from pytorch_metric_learning.utils import common_functions as pml_cf

import wandb


binary_accuracy = BinaryAccuracy(threshold=0.5).to('cuda')


def discriminator_step(
        global_discriminator,
        local_discriminator,
        map_features,
        labels,
        global_discriminator_loss_function,
        local_discriminator_loss_function,
):
    """
    Discriminator step performed between the source and targer domain.
    Input arguments:
      map_features: Tensor = feature map obtained from the feature extractor
      labels: Tensor = ground truth
    Return:
      Tensor = cross entropy loss between the prediction and the ground truth.
    """
    global_outputs, global_context = global_discriminator(map_features['global_features'])
    local_outputs, local_context = local_discriminator(map_features['local_features'])

    # calculate accuracy
    global_discriminator_acc = binary_accuracy(global_outputs, labels['global_labels'])
    local_discriminator_acc = binary_accuracy(local_outputs, labels['local_labels'])
    discriminator_acc = {"global_discriminator_acc": global_discriminator_acc,
                         "local_discriminator_acc": local_discriminator_acc}

    # calculate loss
    global_discriminator_loss = global_discriminator_loss_function(global_outputs, labels['global_labels'].float())
    local_discriminator_loss = local_discriminator_loss_function(local_outputs, labels['local_labels'].float())

    return global_discriminator_loss, local_discriminator_loss, discriminator_acc, global_context, local_context


def compose_discriminator_batch_evaluation(source_features: torch.Tensor,
                                           downsample_2: nn.Module, downsample_4: nn.Module,
                                           labels_source: torch.Tensor,
                                           device: torch.device, shuffle: bool = True):
    # source_features[1] = downsample_2(source_features[1])
    # target_features[1] = downsample_2(target_features[1])

    # only used for yolov3.cfg, not yolov3-tiny.cfg
    if len(source_features) == 3:
        source_features[2] = downsample_4(source_features[2])

    # Create pixel-wise labels
    activation_dims = (source_features[1].shape[2], source_features[1].shape[3], 1)
    labels_source_pixelwise = labels_source.repeat(activation_dims).permute(2,0,1)

    # Combine source and target batches for discriminator
    features = {
        "global_features": source_features[0].to(device),
        "local_features": source_features[1].to(device)
        }
    labels = {
        "global_labels": labels_source.to(device),
        "local_labels": labels_source_pixelwise.to(device)
        }

    if shuffle:
        # Shuffle batch
        idx = torch.randperm(features['global_features'].shape[0])
        features_shuffled = {key:value[idx] for key,value in features.items()}
        labels_shuffled = {key:value[idx] for key,value in labels.items()}
        return features_shuffled, labels_shuffled

    return features, labels



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

    def __init__(self, in_size=255, out_size=1, alpha=1.0, context=False):
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
            nn.AvgPool2d(9),
            nn.Flatten()
        )

        self.out = nn.Sequential(
            nn.Linear(128, out_size),
            nn.Sigmoid()
        )

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

    def __init__(self, in_size=255, alpha=1.0, context = False):
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

        self.context = context

    def forward(self, x):
        x = self.net(x)
        feat = x
        if self.context:
            return self.out(x).squeeze(1), feat
        return self.out(x).squeeze(1), torch.zeros_like(feat)
        # return self.net(torch.flatten(x,1)).squeeze(-1)



