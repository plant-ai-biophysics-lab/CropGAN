# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys
sys.path.append("/home/michael/ucdavis/adaptive_teacher")

import albumentations as aug
from albumentations import functional as F
import numpy as np
import torch
import torch.nn as nn
# from torch.nn import functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from typing import Dict, Tuple, List, Optional

from detectron2.structures import ImageList, Instances

from adapteacher.modeling.meta_arch.rcnn import DAobjTwoStagePseudoLabGeneralizedRCNN


@META_ARCH_REGISTRY.register()
class CropGANDAobjTwoStagePseudoLabGeneralizedRCNN(DAobjTwoStagePseudoLabGeneralizedRCNN):

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = self.preprocess_image(batched_inputs=batched_inputs, image_key="image")

        images_t = self.preprocess_image(batched_inputs=batched_inputs, image_key="image_unlabeled")
        
        return images, images_t

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]], image_key: str = "image"):
        """
        Normalize, pad and batch the input images.
        """
        images = [x[image_key].to(self.device) for x in batched_inputs]
        images = [x / 255. * 2.0 - 1 for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return CropGANDAobjTwoStagePseudoLabGeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results