import sys
sys.path.append("/home/michael/ucdavis/adaptive_teacher")

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from adapteacher import add_ateacher_config
# hacky way to register
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from adapteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from cropgan_adapteacher.modeling.roi_heads.roi_heads import CropGanStandardROIHeadsPseudoLab

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ateacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg