from .base_options import BaseOptions
from .train_options import TrainOptions

class ImageGenOptions(TrainOptions):
    """
    This class includes training options.
    It also includes shared options defined in BaseOptions.
    """
    def initialize(self, parser):
        parser = TrainOptions.initialize(self, parser)
        parser.add_argument('--image_path', type=str, help='Where to read synthetic images.')
        parser.add_argument('--out_path', type=str, help='Where to write modified images.')
        parser.add_argument('--log', action='store_true', default=False, help='Whether to log the output')
        self.isTrain = True        
        
        return parser
