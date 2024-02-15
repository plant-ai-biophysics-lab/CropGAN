from .base_options import BaseOptions


class ImageGenOptions(BaseOptions):
    """
    This class includes training options.
    It also includes shared options defined in BaseOptions.
    """
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--image_path', type=str, help='Where to read synthetic images.')
        parser.add_argument('--out_path', type=str, help='Where to write modified images.')
        parser.add_argument('--phase', type=str, default='gen', help='train, val, test, etc')        
        self.isTrain = False        
        
        return parser
