import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from models.yolo_model import Darknet
import os

class DoubleTaskCycleGanModel(BaseModel):
    """
    This class implements the CycleGAN model with semantic constraint 
    (The translated domain should have same semantic as the original domain), 
    for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Detector B loss: detector task loss on fake B image (G_B(A))
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_detector_b', type=float, default=0.0, help='weight for detector loss on fake B (G_B(A))')
            parser.add_argument('--lambda_detector_a', type=float, default=0.0, help='weight for detector loss on fake A (G_A(B))')
            parser.add_argument('--refine_detector_b_step', type=int, default=0, help='number of steps to refine detector b on one shot image')



        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'detector_b', 'detector_a']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'fake_labeled_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'labeled_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'DetectorA', 'DetectorB']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        # define task detector network
        print("\nInitializing detector network ... ")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Detector Device: ", device)
        #TODO: Can't use Darknet for all detectors
        self.netDetectorA = Darknet(opt.task_model_def, img_size=opt.detector_img_size).to(device)
        self.netDetectorB = Darknet(opt.task_model_def, img_size=opt.detector_img_size).to(device)

        # load detector A weights
        if opt.detector_a_weights != '':
            if opt.detector_a_weights.endswith(".weights"):
                # Load darknet weights
                self.netDetectorA.load_darknet_weights(opt.detector_a_weights)
            else:
                # Load checkpoint weights
                self.netDetectorA.load_state_dict(torch.load(opt.detector_a_weights))
            print("Load detector a weights: ", opt.detector_a_weights)
        else:
            print("No detector a weights loaded ")

        # load detector B weights
        if opt.detector_b_weights != '':
            if opt.detector_b_weights.endswith(".weights"):
                # Load darknet weights
                self.netDetectorB.load_darknet_weights(opt.detector_b_weights)
            else:
                # Load checkpoint weights
                self.netDetectorB.load_state_dict(torch.load(opt.detector_b_weights))
            print("Load detector b weights: ", opt.detector_b_weights)
        else:
            print("No detector b weights loaded ")
        
        # Set in evaluation mode
        self.netDetectorA.eval()  
        self.netDetectorB.eval()  

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_labeled_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_detector_a = torch.optim.Adam(self.netDetectorA.parameters(), lr=opt.lr)
            self.optimizer_detector_b = torch.optim.Adam(self.netDetectorB.parameters(), lr=opt.lr)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_detector_a)
            self.optimizers.append(self.optimizer_detector_b)


            # double task
            self.refine_step = opt.refine_detector_b_step

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.labeled_B = input['labeled_B'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.A_task = input['A_task'].to(self.device)
        self.A_label = input['A_label'].to(self.device)
        self.labeled_B_label = input['labeled_B_label'].to(self.device)

    def set_detector_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['B_paths']
        self.B_label = input['B_label'].to(self.device)

    def get_detector_output(self):
        return self.bbox_outputs

    def save_detector_networks(self, prefix):
        """Save the detector networks to the disk.

        Parameters:
            prefix (string); used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        save_filename = '%s_net_detector_a.pth' % prefix
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.netDetectorA.state_dict(), save_path)
        print("Detector A Model saved at: ", save_path)

        save_filename = '%s_net_detector_b.pth' % prefix
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.netDetectorB.state_dict(), save_path)
        print("Detector B Model saved at: ", save_path)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        self.fake_labeled_A = self.netG_B(self.labeled_B)  # G_B(B)
        self.rec_labeled_B = self.netG_A(self.fake_labeled_A)   # G_A(G_B(B))

    def forward_fake_B(self, no_grad=False):
        """Generate fake B"""
        if no_grad:
            with torch.no_grad():
                self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        else:
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)

    def forward_fake_A(self, no_grad=False):
        """Generate fake A"""
        if no_grad:
            with torch.no_grad():
                self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        else:
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
    
    def load_network_detector_b(self, epoch):
        load_filename = '%s_net_detector_b.pth' % epoch
        load_path = os.path.join(self.save_dir, load_filename)
        self.netDetectorB.load_state_dict(torch.load(load_path))
        print("load detector weights: ", load_path)
        self.netDetectorB.eval()  # Set in evaluation mode


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        lambda_detector_a = self.opt.lambda_detector_a

        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

        if lambda_detector_a > 0:
            fake_labeled_A = self.fake_labeled_A_pool.query(self.fake_labeled_A)
            loss_D_B2 = self.backward_D_basic(self.netD_B, self.real_A, fake_labeled_A)
            self.loss_D_B += loss_D_B2
    
    def backward_Detector_B(self):
        """Calculate Detector B loss for Fake B and label A"""
        lambda_detector_b = self.opt.lambda_detector_b

        if lambda_detector_b > 0:
            loss_detector_b, self.bbox_outputs = self.netDetectorA(self.fake_B*0.5+0.5, self.A_label) # de-normalize the image before feed into the detector net
            self.loss_detector_b = lambda_detector_b * loss_detector_b
        else:
            self.loss_detector_b = 0
        return self.loss_detector_b 

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_detector_a = self.opt.lambda_detector_a
        lambda_detector_b = self.opt.lambda_detector_b
        
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Detector task loss
        if lambda_detector_b > 0:
            loss_detector_b, self.bbox_outputs = self.netDetectorB(self.fake_B*0.5+0.5, self.A_label) # de-normalize the image before feed into the detector net
            self.loss_detector_b = lambda_detector_b * loss_detector_b
        else:
            self.loss_detector_b = 0

        if lambda_detector_a > 0:
            loss_detector_a, self.bbox_outputs_a = self.netDetectorA(self.fake_labeled_A*0.5+0.5, self.labeled_B_label) # de-normalize the image before feed into the detector net
            self.loss_G_B2 = self.criterionGAN(self.netD_B(self.fake_labeled_A), True)
            self.loss_detector_a = lambda_detector_a * loss_detector_a
        else:
            self.loss_detector_a = 0
            self.loss_G_B2 = 0

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + \
                      self.loss_cycle_B + self.loss_idt_A + \
                      self.loss_idt_B + self.loss_detector_b + self.loss_detector_a + self.loss_G_B2
        self.loss_G.backward()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netDetectorA], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

        # # Detector_B
        # self.optimizer_detector_b.zero_grad()  # update D_A and D_B's weights
        # self.backward_Detector_B()   # calculate gradients for Detector B
        # self.optimizer_detector_b.step()  # update D_A and D_B's weights



    def optimize_detector_parameters(self):
        """Generate fake images, calculaye losses, gradients, and update detector weights; called in detector training step"""
        # TODO Waiting implement
        pass

    
    def train_refined_detector_b(self):
        # TODO Waiting implement
        pass
        