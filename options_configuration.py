from vutils.libraries import *
class options_configuration:
  def __init__(self):
    self.opt = None

  def __opt__(self):
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Define the inpainting technique and the path of data
    parser.add_argument('--network', default='CA_Original', help='Type of Inpaiting Technique: CE;GLCG;...')
    parser.add_argument('--dataroot', default='dataset/data', help='Path to dataset')
    parser.add_argument('--dataroot_labels', default='dataset/label', help='Path to dataset')
    parser.add_argument('--datasetType', default='nii', help='Volume Type Extention')
    parser.add_argument('--specificity', default='Square_20', help='Specifications of saved model')

    # Image considerations
    parser.add_argument('--imageSize', type=int, default=256, help='The height/width of the input image')
    parser.add_argument('--nc', type=int, default=1, help='Number of channels')
    parser.add_argument('--imageRot', type=int, default=-90, help='Necessary rotation of the input image')
    parser.add_argument('--imageHouns', default=[1600,-600], help='Houns2intensity Window')
    parser.add_argument('--normalization', default=[0,1], help='Normalization Range')

    # Mask considerations
    parser.add_argument('--maskType', default='Square', help='Mask Type')
    parser.add_argument('--mask', default='1', help='Mask value Initialization: 1 - White; 0 - Gray; -1 - Black; mpv - Mean; random - Random')
    parser.add_argument('--mask_percentage', default=0.2, help='Missing regions area inside ROI')

    # Train Epoch
    parser.add_argument('--batchSize', type=int, default=8, help='Input batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to model train')
    #parser.add_argument('--generator_epochs', type=int, default=10, help='Number of pre-train epochs to generator train')
    #parser.add_argument('--discriminator_epochs', type=int, default=10, help='Number of pre-train epochs to generator train')
    parser.add_argument('--edge_epochs', type=int, default=50, help='Number of pre-train epochs to edge generator train')

    # Train Parameters
    #parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    #parser.add_argument('--beta1', type=float, default=0, help='Beta1 for Adam')
    #parser.add_argument('--beta2', type=float, default=0.9, help='Beta2 for Adam')
    #parser.add_argument('--D2G_lr', type=float, default=0.1, help='D2G Learning Rate for Adam')
    #parser.add_argument('--wtl2', type=float, default=0.98, help='Weight for L2 loss')
    #parser.add_argument('--wtlD', type=float, default=0.02, help='Weight for Discriminator loss')
    #parser.add_argument('--wfm', type=float, default=10, help='Weight for Feture Matching')
    #parser.add_argument('--wstyle', type=float, default=150, help='Weight for Style loss')
    #parser.add_argument('--wperc', type=float, default=0.05, help='Weight for Perceptual loss')
    #parser.add_argument('--wgadv', type=float, default=0.1, help='Weight for GAN adversarial loss')
    #parser.add_argument('--wtl1', type=float, default=50, help='Weight for L1 loss')
    #parser.add_argument('--wtv', type=float, default=0.1, help='Weight for Total Variation loss')

    # GPU options
    parser.add_argument('--gpu', type=int, default=0, help='GPU in use')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')

    # Seed to guarantee same randomizer
    parser.add_argument('--manualSeed', type=int, default=0, help='Manual seed')

    # Parse the command-line arguments
    self.opt = parser.parse_args(args=[])
    return self.opt

  # Informations about execution
  def __description__(self):
    # Print MODEL OVERVIEW
    print("\nMODEL OVERVIEW: ")

    print("Mode: %d (1 - Train; 2 - Test)" % self.opt.mode) if hasattr(self.opt, 'mode') else None
    print("Technique: " + self.opt.network) if hasattr(self.opt, 'network') else None
    print("Specificity: " + self.opt.specificity) if hasattr(self.opt, 'specificity') else None
    print("Image Size: %d" % self.opt.imageSize) if hasattr(self.opt, 'imageSize') else None
    print("Image Rotation: %d" % self.opt.imageRot) if hasattr(self.opt, 'imageRot') else None
    print("Houns2intensity Window: "+str(self.opt.imageHouns)) if hasattr(self.opt, 'imageHouns') else None
    print("Pixels Normalization: " +str(self.opt.normalization)) if hasattr(self.opt, 'normalization') else None
    print("Train Epochs: %d" % self.opt.epochs) if hasattr(self.opt, 'epochs') else None
    print("Batch Size: %d" % self.opt.batchSize) if hasattr(self.opt, 'batchSize') else None
    print("Mask Type: " + self.opt.maskType) if hasattr(self.opt, 'maskType') else None
    print("Holes Number: %d" % self.opt.max_holes) if hasattr(self.opt, 'max_holes') else None
    if hasattr(self.opt, 'maskType') and self.opt.maskType == "Square":
       print("Holes Size: [%d ; %d]\n" % (self.opt.holeMin, self.opt.holeMax)) if hasattr(self.opt, 'holeMin') and hasattr(self.opt, 'holeMax') else None
    print("Beta Range: [%f ; %f]\n" % (self.opt.beta1, self.opt.beta2)) if hasattr(self.opt, 'beta1') and hasattr(self.opt, 'beta2') else None
    print("Generator Learning Rate: %f" % self.opt.lr) if hasattr(self.opt, 'lr') else None
    print("Discriminator Learning Rate: %f" % (self.opt.lr * self.opt.D2G_lr)) if hasattr(self.opt, 'lr') and hasattr(self.opt, 'D2G_lr') else None
    print("Weight for L2 loss: %f" % self.opt.wtl2) if hasattr(self.opt, 'wtl2') else None
    print("Weight for Discriminator loss: %f" % self.opt.wtlD) if hasattr(self.opt, 'wtlD') else None
    print("Weight for Feature Matching: %f" % self.opt.wfm) if hasattr(self.opt, 'wfm') else None
    print("Weight for Style loss: %f" % self.opt.wstyle) if hasattr(self.opt, 'wstyle') else None
    print("Weight for Perceptual loss: %f" % self.opt.wperc) if hasattr(self.opt, 'wperc') else None
    print("Weight for GAN adversarial loss: %f" % self.opt.wgadv) if hasattr(self.opt, 'wgadv') else None
    print("Weight for L1 loss: %f" % self.opt.wtl1) if hasattr(self.opt, 'wtl1') else None
    print("Weight for Total Variation loss: %f" % self.opt.wtv) if hasattr(self.opt, 'wtv') else None


  # Read Options
  def __read__(self,file_path):
    with open(file_path, 'rb') as file:
      self.opt = pickle.load(file)
    return self.opt

  # Write Options
  def __write__(self,opt):
    self.opt=opt
    file_path='models/'+self.opt.network+'/'+self.opt.specificity+'/opt.pkl'
    with open(file_path, 'wb') as file:
      pickle.dump(self.opt, file)
