from vutils.libraries import *
## Loss Functions
class adversarialLoss:
  def __init__(self):
    pass
  def adv_loss(opt,net,fake=None,real=None,crit="BCE"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if crit=="BCE":
      criterion = nn.BCEWithLogits().to(device)
    elif crit=="MSE":
      criterion = nn.MSELoss().to(device)

    # Move label tensors to device if they exist
    real_label = None
    fake_label = None
    if fake is not None:
        real_label = torch.ones_like(fake, requires_grad=True, device=device)
        fake_label = torch.zeros_like(fake, requires_grad=True, device=device)

    # generator adversarial loss
    if net=="netG":
      loss = criterion(fake, real_label)

    # discriminator adversarial loss
    elif net=="netD":
      real_loss = criterion(real, real_label)
      fake_loss = criterion(fake, fake_label)
      loss = (real_loss + fake_loss) / 2

    return loss

class tvLoss:
  def __init__(self):
    pass

  def tv_loss(image):
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))+torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class style_perceptualLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(style_perceptualLoss, self).__init__()
        self.vgg = VGG19().cuda() if torch.cuda.is_available() else VGG19()
        self.criterion = torch.nn.L1Loss().cuda() if torch.cuda.is_available() else torch.nn.L1Loss()
        self.weights = weights

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def forward(self, x, y, masks):
        x_vgg_style, y_vgg_style = self.vgg(x * masks), self.vgg(y * masks)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg_style['relu2_2']), self.compute_gram(y_vgg_style['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg_style['relu3_4']), self.compute_gram(y_vgg_style['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg_style['relu4_4']), self.compute_gram(y_vgg_style['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg_style['relu5_2']), self.compute_gram(y_vgg_style['relu5_2']))

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return style_loss, content_loss

class VGG19(torch.nn.Module):
  def __init__(self):
      super(VGG19, self).__init__()
      features = models.vgg19(pretrained=True).features
      self.relu1_1 = torch.nn.Sequential()
      self.relu1_2 = torch.nn.Sequential()
      self.relu2_1 = torch.nn.Sequential()
      self.relu2_2 = torch.nn.Sequential()
      self.relu3_1 = torch.nn.Sequential()
      self.relu3_2 = torch.nn.Sequential()
      self.relu3_3 = torch.nn.Sequential()
      self.relu3_4 = torch.nn.Sequential()
      self.relu4_1 = torch.nn.Sequential()
      self.relu4_2 = torch.nn.Sequential()
      self.relu4_3 = torch.nn.Sequential()
      self.relu4_4 = torch.nn.Sequential()
      self.relu5_1 = torch.nn.Sequential()
      self.relu5_2 = torch.nn.Sequential()
      self.relu5_3 = torch.nn.Sequential()
      self.relu5_4 = torch.nn.Sequential()

      for x in range(2):
          # Conv+ReLu
          self.relu1_1.add_module(str(x), features[x])
      for x in range(2, 4):
          self.relu1_2.add_module(str(x), features[x])
      for x in range(4, 7):
          self.relu2_1.add_module(str(x), features[x])
      for x in range(7, 9):
          self.relu2_2.add_module(str(x), features[x])
      for x in range(9, 12):
          self.relu3_1.add_module(str(x), features[x])
      for x in range(12, 14):
          self.relu3_2.add_module(str(x), features[x])
      for x in range(14, 16):
          self.relu3_3.add_module(str(x), features[x])
      for x in range(16, 18):
          self.relu3_4.add_module(str(x), features[x])
      for x in range(18, 21):
          self.relu4_1.add_module(str(x), features[x])
      for x in range(21, 23):
          self.relu4_2.add_module(str(x), features[x])
      for x in range(23, 25):
          self.relu4_3.add_module(str(x), features[x])
      for x in range(25, 27):
          self.relu4_4.add_module(str(x), features[x])
      for x in range(27, 30):
          self.relu5_1.add_module(str(x), features[x])
      for x in range(30, 32):
          self.relu5_2.add_module(str(x), features[x])
      for x in range(32, 34):
          self.relu5_3.add_module(str(x), features[x])
      for x in range(34, 36):
          self.relu5_4.add_module(str(x), features[x])

      # don't need the gradients, just want the features
      for param in self.parameters():
          param.requires_grad = False
          param.data = param.data.cuda()

  def forward(self, x):
      x_conc = torch.cat((x, x, x),dim=1)

      relu1_1 = self.relu1_1(x_conc)
      relu1_2 = self.relu1_2(relu1_1)

      relu2_1 = self.relu2_1(relu1_2)
      relu2_2 = self.relu2_2(relu2_1)

      relu3_1 = self.relu3_1(relu2_2)
      relu3_2 = self.relu3_2(relu3_1)
      relu3_3 = self.relu3_3(relu3_2)
      relu3_4 = self.relu3_4(relu3_3)

      relu4_1 = self.relu4_1(relu3_4)
      relu4_2 = self.relu4_2(relu4_1)
      relu4_3 = self.relu4_3(relu4_2)
      relu4_4 = self.relu4_4(relu4_3)

      relu5_1 = self.relu5_1(relu4_4)
      relu5_2 = self.relu5_2(relu5_1)
      relu5_3 = self.relu5_3(relu5_2)
      relu5_4 = self.relu5_4(relu5_3)

      out = {
          'relu1_1': relu1_1,
          'relu1_2': relu1_2,

          'relu2_1': relu2_1,
          'relu2_2': relu2_2,

          'relu3_1': relu3_1,
          'relu3_2': relu3_2,
          'relu3_3': relu3_3,
          'relu3_4': relu3_4,

          'relu4_1': relu4_1,
          'relu4_2': relu4_2,
          'relu4_3': relu4_3,
          'relu4_4': relu4_4,

          'relu5_1': relu5_1,
          'relu5_2': relu5_2,
          'relu5_3': relu5_3,
          'relu5_4': relu5_4,
      }
      return out

class gaborLoss:
  def __init__(self):
    pass

  # Define the Gabor filter
  def gabor_filter(self,Lambda, theta, kernel_size=11, sigma=2.0, psi=0.0, gamma=0.5):
      half_size = kernel_size // 2
      x, y = np.meshgrid(np.arange(-half_size, half_size + 1), np.arange(-half_size, half_size + 1))
      x_theta = x * np.cos(theta) + y * np.sin(theta)
      y_theta = -x * np.sin(theta) + y * np.cos(theta)
      g = np.exp(-(x_theta ** 2 + gamma ** 2 * y_theta ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * x_theta / Lambda + psi)
      g = g - np.mean(g)  # center the Gabor filter
      return g.astype(np.float32)

  # Custom loss function with Gabor term for multiple wavelengths and orientations
  def gabor_loss(self,y_pred, y_true):
    # Define wavelengths and orientations
    wavelengths = [2, 4, 8]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    # Initialize total Gabor loss
    total_gabor_loss = 0

    # Compute Gabor responses for each wavelength and orientation
    for Lambda in wavelengths:
        for theta in orientations:
            kernel_size=11
            # Generate Gabor filter kernel
            gabor_filter_kernel = self.gabor_filter(Lambda, theta)
            gabor_filter_kernel = torch.tensor(gabor_filter_kernel).unsqueeze(0).unsqueeze(0)
            gabor_filter_kernel = gabor_filter_kernel.repeat(1, 1, 1, 1).to(y_true.device)

            # Compute Gabor responses
            gabor_response_real = F.conv2d(y_true, gabor_filter_kernel, padding=kernel_size // 2)
            gabor_response_pred = F.conv2d(y_pred, gabor_filter_kernel, padding=kernel_size // 2)

            # Mean squared error between Gabor responses
            gabor_loss = F.mse_loss(gabor_response_pred, gabor_response_real)

            # Accumulate Gabor loss
            total_gabor_loss += gabor_loss

    # Average Gabor loss over all wavelengths and orientations
    total_gabor_loss /= len(wavelengths) * len(orientations)

    return total_gabor_loss
