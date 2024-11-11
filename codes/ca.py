from vutils.libraries import *
from vutils.losses import *
from frameworks.ca_vutils import * 
import vutils.data_processing as data_processing
import options_configuration

processor = data_processing.data_processing()
config= options_configuration.options_configuration()

class _netG(nn.Module):
    def __init__(self, opt):
        super(_netG, self).__init__()
        self.coarse_generator = _netG_coarse(opt)
        self.fine_generator = _netG_fine(opt)

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x, mask)
        x_stage2 = self.fine_generator(x, x_stage1, mask)
        return x_stage1, x_stage2

class _netG_coarse(nn.Module):
    def __init__(self, opt,cnum=32):
        super(_netG_coarse, self).__init__()
        self.conv1 = gen_conv(opt.nc + 2, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum*2, 3, 2, 1)
        self.conv3 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        self.conv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        self.conv11 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.conv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.conv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.conv17 = gen_conv(cnum//2, opt.nc, 3, 1, 1, activation='none')

    def forward(self, x, mask):
        # For indicating the boundaries of images
        ones = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3])
        if torch.cuda.is_available():
            ones = ones.cuda()

        # 5 x 256 x 256
        x = self.conv1(torch.cat([x, ones, mask], dim=1))
        x = self.conv2_downsample(x)
        # cnum*2 x 128 x 128
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        # cnum*4 x 64 x 64
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum*2 x 128 x 128
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum x 256 x 256
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        # 3 x 256 x 256
        x_stage1 = torch.clamp(x, -1., 1.)

        return x_stage1


class _netG_fine(nn.Module):
    def __init__(self, opt, cnum=32):
        super(_netG_fine, self).__init__()

        # 3 x 256 x 256
        self.conv1 = gen_conv(opt.nc + 2, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.conv3 = gen_conv(cnum, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*2, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.conv5 = gen_conv(cnum*2, cnum*4, 3, 1, 1)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        # attention branch
        # 3 x 256 x 256
        self.pmconv1 = gen_conv(opt.nc + 2, cnum, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.pmconv3 = gen_conv(cnum, cnum*2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.pmconv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1, activation='relu')
        self.contextul_attention = ContextualAttention()
        
        self.pmconv9 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv10 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.allconv11 = gen_conv(cnum*8, cnum*4, 3, 1, 1)
        self.allconv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.allconv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.allconv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.allconv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.allconv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.allconv17 = gen_conv(cnum//2, opt.nc, 3, 1, 1, activation='none')

    def forward(self, xin, x_stage1, mask):
        x1_inpaint = x_stage1 * mask + xin * (1. - mask)
        # For indicating the boundaries of images
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        if torch.cuda.is_available():
            ones = ones.cuda()
        
        # onv branch
        xnow = torch.cat([x1_inpaint, ones, mask], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x = self.contextul_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.clamp(x, -1., 1.)

        return x_stage2
    
class ContextualAttention(nn.Module):
    def __init__(self):
        super(ContextualAttention, self).__init__()

    def forward(self, f, b, mask=None,  ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True):
        
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes

        raw_int_fs = list(f.size())  # b*c*h*w
        raw_int_bs = list(b.size())  # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[rate * stride,
                                               rate * stride],
                                      rates=[1, 1],
                                      padding='same')  # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1. / rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1. / rate, mode='nearest')
        int_fs = list(f.size())  # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksizes=[ksize, ksize],
                                  strides=[stride, stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], ksize, ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]])
            if self.use_cuda:
                mask = mask.cuda()
        else:
            mask = F.interpolate(mask, scale_factor=1. / (4 * rate), mode='nearest')
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksizes=[ksize, ksize],
                                  strides=[stride, stride],
                                  rates=[1, 1],
                                  padding='same')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], ksize, ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)  # m shape: [N, L, C, k, k]
        m = m[0]  # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True) == 0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3)  # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        k = fuse_k
        scale = softmax_scale  # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        if torch.cuda.is_available()==True:
            fuse_weight = fuse_weight.cuda()

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if torch.cuda.is_available()==True:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(reduce_sum(torch.pow(wi, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [ksize, ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
                        
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

            if int_bs != int_fs:
                # Normalize the offset value to match foreground dimension
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat([offset // int_fs[3], offset % int_fs[3]], dim=1)  # 1*2*H*W

            # deconv for patch pasting
            wi_center = raw_wi[0]
#             yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = F.conv_transpose2d(yi, wi_center, stride=rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs)

        return y

class _netD_local(nn.Module):
    def __init__(self, opt):
        super(_netD_local, self).__init__()
        cnum=64
        self.dis_conv_module = DisConvModule(opt.nc, cnum)
        self.linear = nn.Linear(25600, 1)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x= self.sigmoid(x)
        return x


class _netD_global(nn.Module):
    def __init__(self, opt):
        super(_netD_global, self).__init__()
        cnum=64
        self.dis_conv_module = DisConvModule(opt.nc, cnum)
        self.linear = nn.Linear(cnum*4*16*16, 1)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        x= self.sigmoid(x)

        return x


class DisConvModule(nn.Module):
    def __init__(self, input_dim, cnum):
        super(DisConvModule, self).__init__()
        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum*2, 5, 2, 2)
        self.conv3 = dis_conv(cnum*2, cnum*4, 5, 2, 2)
        self.conv4 = dis_conv(cnum*4, cnum*4, 5, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ca_original():
  def __init__(self):
    pass

  def train(self, fold, dataloader,opt):
    resume_epoch=1
    netG = _netG(opt)

    # If exist the file of the pre-trained network
    try:
      netG.load_state_dict(torch.load('models/'+opt.network+'/'+opt.specificity+'/fold_%s_netG.pth'%fold,map_location=lambda storage, location: storage)['state_dict'])
      resume_epoch = torch.load('models/'+opt.network+'/'+opt.specificity+'/fold_%s_netG.pth'%fold)['epoch']
    except Exception :
      print("GENERATOR MODEL NOT DETECTED!")

    # Call the Discriminator
    netD_local = _netD_local(opt)

    try:
      netD_local.load_state_dict(torch.load('models/'+opt.network+'/'+opt.specificity+'/fold_%s_netD_local.pth'%fold,map_location=lambda storage, location: storage)['state_dict'])
      resume_epoch = torch.load('models/'+opt.network+'/'+opt.specificity+'/fold_%s_netD_local.pth'%fold)['epoch']
    except Exception :
      print("LOCAL DISCRIMINATOR MODEL NOTE DETECTED!")

    # Call the Discriminator
    netD_global = _netD_global(opt)
    
    try:
      netD_global.load_state_dict(torch.load('models/'+opt.network+'/'+opt.specificity+'/fold_%s_netD_global.pth'%fold,map_location=lambda storage, location: storage)['state_dict'])
      resume_epoch = torch.load('models/'+opt.network+'/'+opt.specificity+'/fold_%s_netD_global.pth'%fold)['epoch']
    except Exception :
      print("DISCRIMINATOR MODEL NOTE DETECTED!")

    # Call Losses Functions
    l1_loss = nn.L1Loss()

    if torch.cuda.is_available():
      netD_local.cuda()
      netD_global.cuda()
      netG.cuda()
      l1_loss.cuda()

    # Setup optimizer ADAM (net_parameter, learning rate, betas)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    d_params = list(self.netD_local.parameters()) + list(self.netD_global.parameters())
    optimizerD = optim.Adam(d_params, lr=opt.lr*opt.D2G_lr, betas=(opt.beta1, opt.beta2))

    # Create the progress bar
    pbar_epochs = tqdm(total=opt.epochs)
    pbar_epochs.n = resume_epoch
    pbar_epochs.set_description("PRETRAINED EPOCHS MODEL")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Training cycle
    for epoch in range(resume_epoch,opt.epochs+1):
      for i, data in enumerate(dataloader,0):
        print("\n[Fold %s | Epoch %d | Batch %d/%d]"%(fold,epoch,i+1,len(dataloader)-1))
        # Move data to GPU if available and set requires_grad=True
        data_gpu=data.to(device)
        input_real = data_gpu[:, :1, :, :].float()
        masks = data_gpu[:, 1:2, :, :].float()
        input_cropped = ((1 - masks) * input_real + opt.mask * masks).float()

        # Gradient Aplication
        input_real.requires_grad = True
        input_cropped.requires_grad = True
        masks.requires_grad = True

        # generator adversarial loss
        optimizerG.zero_grad()
        fake_coarse, fake_fine, _  = netG(input_cropped,masks)
        fake_coarse= fake_coarse*masks+input_cropped*(1-masks)
        fake_fine= fake_fine*masks+input_cropped*(1-masks)
        crop_masks=processor.local_crop(masks,opt)

        input_real_local=processor.crop(input_real, crop_masks)
        fake_coarse_local=processor.crop(fake_coarse, crop_masks)
        fake_fine_local=processor.crop(fake_fine, crop_masks)

        output_local = netD_local(fake_fine_local)
        output_global = netD_global(fake_fine)
        errG_D=adversarialLoss.adv_loss(opt,"netG",output_local)+adversarialLoss.adv_loss(opt,"netG",output_global)



        

        errG_l1 = l1_loss(fake_fine_local, input_real_local)+l1_loss(fake_coarse_local, input_real_local)+l1_loss(fake_coarse, input_real)+l1_loss(fake_coarse, input_real)
        gen_loss = opt.wtlD * errG_D + opt.wtl1 * errG_l1
        gen_loss.backward()
        optimizerG.step()

        # discriminator loss
        optimizerD.zero_grad()
        dis_fake_local = netD(fake_fine_local.detach())
        dis_real_local = netD(input_real_local)
        dis_fake_global = netD(fake_fine.detach())
        dis_real_global = netD(input_real)
        dis_loss = adversarialLoss.adv_loss(opt,"netD",dis_fake_local,dis_real_local)+adversarialLoss.adv_loss(opt,"netD",dis_fake_global,dis_real_global)
        dis_loss.backward()
        optimizerD.step()

        # Print of step training information
        print('GENERATOR TRAIN LOSS: %.5f \nDISCRIMINATOR TRAIN LOSS: %.5f \nD(x): %.5f \nD(G(z)): %.5f'%(gen_loss.item(), dis_loss.item(), dis_real_global.data.mean(),output_global.data.mean()))

        # Save results if i=0 (in my dataset)
        #if pbar_epochs.n==1 or pbar_epochs.n==opt.epochs:
        #  save_image(fake.data,'debug/'+opt.network+'/'+opt.specificity+'/edges/fold_%s_epoch_%d.png'%(fold,epoch),normalize=True)

      # Do checkpointing
      torch.save({'epoch':pbar_epochs.n,
                  'state_dict':netG.state_dict()},
                  'models/'+opt.network+'/'+opt.specificity+'/fold_%s_netG.pth'%(fold))
      torch.save({'epoch':pbar_epochs.n,
                  'state_dict':netD_global.state_dict()},
                  'models/'+opt.network+'/'+opt.specificity+'/fold_%s_netD_global.pth'%(fold))
      torch.save({'epoch':pbar_epochs.n,
                  'state_dict':netD_local.state_dict()},
                  'models/'+opt.network+'/'+opt.specificity+'/fold_%s_netD_local.pth'%(fold))
      
      pbar_epochs.update()
    pbar_epochs.close()

####################################################################################

  def test(self,fold, dataloader, dataloader_labels, dataloader_tumoral_labels, dataloader_pulmonar_label, opt):
    # Load of generator network already trained
    netG = _netG(opt)
    netG.load_state_dict(torch.load('models/CA_Original/Square_20/fold_%s_netG.pth'%(fold),map_location=lambda storage, location: storage)['state_dict'])
    netG.eval()
    if torch.cuda.is_available():
      netG.cuda()

    array_size=len(dataloader)

    total_mae_list= np.zeros(array_size)
    total_mse_list=np.zeros(array_size)
    total_psnr_list=np.zeros(array_size)
    total_ssim_list=np.zeros(array_size)

    tumour_mae_list=[]
    tumour_mse_list=[]
    tumour_psnr_list=[]
    tumour_ssim_list=[]

    non_tumour_mae_list=[]
    non_tumour_mse_list=[]
    non_tumour_psnr_list=[]
    non_tumour_ssim_list=[]

    pulmonar_mae_list=[]
    pulmonar_mse_list=[]
    pulmonar_psnr_list=[]

    non_pulmonar_mae_list=[]
    non_pulmonar_mse_list=[]
    non_pulmonar_psnr_list=[]

    tumourlabel_mae_list=[]
    tumourlabel_mse_list=[]
    tumourlabel_psnr_list=[]

    ##########
    tumor_metrics = [None]*len(dataloader)
    recon_images = [None]*len(dataloader)
    test_data=[None]*len(dataloader)

    non_tumour_recon_images=[]
    non_tumour_test_data=[]
    
    tumour_recon_images=[]
    tumour_test_data=[]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
      for i, (data,label,tumor_label,pulmonar_label) in enumerate(zip(dataloader,dataloader_labels,dataloader_tumoral_labels,dataloader_pulmonar_label), 0):
        data_gpu=data.to(device)
        input_real = data_gpu[:, :1, :, :].float()
        masks = data_gpu[:, 1:2, :, :].float()
        input_cropped = ((1 - masks) * input_real + opt.mask * masks).float()
        _ , fake  = netG(input_cropped,masks)
        masks=(masks == 1).float()
        # save_image((masks>0).float() - (masks==1).float(), 'masks.png',normalize=False)
        # masks=(masks == 1).float()
        fake = fake*masks+input_real*(1-masks)
        fake=torch.clamp(fake, 0,1)

        tumor_real_inside_missing_condition=(masks*tumor_label*input_real)
        tumor_fake_inside_missing_condition=(masks*tumor_label*fake)

        pulmonar_real_inside_missing_condition=(masks*pulmonar_label*(1-tumor_label)*input_real)
        pulmonar_fake_inside_missing_condition=(masks*pulmonar_label*(1-tumor_label)*fake)

        non_pulmonar_real_inside_missing_condition=(masks*(1-pulmonar_label)*(1-tumor_label)*input_real)
        non_pulmonar_fake_inside_missing_condition=(masks*(1-pulmonar_label)*(1-tumor_label)*fake)

        #torch.save(torch.cat((masks,tumor_label,input_real,fake,tumor_real_inside_missing_condition,tumor_fake_inside_missing_condition,tumor_real_condition),dim=0),"tumour.pth")

        if torch.sum(tumor_real_inside_missing_condition>0.1875)>=1:
          pixels_inside_real=torch.sum(tumor_real_inside_missing_condition>0.1875)
          pixels_inside_fake=torch.sum(tumor_fake_inside_missing_condition>0.1875)
          pixels_all_real=torch.sum(tumor_label*input_real>0.1875)

          ratio_inside_real=pixels_inside_real/pixels_all_real                                        
          ratio_inside_fake=pixels_inside_fake/pixels_all_real
          ratio_fake_real=pixels_inside_fake/pixels_inside_real # value<100 - less tumor; velue>100 - more tumor

          mae_tumourlabel=torch.abs(torch.sum(tumor_real_inside_missing_condition-tumor_fake_inside_missing_condition))/torch.sum(masks*tumor_label)
          mse_tumourlabel=torch.sqrt(torch.sum((tumor_real_inside_missing_condition-tumor_fake_inside_missing_condition)**2)/torch.sum(masks*tumor_label))
          psnr_tumorlabel=10*torch.log10(1/mse_tumourlabel**2)
          tumourlabel_mae_list.append(mae_tumourlabel)
          tumourlabel_mse_list.append(mse_tumourlabel)
          tumourlabel_psnr_list.append(psnr_tumorlabel)

          tumors_F1Score=2*(torch.sum((tumor_fake_inside_missing_condition>0.1875) & (tumor_real_inside_missing_condition>0.1875)))/(pixels_inside_real+pixels_inside_fake)
          tumors_IoU=torch.sum((tumor_fake_inside_missing_condition>0.1875) & (tumor_real_inside_missing_condition>0.1875))/torch.sum((tumor_fake_inside_missing_condition>0.1875) | (tumor_real_inside_missing_condition>0.1875))

          tumor_metrics[i]=[ratio_inside_real,ratio_inside_fake,ratio_fake_real,tumors_F1Score,tumors_IoU]
          # tumors[i][j]=[pixels_inside_real,pixels_inside_fake,pixels_all_real,ratio_inside_real,ratio_inside_fake,ratio_fake_real,tumors_F1Score,tumors_IoU,tumors_intensity,std_difference,tumor_hausdorff_distance,tumor_frechet_distance] 

        if torch.sum(torch.sum(masks*pulmonar_label))>=1:
            mae_pulmonar=torch.abs(torch.sum(pulmonar_real_inside_missing_condition-pulmonar_fake_inside_missing_condition))/torch.sum(masks*pulmonar_label*(1-tumor_label))
            mse_pulmonar=torch.sqrt(torch.sum((pulmonar_real_inside_missing_condition-pulmonar_fake_inside_missing_condition)**2)/torch.sum(masks*pulmonar_label*(1-tumor_label)))
            psnr_pulmonar=10*torch.log10(1/mse_pulmonar**2)
            pulmonar_mae_list.append(mae_pulmonar)
            pulmonar_mse_list.append(mse_pulmonar)
            pulmonar_psnr_list.append(psnr_pulmonar)

        if torch.sum(torch.sum(masks*(1-pulmonar_label)))>=1:
            mae_non_pulmonar=torch.abs(torch.sum(non_pulmonar_real_inside_missing_condition-non_pulmonar_fake_inside_missing_condition))/torch.sum(masks*(1-pulmonar_label)*(1-tumor_label))
            mse_non_pulmonar=torch.sqrt(torch.sum((non_pulmonar_real_inside_missing_condition-non_pulmonar_fake_inside_missing_condition)**2)/torch.sum(masks*(1-pulmonar_label)*(1-tumor_label)))
            psnr_non_pulmonar=10*torch.log10(1/mse_non_pulmonar**2)
            non_pulmonar_mae_list.append(mae_non_pulmonar)
            non_pulmonar_mse_list.append(mse_non_pulmonar)
            non_pulmonar_psnr_list.append(psnr_non_pulmonar)

        total_mae_list[i] = torch.mean(torch.abs(input_real - fake))
        total_mse_list[i] = torch.mean((input_real - fake)**2)
        total_psnr_list[i] = piq.psnr(input_real,fake)
        total_ssim_list[i] = piq.ssim(input_real,fake)
        recon_images[i]=fake[0,:,:,:]
        test_data[i]=input_real[0,:,:,:]

        if label[1]==0:
          non_tumour_mae_list.append(torch.mean(torch.abs(input_real - fake)))
          non_tumour_mse_list.append(torch.mean((input_real - fake)**2))
          non_tumour_psnr_list.append(piq.psnr(input_real,fake))
          non_tumour_ssim_list.append(piq.ssim(input_real,fake))
          non_tumour_recon_images.append(fake[0,:,:,:])
          non_tumour_test_data.append(input_real[0,:,:,:])
        
        else:
          tumour_mae_list.append(torch.mean(torch.abs(input_real - fake)))
          tumour_mse_list.append(torch.mean((input_real - fake)**2))
          tumour_psnr_list.append(piq.psnr(input_real,fake))
          tumour_ssim_list.append(piq.ssim(input_real,fake))
          tumour_recon_images.append(fake[0,:,:,:])
          tumour_test_data.append(input_real[0,:,:,:])
  
        if i%100==0:
          print("SAVING IMAGE %d"%i)
          image_list=[input_real[0,:,:,:],input_cropped[0,:,:,:],fake[0,:,:,:]]
          grid_image = make_grid(image_list, nrow=3, normalize=True)
          save_image(grid_image, 'test/'+opt.network+'/'+opt.specificity+'/fold%s_test%d.png'%(fold,i),normalize=False)

      recon_images_rgb = [torch.cat([img, img, img], dim=0) for img in recon_images]
      test_data_rgb = [torch.cat([img, img, img], dim=0) for img in test_data]
      recon_dataloader = torch.utils.data.DataLoader(recon_images_rgb, batch_size=1, shuffle=False, num_workers=0)
      test_dataloader= torch.utils.data.DataLoader(test_data_rgb, batch_size=1, shuffle=False, num_workers=0)

      fid_metric = piq.FID()
      fake_feats=fid_metric.compute_feats(recon_dataloader,device=device)
      real_feats=fid_metric.compute_feats(test_dataloader,device=device)
      all_fid = fid_metric(fake_feats, real_feats)
      all_iscore, _ = piq.inception_score(fake_feats)

      non_tumor_recon_images_rgb = [torch.cat([img, img, img], dim=0) for img in non_tumour_recon_images]
      non_tumor_test_data_rgb = [torch.cat([img, img, img], dim=0) for img in non_tumour_test_data]
      non_tumor_recon_dataloader = torch.utils.data.DataLoader(non_tumor_recon_images_rgb, batch_size=1, shuffle=False, num_workers=0)
      non_tumor_test_dataloader= torch.utils.data.DataLoader(non_tumor_test_data_rgb, batch_size=1, shuffle=False, num_workers=0)

      non_tumor_fake_feats=fid_metric.compute_feats(non_tumor_recon_dataloader,device=device)
      non_tumor_real_feats=fid_metric.compute_feats(non_tumor_test_dataloader,device=device)
      non_tumor_fid = fid_metric(non_tumor_fake_feats, non_tumor_real_feats)
      non_tumor_iscore, _ = piq.inception_score(non_tumor_fake_feats)

      tumor_recon_images_rgb = [torch.cat([img, img, img], dim=0) for img in tumour_recon_images]
      tumor_test_data_rgb = [torch.cat([img, img, img], dim=0) for img in tumour_test_data]
      tumor_recon_dataloader = torch.utils.data.DataLoader(tumor_recon_images_rgb, batch_size=1, shuffle=False, num_workers=0)
      tumor_test_dataloader= torch.utils.data.DataLoader(tumor_test_data_rgb, batch_size=1, shuffle=False, num_workers=0)

      fid_metric = piq.FID()
      tumor_fake_feats=fid_metric.compute_feats(tumor_recon_dataloader,device=device)
      tumor_real_feats=fid_metric.compute_feats(tumor_test_dataloader,device=device)
      tumor_fid = fid_metric(tumor_fake_feats, tumor_real_feats)
      tumor_iscore, _ = piq.inception_score(tumor_fake_feats)

      # metrics=[np.mean(total_mae_list),np.mean(total_mse_list),np.mean(total_psnr_list),np.mean(total_ssim_list),np.mean(total_ms_ssim_list),fid,iscore,np.mean(mae_list),np.mean(mse_list),np.mean(psnr_list),np.mean(ssim_list)]

      mae=[total_mae_list,tumour_mae_list,non_tumour_mae_list,tumourlabel_mae_list,pulmonar_mae_list,non_pulmonar_mae_list]
      mse=[total_mse_list,tumour_mse_list,non_tumour_mse_list,tumourlabel_mse_list,pulmonar_mse_list,non_pulmonar_mse_list]
      psnr=[total_psnr_list,tumour_psnr_list,non_tumour_psnr_list,tumourlabel_psnr_list,pulmonar_psnr_list,non_pulmonar_psnr_list]
      ssim=[total_ssim_list,tumour_ssim_list,non_tumour_ssim_list]

      #mae=[total_mae_list,lowComplexity_mae_list,mediumComplexity_mae_list,highComplexity_mae_list,non_tumour_mae_list,tumour_mae_list]
      #mse=[total_mse_list,lowComplexity_mse_list,mediumComplexity_mse_list,highComplexity_mse_list,non_tumour_mse_list,tumour_mse_list]
      #psnr=[total_psnr_list,lowComplexity_psnr_list,mediumComplexity_psnr_list,highComplexity_psnr_list,non_tumour_psnr_list,tumour_psnr_list]
      
      fid=[all_fid,tumor_fid,non_tumor_fid]
      iscore=[all_iscore,tumor_iscore,non_tumor_iscore]

      metrics=[mae,mse,psnr,ssim,fid,iscore,tumor_metrics]

      torch.save(metrics,'test/%s/%s/complementar_all_metrics_fold_%s.pth'%(opt.network,opt.specificity,fold))

    print("-- END TEST --")

