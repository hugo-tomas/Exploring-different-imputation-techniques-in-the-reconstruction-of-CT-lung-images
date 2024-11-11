from vutils.libraries import *
from vutils.losses import *
# from vutils.metrics import *
import vutils.data_processing
import options_configuration

processor = vutils.data_processing.data_processing()
config=options_configuration.options_configuration()

class _netG_edge(nn.Module):
    def __init__(self, opt, use_spectral_norm=True):
        super(_netG_edge, self).__init__()
        self.residual_blocks=8
        self.use_spectral_norm=use_spectral_norm
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            self.spectral_norm(nn.Conv2d(in_channels=opt.nc+2, out_channels=64, kernel_size=7, padding=0), self.use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            self.spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), self.use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            self.spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), self.use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(self.residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=self.use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            self.spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), self.use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            self.spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), self.use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

    def spectral_norm(self,module, mode=True):
      if mode:
          return nn.utils.spectral_norm(module)
      return module

    def forward(self, input, masks):
        if torch.cuda.is_available():
          self.encoder.cuda()
          self.middle.cuda()
          self.decoder.cuda()

        output = self.encoder(input)
        output = self.middle(output)
        output = self.decoder(output)
        output = torch.sigmoid(output)
        output = output*masks+(1-masks)*input[:,1:2,:,:]
        return output

class _netG_inpainting(nn.Module):
    def __init__(self, opt):
        super(_netG_inpainting, self).__init__()
        self.residual_blocks=8
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=opt.nc+1, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(self.residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=opt.nc, kernel_size=7, padding=0),
        )

    def forward(self, input, masks):
        output = self.encoder(input)
        output = self.middle(output)
        output = self.decoder(output)
        output = torch.sigmoid(output)
        output = output*masks+(1-masks)*input[:,:1,:,:]
        return output

class _netD(nn.Module):
    def __init__(self, opt, in_channels=1, use_spectral_norm=True):
        super(_netD, self).__init__()
        self.ndf = 64
        self.use_spectral_norm=use_spectral_norm
        self.in_channels=in_channels
        self.conv1 = self.features = nn.Sequential(
            self.spectral_norm(nn.Conv2d(in_channels=self.in_channels, out_channels=self.ndf, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            self.spectral_norm(nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf*2, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            self.spectral_norm(nn.Conv2d(in_channels=self.ndf*2, out_channels=self.ndf*4, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            self.spectral_norm(nn.Conv2d(in_channels=self.ndf*4, out_channels=self.ndf*8, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            self.spectral_norm(nn.Conv2d(in_channels=self.ndf*8, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm))
        )

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        outputs = torch.sigmoid(conv5)
        return outputs, [conv1, conv2, conv3, conv4, conv5]

    def spectral_norm(self, module):
        if self.use_spectral_norm:
            return nn.utils.spectral_norm(module)
        else:
            return module

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=2, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            self.spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(dilation),
            self.spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def spectral_norm(self,module, mode=True):
      if mode:
          return nn.utils.spectral_norm(module)
      return module

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class edge_connected():
  def __init__(self):
    pass

  def weights_init(self, m, init_type='normal', gain=0.02):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv2d') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=gain)

        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, gain)
        nn.init.constant_(m.bias.data, 0.0)

  def train_edges(self, fold, dataloader,opt):
    resume_epoch=1
    netG = _netG_edge(opt)
    netG.apply(self.weights_init)

    # If exist the file of the pre-trained network
    try:
      netG.load_state_dict(torch.load('models/'+opt.network+'/'+opt.specificity+'/edges/fold_%s_netG.pth'%fold,map_location=lambda storage, location: storage)['state_dict'])
      resume_epoch = torch.load(opt.netG)['epoch']
    except Exception :
      print("GENERATOR MODEL NOT DETECTED!")

    # Call the Discriminator
    netD = _netD(opt)
    netD.apply(self.weights_init)

    try:
      netD.load_state_dict(torch.load('models/'+opt.network+'/'+opt.specificity+'/edges/fold_%s_netD.pth'%fold,map_location=lambda storage, location: storage)['state_dict'])
      resume_epoch = torch.load(opt.netD)['epoch']
    except Exception :
      print("DISCRIMINATOR MODEL NOTE DETECTED!")

    # Call Losses Functions
    l1_loss = nn.L1Loss()

    if torch.cuda.is_available():
      netD.cuda()
      netG.cuda()
      l1_loss.cuda()


    # Setup optimizer ADAM (net_parameter, learning rate, betas)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr*opt.D2G_lr, betas=(opt.beta1, opt.beta2))

    # Create the progress bar
    pbar_epochs = tqdm(total=opt.edge_epochs)
    pbar_epochs.n = resume_epoch
    pbar_epochs.set_description("PRETRAINED EPOCHS | EDGE MODEL")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Training cycle
    for epoch in range(resume_epoch,opt.edge_epochs+1):
      for i, data in enumerate(dataloader,0):
        print("\n[Fold %s | Epoch %d | Batch %d/%d]"%(fold,epoch,i+1,len(dataloader)-1))
        # Move data to GPU if available and set requires_grad=True
        data_gpu=data.to(device)
        input_real = data_gpu[:, :1, :, :].float()
        masks = data_gpu[:, 3:4, :, :].float()
        input_cropped = ((1 - masks) * input_real + opt.mask * masks).float()
        edges=data_gpu[:, 1:2, :, :].float()
        edges_cropped=edges*(1 - masks)

        # Gradient Aplication
        input_real.requires_grad = True
        input_cropped.requires_grad = True
        edges.requires_grad = True
        edges_cropped.requires_grad = True
        masks.requires_grad = True

        # zero optimizers
        optimizerG.zero_grad()

        # generator adversarial loss
        fake = netG(torch.cat((input_cropped, edges_cropped, masks), dim=1).float(),masks)
        fake = fake*masks+(1-masks)*edges_cropped

        gen_fake, gen_fake_feat = netD(fake)
        gen_gan_loss = adversarialLoss.adv_loss(opt,"netG",gen_fake)

        dis_real , dis_real_feat = netD(edges.float())
        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())

        # Total loss
        gen_loss = opt.wgadv*gen_gan_loss+opt.wfm*gen_fm_loss
        gen_loss.backward()
        optimizerG.step()

        optimizerD.zero_grad()

        # discriminator loss
        dis_fake, _ = netD(fake.detach())
        dis_loss = adversarialLoss.adv_loss(opt,"netD",dis_fake,dis_real)
        dis_loss.backward()
        optimizerD.step()

        # Print of step training information
        print('GENERATOR TRAIN LOSS: %.5f \nDISCRIMINATOR TRAIN LOSS: %.5f \nD(x): %.5f \nD(G(z)): %.5f'%(gen_loss.item(), dis_loss.item(), dis_real.data.mean(),gen_fake.data.mean()))

        # Save results if i=0 (in my dataset)
        if pbar_epochs.n==1 or pbar_epochs.n%10==0:
          save_image(fake.data,'debug/'+opt.network+'/'+opt.specificity+'/edges/fold_%s_epoch_%d.png'%(fold,epoch),normalize=True)
#          save_image(edges_cropped.data,'debug/'+opt.network+'/'+opt.specificity+'/edges/fold_%s_epoch_%d.png'%(fold,epoch),normalize=True)

      # Do checkpointing
      torch.save({'epoch':pbar_epochs.n,
                  'state_dict':netG.state_dict()},
                  'models/'+opt.network+'/'+opt.specificity+'/edges/fold_%s_netG.pth'%(fold))
      torch.save({'epoch':pbar_epochs.n,
                  'state_dict':netD.state_dict()},
                  'models/'+opt.network+'/'+opt.specificity+'/edges/fold_%s_netD.pth'%(fold))

      pbar_epochs.update()
    pbar_epochs.close()


  def train(self, fold, dataloader,opt):
    resume_epoch=1
    netG_edge = _netG_edge(opt)
    netG_edge.apply(self.weights_init)

    # If exist the file of the pre-trained network
    try:
      netG_edge.load_state_dict(torch.load('models/'+opt.network+'/'+opt.specificity+'/edges/fold_%s_netG.pth'%(fold),map_location=lambda storage, location: storage)['state_dict'])
    except Exception :
      print("EDGE GENERATOR MODEL NOT DETECTED!")

    netG = _netG_inpainting(opt)
    netG.apply(self.weights_init)
    # If exist the file of the pre-trained network
    try:
      netG.load_state_dict(torch.load('models/'+opt.network+'/'+opt.specificity+'/fold_%s_netG.pth'%(fold),map_location=lambda storage, location: storage)['state_dict'])
      resume_epoch = torch.load(opt.netG)['epoch']
    except Exception :
      print("GENERATOR MODEL NOT DETECTED!")

    # Call the Discriminator
    netD = _netD(opt)
    netD.apply(self.weights_init)

    try:
      netD.load_state_dict(torch.load('models/'+opt.network+'/'+opt.specificity+'/fold_%s_netD.pth'%(fold),map_location=lambda storage, location: storage)['state_dict'])
      resume_epoch = torch.load(opt.netD)['epoch']
    except Exception :
      print("DISCRIMINATOR MODEL NOTE DETECTED!")

    # Call Losses Functions
    l1_loss = nn.L1Loss()
    style_per_loss=style_perceptualLoss()

    if torch.cuda.is_available():
      netG_edge.cuda()
      netD.cuda()
      netG.cuda()
      l1_loss.cuda()
      style_per_loss.cuda()

    # Setup optimizer ADAM (net_parameter, learning rate, betas)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr*opt.D2G_lr, betas=(opt.beta1, opt.beta2))

    # Create the progress bar
    pbar_epochs = tqdm(total=opt.edge_epochs)
    pbar_epochs.n = resume_epoch
    pbar_epochs.set_description("TRAINED EPOCHS | INPAINTING MODEL")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Training cycle
    for epoch in range(resume_epoch,opt.epochs+1):
      for i, data in enumerate(dataloader, 0):
        if i == len(dataloader)-1:
            break
        print("\n[Fold %s | Epoch %d | Batch %d/%d]"%(fold,epoch,i+1,len(dataloader)-1))
        # Move data to GPU if available and set requires_grad=True
        data_gpu=data.to(device)
        input_real = data_gpu[:, :1, :, :].float()
        masks = data_gpu[:, 3:4, :, :].float()
        input_cropped = ((1 - masks) * input_real + opt.mask * masks).float()
        edges=data_gpu[:, 1:2, :, :].float()
        edges_cropped=edges*(1 - masks)

        # Gradient Aplication
        input_real.requires_grad = True
        input_cropped.requires_grad = True
        edges.requires_grad = True
        edges_cropped.requires_grad = True
        masks.requires_grad = True

        # zero optimizers
        optimizerG.zero_grad()

        # generator adversarial loss
        fake_edge = netG_edge(torch.cat((input_cropped, edges_cropped, masks), dim=1).float(),masks)
        fake_edge = fake_edge*masks+(1-masks)*edges

        fake=netG(torch.cat((input_cropped,fake_edge),dim=1).float(),masks)
        fake = fake*masks+(1-masks)*input_real

        gen_fake, gen_fake_feat = netD(fake)
        gen_gan_loss = adversarialLoss.adv_loss(opt,"netG",gen_fake)

        dis_real , dis_real_feat = netD(input_real.float())

        gen_l1_loss = l1_loss(fake, input_real)
        gen_style_loss, gen_content_loss = style_per_loss(fake, input_real, masks)

        # Total loss
        gen_loss = opt.wstyle*gen_style_loss+opt.wperc*gen_content_loss+opt.wtl1*gen_l1_loss+opt.wgadv*gen_gan_loss
        gen_loss.backward()
        optimizerG.step()

        optimizerD.zero_grad()

        # discriminator loss
        dis_fake, _ = netD(fake.detach())
        dis_loss = adversarialLoss.adv_loss(opt,"netD",dis_fake,dis_real)
        dis_loss.backward()
        optimizerD.step()

        # Print of step training information
        print('GENERATOR TRAIN LOSS: %.5f \nDISCRIMINATOR TRAIN LOSS: %.5f \nD(x): %.5f \nD(G(z)): %.5f'%(gen_loss.item(), dis_loss.item(), dis_real.data.mean(),gen_fake.data.mean()))

        if pbar_epochs.n==1 or pbar_epochs.n%10==0:
          save_image(fake.data,'debug/'+opt.network+'/'+opt.specificity+'/fold_%s_epoch_%d.png'%(fold,epoch),normalize=True)

      # Do checkpointing
      torch.save({'epoch':pbar_epochs.n,
                  'state_dict':netG.state_dict()},
                  'models/'+opt.network+'/'+opt.specificity+'/fold_%s_netG.pth'%(fold))
      torch.save({'epoch':pbar_epochs.n,
                  'state_dict':netD.state_dict()},
                  'models/'+opt.network+'/'+opt.specificity+'/fold_%s_netD.pth'%(fold))

      pbar_epochs.update()


  def test(self,fold,dataloader,dataloader_labels,dataloader_tumoral_labels,dataloader_pulmonar_label,type,opt):
    # Load of generator network already trained
    if type=="edges":
      netG = _netG_edge(opt)
      netG.load_state_dict(torch.load('models/'+opt.network+'/'+opt.specificity+'/edges/fold_%s_netG.pth'%(fold),map_location=lambda storage, location: storage)['state_dict'])
      netG.eval()
      if torch.cuda.is_available():
        netG.cuda()

    elif type=="inpainting":
      netG_edge = _netG_edge(opt)
      netG_edge.load_state_dict(torch.load('models/'+opt.network+'/'+opt.specificity+'/edges/fold_%s_netG.pth'%(fold),map_location=lambda storage, location: storage)['state_dict'])
      netG = _netG_inpainting(opt)
      netG.load_state_dict(torch.load('models/'+opt.network+'/'+opt.specificity+'/fold_%s_netG.pth'%(fold),map_location=lambda storage, location: storage)['state_dict'])
      netG_edge.eval()
      netG.eval()
      if torch.cuda.is_available():
        netG_edge.cuda()
        netG.cuda()

    array_size=len(dataloader)

    total_mae_list= np.zeros(array_size)
    total_mse_list=np.zeros(array_size)
    total_psnr_list=np.zeros(array_size)
    total_ssim_list=np.zeros(array_size)

    #lowComplexity_mae_list=[]
    #lowComplexity_mse_list=[]
    #lowComplexity_psnr_list=[]
    #lowComplexity_ssim_list=[]

    #mediumComplexity_mae_list=[]
    #mediumComplexity_mse_list=[]
    #mediumComplexity_psnr_list=[]
    #mediumComplexity_ssim_list=[]

    #highComplexity_mae_list=[]
    #highComplexity_mse_list=[]
    #highComplexity_psnr_list=[]
    #highComplexity_ssim_list=[]

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

    #low_recon_images = []
    #low_test_data = []

    #medium_recon_images = []

    #high_recon_images = []
    #high_test_data = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
      for i, (data,label,tumor_label,pulmonar_label) in enumerate(zip(dataloader,dataloader_labels,dataloader_tumoral_labels,dataloader_pulmonar_label), 0):
        data_gpu=data.to(device)
        input_real = data_gpu[:, :1, :, :].float()
        masks = data_gpu[:, 3:4, :, :].float()
        input_cropped = ((1 - masks) * input_real + opt.mask * masks).float()
        edges=data_gpu[:, 1:2, :, :].float()
        edges_cropped=edges*(1 - masks)

        tumor_label=tumor_label.cuda()
        pulmonar_label=pulmonar_label.cuda()

        if type=="edges":
          fake = netG(torch.cat((input_cropped, edges_cropped, masks), dim=1).float(),masks)

          masks=(masks == 1).float()
          edges_cropped = (1 - masks) * input_real + int(opt.masks) * masks
          fake = (1 - masks) * edges_cropped + fake * masks

        elif type=="inpainting":
          fake_edge=netG_edge(torch.cat((input_cropped, edges, masks), dim=1).float(),masks)
          fake_edge = fake_edge*masks+(1-masks)*edges_cropped
          fake = netG(torch.cat((input_cropped, fake_edge), dim=1).float(),masks)

          masks=(masks == 1).float()
          input_cropped = (1 - masks) * input_real + int(opt.mask) * masks
          fake = (1 - masks) * input_cropped + fake * masks

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

