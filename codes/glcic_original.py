from vutils.libraries import *
from vutils.losses import *
import vutils.data_processing
import options_configuration

processor = vutils.data_processing.data_processing()
config=options_configuration.options_configuration()

class Flatten(nn.Module):
  def __init__(self):
      super(Flatten, self).__init__()

  def forward(self, x):
      return x.view(x.shape[0], -1)

class Concatenate(nn.Module):
  def __init__(self, dim=-1):
      super(Concatenate, self).__init__()
      self.dim = dim

  def forward(self, x):
      return torch.cat(x, dim=self.dim)

class _netG(nn.Module):
  def __init__(self,opt):
    super(_netG, self).__init__()
    self.ngpu = opt.ngpu
    self.nef=64
    self.ngf=64
    self.main = nn.Sequential(
    # input_shape: (None, opt.nc+mask, img_h, img_w)
    nn.Conv2d(opt.nc, self.nef, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(self.nef),
    nn.ReLU(),
    # input_shape: (None, opt.nef, img_h, img_w)
    nn.Conv2d(self.nef, self.nef*2, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(self.nef*2),
    nn.ReLU(),
    # input_shape: (None, opt.nef*2, img_h//2, img_w//2)
    nn.Conv2d(self.nef*2, self.nef*2, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(self.nef*2),
    nn.ReLU(),
    # input_shape: (None, opt.nef*2, img_h//2, img_w//2)
    nn.Conv2d(self.nef*2, self.nef*4, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(self.nef*4),
    nn.ReLU(),
    # input_shape: (None, opt.nef*4, img_h//4, img_w//4)
    nn.Conv2d(self.nef*4, self.nef*4, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(self.nef*4),
    nn.ReLU(),
    # input_shape: (None, opt.nef*4, img_h//4, img_w//4)
    nn.Conv2d(self.nef*4, self.nef*4, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(self.nef*4),
    nn.ReLU(),
    # input_shape: (None, opt.nef*4, img_h//4, img_w//4)
    nn.Conv2d(self.nef*4, self.nef*4, kernel_size=3, stride=1, dilation=2, padding=2),
    nn.BatchNorm2d(self.nef*4),
    nn.ReLU(),
    # input_shape: (None, opt.nef*4, img_h//4, img_w//4)
    nn.Conv2d(self.nef*4, self.nef*4, kernel_size=3, stride=1, dilation=4, padding=4),
    nn.BatchNorm2d(self.nef*4),
    nn.ReLU(),
    # input_shape: (None, opt.nef*4, img_h//4, img_w//4)
    nn.Conv2d(self.nef*4, self.ngf*4, kernel_size=3, stride=1, dilation=8, padding=8),
    nn.BatchNorm2d(self.ngf*4),
    nn.ReLU(),
    # input_shape: (None, opt.nef*4, img_h//4, img_w//4)
    nn.Conv2d(self.ngf*4, self.ngf*4, kernel_size=3, stride=1, dilation=16, padding=16),
    nn.BatchNorm2d(self.ngf*4),
    nn.ReLU(),
    # input_shape: (None, opt.nef*4, img_h//4, img_w//4)
    nn.Conv2d(self.ngf*4, self.ngf*4, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(self.ngf*4),
    nn.ReLU(),
    # input_shape: (None, opt.nef*4, img_h//4, img_w//4)
    nn.Conv2d(self.ngf*4, self.ngf*4, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(self.ngf*4),
    nn.ReLU(),
    # input_shape: (None, opt.nef*4, img_h//4, img_w//4)
    nn.ConvTranspose2d(self.ngf*4, self.ngf*2, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(self.ngf*2),
    nn.ReLU(),
    # input_shape: (None, opt.nef*2, img_h//2, img_w//2)
    nn.Conv2d(self.ngf*2, self.ngf*2, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(self.ngf*2),
    nn.ReLU(),
    # input_shape: (None, opt.nef*2, img_h//2, img_w//2)
    nn.ConvTranspose2d(self.ngf*2, self.ngf, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(self.ngf),
    nn.ReLU(),
    # input_shape: (None, opt.nef, img_h, img_w)
    nn.Conv2d(self.ngf, self.ngf//2, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(self.ngf//2),
    nn.ReLU(),
    nn.Conv2d(self.ngf//2, opt.nc, kernel_size=3, stride=1, padding=1),
    nn.Sigmoid()
    # output_shape: (None, opt.nc, img_h. img_w)
    )

  def forward(self,input,masks):
    # Run with GPU in parallel
    if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
      output = self.main(input)

    # Final Output only have the mask region modified
    img = torch.empty_like(input)
    for j in range(input.shape[0]):
        img[j,:,:,:] = (1 - masks[j,:,:,:]) * input[j,:,:,:] + masks[j,:,:,:] * output[j,:,:,:]
    return img

class _netlocal_globalD(nn.Module):
  def __init__(self, opt):
    super(_netlocal_globalD, self).__init__()
    self.ndf=64
    self.output_shape = self.ndf*16
    self.ngpu = opt.ngpu
    self.main = nn.Sequential(
    # input_shape: (None, opt.nc, img_h, img_w)
    nn.Conv2d(opt.nc, self.ndf, kernel_size=5, stride=2, padding=2),
    nn.BatchNorm2d(self.ndf),
    nn.LeakyReLU(0.2, inplace=True),
    # input_shape: (None, opt.ndf, img_h//2, img_w//2)
    nn.Conv2d(self.ndf, self.ndf*2, kernel_size=5, stride=2, padding=2),
    nn.BatchNorm2d(self.ndf*2),
    nn.LeakyReLU(0.2, inplace=True),
    # input_shape: (None, opt.ndf*2, img_h//4, img_w//4)
    nn.Conv2d(self.ndf*2, self.ndf*4, kernel_size=5, stride=2, padding=2),
    nn.BatchNorm2d(self.ndf*4),
    nn.LeakyReLU(0.2, inplace=True),
    # input_shape: (None, opt.ndf*4, img_h//8, img_w//8)
    nn.Conv2d(self.ndf*4, self.ndf*8, kernel_size=5, stride=2, padding=2),
    nn.BatchNorm2d(self.ndf*8),
    nn.LeakyReLU(0.2, inplace=True),
    # input_shape: (None, opt.ndf*8, img_h//16, img_w//16)
    nn.Conv2d(self.ndf*8, self.ndf*8, kernel_size=5, stride=2, padding=2),
    nn.BatchNorm2d(self.ndf*8),
    nn.LeakyReLU(0.2, inplace=True),
    # input_shape (None, opt.ndf*8, img_h//64, img_w//64)
    )
    self.global_layer= nn.Sequential(nn.Conv2d(self.ndf*8, self.ndf*8, kernel_size=5, stride=2, padding=2),
                                     nn.BatchNorm2d(self.ndf*8),
                                     nn.LeakyReLU(0.2, inplace=True))
    self.Flatten=Flatten()
    # Linear layer without specifying in_features
    self.linear = nn.Linear(in_features=1, out_features=self.output_shape)
    self.ReLu = nn.ReLU()

  def forward(self,type,input):
    if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
      if type=='local':
        output = nn.parallel.data_parallel(self.Flatten(self.main), input, range(self.ngpu))
      else:
        output = nn.parallel.data_parallel(self.Flatten(self.global_layer(self.main, input)), range(self.ngpu))
    else:
      if type=='local':
        output = self.Flatten(self.main(input))
      else:
        output = self.Flatten(self.global_layer(self.main(input)))

    device = next(self.parameters()).device

    if self.linear.in_features != output.size(1):
      self.linear = nn.Linear(in_features=output.size(1), out_features=self.output_shape).to(device)

    output = self.ReLu(self.linear(output))
    return output


class _netD(nn.Module):
  def __init__(self, opt):
    super(_netD, self).__init__()
    self.model_ld = _netlocal_globalD(opt)
    self.model_gd = _netlocal_globalD(opt)
    # input_shape: [(None, opt.ndf*16), (None, opt.ndf*16)]
    self.in_features = self.model_ld.output_shape + self.model_gd.output_shape
    self.concat = Concatenate(dim=-1)
    # input_shape: (None, 2048)
    self.linear = nn.Linear(self.in_features, 1)
    self.act = nn.Sigmoid()
    # output_shape: (None, 1)

  def forward(self, input_ld, input_gd):
    input_local = self.model_ld('local',input_ld)
    input_global = self.model_gd('global',input_gd)
    output = self.act(self.linear(self.concat([input_local, input_global])))
    return output.view(-1,1)

class globally_locally_consistent():
  def __init__(self):
    pass

  def weights_init(self,m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
      m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
      m.weight.data.normal_(1.0, 0.02)
      m.bias.data.fill_(0)

  def train_generator(self, fold, dataloader,opt):
    # Initial epoch
    resume_epoch=1

    # Call the Generator
    netG = _netG(opt)
    netG.apply(self.weights_init)

    try:
      os.makedirs('debug/'+opt.network+'/'+opt.specificity+'/generator_train')
      os.makedirs('models/'+opt.network+'/'+opt.specificity+'/phase1')
    except OSError:
      print("GENERATOR MODEL NOTE DETECTED!")

    # Used Losses (Mean Squared Error)
    criterionMSE = nn.MSELoss()

    if torch.cuda.is_available():
        netG = netG.cuda()
        criterionMSE = criterionMSE.cuda()

    netG_opt = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    # Create the progress bar
    print("-- BEGIN PRETRAIN --")
    pbar_epochs = tqdm(total=opt.generator_epochs)
    pbar_epochs.n = resume_epoch
    pbar_epochs.set_description("TRAINED GENERATOR EPOCHS | GLCIC [ORIGINAL]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(resume_epoch,opt.generator_epochs+1):
      for i, data in enumerate(dataloader, 0):
        print("\n[Fold %s| Epoch %d | Batch %d/%d]"%(fold,epoch,i+1,len(dataloader)-1))
        # Move data to GPU if available and set requires_grad=True
        data_gpu=data.to(device)
        input_real = data_gpu[:, :1, :, :].float()
        masks = data_gpu[:, 1:2, :, :].float()
        input_cropped = ((1 - masks) * input_real + opt.mask * masks).float()
        crop_masks=processor.local_crop(masks,opt)

        # Gradient Aplication
        input_real.requires_grad = True
        input_cropped.requires_grad = True
        masks.requires_grad = True
        crop_masks.requires_grad = True

        netG.zero_grad()
        fake = netG(input_cropped,masks)
        errG = criterionMSE(processor.crop(fake, crop_masks),processor.crop(input_real, crop_masks))
        errG.backward()

        # optimize
        netG_opt.step()

        print('GENERATOR TARINING LOSS: {:.5f}'.format(errG))

      # Do checkpointing
      torch.save({'epoch':pbar_epochs.n,
                'state_dict':netG.state_dict()},
                'models/'+opt.network+'/'+opt.specificity+'/phase1/fold%s_netG.pth'%(fold))

      pbar_epochs.update()
    pbar_epochs.close()

  def train_discriminator(self, fold, dataloader,opt):
    resume_epoch=1

    netD = _netD(opt)
    netD.apply(self.weights_init)

    netG = _netG(opt)
    try:
      netG.load_state_dict(torch.load('models/'+opt.network+'/'+opt.specificity+'/phase1/fold%s_netG.pth'%(fold), map_location=lambda storage, location: storage)['state_dict'])
    except OSError:
      netG.apply(self.weights_init)

    try:
      os.makedirs('models/'+opt.network+'/'+opt.specificity+'/phase2')
    except OSError:
      print("DISCRIMINATOR MODEL NOTE DETECTED!")

    if torch.cuda.is_available():
        netG = netG.cuda()
        netD = netD.cuda()

    # Setup optimizer ADAM (net_parameter, learning rate, betas)
    netD_opt = optim.Adam(netD.parameters(), lr=opt.lr*opt.D2G_lr, betas=(opt.beta1, opt.beta2))

    # Create the progress bar
    print("-- BEGIN PRETRAIN --")
    pbar_epochs = tqdm(total=opt.discriminator_epochs)
    pbar_epochs.n = resume_epoch
    pbar_epochs.set_description("TRAINED DISCRIMINATOR EPOCHS | GLCIC [ORIGINAL]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(resume_epoch,opt.discriminator_epochs+1):
      for i, data in enumerate(dataloader, 0):
        print("\n[Fold %s | Epoch %d | Batch %d/%d]"%(fold,epoch,i+1,len(dataloader)-1))
        # Extract input_real, input_cropped, and masks directly from the data tensor
        data_gpu=data.to(device)
        input_real = data_gpu[:, :1, :, :].float()
        masks = data_gpu[:, 1:2, :, :].float()
        input_cropped = ((1 - masks) * input_real + opt.mask * masks).float()
        crop_masks=processor.local_crop(masks,opt)

        # Gradient Aplication
        input_real.requires_grad = True
        input_cropped.requires_grad = True
        masks.requires_grad = True
        crop_masks.requires_grad = True

        # Zero Gradient (each iteration)
        netD.zero_grad()

        fake = netG(input_cropped,masks)
        output_fake = netD(processor.crop(fake, crop_masks).detach(), fake.detach())
        output_real = netD(processor.crop(input_real, crop_masks),input_real)
        loss = adversarialLoss.adv_loss(opt,"netD",output_fake,output_real)
        loss.backward()
        netD_opt.step()

        print('Discriminator Training Loss: {:.5f}'.format(loss))

      # Do checkpointing
      torch.save({'epoch':pbar_epochs.n,
                'state_dict':netD.state_dict()},
                'models/'+opt.network+'/'+opt.specificity+'/phase2/fold%s_netD.pth'%(fold))

      pbar_epochs.update()
    pbar_epochs.close()


  def train(self, fold, dataloader,opt):
    # Initial epoch
    resume_epoch=1

    # Call the Generator
    netG = _netG(opt)

    # If exist the file of the pre-trained network
    try:
      netG.load_state_dict(torch.load('models/'+opt.network+'/'+opt.specificity+'/fold%s_netG.pth'%(fold),map_location=lambda storage, location: storage)['state_dict'])
      resume_epoch = torch.load(opt.netG)['epoch']
    except Exception :
      print("GENERATOR MODEL NOT DETECTED!")
      netG.apply(self.weights_init)

    # Call the Discriminator
    netD = _netD(opt)
    netD.apply(self.weights_init)

    try:
      netD.load_state_dict(torch.load('models/'+opt.network+'/'+opt.specificity+'/fold%s_netD.pth'%(fold),map_location=lambda storage, location: storage)['state_dict'])
      resume_epoch = torch.load(opt.netD)['epoch']
    except Exception :
      print("DISCRIMINATOR MODEL NOTE DETECTED!")
      netD.apply(self.weights_init)

    # Used Losses
    criterionMSE = nn.MSELoss()

    # GAN Management
    if torch.cuda.is_available():
      netD.cuda()
      netG.cuda()
      criterionMSE.cuda()

    # Setup optimizer ADAM (net_parameter, learning rate, betas)
    netD_opt = optim.Adam(netD.parameters(), lr=opt.lr*opt.D2G_lr, betas=(opt.beta1, opt.beta2))
    netG_opt = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    # Create the progress bar
    print("-- BEGIN TRAIN --")
    pbar_epochs = tqdm(total=opt.epochs)
    pbar_epochs.n = resume_epoch
    pbar_epochs.set_description("TRAINED EPOCHS | GLCIC [ORIGINAL]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(resume_epoch,opt.epochs+1):
      for i, data in enumerate(dataloader, 0):
        print("\n[Fold %s | Epoch %d | Batch %d/%d]"%(fold,epoch,i+1,len(dataloader)-1))
        # Extract input_real, input_cropped, and masks directly from the data tensor
        data_gpu=data.to(device)
        input_real = data_gpu[:, :1, :, :].float()
        masks = data_gpu[:, 1:2, :, :].float()
        masks=(masks==1).float()
        input_cropped = ((1 - masks) * input_real + opt.mask * masks).float()
        crop_masks=processor.local_crop(masks,opt)

        # Gradient Aplication
        input_real.requires_grad = True
        input_cropped.requires_grad = True
        masks.requires_grad = True
        crop_masks.requires_grad = True

        # forward model_cn
        netG.zero_grad()
        fake = netG(input_cropped,masks)
        output = netD(processor.crop(fake, crop_masks), fake)
        errG_D = adversarialLoss.adv_loss(opt,"netG",output)
        errG_l2 = criterionMSE(processor.crop(fake, crop_masks),processor.crop(input_real, crop_masks))
        errG = opt.wtlD * errG_D + opt.wtl2 * errG_l2
        # backward model_cn
        errG.backward()
        # optimize
        netG_opt.step()

        # Train D with real image
        netD.zero_grad()
        output_real = netD(processor.crop(input_real, crop_masks), input_real)
        D_x = output_real.data.mean()

        # Train D with fake image
        output_fake = netD(processor.crop(fake, crop_masks).detach(), fake.detach())
        D_G_z = output_fake.data.mean()
        errD = adversarialLoss.adv_loss(opt,"netD",output_fake,output_real)
        errD.backward()
        netD_opt.step()

        # Print of step training information
        print('Generator train loss: %.5f \nDiscriminator train loss: %.5f \nD(x): %.5f \nD(G(z)): %.5f'%(errG.item(), errD.item(), D_x,D_G_z))
        # Save results if i=0 (in my dataset)
        # if pbar_epochs.n==1 or pbar_epochs.n==opt.epochs:
        #   save_image(fake.data,'debug/'+opt.network+'/'+opt.specificity+'/fold_%s_epoch_%d.png'%(fold,epoch),normalize=True)

      # Do checkpointing
      torch.save({'epoch':pbar_epochs.n,
                'state_dict':netD.state_dict()},
                'models/'+opt.network+'/'+opt.specificity+'/fold%s_netD.pth'%(fold))
      torch.save({'epoch':pbar_epochs.n,
                'state_dict':netG.state_dict()},
                'models/'+opt.network+'/'+opt.specificity+'/fold%s_netG.pth'%(fold))

      pbar_epochs.update()
    pbar_epochs.close()
    print("-- END TRAIN --")


  def test(self,fold, dataloader, dataloader_labels, dataloader_tumoral_labels,dataloader_pulmonar_label, opt):
    # Load of generator network already trained
    netG = _netG(opt)
    netG.load_state_dict(torch.load('models/'+opt.network+'/'+opt.specificity+'/fold%s_netG.pth'%(fold),map_location=lambda storage, location: storage)['state_dict'])
    netG=netG.eval()

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
        if i==0:
          print("-- BEGIN TEST --")

        data_gpu=data.to(device)
        input_real = data_gpu[:, :1, :, :].float()
        masks = data_gpu[:, 1:2, :, :].float()
        masks = (masks==1).float()
        input_cropped = ((1 - masks) * input_real + opt.mask * masks).float()

        fake = netG(input_cropped,masks)
        tumor_label=tumor_label.cuda()
        pulmonar_label=pulmonar_label.cuda()

        #masks=(masks == 1).float()
        #input_cropped = (1 - masks) * input_real + int(opt.mask) * masks
        #fake = (1 - masks) * input_cropped + fake * masks

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

