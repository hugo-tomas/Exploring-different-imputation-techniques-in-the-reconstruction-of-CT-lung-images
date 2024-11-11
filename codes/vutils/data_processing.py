from vutils.libraries import *
class data_processing:
  def __init__(self):
    pass

  ### .NII Images
  def load_nii(self,path):
    # volume (xDim,yDim,n_slices)
    volume = nib.load(path).get_fdata()
    return volume

  # Load all .nii files inside data dataroot
  def load_label(self,opt):
    nii_file_list = os.listdir(opt.dataroot_labels)
    dataset = [[] for _ in range(len(nii_file_list))]
    for id,nii in enumerate(nii_file_list):
      nii_path=opt.dataroot+"/"+nii
      volume=self.load_nii(nii_path)
      dataset[id]=self.list_volume(volume,opt.imageRot,opt.imageHouns,opt)
    return dataset

  # Load all .nii files inside data dataroot
  def load_lung(self,opt):
    file_list = os.listdir(opt.dataroot)
    all_images = defaultdict(list)
    for id,filename in enumerate(file_list):
      file_path = os.path.join(opt.dataroot, filename)
      label_path = os.path.join(opt.dataroot_labels, filename)
      print("> LOAD IMAGE: "+file_path)

      volume = filename.split(".")[0]+".nii"
      img=mpimg.imread(file_path)[:,:,0]
      label=mpimg.imread(label_path)[:,:,0]
      threshold=(-300/opt.imageHouns[0])-(opt.imageHouns[1]/opt.imageHouns[0])
      smoothed_image = cv2.GaussianBlur(img, (5, 5), 0)
      _, binarized_image = cv2.threshold(smoothed_image, threshold, 1, cv2.THRESH_BINARY)

      kernel = np.ones((5, 5), np.uint8)
      filled_image = cv2.dilate(1-binarized_image, kernel, iterations=1).astype(np.uint8)
      _ , labeled_image = cv2.connectedComponents((filled_image).astype(np.uint8))

      histogram, _ = np.histogram(labeled_image.flatten(), bins=np.arange(0, np.max(labeled_image) + 2))
      histogram = histogram.flatten()
      histogram[labeled_image[0,0]]=0 #Remove the background
      histogram[labeled_image[len(labeled_image)-1,len(labeled_image)-1]]=0 #Remove the label of [0,0] and [512,512] pixels

      lung_mask = np.zeros_like(labeled_image)
      lung_mask[np.isin(labeled_image, np.where(histogram != 0)[0]) * np.isin(labeled_image, np.where(histogram > 5000)[0])] = 1
      contours, _ = cv2.findContours((filled_image-(1-lung_mask.astype(np.uint8))).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      mask = np.zeros_like(labeled_image)
      
      for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        mask[y:y + h,x:x + w]=1

      info=torch.cat((torch.tensor(img).unsqueeze(0),torch.tensor(filled_image-(1-lung_mask)).unsqueeze(0),torch.tensor(mask).unsqueeze(0),torch.tensor(label).unsqueeze(0)),dim=0)
      all_images[volume].append(info)

    dataset=list(all_images.values())
    return dataset #[[tensor3D,tensor3D,tensor3D,...],[], ... , []] -> tensor3D=Slice Norm;Lung mask; ROI


  ### General
  def list_volume(self,volume,rot,window,opt):
    dataset = [[] for _ in range(volume.shape[2])]
    opt.normalization_mean=(opt.normalization[0])/(opt.normalization[0]-opt.normalization[1])
    opt.normalization_std=-1/(opt.normalization[0]-opt.normalization[1])

    # Dataset Tranformation (Resized and Normalized)
    transform = transforms.Compose([transforms.Resize(opt.imageSize,antialias=True),
                                      transforms.Normalize(opt.normalization_mean, opt.normalization_std),
                                      transforms.CenterCrop(opt.imageSize)])

    for i in range(volume.shape[2]):
      slice_img = rotate(volume[:,:,i], rot)
      if len(window)!=0:
        slice_img=self.houns2intensity(slice_img,window)

      dataset[i]=transform(torch.tensor(slice_img).unsqueeze(0))
    return dataset

  # Crop one data dimension
  def compact_dataset(self,dataset):
    sum=0
    for volume in dataset:
      sum=sum+len(volume)
    new_dataset = [[] for _ in range(sum)]
    id=0
    for i in range(len(dataset)):
      for j in range(len(dataset[i])):
        new_dataset[id]=dataset[i][j]
        id=id+1
    return new_dataset

  def houns2intensity(self,slice_img,window):
    new_slice_img = slice_img/window[0]-window[1]/window[0]
    new_slice_img[new_slice_img<0] = 0
    new_slice_img[new_slice_img>1] = 1
    new_slice_img=new_slice_img
    return new_slice_img

  def mean_pixel_value(self,dataset):
    # Only one channel
    mpv = np.zeros(shape=(1,))
    for img in dataset:
      x = np.array(img)
      mpv += x.mean(axis=(1, 2))
    mpv /= len(dataset)
    return mpv

  def renormalization(self,before,after,images):
    renormalization_std=(before[1]-before[0])/(after[1]-after[0])
    renormalization_mean=-after[0]*renormalization_std+before[0]
    pixelScale = transforms.Compose([transforms.Normalize(mean=renormalization_mean, std=renormalization_std)])
    for i in range(images.shape[0]):
      images[i,:,:,:]=pixelScale(images[i,:,:,:])
    return images



  ### Masks
  def create_mask(self,dataset,opt):
    all_masks = [None] * len(dataset)
    new_dataset = [None] * len(dataset)
    for volume in range(len(dataset)):
      print("CREATE MASKS - Volume %s"%volume)
      mask_list = []
      volume_list=[]
      for slice in range(len(dataset[volume])):
        mask = torch.zeros(1, dataset[volume][slice].shape[1], dataset[volume][slice].shape[2])
        total_area=dataset[volume][slice][1,:,:].sum()
        mask_area=(total_area*opt.mask_percentage).numpy()

        if opt.maskType=="Square" and mask_area>0:
          dim1=round(np.sqrt(mask_area))
          dim2=round(mask_area/dim1)
          while dim1 * dim2 < mask_area:
            dim1 += 1
            dim2 += 1

          # Find the indices of valid pixels in the larger mask (where the value is 1)
          valid_indices = dataset[volume][slice][1,:,:].nonzero()
          random.shuffle(valid_indices)
          random_element = valid_indices[random.randint(0, len(valid_indices) - 1)]
          x, y = random_element[0], random_element[1]

          while (((dataset[volume][slice][1, y-dim2//2:y+dim2//2, x-dim1//2:x+dim1//2].sum())<0.6*dim2*dim1) or ((torch.all(dataset[volume][slice][2, y-dim2//2:y+dim2//2, x-dim1//2:x+dim1//2] == 1))==False)):
            eliminate_bool = torch.all(valid_indices == random_element, dim=1)
            valid_indices = valid_indices[~eliminate_bool]
            if len(valid_indices)==0:
              remove==True
              break
            else:
              remove=False
              random_element = valid_indices[random.randint(0, len(valid_indices) - 1)]
              x, y = random_element[1], random_element[0]

          if remove==False:
            mask[:, y-dim2//2:y+dim2//2, x-dim1//2:x+dim1//2]=1
            volume_list.append(dataset[volume][slice])
            mask_list.append(mask)

        elif opt.maskType=="Random" and mask_area>0:
          valid_indices = dataset[volume][slice][1,:,:].nonzero()
          num_valid_indices = len(valid_indices)
          random_indices = np.random.choice(num_valid_indices, int(mask_area))
          for indexes in random_indices:
            select_pixel=valid_indices[indexes]
            mask[:,select_pixel[0],select_pixel[1]] = 1
            volume_list.append(slice)
            mask_list.append(mask)

      all_masks[volume]=mask_list
      new_dataset[volume]=volume_list
    return new_dataset, all_masks



  ### Data Manipulation
  def crop(self, input, masks):
    cropped_inputs = []
    for i, mask in enumerate(masks, 0):
        nonzero_indices = mask.nonzero()
        ymin = nonzero_indices[:, 2].min().item()
        xmin = nonzero_indices[:, 1].min().item()
        ymax = nonzero_indices[:, 2].max().item()
        xmax = nonzero_indices[:, 1].max().item()

        cropped_input = input[i, :, xmin:xmax, ymin:ymax].clone()  # Create a clone of the cropped region

        cropped_inputs.append(cropped_input)

    cropped_inputs = torch.stack(cropped_inputs)  # Convert list of tensors to a single tensor
    if torch.cuda.is_available():
      cropped_inputs=cropped_inputs.cuda()
    return cropped_inputs

  def local_crop(self,masks,opt):
    cropped_inputs = []
    half=round(np.sqrt(opt.mask_percentage*(opt.imageSize*opt.imageSize))//2)
    for i, mask in enumerate(masks, 0):
      nonzero_indices = mask.nonzero()
      ymin = nonzero_indices[:, 2].min().item()
      xmin = nonzero_indices[:, 1].min().item()
      ymax = nonzero_indices[:, 2].max().item()
      xmax = nonzero_indices[:, 1].max().item()
      center_x=(xmin+xmax)//2
      center_y=(ymin+ymax)//2

      local=torch.zeros_like(mask)

      if center_x-half<0:
        xi=0
        xf=center_x+half-(center_x-half)
      elif center_x+half>opt.imageSize:
        xi=center_x-half-(center_x+half-opt.imageSize)
        xf=opt.imageSize
      else:
        xi=center_x-half
        xf=center_x+half
      #
      if center_y-half<0:
        yi=0
        yf=center_y+half-(center_y-half)
      elif center_y+half>opt.imageSize:
        yi=center_y-half-(center_y+half-opt.imageSize)
        yf=opt.imageSize
      else:
        yi=center_y-half
        yf=center_y+half

      local[:,xi:xf,yi:yf]=1
      cropped_inputs.append(local)

    cropped_inputs = torch.stack(cropped_inputs)
    if torch.cuda.is_available():
      cropped_inputs=cropped_inputs.cuda()
    return cropped_inputs

  def create_edges(self, img, sigma=2):
      canny_edges_batch = torch.empty_like(img)
      for i in range(img.shape[0]):
          for j in range(img.shape[1]):
              # Convert tensor to numpy array
              image_array = img[i, j].detach().cpu().numpy()
              # Apply Canny edge detection
              edges = canny(image_array, sigma=sigma)
              # Convert numpy array back to tensor and move to GPU
              canny_edges_batch[i, j] = torch.from_numpy(edges).to(img.device)
      return canny_edges_batch


  ### Top Similar Images
  def extract_features(self, type, image, mask, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = []

    if type == "MSE":
        for img in dataset:
            _, mse_val = mae_mse(img[0, :, :] * (1 - mask).numpy(), image[0, :, :].numpy())
            metrics.append(mse_val)

    elif type == "SSIM":
        for img in dataset:
            ssim_val = ssim(img[0, :, :] * (1 - mask).numpy(), image[0, :, :].numpy())
            metrics.append(ssim_val)

    elif type == "RESNET18":
        # Load pre-trained ResNet model and move it to GPU
        resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True).to(device)
        resnet.eval()

        features = []
        for img in dataset:
            img.unsqueeze(0)
            img3ch = torch.cat((img * (1 - mask), img * (1 - mask), img * (1 - mask)), dim=0).to(device)
            with torch.no_grad():
                feature = resnet(img3ch.unsqueeze(0).to(device))
                features.append(feature)

        metrics = [item for sublist in self.compute_similarity(features, resnet(
            torch.cat((image, image, image), dim=0).unsqueeze(0).to(device))) for item in sublist]

    elif type == "VGG19":
        vgg = models.vgg19(pretrained=True).to(device)
        vgg.eval()

        features = []
        for img in dataset:
            img3ch = torch.cat((img * (1 - mask), img * (1 - mask), img * (1 - mask)), dim=0).to(device)
            with torch.no_grad():
                feature = vgg(img3ch.unsqueeze(0))
                features.append(feature)

        metrics = [item for sublist in self.compute_similarity(features, vgg(
            torch.cat((image, image, image), dim=0).unsqueeze(0))) for item in sublist]

    return metrics

  def compute_similarity(self,query_features, database_features):
     # Flatten each tensor and convert them to NumPy arrays
     query_features_np = [tensor.cpu().view(-1).detach().numpy() for tensor in query_features]
     database_features_np = [tensor.cpu().view(-1).detach().numpy() for tensor in database_features]

     # Convert lists of flattened arrays to 2D NumPy arrays
     query_features_np = np.stack(query_features_np, axis=0)
     database_features_np = np.stack(database_features_np, axis=0)

     similarity_matrix = cosine_similarity(query_features_np, database_features_np)
     return similarity_matrix

  # Example code to retrieve top similar images
  def retrieve_top_similar_images(self, metrics, image, mask, database_images, num_similar_images=5):
     # Get indices of top similar images
     descending_indices = np.argsort(metrics, axis=0)[::-1]
     top_indices = descending_indices[:num_similar_images+1]
     # Create an empty array to store the top similar images
     top_similar_images = [None]*len(top_indices)

     # Fill the array with the similar images
     for idx,indices_row in enumerate(top_indices):
       similar_images = database_images[indices_row].unsqueeze(0)
       top_similar_images[idx]=similar_images

     if len(top_similar_images)==num_similar_images+1:
       top_similar_images=top_similar_images[1:]

     image_list=[image[:1,:,:]]+top_similar_images
     grid_image = make_grid(image_list, nrow=6, normalize=True)
     save_image(grid_image, '/kaggle/working/test.png', normalize=False)

     return top_similar_images



class TSmooth():
  def __init__(self):
    pass

  def tsmooth(self, img, lambda_=0.05, sigma=5, sharpness=0.02, maxIter=4):
    tsmooth_batch = torch.empty((img.shape[0], img.shape[1], img.shape[2], img.shape[3]))

    for k in range(img.shape[0]):
      I = (img[k, 0].unsqueeze(-1)).expand(img.shape[2], img.shape[3], 3)

      sigma_iter = sigma
      lambda_ = lambda_ / 2.0
      dec = 2.0
      for _ in range(maxIter):
        wx, wy = self.computeTextureWeights(I, sigma_iter, sharpness)
        x = self.solveLinearEquation(I, wx, wy, lambda_)
        sigma_iter /= dec
        if sigma_iter < 0.5:
          sigma_iter = 0.5

      tsmooth_batch[k, 0] = x[:, :, 0]

    return tsmooth_batch.cuda() if torch.cuda.is_available() else tsmooth_batch

  def computeTextureWeights(self, fin, sigma, sharpness):
    fx = torch.diff(fin, dim=1)
    fx = F.pad(fx, (0, 0, 0, 1, 0, 0), mode='constant')
    fy = torch.diff(fin, dim=0)
    fy = F.pad(fy, (0, 0, 0, 0, 0, 1), mode='constant')
    vareps_s = vareps = torch.tensor(sharpness)
    vareps = torch.tensor(0.001)
    wto = torch.maximum(torch.sum(torch.sqrt(fx**2 + fy**2), dim=2) / fin.shape[2], vareps_s)**(-1)

    fbin = self.lpfilter(fin, sigma)

    gfx = torch.diff(fbin, dim=1)
    gfx = F.pad(gfx, (0, 0, 0, 1, 0, 0), mode='constant')
    gfy = torch.diff(fbin, dim=0)
    gfy = F.pad(gfy, (0, 0, 0, 0, 0, 1), mode='constant')

    wtbx = torch.maximum(torch.sum(torch.abs(gfx), dim=2) / fin.shape[2], vareps)**(-1)
    wtby = torch.maximum(torch.sum(torch.abs(gfy), dim=2) / fin.shape[2], vareps)**(-1)

    retx = wtbx * wto
    rety = wtby * wto

    retx[:, -1] = 0
    rety[-1, :] = 0

    return retx, rety

  def fspecial_gaussian(self, shape, sigma):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = torch.meshgrid(torch.arange(-m, m + 1), torch.arange(-n, n + 1), indexing='xy')
    h = torch.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
      h /= sumh
    return h

  def conv2(self, x, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x=x.to(device)
    y=y.to(device)
    padding_height = (y.size(-2) - 1) // 2
    padding_width = (y.size(-1) - 1) // 2
    output = F.conv2d(x.unsqueeze(0).unsqueeze(0), y.unsqueeze(0).unsqueeze(0), padding=(padding_height, padding_width)).squeeze()
    return output

  def conv2_sep(self, im, sigma):
    ksize = torch.tensor(int(5 * sigma))
    ksize = ksize if ksize % 2 != 0 else ksize + 1  # Ensure ksize is odd
    g = self.fspecial_gaussian((1, ksize), sigma)
    ret = self.conv2(im, g)
    ret = self.conv2(im, g.T)
    return ret

  def lpfilter(self, FImg, sigma):
    FBImg = torch.zeros_like(FImg)
    for ic in range(FImg.shape[2]):
      FBImg[:, :, ic] = self.conv2_sep(FImg[:, :, ic], sigma)
      return FBImg

  def solveLinearEquation(self,IN, wx, wy, lambda_):
    r, c, ch = IN.shape
    k = r * c
    dx = -lambda_ * wx.view(-1)
    dy = -lambda_ * wy.view(-1)
    dx=dx.cpu()
    dy=dy.cpu()
    B = np.column_stack((dx, dy))
    A = spdiags(B.T, [-r, -1], k, k)

    e = dx
    w = np.pad(dx, (r, 0), mode='constant')[:-r]
    s = dy
    n = np.pad(dy, (1, 0), mode='constant')[:-1]
    D = 1 - (e + w + s + n)
    A = A + A.T + spdiags(D, 0, k, k)

    OUT = torch.zeros_like(IN)
    for ii in range(ch):
      tin = IN[:, :, ii].cpu().view(-1)
      tout, _ = cg(A, tin, tol=0.01, maxiter=1000)
      OUT[:, :, ii] = torch.tensor(tout.reshape(r, c))

    return OUT


