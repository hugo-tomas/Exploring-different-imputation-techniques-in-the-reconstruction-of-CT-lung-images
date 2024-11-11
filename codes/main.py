# Change directory to where your files are located
import os
input_dir = '/home/cruncher/Documents/Hugo/codes/'
os.chdir(input_dir)

## Import External Classes
# Libraries import
from vutils.libraries import *
import vutils.losses
import vutils.data_processing
import options_configuration

# Objects initialization
processor = vutils.data_processing.data_processing()
config=options_configuration.options_configuration()
opt=config.__opt__()

# GPU Device initialization
torch.cuda.set_device(opt.gpu)

## Create if directories don't exist
try:
  os.makedirs("models")
  os.makedirs("test")
  os.makedirs("debug")
  os.makedirs("dataset/processed/")
except OSError:
  pass

## Define the Model's State
try:
  # If model exists
  opt=config.__read__('models/'+opt.network+'/'+opt.specificity+'/opt.pkl')
  state=1

except OSError:
  # If model doesn't exist
  os.makedirs("models/"+opt.network+'/'+opt.specificity)
  os.makedirs("test/"+opt.network+'/'+opt.specificity)
  os.makedirs("debug/"+opt.network+'/'+opt.specificity)
  state=0
  pass

# Warning if GPU is available and not used
if torch.cuda.is_available():
  print("[Warning] You have a CUDA device, so you should probably run with GPU mode")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########################################################################################################################################

# Never trained before
if state==0:
  config.__write__(opt)

  # Seed Generation (in order to mantain the random values if we want to run again the code)
  # random.seed(opt.manualSeed)
  torch.manual_seed(opt.manualSeed)

  # Cuda otimization (integration of GPU)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.manualSeed)

  # Fixed input size leads to a betters performances
  cudnn.benchmark = True

  # Load the processed data
  try:
    all_volumes=torch.load('dataset/processed/volumes_'+str(opt.normalization)+'_'+str(opt.imageHouns)+'.pth')

  except OSError:
    # Load of Dataset
    if opt.datasetType=='nii':
      all_volumes=processor.load_lung(opt)
      all_volumes,all_masks=processor.create_mask(all_volumes,opt)

      # Save Datasets
      torch.save(all_volumes, 'dataset/processed/volumes_'+str(opt.normalization)+'_'+str(opt.imageHouns)+'.pth')
      torch.save(all_masks, 'dataset/processed/masks_'+str(opt.maskType)+'_'+str(opt.mask_percentage)+'.pth')
    pass

  try:
    # Load Masks
    all_masks = torch.load('dataset/processed/masks_'+str(opt.maskType)+'_'+str(opt.mask_percentage)+'.pth')

  except OSError:
    # Create Masks
    _ , all_masks=processor.create_mask(all_volumes,opt)

    # Save Datasets
    torch.save(all_masks, 'dataset/processed/masks_'+str(opt.maskType)+'_'+str(opt.mask_percentage)+'.pth')

## Continue the train of an existent model
elif state==1:

  # Seed Generation
  # random.seed(opt.manualSeed)
  torch.manual_seed(opt.manualSeed)

  # Load Train Data
  try:
    all_volumes=torch.load('dataset/processed/volumes_'+str(opt.normalization)+'_'+str(opt.imageHouns)+'.pth')
    all_masks = torch.load('dataset/processed/masks_'+str(opt.maskType)+'_'+str(opt.mask_percentage)+'.pth')

  except OSError:
    print("[Error]: Train Data with this specifications doesn't exist!")
    sys.exit()

else:
  print("[Error]: Something wrong happend!")
  sys.exit()

#################################
data_type=2 # 0 - normal inpainting methods; 1 - edges and strucures; 2 - most similar images (Self-proposed approach not available at the moment)
if data_type==1:
  try:
    all_volumes=torch.load('dataset/processed/volumes_'+str(opt.normalization)+'_'+str(opt.imageHouns)+'_with_edges_and_structure.pth')

  except OSError:
    TSmooth = vutils.data_processing.TSmooth()
    for i, volume in enumerate(all_volumes):
      print("Volume %s"%i)
      for j,image in enumerate(volume):
        print("Image %s"%j)
        all_volumes[i][j]=torch.cat((all_volumes[i][j][:1,:,:].to(device),processor.create_edges(all_volumes[i][j][:1,:,:].unsqueeze(0))[0,:,:,:].to(device),TSmooth.tsmooth(all_volumes[i][j][:1,:,:].unsqueeze(0))[0,:,:,:].to(device)),dim=0)
    torch.save(all_volumes, 'dataset/processed/volumes_'+str(opt.normalization)+'_'+str(opt.imageHouns)+'_with_edges_and_structure.pth')

if data_type==2:
  try:
    all_volumes=torch.load('dataset/processed/volumes_'+str(opt.normalization)+'_'+str(opt.imageHouns)+'_masks_'+str(opt.maskType)+'_'+str(opt.mask_percentage)+".pth")

  except OSError:
    opt.nslices=5
    new_all_volumes=[None]*len(all_volumes)
    for k,(img_list,msk_list) in enumerate(zip(all_volumes,all_masks)):
      new_img_list=[None]*len(img_list)
      print("Volume "+str(k))
      for j,(img,msk) in enumerate(zip(img_list,msk_list)):
        print("Image "+str(j))
        metrics = processor.extract_features("RESNET18", (img[:1,:,:] * (1 - msk)), msk, train_data)
        top_similar_images=processor.retrieve_top_similar_images(metrics, img, msk, train_data, num_similar_images=opt.nslices)
        new_img_list[j]=torch.cat((img[:1,:,:].to(device),torch.cat(top_similar_images, dim=0).to(device)),dim=0)
      new_all_volumes[k]=new_img_list
    all_volumes=new_all_volumes
    torch.save(new_all_volumes, 'dataset/processed/volumes_'+str(opt.normalization)+'_'+str(opt.imageHouns)+'_masks_'+str(opt.maskType)+'_'+str(opt.mask_percentage)+".pth")

#################################

# 10 K-Cross-Validation
indexes = np.arange(len(all_volumes))
np.random.seed(opt.manualSeed)
np.random.shuffle(indexes)
torch.save(indexes, 'test/'+opt.network+'/'+opt.specificity+'/indexes.pth')

k=10 # K fold
fold_size = len(all_volumes) // k
fold_indexes = []
test_results=[None]*k
test_order=[None]*k

# Mean and Standard Deviation of Normalization (From [0,1] to [opt.normalization[0],opt.normalization[1]])
opt.normalization_mean=(opt.normalization[0])/(opt.normalization[0]-opt.normalization[1])
opt.normalization_std=-1/(opt.normalization[0]-opt.normalization[1])

# Dataset Tranformation (Resized and Normalized)
transform = transforms.Compose([transforms.Resize(opt.imageSize,antialias=True),
                                      transforms.Normalize(opt.normalization_mean, opt.normalization_std),
                                      transforms.CenterCrop(opt.imageSize)])

# Divide os índices em folds
for fold in range(k):
  start = fold * fold_size
  end = (fold + 1) * fold_size if fold < k - 1 else len(all_volumes)
  test_indexes = indexes[start:end]
  train_indexes = np.concatenate([indexes[:start], indexes[end:]])

  test_order[fold]=test_indexes
  torch.save(test_order, 'test/'+opt.network+'/'+opt.specificity+'/test_order.pth')

  test_data=processor.compact_dataset([volume for i, volume in enumerate(all_volumes) if i in test_indexes])
  train_data=processor.compact_dataset([volume for i, volume in enumerate(all_volumes) if i not in test_indexes])

  test_masks=processor.compact_dataset([volume for i, volume in enumerate(all_masks) if i in test_indexes])
  train_masks=processor.compact_dataset([volume for i, volume in enumerate(all_masks) if i not in test_indexes])

  # Turn the mask value equal to the mean pixel value
  if str(opt.mask)=="mpv":
      mpv=processor.mean_pixel_value(train_data)
      opt.mask=mpv[0]

  opt.mask=float(opt.mask)

  # Creation of a dataloader (batch of shuffle images to train)
  # Concatenate only the Original Image with the Missing Mask
  if data_type==1 or data_type==2:
    concatenated_data = [transform(torch.cat([train_data[i].to(device), train_masks[i].to(device)], dim=0)) for i in range(len(train_masks))]
    dataloader = torch.utils.data.DataLoader(concatenated_data, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))
  else:
    concatenated_data = [transform(torch.cat([train_data[i][:1,:,:], train_masks[i]], dim=0)) for i in range(len(train_masks))]
    dataloader = torch.utils.data.DataLoader(concatenated_data, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

  if data_type==1 or data_type==2:
    # Concatenate only the Original Image with the Missing Mask
    concatenated_data_test = [transform(torch.cat([test_data[i].to(device), test_masks[i].to(device)], dim=0)) for i in range(len(test_masks))]
    dataloader_test = torch.utils.data.DataLoader(concatenated_data_test, batch_size=1,
                                          shuffle=False, num_workers=int(opt.workers))
  else:
    concatenated_data_test = [transform(torch.cat([transform(test_data[i][:1,:,:]), transform(test_masks[i])], dim=0)) for i in range(len(test_masks))]
    dataloader_test = torch.utils.data.DataLoader(concatenated_data_test, batch_size=1,
                                          shuffle=False, num_workers=int(opt.workers))

  if opt.network=='CE_Modified_GlobalDiscriminator':
    # Hyper-parameters
    opt.beta1=0.5
    opt.beta2=0.999
    opt.lr=0.0001
    opt.D2G_lr=0.1
    opt.wtlD=0.001
    opt.wtl2=0.999
    config.__write__(opt)

    from frameworks.CE.ce_modified_globalDiscriminator import context_encoder
    CE=context_encoder()
    CE.train(fold,dataloader,opt)
    print("\n- FOLD %s | MODEL FINISH THE TRAIN | %d epochs!"%(fold, opt.epochs))
    metrics_list=CE.test(fold,dataloader_test,opt)
    test_results.append(metrics_list)
    print("\n- FOLD %s | MODEL FINISH THE TEST | %d IMAGES!"%(fold, len(test_data)))

  elif opt.network=='CE_Modified_LocalDiscriminator':
    # Options
    opt.beta1=0
    opt.beta2=0.9
    opt.lr=0.0001
    opt.D2G_lr=0.1
    opt.wtlD=0.001
    opt.wtl2=0.999
    config.__write__(opt)

    from frameworks.CE.ce_modified_localDiscriminator import context_encoder
    CE=context_encoder()
    CE.train(fold,dataloader,opt)
    print("\n- FOLD %s | MODEL FINISH THE TRAIN | %d epochs!"%(fold, opt.epochs))
    metrics_list=CE.test(fold,dataloader_test,opt)
    test_results.append(metrics_list)
    print("\n- FOLD %s | MODEL FINISH THE TEST | %d IMAGES!"%(fold, len(test_data)))

  elif opt.network=='GLCIC_Original':

    # Options
    opt.beta1=0.5
    opt.beta2=0.999
    opt.lr=0.0001
    opt.D2G_lr=0.01
    opt.wtlD=0.0004
    opt.wtl2=1-opt.wtlD

    config.__description__()
    from frameworks.GLCIC.glcic_original import globally_locally_consistent
    GLCIC=globally_locally_consistent()
    # GLCIC.train_generator(fold,dataloader,opt)
    # print("\n- FOLD %s | GENERATOR MODEL FINISH THE TRAIN | %d epochs!"%(fold,opt.generator_epochs))
    # GLCIC.train_discriminator(fold,dataloader,opt)
    # print("\n- FOLD %s | DISCRIMINATOR MODEL FINISH THE TRAIN | %d epochs!"%(fold,opt.discriminator_epochs))
    GLCIC.train(fold,dataloader,opt)
    print("\n- FOLD %s | MODEL FINISH THE TRAIN | %d epochs!"%(fold, opt.epochs))
    metrics_list=GLCIC.test(fold,dataloader_test,opt)
    test_results.append(metrics_list)
    print("\n- FOLD %s | MODEL FINISH THE TEST | %d IMAGES!"%(fold, len(test_data)))


  elif opt.network=='EC_Original':
    from frameworks.EC.ec_original import edge_connected
    try:
      os.makedirs("debug/"+opt.network+'/'+opt.specificity+'/edges')
      os.makedirs("models/"+opt.network+'/'+opt.specificity+'/edges')
      os.makedirs("test/"+opt.network+'/'+opt.specificity+'/edges')
    except OSError:
      pass

    #Options
    opt.lr=0.0001
    opt.D2G_lr=0.1
    opt.beta1=0
    opt.beta2=0.9
    opt.wfm=10
    opt.wgadv=1

    EC=edge_connected()
    EC.train_edges(fold,dataloader,opt)
    print("\n- FOLD %s | EDGE MODEL FINISH THE TRAIN | %d epochs!"%(fold,opt.edge_epochs))

    #Options
    opt.lr=0.0001
    opt.D2G_lr=0.1
    opt.beta1=0
    opt.beta2=0.9
    opt.wtl1=1
    opt.wgadv=0.1
    opt.wperc=0.1
    opt.wstyle=250

    EC.train(fold,dataloader,opt)
    print("\n- FOLD %s | INPAINTING MODEL FINISH THE TRAIN | %d epochs!"%(fold,opt.epochs))
    metrics_list=EC.test(fold,dataloader_test,"inpainting",opt)
    test_results.append(metrics_list)
    print("\n- FOLD %s | MODEL FINISH THE TEST | %d IMAGES!"%(fold, len(test_data)))

  elif opt.network=='ESMII_Original':
    from frameworks.ESMII.esmii import esmii

    #Options
    opt.lr=0.0002
    opt.D2G_lr=0.1
    opt.beta1=0.5
    opt.beta2=0.999
    opt.wfm=10
    opt.wtl1=50
    opt.wgadv=0.1
    opt.wperc=0.05
    opt.wstyle=150

    ESMII=esmii()
    ESMII.train(fold,dataloader,opt)
    print("\n- FOLD %s | MODEL FINISH THE TRAIN | %d epochs!"%(fold,opt.epochs))
    metrics_list=ESMII.test(fold,dataloader_test,opt)
    test_results.append(metrics_list)
    print("\n- FOLD %s | MODEL FINISH THE TEST | %d IMAGES!"%(fold, len(test_data)))

  elif opt.network=='Multi_Slices': #(Self-proposed approach not available at the moment)
    #Options
    opt.lr=0.0001
    opt.D2G_lr=0.01
    opt.beta1=0.5
    opt.beta2=0.999
    opt.wfm=10
    opt.wtl1=50
    opt.wgadv=0.1
    opt.wperc=0.05
    opt.wstyle=150

    from frameworks.multi_slices import multi_slices
    MS=multi_slices()
    MS.train(fold, dataloader, opt)
    print("\n- FOLD %s | MODEL FINISH THE TRAIN | %d epochs!"%(fold,opt.epochs))
    metrics_list=MS.test(fold,dataloader_test,opt)
    test_results.append(metrics_list)
    print("\n- FOLD %s | MODEL FINISH THE TEST | %d IMAGES!"%(fold, len(test_data)))


torch.save(test_order, 'test/'+opt.network+'/'+opt.specificity+'/test_order.pth')
torch.save(test_results, 'test/'+opt.network+'/'+opt.specificity+'/test_results.pth')

### This doesn't work at the moment, but the evaluation results are saved in a .pth file, that can be easly acessed after!
# transposed_data = list(zip(*test_results))
# means = [np.mean(column) for column in transposed_data]
# stds = [np.std(column) for column in transposed_data]

# # Metrics Print
# print("------------------------------")
# print("METRICS MEAN VALUE:")
# print("- TOTAL IMAGE -")
# print("| Image Metrics |")
# print("Mean Absolute Error: %.3f ± %.3f"%(means[0], stds[0]))
# print("Mean Squared Error: %.3f ± %.3f"%(means[1], stds[1]))
# print("PSNR Metric: %.3f dB ± %.3f"%(means[2], stds[2]))
# print("SSIM Metric: %.3f ± %.3f"%(means[3], stds[3]))
# print("MS-SSIM Metric: %.3f ± %.3f"%(means[4], stds[4]))

# print("| Features Metrics |")
# print("FID Metric: %.3f ± %.3f"%(means[5], stds[5]))
# print("IS Metric: %.3f ± %.3f"%(means[6], stds[6]))
# print("------------------------------")

