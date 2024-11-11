import torch
import numpy as np


# std_difference=[]
# TUMOUR_hausdorff_distance=[]
# TUMOUR_frechet_distance=[]

# GENERAL TUMOUR
# method='CE_Modified_LocalDiscriminator'
method='CE_Modified_LocalDiscriminator'
mask_percentage_list=[10,20,30,40]

excel = [
    ['Method', 'Missing','Condition', 'MAE', 'MSE', 'PSNR', 'SSIM', 'FID', 'IS']]

# excel1 = [
#     ['Method', 'Missing', 'ratio_inside_real', 'ratio_inside_fake', 'ratio_fake_real', 'TUMOURs_F1Score', 'TUMOURs_IoU', 'TUMOURs_intensity']]

excel1 = [
    ['Method', 'Missing', 'TUMOURs_F1Score_0_25', 'TUMOURs_F1Score_25_50', 'TUMOURs_F1Score_50_75', 'TUMOURs_F1Score_75_100', 'TUMOURs_IoU_0_25', 'TUMOURs_IoU_25_50','TUMOURs_IoU_50_75','TUMOURs_IoU_75_100']]


for mask_percentage in mask_percentage_list:
    # ALL
    MAE=[]
    MSE=[]
    PSNR=[]
    SSIM=[]
    FID=[]
    IS=[]

    # TUMOUR
    TUMOUR_MAE=[]
    TUMOUR_MSE=[]
    TUMOUR_PSNR=[]
    TUMOUR_SSIM=[]
    TUMOUR_FID=[]
    TUMOUR_IS=[]

    # NON_TUMOUR
    NON_TUMOUR_MAE=[]
    NON_TUMOUR_MSE=[]
    NON_TUMOUR_PSNR=[]
    NON_TUMOUR_SSIM=[]
    NON_TUMOUR_FID=[]
    NON_TUMOUR_IS=[]

    # TUMOUR
    TUMOURLABEL_MAE=[]
    TUMOURLABEL_MSE=[]
    TUMOURLABEL_PSNR=[]
    
    PULMONAR_MAE=[]
    PULMONAR_MSE=[]
    PULMONAR_PSNR=[]

    NON_PULMONAR_MAE=[]
    NON_PULMONAR_MSE=[]
    NON_PULMONAR_PSNR=[]

    TUMOURs_F1Score_0_25=[]
    TUMOURs_IoU_0_25=[]

    TUMOURs_F1Score_25_50=[]
    TUMOURs_IoU_25_50=[]

    TUMOURs_F1Score_50_75=[]
    TUMOURs_IoU_50_75=[]

    TUMOURs_F1Score_75_100=[]
    TUMOURs_IoU_75_100=[]

    # TUMOURs_intensity=[]
    for i in range(10):
        file_path = 'test/%s/Square_%s/complementar_all_metrics_fold_%s.pth'%(method,mask_percentage,i)
        # file_path = 'Square_%s/complementar_all_metrics_fold_%s.pth'%(mask_percentage,i)
        # file_path = 'Square_%s/all_metrics_fold_%s.pth'%(mask_percentage,i)
        data = torch.load(file_path,map_location="cpu")

        MAE.append(np.mean(data[0][0]))
        TUMOUR_MAE.append(np.mean(data[0][1]))
        NON_TUMOUR_MAE.append(np.mean(data[0][2]))
        TUMOURLABEL_MAE.append(np.mean(data[0][3]))
        PULMONAR_MAE.append(np.mean(data[0][4]))
        NON_PULMONAR_MAE.append(np.mean(data[0][5]))

        MSE.append(np.mean(data[1][0]))
        TUMOUR_MSE.append(np.mean(data[1][1]))
        NON_TUMOUR_MSE.append(np.mean(data[1][2]))
        TUMOURLABEL_MSE.append(np.mean(data[1][3]))
        PULMONAR_MSE.append(np.mean(data[1][4]))
        NON_PULMONAR_MSE.append(np.mean(data[1][5]))

        PSNR.append(np.mean(data[2][0]))
        TUMOUR_PSNR.append(np.mean(data[2][1]))
        NON_TUMOUR_PSNR.append(np.mean(data[2][2]))
        TUMOURLABEL_PSNR.append(np.mean(data[2][3]))
        PULMONAR_PSNR.append(np.mean(data[2][4]))
        NON_PULMONAR_PSNR.append(np.mean(data[2][5]))

        SSIM.append(np.mean(data[3][0]))
        TUMOUR_SSIM.append(np.mean(data[3][1]))
        NON_TUMOUR_SSIM.append(np.mean(data[3][2]))

        FID.append(data[4][0].numpy())
        TUMOUR_FID.append(data[4][1].numpy())
        NON_TUMOUR_FID.append(data[4][2].numpy())

        IS.append(data[5][0].numpy())
        TUMOUR_IS.append(data[5][2].numpy())
        NON_TUMOUR_IS.append(data[5][1].numpy())

        for k in range(len(data[6])):
            if data[6][k] is not None:
                
                # ratio_inside_real.append(data[6][k][0].numpy())
                # ratio_inside_fake.append(data[6][k][1].numpy())
                # ratio_fake_real.append(data[6][k][2].numpy())

                if data[6][k][0].numpy() < 0.25:
                    TUMOURs_F1Score_0_25.append(data[6][k][3].numpy())
                    TUMOURs_IoU_0_25.append(data[6][k][4].numpy())
                elif data[6][k][0].numpy() >= 0.25 and data[6][k][0].numpy() < 0.5:
                    TUMOURs_F1Score_25_50.append(data[6][k][3].numpy())
                    TUMOURs_IoU_25_50.append(data[6][k][4].numpy())
                elif data[6][k][0].numpy() >= 0.5 and data[6][k][0].numpy() < 0.75:
                    TUMOURs_F1Score_50_75.append(data[6][k][3].numpy())
                    TUMOURs_IoU_50_75.append(data[6][k][4].numpy())
                else:
                    TUMOURs_F1Score_75_100.append(data[6][k][3].numpy())
                    TUMOURs_IoU_75_100.append(data[6][k][4].numpy())

                # TUMOURs_F1Score.append(data[6][k][3].numpy())
                # TUMOURs_IoU.append(data[6][k][4].numpy())
                # TUMOURs_intensity.append(data[6][k][5].numpy())
                # std_difference.append(data[6][k][6].numpy())
                # TUMOUR_hausdorff_distance.append(data[6][k][7].numpy())
                # TUMOUR_frechet_distance.append(data[6][k][8].numpy())


    excel=excel+[
        [method, str(mask_percentage)+'%','Overall','%s±%s'%(str(round(np.mean(MAE),8)),str(round(np.std(MAE),8))),'%s±%s'%(str(round(np.mean(MSE),8)),str(round(np.std(MSE),8))),'%s±%s'%(str(round(np.mean(PSNR),8)),str(round(np.std(PSNR),8))),'%s±%s'%(str(round(np.mean(SSIM),8)),str(round(np.std(SSIM),8))),'%s±%s'%(str(round(np.mean(FID),8)),str(round(np.std(FID),8))),'%s±%s'%(str(round(np.mean(IS),8)),str(round(np.std(IS),8)))],
        [method, str(mask_percentage)+'%','Tumour Overall','%s±%s'%(str(round(np.mean(TUMOUR_MAE),8)),str(round(np.std(TUMOUR_MAE),8))),'%s±%s'%(str(round(np.mean(TUMOUR_MSE),8)),str(round(np.std(TUMOUR_MSE),8))),'%s±%s'%(str(round(np.mean(TUMOUR_PSNR),8)),str(round(np.std(TUMOUR_PSNR),8))),'%s±%s'%(str(round(np.mean(TUMOUR_SSIM),8)),str(round(np.std(TUMOUR_SSIM),8))),'%s±%s'%(str(round(np.mean(TUMOUR_FID),8)),str(round(np.std(TUMOUR_FID),8))),'%s±%s'%(str(round(np.mean(TUMOUR_IS),8)),str(round(np.std(TUMOUR_IS),8)))],
        [method, str(mask_percentage)+'%','Non-Tumour Overall','%s±%s'%(str(round(np.mean(NON_TUMOUR_MAE),8)),str(round(np.std(NON_TUMOUR_MAE),8))),'%s±%s'%(str(round(np.mean(NON_TUMOUR_MSE),8)),str(round(np.std(NON_TUMOUR_MSE),8))),'%s±%s'%(str(round(np.mean(NON_TUMOUR_PSNR),8)),str(round(np.std(NON_TUMOUR_PSNR),8))),'%s±%s'%(str(round(np.mean(NON_TUMOUR_SSIM),8)),str(round(np.std(NON_TUMOUR_SSIM),8))),'%s±%s'%(str(round(np.mean(NON_TUMOUR_FID),8)),str(round(np.std(NON_TUMOUR_FID),8))),'%s±%s'%(str(round(np.mean(NON_TUMOUR_IS),8)),str(round(np.std(NON_TUMOUR_IS),8)))],
        [method, str(mask_percentage)+'%','Tumour Tissue','%s±%s'%(str(round(np.mean(TUMOURLABEL_MAE),8)),str(round(np.std(TUMOURLABEL_MAE),8))),'%s±%s'%(str(round(np.mean(TUMOURLABEL_MSE),8)),str(round(np.std(TUMOURLABEL_MSE),8))),'%s±%s'%(str(round(np.mean(TUMOURLABEL_PSNR),8)),str(round(np.std(TUMOURLABEL_PSNR),8))),'None','None','None'],
        [method, str(mask_percentage)+'%','Pulmonar Tissue','%s±%s'%(str(round(np.mean(PULMONAR_MAE),8)),str(round(np.std(PULMONAR_MAE),8))),'%s±%s'%(str(round(np.mean(PULMONAR_MSE),8)),str(round(np.std(PULMONAR_MSE),8))),'%s±%s'%(str(round(np.mean(PULMONAR_PSNR),8)),str(round(np.std(PULMONAR_PSNR),8))),'None','None','None'],
        [method, str(mask_percentage)+'%','Non-Pulmonar Tissue','%s±%s'%(str(round(np.mean(NON_PULMONAR_MAE),8)),str(round(np.std(NON_PULMONAR_MAE),8))),'%s±%s'%(str(round(np.mean(NON_PULMONAR_MSE),8)),str(round(np.std(NON_PULMONAR_MSE),8))),'%s±%s'%(str(round(np.mean(NON_PULMONAR_PSNR),8)),str(round(np.std(NON_PULMONAR_PSNR),8))),'None','None','None'],
        [''] * 9
    ]

    # excel1=excel1+[
    #     [method, str(mask_percentage)+'%','%s±%s'%(str(round(np.mean(ratio_inside_real),8)),str(round(np.std(ratio_inside_real),8))),'%s±%s'%(str(round(np.mean(ratio_inside_fake),8)),str(round(np.std(ratio_inside_fake),8))),'%s±%s'%(str(round(np.mean(ratio_fake_real),8)),str(round(np.std(ratio_fake_real),8))),'%s±%s'%(str(round(np.mean(TUMOURs_F1Score),8)),str(round(np.std(TUMOURs_F1Score),8))),'%s±%s'%(str(round(np.mean(TUMOURs_IoU),8)),str(round(np.std(TUMOURs_IoU),8))),'%s±%s'%(str(round(np.mean(TUMOURs_intensity),8)),str(round(np.std(TUMOURs_intensity),8)))],
    # ]

    excel1=excel1+[
        [method, str(mask_percentage)+'%','%s±%s'%(str(round(np.mean(TUMOURs_F1Score_0_25),8)),str(round(np.std(TUMOURs_F1Score_0_25),8))),'%s±%s'%(str(round(np.mean(TUMOURs_F1Score_25_50),8)),str(round(np.std(TUMOURs_F1Score_25_50),8))),'%s±%s'%(str(round(np.mean(TUMOURs_F1Score_50_75),8)),str(round(np.std(TUMOURs_F1Score_50_75),8))),'%s±%s'%(str(round(np.mean(TUMOURs_F1Score_75_100),8)),str(round(np.std(TUMOURs_F1Score_75_100),8))),'%s±%s'%(str(round(np.mean(TUMOURs_IoU_0_25),8)),str(round(np.std(TUMOURs_IoU_0_25),8))),'%s±%s'%(str(round(np.mean(TUMOURs_IoU_25_50),8)),str(round(np.std(TUMOURs_IoU_25_50),8))),'%s±%s'%(str(round(np.mean(TUMOURs_IoU_50_75),8)),str(round(np.std(TUMOURs_IoU_50_75),8))),'%s±%s'%(str(round(np.mean(TUMOURs_IoU_75_100),8)),str(round(np.std(TUMOURs_IoU_75_100),8)))],
    ]

import pandas as pd
df1 = pd.DataFrame(excel[1:], columns=excel[0])
df2 = pd.DataFrame(excel1[1:], columns=excel1[0])
with pd.ExcelWriter('%s_new.xlsx'%(method), engine='openpyxl') as writer:
    df1.to_excel(writer, sheet_name='Metrics', index=False)
    df2.to_excel(writer, sheet_name='Tumour', index=False)