import torch
import numpy as np


# std_difference=[]
# tumor_hausdorff_distance=[]
# tumor_frechet_distance=[]

# GENERAL TUMOR
# method='CE_Modified_LocalDiscriminator'
# methods=['GLCIC_Original']

methods=['CE_Modified_LocalDiscriminator','CE_Modified_GlobalDiscriminator','GLCIC_Original','CA_Original','EC_Original','ESMII_Original','Multi_Slices']

mask_percentage_list=[10,20,30,40]

# excel = [
#     ['Method', 'Missing','Condition', 'MAE', 'MSE', 'PSNR', 'SSIM', 'FID', 'IS']]

# # excel1 = [
# #     ['Method', 'Missing', 'ratio_inside_real', 'ratio_inside_fake', 'ratio_fake_real', 'tumors_F1Score', 'tumors_IoU', 'tumors_intensity']]

# excel1 = [
#     ['Method', 'Missing', 'tumors_F1Score_0_25', 'tumors_F1Score_25_50', 'tumors_F1Score_50_75', 'tumors_F1Score_75_100', 'tumors_IoU_0_25', 'tumors_IoU_25_50','tumors_IoU_50_75','tumors_IoU_75_100']]

all_tumors_F1Score_0_25=[]
all_tumors_IoU_0_25=[]

all_tumors_F1Score_25_50=[]
all_tumors_IoU_25_50=[]

all_tumors_F1Score_50_75=[]
all_tumors_IoU_50_75=[]

all_tumors_F1Score_75_100=[]
all_tumors_IoU_75_100=[]


for ii,method in enumerate(methods):
    mask_tumors_F1Score_0_25=[]
    mask_tumors_IoU_0_25=[]

    mask_tumors_F1Score_25_50=[]
    mask_tumors_IoU_25_50=[]

    mask_tumors_F1Score_50_75=[]
    mask_tumors_IoU_50_75=[]

    mask_tumors_F1Score_75_100=[]
    mask_tumors_IoU_75_100=[]
    for mask_percentage in mask_percentage_list:
        # ALL
        MAE=[]
        MSE=[]
        PSNR=[]
        SSIM=[]
        FID=[]
        IS=[]

        # LOW
        LOW_MAE=[]
        LOW_MSE=[]
        LOW_PSNR=[]
        LOW_SSIM=[]
        LOW_FID=[]
        LOW_IS=[]

        # MEDIUM
        MEDIUM_MAE=[]
        MEDIUM_MSE=[]
        MEDIUM_PSNR=[]
        MEDIUM_SSIM=[]
        MEDIUM_FID=[]
        MEDIUM_IS=[]

        # HIGH
        HIGH_MAE=[]
        HIGH_MSE=[]
        HIGH_PSNR=[]
        HIGH_SSIM=[]
        HIGH_FID=[]
        HIGH_IS=[]

        # TUMOR
        TUMOR_MAE=[]
        TUMOR_MSE=[]
        TUMOR_PSNR=[]
        TUMOR_SSIM=[]
        TUMOR_FID=[]
        TUMOR_IS=[]

        # NON-TUMOR
        NON_TUMOR_MAE=[]
        NON_TUMOR_MSE=[]
        NON_TUMOR_PSNR=[]
        NON_TUMOR_SSIM=[]
        NON_TUMOR_FID=[]
        NON_TUMOR_IS=[]

        ratio_inside_real=[]
        ratio_inside_fake=[]
        ratio_fake_real=[]
        tumors_F1Score_0_25=[]
        tumors_IoU_0_25=[]

        tumors_F1Score_25_50=[]
        tumors_IoU_25_50=[]

        tumors_F1Score_50_75=[]
        tumors_IoU_50_75=[]

        tumors_F1Score_75_100=[]
        tumors_IoU_75_100=[]

        # tumors_intensity=[]
        for i in range(10):
            file_path = 'test/%s/Square_%s/all_metrics_fold_%s.pth'%(method,mask_percentage,i)
            # file_path = 'Square_%s/all_metrics_fold_%s.pth'%(mask_percentage,i)
            data = torch.load(file_path,map_location="cpu")

            MAE.append(np.mean(data[0][0]))
            LOW_MAE.append(np.mean(data[0][1]))
            MEDIUM_MAE.append(np.mean(data[0][2]))
            HIGH_MAE.append(np.mean(data[0][3]))
            NON_TUMOR_MAE.append(np.mean(data[0][4]))
            TUMOR_MAE.append(np.mean(data[0][5]))

            MSE.append(np.mean(data[1][0]))
            LOW_MSE.append(np.mean(data[1][1]))
            MEDIUM_MSE.append(np.mean(data[1][2]))
            HIGH_MSE.append(np.mean(data[1][3]))
            NON_TUMOR_MSE.append(np.mean(data[1][4]))
            TUMOR_MSE.append(np.mean(data[1][5]))

            PSNR.append(np.mean(data[2][0]))
            LOW_PSNR.append(np.mean(data[2][1]))
            MEDIUM_PSNR.append(np.mean(data[2][2]))
            HIGH_PSNR.append(np.mean(data[2][3]))
            NON_TUMOR_PSNR.append(np.mean(data[2][4]))
            TUMOR_PSNR.append(np.mean(data[2][5]))

            SSIM.append(np.mean(data[3][0]))
            LOW_SSIM.append(np.mean(data[3][1]))
            MEDIUM_SSIM.append(np.mean(data[3][2]))
            HIGH_SSIM.append(np.mean(data[3][3]))
            NON_TUMOR_SSIM.append(np.mean(data[3][4]))
            TUMOR_SSIM.append(np.mean(data[3][5]))

            FID.append(data[4][0].numpy())
            LOW_FID.append(data[4][1].numpy())
            MEDIUM_FID.append(data[4][2].numpy())
            HIGH_FID.append(data[4][3].numpy())

            IS.append(data[5][0].numpy())
            LOW_IS.append(data[5][1].numpy())
            MEDIUM_IS.append(data[5][2].numpy())
            HIGH_IS.append(data[5][3].numpy())

            for k in range(len(data[6])):
                if data[6][k] is not None:
                    
                    # ratio_inside_real.append(data[6][k][0].numpy())
                    # ratio_inside_fake.append(data[6][k][1].numpy())
                    # ratio_fake_real.append(data[6][k][2].numpy())

                    if data[6][k][0].numpy() < 0.25:
                        tumors_F1Score_0_25.append(data[6][k][3].numpy())
                        tumors_IoU_0_25.append(data[6][k][4].numpy())
                    elif data[6][k][0].numpy() >= 0.25 and data[6][k][0].numpy() < 0.5:
                        tumors_F1Score_25_50.append(data[6][k][3].numpy())
                        tumors_IoU_25_50.append(data[6][k][4].numpy())
                    elif data[6][k][0].numpy() >= 0.5 and data[6][k][0].numpy() < 0.75:
                        tumors_F1Score_50_75.append(data[6][k][3].numpy())
                        tumors_IoU_50_75.append(data[6][k][4].numpy())
                    else:
                        tumors_F1Score_75_100.append(data[6][k][3].numpy())
                        tumors_IoU_75_100.append(data[6][k][4].numpy())

                # tumors_F1Score.append(data[6][k][3].numpy())
                # tumors_IoU.append(data[6][k][4].numpy())
                # tumors_intensity.append(data[6][k][5].numpy())
                # std_difference.append(data[6][k][6].numpy())
                # tumor_hausdorff_distance.append(data[6][k][7].numpy())
                # tumor_frechet_distance.append(data[6][k][8].numpy())

        mask_tumors_F1Score_0_25.append(tumors_F1Score_0_25)
        mask_tumors_IoU_0_25.append(tumors_IoU_0_25)

        mask_tumors_F1Score_25_50.append(tumors_F1Score_25_50)
        mask_tumors_IoU_25_50.append(tumors_IoU_25_50)

        mask_tumors_F1Score_50_75.append(tumors_F1Score_50_75)
        mask_tumors_IoU_50_75.append(tumors_IoU_50_75)

        mask_tumors_F1Score_75_100.append(tumors_F1Score_75_100)
        mask_tumors_IoU_75_100.append(tumors_IoU_75_100)


    all_tumors_F1Score_0_25.append(mask_tumors_F1Score_0_25)
    all_tumors_IoU_0_25.append(mask_tumors_IoU_0_25)

    all_tumors_F1Score_25_50.append(mask_tumors_F1Score_25_50)
    all_tumors_IoU_25_50.append(mask_tumors_IoU_25_50)

    all_tumors_F1Score_50_75.append(mask_tumors_F1Score_50_75)
    all_tumors_IoU_50_75.append(mask_tumors_IoU_50_75)

    all_tumors_F1Score_75_100.append(mask_tumors_F1Score_75_100)
    all_tumors_IoU_75_100.append(mask_tumors_IoU_75_100)

        # excel=excel+[
        #     [method, str(mask_percentage)+'%','Overall','%s±%s'%(str(round(np.mean(MAE),8)),str(round(np.std(MAE),8))),'%s±%s'%(str(round(np.mean(MSE),8)),str(round(np.std(MSE),8))),'%s±%s'%(str(round(np.mean(PSNR),8)),str(round(np.std(PSNR),8))),'%s±%s'%(str(round(np.mean(SSIM),8)),str(round(np.std(SSIM),8))),'%s±%s'%(str(round(np.mean(FID),8)),str(round(np.std(FID),8))),'%s±%s'%(str(round(np.mean(IS),8)),str(round(np.std(IS),8)))],
        #     [method, str(mask_percentage)+'%','Low Complexity','%s±%s'%(str(round(np.mean(LOW_MAE),8)),str(round(np.std(LOW_MAE),8))),'%s±%s'%(str(round(np.mean(LOW_MSE),8)),str(round(np.std(LOW_MSE),8))),'%s±%s'%(str(round(np.mean(LOW_PSNR),8)),str(round(np.std(LOW_PSNR),8))),'%s±%s'%(str(round(np.mean(LOW_SSIM),8)),str(round(np.std(LOW_SSIM),8))),'%s±%s'%(str(round(np.mean(LOW_FID),8)),str(round(np.std(LOW_FID),8))),'%s±%s'%(str(round(np.mean(LOW_IS),8)),str(round(np.std(LOW_IS),8)))],
        #     [method, str(mask_percentage)+'%','Medium Complexity','%s±%s'%(str(round(np.mean(MEDIUM_MAE),8)),str(round(np.std(MEDIUM_MAE),8))),'%s±%s'%(str(round(np.mean(MEDIUM_MSE),8)),str(round(np.std(MEDIUM_MSE),8))),'%s±%s'%(str(round(np.mean(MEDIUM_PSNR),8)),str(round(np.std(MEDIUM_PSNR),8))),'%s±%s'%(str(round(np.mean(MEDIUM_SSIM),8)),str(round(np.std(MEDIUM_SSIM),8))),'%s±%s'%(str(round(np.mean(MEDIUM_FID),8)),str(round(np.std(MEDIUM_FID),8))),'%s±%s'%(str(round(np.mean(MEDIUM_IS),8)),str(round(np.std(MEDIUM_IS),8)))],
        #     [method, str(mask_percentage)+'%','High Complexity','%s±%s'%(str(round(np.mean(HIGH_MAE),8)),str(round(np.std(HIGH_MAE),8))),'%s±%s'%(str(round(np.mean(HIGH_MSE),8)),str(round(np.std(HIGH_MSE),8))),'%s±%s'%(str(round(np.mean(HIGH_PSNR),8)),str(round(np.std(HIGH_PSNR),8))),'%s±%s'%(str(round(np.mean(HIGH_SSIM),8)),str(round(np.std(HIGH_SSIM),8))),'%s±%s'%(str(round(np.mean(HIGH_FID),8)),str(round(np.std(HIGH_FID),8))),'%s±%s'%(str(round(np.mean(HIGH_IS),8)),str(round(np.std(HIGH_IS),8)))],
        #     [method, str(mask_percentage)+'%','Non-Tumour Complexity','%s±%s'%(str(round(np.mean(NON_TUMOR_MAE),8)),str(round(np.std(NON_TUMOR_MAE),8))),'%s±%s'%(str(round(np.mean(NON_TUMOR_MSE),8)),str(round(np.std(NON_TUMOR_MSE),8))),'%s±%s'%(str(round(np.mean(NON_TUMOR_PSNR),8)),str(round(np.std(NON_TUMOR_PSNR),8))),'%s±%s'%(str(round(np.mean(NON_TUMOR_SSIM),8)),str(round(np.std(NON_TUMOR_SSIM),8))),'None','None'],
        #     [method, str(mask_percentage)+'%','Tumour Complexity','%s±%s'%(str(round(np.mean(TUMOR_MAE),8)),str(round(np.std(TUMOR_MAE),8))),'%s±%s'%(str(round(np.mean(TUMOR_MSE),8)),str(round(np.std(TUMOR_MSE),8))),'%s±%s'%(str(round(np.mean(TUMOR_PSNR),8)),str(round(np.std(TUMOR_PSNR),8))),'%s±%s'%(str(round(np.mean(TUMOR_SSIM),8)),str(round(np.std(TUMOR_SSIM),8))),'None','None'],
        #     [''] * 9
        # ]

    # excel1=excel1+[
    #     [method, str(mask_percentage)+'%','%s±%s'%(str(round(np.mean(ratio_inside_real),8)),str(round(np.std(ratio_inside_real),8))),'%s±%s'%(str(round(np.mean(ratio_inside_fake),8)),str(round(np.std(ratio_inside_fake),8))),'%s±%s'%(str(round(np.mean(ratio_fake_real),8)),str(round(np.std(ratio_fake_real),8))),'%s±%s'%(str(round(np.mean(tumors_F1Score),8)),str(round(np.std(tumors_F1Score),8))),'%s±%s'%(str(round(np.mean(tumors_IoU),8)),str(round(np.std(tumors_IoU),8))),'%s±%s'%(str(round(np.mean(tumors_intensity),8)),str(round(np.std(tumors_intensity),8)))],
    # ]

#         excel1=excel1+[
#             [method, str(mask_percentage)+'%','%s±%s'%(str(round(np.mean(tumors_F1Score_0_25),8)),str(round(np.std(tumors_F1Score_0_25),8))),'%s±%s'%(str(round(np.mean(tumors_F1Score_25_50),8)),str(round(np.std(tumors_F1Score_25_50),8))),'%s±%s'%(str(round(np.mean(tumors_F1Score_50_75),8)),str(round(np.std(tumors_F1Score_50_75),8))),'%s±%s'%(str(round(np.mean(tumors_F1Score_75_100),8)),str(round(np.std(tumors_F1Score_75_100),8))),'%s±%s'%(str(round(np.mean(tumors_IoU_0_25),8)),str(round(np.std(tumors_IoU_0_25),8))),'%s±%s'%(str(round(np.mean(tumors_IoU_25_50),8)),str(round(np.std(tumors_IoU_25_50),8))),'%s±%s'%(str(round(np.mean(tumors_IoU_50_75),8)),str(round(np.std(tumors_IoU_50_75),8))),'%s±%s'%(str(round(np.mean(tumors_IoU_75_100),8)),str(round(np.std(tumors_IoU_75_100),8)))],
#         ]

# import pandas as pd
# df1 = pd.DataFrame(excel[1:], columns=excel[0])
# df2 = pd.DataFrame(excel1[1:], columns=excel1[0])
# with pd.ExcelWriter('%s.xlsx'%(method), engine='openpyxl') as writer:
#     df1.to_excel(writer, sheet_name='Metrics', index=False)
#     df2.to_excel(writer, sheet_name='Tumour', index=False)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define your methods
methods = ['CE_Modified_LocalDiscriminator', 'CE_Modified_GlobalDiscriminator', 'GLCIC_Original',
           'CA_Original', 'CE_Original', 'ESMII_Original', 'Multi_Slices']

# Initialize data storage
data = {
    'Method': [],
    'Mask_Percentage': [],
    'Range': [],
    'F1Score': [],
}

for ii,method in enumerate(methods):
    mask_percentages = [10, 20, 30, 40]  # Adjust these if needed
    ranges = ['[0,25%]', '[25,50%]', '[50,75%]', '[75,100%]']
    
    # Append data to the dictionary for F1 Scores
    data['Method'].extend([method] * 16)
    data['Mask_Percentage'].extend(['10%'] * 4)
    data['Range'].extend(['[0,25%]','[25,50%]','[50,75%]','[75,100%]'])
    data['F1Score'].extend([all_tumors_F1Score_0_25[ii][0],all_tumors_F1Score_25_50[ii][0],all_tumors_F1Score_50_75[ii][0],all_tumors_F1Score_75_100[ii][0]])

    data['Mask_Percentage'].extend(['20%'] * 4)
    data['Range'].extend(['[0,25%]','[25,50%]','[50,75%]','[75,100%]'])
    data['F1Score'].extend([all_tumors_F1Score_0_25[ii][1],all_tumors_F1Score_25_50[ii][1],all_tumors_F1Score_50_75[ii][1],all_tumors_F1Score_75_100[ii][1]])

    data['Mask_Percentage'].extend(['30%'] * 4)
    data['Range'].extend(['[0,25%]','[25,50%]','[50,75%]','[75,100%]'])
    data['F1Score'].extend([all_tumors_F1Score_0_25[ii][2],all_tumors_F1Score_25_50[ii][2],all_tumors_F1Score_50_75[ii][2],all_tumors_F1Score_75_100[ii][2]])

    data['Mask_Percentage'].extend(['40%'] * 4)
    data['Range'].extend(['[0,25%]','[25,50%]','[50,75%]','[75,100%]'])
    data['F1Score'].extend([all_tumors_F1Score_0_25[ii][3],all_tumors_F1Score_25_50[ii][3],all_tumors_F1Score_50_75[ii][3],all_tumors_F1Score_75_100[ii][3]])

# Convert the data dictionary to a DataFrame
df = pd.DataFrame(data)
df_expanded = df.explode('F1Score')
df_expanded['F1Score'] = df_expanded['F1Score'].astype(str).str.strip()
df_expanded['F1Score'] = pd.to_numeric(df_expanded['F1Score'], errors='coerce')
df_expanded['Range'] = pd.Categorical(df_expanded['Range'], categories=['[0,25%]', '[25,50%]', '[50,75%]', '[75,100%]'], ordered=True)

import matplotlib.pyplot as plt
import seaborn as sns

# Define your unique mask percentages and methods
mask_percentages = df['Mask_Percentage'].unique()
methods = df['Method'].unique()
range = df['Range'].unique()

palette = sns.color_palette("Blues", n_colors=len(df_expanded['Range'].unique()))

methods_name=["CE LOCAL","CE GLOBAL","GLCIC","CA","EC","ESMII","MS"]

# Create subplots
fig, axs = plt.subplots(len(mask_percentages), len(methods), figsize=(12, 12), sharex=True, sharey=True)

for row, mask_percentage in enumerate(mask_percentages):
    for col, method in enumerate(methods):
        subset = df_expanded[(df_expanded['Method'] == method) & (df_expanded['Mask_Percentage'] == mask_percentage)]
        
        sns.boxplot(x='Range', y='F1Score', data=subset, ax=axs[row, col], palette=palette)
        
        # Set the title for the first row with the method name
        if row == 0:
            axs[row, col].set_title(methods_name[col], fontsize=12, fontweight='bold')
        else:
            axs[row, col].set_title('')

        # Set y-label only for the first column
        if col == 0:
            axs[row, col].set_ylabel(mask_percentage, fontsize=12, fontweight='bold')
        else:
            axs[row, col].set_ylabel('')

        # Rotate x-axis labels for better readability
        axs[row, col].tick_params(axis='x', rotation=90)

# Adjust layout
plt.tight_layout()
plt.savefig('tumourDICE_graph.png', dpi=300, bbox_inches='tight')
plt.show()

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define your methods and mask percentages
methods = ['CE_Modified_LocalDiscriminator', 'CE_Modified_GlobalDiscriminator', 'GLCIC_Original',
           'CA_Original', 'EC_Original', 'ESMII_Original', 'Multi_Slices']
mask_percentage_list = [10, 20, 30, 40]

# Initialize data storage
all_tumors_IoU_0_25 = []
all_tumors_IoU_25_50 = []
all_tumors_IoU_50_75 = []
all_tumors_IoU_75_100 = []

for ii, method in enumerate(methods):
    mask_tumors_IoU_0_25 = []
    mask_tumors_IoU_25_50 = []
    mask_tumors_IoU_50_75 = []
    mask_tumors_IoU_75_100 = []
    
    for mask_percentage in mask_percentage_list:
        tumors_IoU_0_25 = []
        tumors_IoU_25_50 = []
        tumors_IoU_50_75 = []
        tumors_IoU_75_100 = []

        for i in range(10):
            file_path = f'test/{method}/Square_{mask_percentage}/all_metrics_fold_{i}.pth'
            data = torch.load(file_path, map_location="cpu")

            for k in range(len(data[6])):
                if data[6][k] is not None:
                    if data[6][k][0].numpy() < 0.25:
                        tumors_IoU_0_25.append(data[6][k][4].numpy())
                    elif 0.25 <= data[6][k][0].numpy() < 0.5:
                        tumors_IoU_25_50.append(data[6][k][4].numpy())
                    elif 0.5 <= data[6][k][0].numpy() < 0.75:
                        tumors_IoU_50_75.append(data[6][k][4].numpy())
                    else:
                        tumors_IoU_75_100.append(data[6][k][4].numpy())

        mask_tumors_IoU_0_25.append(tumors_IoU_0_25)
        mask_tumors_IoU_25_50.append(tumors_IoU_25_50)
        mask_tumors_IoU_50_75.append(tumors_IoU_50_75)
        mask_tumors_IoU_75_100.append(tumors_IoU_75_100)

    all_tumors_IoU_0_25.append(mask_tumors_IoU_0_25)
    all_tumors_IoU_25_50.append(mask_tumors_IoU_25_50)
    all_tumors_IoU_50_75.append(mask_tumors_IoU_50_75)
    all_tumors_IoU_75_100.append(mask_tumors_IoU_75_100)

# Prepare the data for plotting
data = {
    'Method': [],
    'Mask_Percentage': [],
    'Range': [],
    'IoU': [],
}

for ii, method in enumerate(methods):
    mask_percentages = mask_percentage_list
    ranges = ['[0,25%]', '[25,50%]', '[50,75%]', '[75,100%]']
    
    # Append data to the dictionary for IoU
    data['Method'].extend([method] * 16)
    data['Mask_Percentage'].extend(['10%'] * 4)
    data['Range'].extend(['[0,25%]','[25,50%]','[50,75%]','[75,100%]'])
    data['IoU'].extend([all_tumors_IoU_0_25[ii][0],all_tumors_IoU_25_50[ii][0],all_tumors_IoU_50_75[ii][0],all_tumors_IoU_75_100[ii][0]])

    data['Mask_Percentage'].extend(['20%'] * 4)
    data['Range'].extend(['[0,25%]','[25,50%]','[50,75%]','[75,100%]'])
    data['IoU'].extend([all_tumors_IoU_0_25[ii][1],all_tumors_IoU_25_50[ii][1],all_tumors_IoU_50_75[ii][1],all_tumors_IoU_75_100[ii][1]])

    data['Mask_Percentage'].extend(['30%'] * 4)
    data['Range'].extend(['[0,25%]','[25,50%]','[50,75%]','[75,100%]'])
    data['IoU'].extend([all_tumors_IoU_0_25[ii][2],all_tumors_IoU_25_50[ii][2],all_tumors_IoU_50_75[ii][2],all_tumors_IoU_75_100[ii][2]])

    data['Mask_Percentage'].extend(['40%'] * 4)
    data['Range'].extend(['[0,25%]','[25,50%]','[50,75%]','[75,100%]'])
    data['IoU'].extend([all_tumors_IoU_0_25[ii][3],all_tumors_IoU_25_50[ii][3],all_tumors_IoU_50_75[ii][3],all_tumors_IoU_75_100[ii][3]])

# Convert the data dictionary to a DataFrame
df = pd.DataFrame(data)
df_expanded = df.explode('IoU')
df_expanded['IoU'] = df_expanded['IoU'].astype(str).str.strip()
df_expanded['IoU'] = pd.to_numeric(df_expanded['IoU'], errors='coerce')
df_expanded['Range'] = pd.Categorical(df_expanded['Range'], categories=['[0,25%]', '[25,50%]', '[50,75%]', '[75,100%]'], ordered=True)

# Create plots for IoU
palette = sns.color_palette("Blues", n_colors=len(df_expanded['Range'].unique()))

methods_name = ["CE LOCAL", "CE GLOBAL", "GLCIC", "CA", "EC", "ESMII", "MS"]

fig, axs = plt.subplots(len(mask_percentage_list), len(methods), figsize=(12, 12), sharex=True, sharey=True)

for row, mask_percentage in enumerate(mask_percentage_list):
    for col, method in enumerate(methods):
        subset = df_expanded[(df_expanded['Method'] == method) & (df_expanded['Mask_Percentage'] == f"{mask_percentage}%")]
        
        sns.boxplot(x='Range', y='IoU', data=subset, ax=axs[row, col], palette=palette)
        
        # Set the title for the first row with the method name
        if row == 0:
            axs[row, col].set_title(methods_name[col], fontsize=12, fontweight='bold')
        else:
            axs[row, col].set_title('')

        # Set y-label only for the first column
        if col == 0:
            axs[row, col].set_ylabel(f"{mask_percentage}%", fontsize=12, fontweight='bold')
        else:
            axs[row, col].set_ylabel('')

        # Rotate x-axis labels for better readability
        axs[row, col].tick_params(axis='x', rotation=90)

# Adjust layout
plt.tight_layout()
plt.savefig('tumourIoU.png', dpi=300, bbox_inches='tight')
plt.show()
