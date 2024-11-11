from vutils.libraries import *
from scipy.stats import f_oneway, shapiro, ttest_ind, mannwhitneyu
import itertools

# Two lists
methods_names = ['GLCIC_Original','CA_Original','EC_Original','ESMII_Original','Multi_Slices']
masks_percentage_names = ['10%','20%','30%','40%']
complexity_names = ['Lung','Tumour','External-Lung']

methods = range(5)
masks_percentage = range(4)
complexity = range(3)
combinations = [[a,b,c] for a, b, c in itertools.product(methods, masks_percentage,complexity)]
combinations_stats=list(itertools.combinations(combinations, 2))
p_value=0.05

excel = [['Metrics','Distribution 1','Distribution 2', 'Test', 'p_value', 'Dif. Significativa?']]

ALL_MAE=[];ALL_MSE=[];ALL_PSNR=[];ALL_SSIM=[];ALL_FID=[]
ALL_TUMOR_MAE=[];ALL_TUMOR_MSE=[];ALL_TUMOR_PSNR=[];ALL_TUMOR_SSIM=[];ALL_TUMOR_FID=[]
ALL_NON_TUMOR_MAE=[];ALL_NON_TUMOR_MSE=[];ALL_NON_TUMOR_PSNR=[];ALL_NON_TUMOR_SSIM=[];ALL_NON_TUMOR_FID=[]
ALL_TUMOR_F1SCORE=[];ALL_TUMOR_IoU=[]

for method in methods_names:
    METHOD_MAE=[];METHOD_MSE=[];METHOD_PSNR=[];METHOD_SSIM=[];METHOD_FID=[]
    METHOD_TUMOR_MAE=[];METHOD_TUMOR_MSE=[];METHOD_TUMOR_PSNR=[];METHOD_TUMOR_SSIM=[];METHOD_TUMOR_FID=[]
    METHOD_NON_TUMOR_MAE=[];METHOD_NON_TUMOR_MSE=[];METHOD_NON_TUMOR_PSNR=[];METHOD_NON_TUMOR_SSIM=[];METHOD_NON_TUMOR_FID=[]
    METHOD_TUMOR_F1SCORE=[];METHOD_TUMOR_IoU=[]

    for mask_percentage in [10,20,30,40]:
        MAE=[];MSE=[];PSNR=[];SSIM=[];FID=[];IS=[]
        TUMOR_MAE=[];TUMOR_MSE=[];TUMOR_PSNR=[];TUMOR_SSIM=[];TUMOR_FID=[]
        NON_TUMOR_MAE=[];NON_TUMOR_MSE=[];NON_TUMOR_PSNR=[];NON_TUMOR_SSIM=[];NON_TUMOR_FID=[]
        # TUMOR_F1SCORE=[];TUMOR_IoU=[]

        for i in range(10):
            try:
                file_path = 'test/%s/Square_%s/all_metrics_fold_%s.pth'%(method,mask_percentage,i)
                # file_path = 'Square_%s/all_metrics_fold_%s.pth'%(mask_percentage,i)
                data = torch.load(file_path,map_location="cpu")

                MAE.append(np.mean(data[0][5]))
                NON_TUMOR_MAE.append(np.mean(data[0][4]))
                TUMOR_MAE.append(np.mean(data[0][3]))
                
                MSE.append(np.mean(data[1][5]))
                NON_TUMOR_MSE.append(np.mean(data[1][4]))
                TUMOR_MSE.append(np.mean(data[1][3]))

                PSNR.append(np.mean(data[2][5]))
                NON_TUMOR_PSNR.append(np.mean(data[2][4]))
                TUMOR_PSNR.append(np.mean(data[2][3]))

                # SSIM.append(np.mean(data[3][5]))
                # NON_TUMOR_SSIM.append(np.mean(data[3][4]))
                # TUMOR_SSIM.append(np.mean(data[3][3]))

                # FID.append(data[4][5].numpy())
                # NON_TUMOR_FID.append(data[4][4].numpy())
                # TUMOR_FID.append(data[4][3].numpy())

                # for k in range(len(data[6])):
                #     if data[6][k] is not None:
                #         TUMOR_F1SCORE.append(data[6][k][3].numpy())
                #         TUMOR_IoU.append(data[6][k][4].numpy())
            
            except:
                pass
            

        METHOD_MAE.append(MAE)
        METHOD_NON_TUMOR_MAE.append(NON_TUMOR_MAE)
        METHOD_TUMOR_MAE.append(TUMOR_MAE)

        METHOD_MSE.append(MSE)
        METHOD_NON_TUMOR_MSE.append(NON_TUMOR_MSE)
        METHOD_TUMOR_MSE.append(TUMOR_MSE)

        METHOD_PSNR.append(PSNR)
        METHOD_NON_TUMOR_PSNR.append(NON_TUMOR_PSNR)
        METHOD_TUMOR_PSNR.append(TUMOR_PSNR)

        # METHOD_SSIM.append(SSIM)
        # METHOD_NON_TUMOR_SSIM.append(NON_TUMOR_SSIM)
        # METHOD_TUMOR_SSIM.append(TUMOR_SSIM)

        # METHOD_FID.append(FID)
        # METHOD_NON_TUMOR_FID.append(NON_TUMOR_FID)
        # METHOD_TUMOR_FID.append(TUMOR_FID)

        # METHOD_TUMOR_F1SCORE.append(NON_TUMOR_SSIM)
        # METHOD_TUMOR_IoU.append(TUMOR_SSIM)


    ALL_MAE.append(METHOD_MAE)
    ALL_NON_TUMOR_MAE.append(METHOD_NON_TUMOR_MAE)
    ALL_TUMOR_MAE.append(METHOD_TUMOR_MAE)

    ALL_MSE.append(METHOD_MSE)
    ALL_NON_TUMOR_MSE.append(METHOD_NON_TUMOR_MSE)
    ALL_TUMOR_MSE.append(METHOD_TUMOR_MSE)

    ALL_PSNR.append(METHOD_PSNR)
    ALL_NON_TUMOR_PSNR.append(METHOD_NON_TUMOR_PSNR)
    ALL_TUMOR_PSNR.append(METHOD_TUMOR_PSNR)

    # ALL_SSIM.append(METHOD_SSIM)
    # ALL_NON_TUMOR_SSIM.append(METHOD_NON_TUMOR_SSIM)
    # ALL_TUMOR_SSIM.append(METHOD_TUMOR_SSIM)

    # ALL_FID.append(METHOD_FID)
    # ALL_NON_TUMOR_FID.append(METHOD_NON_TUMOR_FID)
    # ALL_TUMOR_FID.append(METHOD_TUMOR_FID)

for comb in combinations_stats:
    data1=comb[0]
    data2=comb[1]

    if data1[2]==2:
        MAE1=ALL_MAE[data1[0]][data1[1]]
        MSE1=ALL_MSE[data1[0]][data1[1]]
        PSNR1=ALL_PSNR[data1[0]][data1[1]]
        # SSIM1=ALL_SSIM[data1[0]][data1[1]]
        # FID1=ALL_FID[data1[0]][data1[1]]

    if data1[2]==0: # NON-TUMOUR
        MAE1=ALL_NON_TUMOR_MAE[data1[0]][data1[1]]
        MSE1=ALL_NON_TUMOR_MSE[data1[0]][data1[1]]
        PSNR1=ALL_NON_TUMOR_PSNR[data1[0]][data1[1]]
        # SSIM1=ALL_NON_TUMOR_SSIM[data1[0]][data1[1]]
        # FID1=ALL_NON_TUMOR_FID[data1[0]][data1[1]]

    elif data1[2]==1: # TUMOUR
        MAE1=ALL_TUMOR_MAE[data1[0]][data1[1]]
        MSE1=ALL_TUMOR_MSE[data1[0]][data1[1]]
        PSNR1=ALL_TUMOR_PSNR[data1[0]][data1[1]]
        # SSIM1=ALL_TUMOR_SSIM[data1[0]][data1[1]]
        # FID1=ALL_TUMOR_FID[data1[0]][data1[1]]

    if data2[2]==2:
        MAE2=ALL_MAE[data2[0]][data2[1]]
        MSE2=ALL_MSE[data2[0]][data2[1]]
        PSNR2=ALL_PSNR[data2[0]][data2[1]]
        # SSIM2=ALL_SSIM[data2[0]][data2[1]]
        # FID2=ALL_FID[data2[0]][data2[1]]

    if data2[2]==0:
        MAE2=ALL_NON_TUMOR_MAE[data2[0]][data2[1]]
        MSE2=ALL_NON_TUMOR_MSE[data2[0]][data2[1]]
        PSNR2=ALL_NON_TUMOR_PSNR[data2[0]][data2[1]]
        # SSIM2=ALL_NON_TUMOR_SSIM[data2[0]][data2[1]]
        # FID2=ALL_NON_TUMOR_FID[data2[0]][data2[1]]

    elif data2[2]==1:
        MAE2=ALL_TUMOR_MAE[data2[0]][data2[1]]
        MSE2=ALL_TUMOR_MSE[data2[0]][data2[1]]
        PSNR2=ALL_TUMOR_PSNR[data2[0]][data2[1]]
        # SSIM2=ALL_TUMOR_SSIM[data2[0]][data2[1]]
        # FID2=ALL_NON_TUMOR_FID[data2[0]][data2[1]]


    try:
        if shapiro(MAE1).pvalue>p_value and shapiro(MAE2).pvalue>p_value:
            if len(MAE1)==len(MAE2):
                stat=ttest_ind(MAE1, MAE2, equal_var=True)
                test_name = "T-Student Independente"
            else:
                stat=ttest_ind(MAE1, MAE2, equal_var=False)
                test_name = "Welch's T-Test"
            
            if stat.pvalue>p_value:
                conclusion="NÃO"
            else:
                conclusion="SIM"

            excel = excel+[["MAE",methods_names[data1[0]]+' | '+masks_percentage_names[data1[1]]+' | '+complexity_names[data1[2]], methods_names[data2[0]]+' | '+masks_percentage_names[data2[1]]+' | '+complexity_names[data2[2]],test_name,stat.pvalue,conclusion]]
        
        else:
            stat=mannwhitneyu(MAE1, MAE2)
            if stat.pvalue>p_value:
                conclusion="NÃO"
            else:
                conclusion="SIM"

            excel = excel+[["MAE",methods_names[data1[0]]+' | '+masks_percentage_names[data1[1]]+' | '+complexity_names[data1[2]], methods_names[data2[0]]+' | '+masks_percentage_names[data2[1]]+' | '+complexity_names[data2[2]],"Mann-Whitney",stat.pvalue,conclusion]]
        

        if shapiro(MSE1).pvalue>p_value and shapiro(MSE2).pvalue>p_value:
            if len(MSE1)==len(MSE2):
                stat=ttest_ind(MSE1, MSE2, equal_var=True)
                test_name = "T-Student Independente"
            else:
                stat=ttest_ind(MSE1, MSE2, equal_var=False)
                test_name = "Welch's T-Test"

            if stat.pvalue>p_value:
                conclusion="NÃO"
            else:
                conclusion="SIM"

            excel = excel+[["MSE",methods_names[data1[0]]+' | '+masks_percentage_names[data1[1]]+' | '+complexity_names[data1[2]], methods_names[data2[0]]+' | '+masks_percentage_names[data2[1]]+' | '+complexity_names[data2[2]],test_name,stat.pvalue,conclusion]]
        
        else:
            stat=mannwhitneyu(MSE1, MSE2)
            if stat.pvalue>p_value:
                conclusion="NÃO"
            else:
                conclusion="SIM"

            excel = excel+[["MSE",methods_names[data1[0]]+' | '+masks_percentage_names[data1[1]]+' | '+complexity_names[data1[2]], methods_names[data2[0]]+' | '+masks_percentage_names[data2[1]]+' | '+complexity_names[data2[2]],"Mann-Whitney",stat.pvalue,conclusion]]
        
        
        if shapiro(PSNR1).pvalue>p_value and shapiro(PSNR2).pvalue>p_value:
            if len(PSNR1)==len(PSNR2):
                stat=ttest_ind(PSNR1, PSNR2, equal_var=True)
                test_name = "T-Student Independente"
            else:
                stat=ttest_ind(PSNR1, PSNR2, equal_var=False)
                test_name = "Welch's T-Test"
                
            if stat.pvalue>p_value:
                conclusion="NÃO"
            else:
                conclusion="SIM"

            excel = excel+[["PSNR",methods_names[data1[0]]+' | '+masks_percentage_names[data1[1]]+' | '+complexity_names[data1[2]], methods_names[data2[0]]+' | '+masks_percentage_names[data2[1]]+' | '+complexity_names[data2[2]],test_name,stat.pvalue,conclusion]]
        
        else:
            stat=mannwhitneyu(PSNR1, PSNR2)
            if stat.pvalue>p_value:
                conclusion="NÃO"
            else:
                conclusion="SIM"

            excel = excel+[["PSNR",methods_names[data1[0]]+' | '+masks_percentage_names[data1[1]]+' | '+complexity_names[data1[2]], methods_names[data2[0]]+' | '+masks_percentage_names[data2[1]]+' | '+complexity_names[data2[2]],"Mann-Whitney",stat.pvalue,conclusion]]
        

        # if shapiro(SSIM1).pvalue>p_value and shapiro(SSIM2).pvalue>p_value:
        #     if len(SSIM1)==len(SSIM2):
        #         stat=ttest_ind(SSIM1, SSIM2, equal_var=True)
        #         test_name = "T-Student Independente"
        #     else:
        #         stat=ttest_ind(SSIM1, SSIM2, equal_var=False)
        #         test_name = "Welch's T-Test"
                
        #     if stat.pvalue>p_value:
        #         conclusion="NÃO"
        #     else:
        #         conclusion="SIM"

        #     excel = excel+[["SSIM",methods_names[data1[0]]+' | '+masks_percentage_names[data1[1]]+' | '+complexity_names[data1[2]], methods_names[data2[0]]+' | '+masks_percentage_names[data2[1]]+' | '+complexity_names[data2[2]],test_name,stat.pvalue,conclusion]]
        
        # else:
        #     stat=mannwhitneyu(SSIM1, SSIM2)
        #     if stat.pvalue>p_value:
        #         conclusion="NÃO"
        #     else:
        #         conclusion="SIM"

        #     excel = excel+[["SSIM",methods_names[data1[0]]+' | '+masks_percentage_names[data1[1]]+' | '+complexity_names[data1[2]], methods_names[data2[0]]+' | '+masks_percentage_names[data2[1]]+' | '+complexity_names[data2[2]],"Mann-Whitney",stat.pvalue,conclusion]]

        # if shapiro(FID1).pvalue>p_value and shapiro(FID2).pvalue>p_value:
        #     if len(FID1)==len(FID2):
        #         stat=ttest_ind(FID1, FID2, equal_var=True)
        #         test_name = "T-Student Independente"
        #     else:
        #         stat=ttest_ind(FID1, FID2, equal_var=False)
        #         test_name = "Welch's T-Test"
                
        #     if stat.pvalue>p_value:
        #         conclusion="NÃO"
        #     else:
        #         conclusion="SIM"

        #     excel = excel+[["FID",methods_names[data1[0]]+' | '+masks_percentage_names[data1[1]]+' | '+complexity_names[data1[2]], methods_names[data2[0]]+' | '+masks_percentage_names[data2[1]]+' | '+complexity_names[data2[2]],test_name,stat.pvalue,conclusion]]
        
        # else:
        #     stat=mannwhitneyu(FID1, FID2)
        #     if stat.pvalue>p_value:
        #         conclusion="NÃO"
        #     else:
        #         conclusion="SIM"

        #     excel = excel+[["FID",methods_names[data1[0]]+' | '+masks_percentage_names[data1[1]]+' | '+complexity_names[data1[2]], methods_names[data2[0]]+' | '+masks_percentage_names[data2[1]]+' | '+complexity_names[data2[2]],"Mann-Whitney",stat.pvalue,conclusion]]
        
        excel=excel+[[''] * 6]

    except:
        excel = excel+[["WITHOUT DATA",methods_names[data1[0]]+' | '+masks_percentage_names[data1[1]]+' | '+complexity_names[data1[2]], methods_names[data2[0]]+' | '+masks_percentage_names[data2[1]]+' | '+complexity_names[data2[2]],"None","None","None"]]
        excel=excel+[[''] * 6]
    
import pandas as pd
df1 = pd.DataFrame(excel[1:], columns=excel[0])
with pd.ExcelWriter("STATISTICS_TISSUES.xlsx", engine='openpyxl') as writer:
    df1.to_excel(writer, sheet_name='Stats_Tumours', index=False)

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.colors as mcolors

# # Method names and mask percentages
# methods_names = ['CE_Modified_LocalDiscriminator', 'CE_Modified_GlobalDiscriminator', 'GLCIC_Original', 'CA_Original', 'EC_Original', 'ESMII_Original', 'Multi_Slices']
# mask_percentages = [10, 20, 30, 40]

# # Metrics data organized by case type
# metrics_data = {
#     'MAE ↓': [ALL_NON_TUMOR_MAE, ALL_TUMOR_MAE],
#     'MSE ↓': [ALL_NON_TUMOR_MSE, ALL_TUMOR_MSE],
#     'PSNR ↑': [ALL_NON_TUMOR_PSNR, ALL_TUMOR_PSNR],
#     'SSIM ↑': [ALL_NON_TUMOR_SSIM, ALL_TUMOR_SSIM],
# }

# # Case types and their colors
# case_types = ['Non-Tumoral', 'Tumoral']
# case_colors = ['#cce3ff', '#ffcccc']  # Light Blue, Light Red
# position = [0.3, 0.72]

# # Custom color map and markers
# custom_colors = [
#     '#000000',  # Black
#     '#003f5c',  # Very Dark Blue
#     '#004e8c',  # Dark Blue
#     '#0069d9',  # Medium Blue
#     '#3389d6',  # Bright Blue
#     '#66aaff',  # Light Blue
#     '#99c2ff',  # Lighter Blue
#     '#cce3ff'   # Very Light Blue
# ]

# cmap = mcolors.LinearSegmentedColormap.from_list('custom_cool', custom_colors, N=len(custom_colors))
# markers = ['o', '^', 's', 'D', 'p', '*', 'H']

# # Create a 4x2 subplot grid
# fig, axes = plt.subplots(4, 2, figsize=(8, 8))
# axes = axes.flatten()

# # Increase the overall font size for better readability
# plt.rcParams.update({'font.size': 8})

# # Plotting each metric for each case type
# for i, (metric_name, metric_case_data) in enumerate(metrics_data.items()):
#     # Find common y-axis limits for the current metric
#     min_y, max_y = float('inf'), float('-inf')
#     for j in range(len(case_types)):
#         ALL_METRIC = metric_case_data[j]
#         for method_index in range(len(methods_names)):
#             for mask_index in range(len(mask_percentages)):
#                 metric_values = ALL_METRIC[method_index][mask_index]
#                 if metric_values is not None and len(metric_values) > 0:
#                     mean_value = np.mean(metric_values)
#                     min_y = min(min_y, mean_value)
#                     max_y = max(max_y, mean_value)

#     # Adjust the limits slightly to provide some padding
#     y_padding = (max_y - min_y) * 0.1
#     min_y -= y_padding
#     max_y += y_padding

#     for j, (case_type, color) in enumerate(zip(case_types, case_colors)):
#         ax = axes[i * 2 + j]  # Calculate position in the 4x2 grid
#         ALL_METRIC = metric_case_data[j]

#         # Set the same y-axis limits for the entire row
#         ax.set_ylim(min_y, max_y)

#         ax.set_xlim(9, 41)
#         ax.axvspan(0, 50, color=color, alpha=0.25, zorder=-1)

#         for method_index, method_name in enumerate(methods_names):
#             y_values = []

#             for mask_index in range(len(mask_percentages)):  # Iterate over mask percentages
#                 metric_values = ALL_METRIC[method_index][mask_index]

#                 mean_value = np.mean(metric_values) if metric_values is not None else 0
#                 y_values.append(mean_value)

#             plot_color = cmap(method_index)
#             ax.plot(mask_percentages, y_values, color=plot_color, marker=markers[method_index], markersize=3, linestyle='-', linewidth=0.3, label=method_name)

#         # Subtle vertical grid lines
#         ax.grid(True, which='both', axis='y', linestyle='--', color='gray', alpha=0.5)
#         ax.grid(True, which='both', axis='x', linestyle='--', color='gray', alpha=0.5)

#         if i == 3:  # For the last row
#             ax.set_xlabel('Mask Percentage', fontsize=8, fontweight='bold')  # X-axis label for each row

#         if j == 0:
#             # Set Y-axis label and rotate it vertically
#             ax.set_ylabel(metric_name, fontsize=12, fontweight='bold', rotation=90, labelpad=15)

#         ax.set_xticks(mask_percentages)
#         ax.set_xticklabels([f'{mp}%' for mp in mask_percentages])  # Add '%' symbol to x-axis labels

# # Add case type labels at the top of each column
# for j, case_type in enumerate(case_types):
#     plt.text(position[j], 0.91, case_type, ha='center', va='top', fontsize=12, fontweight='bold', transform=fig.transFigure)

# # Adjust layout and add a single legend at the bottom without overlapping
# handles, labels = axes[-1].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.92), ncol=4, fontsize=8)
# plt.tight_layout(rect=[0, 0, 1, 0.8])

# # Save the figure with high resolution
# plt.savefig('subplot_for_thesis_case_types.png', dpi=500, bbox_inches='tight')
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.colors as mcolors

# def format_ytick(ytick):
#     if ytick == 0:
#         return '0'  # Show zero as "0" without decimal places
#     elif abs(ytick) >= 2:
#         return f'{ytick:.0f}'  # Two decimal places for larger values
#     elif abs(ytick) >= 0.01 and abs(ytick) <= 1:
#         return f'{ytick:.3f}'  # Four decimal places for smaller values
#     elif ((abs(ytick) >= 1.027) and (abs(ytick) <= 1.030)):
#         return f'{ytick:.4f}'  # Four decimal places for smaller values
#     elif abs(ytick) < 0.01:
#         return f'{ytick:.1e}'  # Scientific notation for very small values

# # Method names and mask percentages
# methods_names = ['CE LOCAL', 'CE GLOBAL', 'GLCIC', 'CA', 'EC', 'ESMII', 'MS']
# mask_percentages = [10, 20, 30, 40]

# # Metrics data organized by case type
# metrics_data = {
#     'MAE ↓': [ALL_NON_TUMOR_MAE, ALL_TUMOR_MAE],
#     'MSE ↓': [ALL_NON_TUMOR_MSE, ALL_TUMOR_MSE],
#     'PSNR ↑': [ALL_NON_TUMOR_PSNR, ALL_TUMOR_PSNR],
#     'SSIM ↑': [ALL_NON_TUMOR_SSIM, ALL_TUMOR_SSIM],
# }

# # Case types and their colors
# case_types = ['NON-TUMOURAL', 'TUMOURAL']
# case_colors = ['#cce3ff', '#ffcccc']  # Light Blue, Light Red
# position = [0.22,0.53]

# # Custom color map and markers
# custom_colors = [
#     '#000000',  # Black
#     '#003f5c',  # Very Dark Blue
#     '#004e8c',  # Dark Blue
#     '#0069d9',  # Medium Blue
#     '#3389d6',  # Bright Blue
#     '#66aaff',  # Light Blue
#     '#99c2ff',  # Lighter Blue
#     '#cce3ff'   # Very Light Blue
# ]

# cmap = mcolors.LinearSegmentedColormap.from_list('custom_cool', custom_colors, N=len(custom_colors))
# markers = ['o', '^', 's', 'D', 'p', '*', 'H']

# # Create a 4x3 subplot grid
# fig, axes = plt.subplots(4, 3, figsize=(12, 8))
# axes = axes.flatten()

# # Increase the overall font size for better readability
# plt.rcParams.update({'font.size': 8})

# # Plotting each metric for each case type
# for i, (metric_name, metric_case_data) in enumerate(metrics_data.items()):
#     # Find common y-axis limits for the current metric
#     min_y, max_y = float('inf'), float('-inf')
#     for j in range(len(case_types)):
#         ALL_METRIC = metric_case_data[j]
#         for method_index in range(len(methods_names)):
#             for mask_index in range(len(mask_percentages)):
#                 metric_values = ALL_METRIC[method_index][mask_index]
#                 if metric_values is not None and len(metric_values) > 0:
#                     mean_value = np.mean(metric_values)
#                     min_y = min(min_y, mean_value)
#                     max_y = max(max_y, mean_value)

#     # Adjust the limits slightly to provide some padding
#     y_padding = (max_y - min_y) * 0.1
#     min_y -= y_padding
#     max_y += y_padding

#     for j, (case_type, color) in enumerate(zip(case_types, case_colors)):
#         ax = axes[i * 3 + j]  # Calculate position in the 4x3 grid
#         ALL_METRIC = metric_case_data[j]

#         # Set the same y-axis limits for the entire row
#         ax.set_ylim(min_y, max_y)

#         ax.set_xlim(9, 41)
#         ax.axvspan(0, 50, color=color, alpha=0.25, zorder=-1)

#         for method_index, method_name in enumerate(methods_names):
#             y_values = []

#             for mask_index in range(len(mask_percentages)):  # Iterate over mask percentages
#                 metric_values = ALL_METRIC[method_index][mask_index]

#                 mean_value = np.mean(metric_values) if metric_values is not None else 0
#                 y_values.append(mean_value)

#             plot_color = cmap(method_index)
#             ax.plot(mask_percentages, y_values, color=plot_color, marker=markers[method_index], markersize=3, linestyle='-', linewidth=0.3, label=method_name)

#         # Subtle vertical grid lines
#         ax.grid(True, which='both', axis='y', linestyle='--', color='gray', alpha=0.5)
#         ax.grid(True, which='both', axis='x', linestyle='--', color='gray', alpha=0.5)

#         if i == 3:  # For the last row
#             ax.set_xlabel('MASK PERCENTAGE', fontsize=6, fontweight='bold')  # X-axis label for each row

#         if j == 0:
#             # Set Y-axis label and rotate it vertically
#             ax.set_ylabel(metric_name, fontsize=12, fontweight='bold', rotation=90, labelpad=15)

#         ax.set_yticklabels([format_ytick(ytick) for ytick in ax.get_yticks()], fontsize=6)
#         ax.set_xticklabels([f'{mp}%' for mp in mask_percentages],fontsize=6)  # Add '%' symbol to x-axis labels
#         ax.set_xticks(mask_percentages)
#         ax.set_xticklabels([f'{mp}%' for mp in mask_percentages], fontsize=6)  # Add '%' symbol to x-axis labels

    
#     # Add the third column for the difference
#     ax_diff = axes[i * 3 + 2]
#     ALL_NON_TUMOR, ALL_TUMOR = metric_case_data

#     for method_index in range(len(methods_names)):
#         y_diff_values = []
#         # ax_diff.set_ylim(min_y, max_y)
#         for mask_index in range(len(mask_percentages)):
#             non_tumor_values = ALL_NON_TUMOR[method_index][mask_index]
#             tumor_values = ALL_TUMOR[method_index][mask_index]

#             mean_non_tumor = np.mean(non_tumor_values) if non_tumor_values is not None else 0
#             mean_tumor = np.mean(tumor_values) if tumor_values is not None else 0

#             y_diff_values.append(mean_non_tumor - mean_tumor)

#         plot_color = cmap(method_index)
#         ax_diff.plot(mask_percentages, y_diff_values, color=plot_color, marker=markers[method_index], markersize=3, linestyle='-', linewidth=0.3, label=methods_names[method_index])

#         if i == 3:  # For the last row
#             ax_diff.set_xlabel('MASK PERCENTAGE', fontsize=6, fontweight='bold')  # X-axis label for each row

#         ax.set_xticks(mask_percentages)
#         ax.set_xticklabels([f'{mp}%' for mp in mask_percentages])  # Add '%' symbol to x-axis labels

#     # Set y-axis limits for the difference plot
#     ax_diff.set_xlim(9, 41)
#     ax_diff.axhline(0, color='gray', linestyle='--')
#     ax_diff.grid(True, which='both', axis='both', linestyle='--', color='gray', alpha=0.5)

#     ax_diff.set_yticklabels([format_ytick(ytick) for ytick in ax.get_yticks()], fontsize=6)
#     ax_diff.set_xticklabels([f'{mp}%' for mp in mask_percentages],fontsize=6)  # Add '%' symbol to x-axis labels
#     ax_diff.set_xticks(mask_percentages)
#     ax_diff.set_xticklabels([f'{mp}%' for mp in mask_percentages], fontsize=6)  # Add '%' symbol to x-axis labels



# # Add case type labels at the top of each column
# for j, case_type in enumerate(case_types):
#     plt.text(position[j], 0.91, case_type, ha='center', va='top', fontsize=12, fontweight='bold', transform=fig.transFigure)
# plt.text(0.83, 0.91, "DIFFERENCES", ha='center', va='top', fontsize=12, fontweight='bold', transform=fig.transFigure)

# plt.subplots_adjust(left=0.1, right=0.95, top=0.89, bottom=0.1, hspace=0.27, wspace=0.3)
# # Adjust layout and add a single legend at the bottom without overlapping
# handles, labels = axes[-3].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.92), ncol=7, fontsize=10)
# plt.tight_layout(rect=[0, 0, 1, 0.8])

# # Save the figure with high resolution
# plt.savefig('tumour_graphs.png', dpi=300, bbox_inches='tight')
# plt.show()
