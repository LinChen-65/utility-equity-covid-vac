# python plot_curves.py (for Fig. 1(b)(c), Supplementary Fig.1)
# python plot_curves.py --with_vac (for Supplementary Fig.18)


import socket
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import constants

# root
root = os.getcwd()
dataroot = os.path.join(root, 'data')
result_root = os.path.join(root, 'results')
fig_save_root = os.path.join(root, 'figures')

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--vaccination_time', type=int, default=31,
                    help='Time to distribute vaccines.')
parser.add_argument('--vaccination_ratio' , type=float, default=0.1,
                    help='Vaccination ratio relative to MSA population.')
parser.add_argument('--safegraph_root', default=dataroot, 
                    help='Safegraph data root.')                    
parser.add_argument('--with_vac', default=False, action='store_true',
                    help='If true, plot with comprehensive vaccination results.')
args = parser.parse_args()



msa_name_list = ['Atlanta', 'Chicago', 'Dallas', 'Houston', 'LosAngeles', 'Miami', 'Philadelphia', 'SanFrancisco', 'WashingtonDC']
anno_list = ['Atlanta', 'Chicago', 'Dallas', 'Houston', 'Los Angeles', 'Miami', 'Philadelphia', 'San Francisco', 'Washington D.C.']
this_recheck_interval = 0.01

def get_mean_max_min(history_D2): #20220313
    deaths_total_mean = np.mean(np.sum(history_D2, axis=2), axis=1) #(63,30,3130)->(63,30)->(63,)
    deaths_total_max = np.max(np.sum(history_D2, axis=2), axis=1) 
    deaths_total_min = np.min(np.sum(history_D2, axis=2), axis=1) 
    # Transform into daily results
    mean_list = [0]
    max_list = [0]
    min_list = [0]
    for i in range(1,len(deaths_total_mean)):
        mean_list.append(deaths_total_mean[i]-deaths_total_mean[i-1])
        max_list.append(deaths_total_max[i]-deaths_total_max[i-1])
        min_list.append(deaths_total_min[i]-deaths_total_min[i-1])
    return mean_list, max_list, min_list

####################################################################################################
# Fig. 1(c)

from math import sqrt
from sklearn.metrics import mean_squared_error

# Normalized by the mean of daily deaths smooth
deaths_daily_rmse_seed_age_agnostic_msa = np.load(os.path.join(result_root,'20210206_norm_by_mean_deaths_daily_rmse_seed_age_agnostic_msa.npy'),
                                                  allow_pickle=True).item()
deaths_daily_rmse_seed_no_vaccination_msa = np.load(os.path.join(result_root,'20210206_norm_by_mean_deaths_daily_rmse_seed_no_vaccination_msa.npy'),
                                                    allow_pickle=True).item()
bar_standardSEIR = [0.46, 0.97, 0.95, 0.41, 0.77, 0.84, 1.13, 0.75, 1.05] # generated by standard_seir.py

count=0
rmse_no_vaccination = np.zeros(9)
rmse_age_agnostic = np.zeros(9)
for j in range(len(constants.MSA_NAME_LIST)):
    if(constants.MSA_NAME_LIST[j]=='NewYorkCity'):continue
    MSA_NAME = constants.MSA_NAME_LIST[j];print(MSA_NAME)
    print(count)
    deaths_total_no_vaccination = np.load(os.path.join(result_root,'20210206_deaths_total_no_vaccination_%s.npy'%MSA_NAME))
    deaths_total_age_agnostic = np.load(os.path.join(result_root,'20210206_deaths_total_age_agnostic_%s.npy'%MSA_NAME))

    deaths_daily_total_no_vaccination = [0]
    for i in range(1,len(deaths_total_no_vaccination)):
        deaths_daily_total_no_vaccination.append(deaths_total_no_vaccination[i]-deaths_total_no_vaccination[i-1])

    deaths_daily_total_age_agnostic = [0]
    for i in range(1,len(deaths_total_age_agnostic)):
        deaths_daily_total_age_agnostic.append(deaths_total_age_agnostic[i]-deaths_total_age_agnostic[i-1])

    deaths_daily_smooth = np.load(os.path.join(result_root,'20210206_deaths_daily_smooth_nyt_%s.npy'%MSA_NAME))
    
    rmse_no_vaccination[count] = sqrt(mean_squared_error(deaths_daily_smooth,deaths_daily_total_no_vaccination))/np.mean(deaths_daily_smooth)
    rmse_age_agnostic[count] = sqrt(mean_squared_error(deaths_daily_smooth,deaths_daily_total_age_agnostic))/np.mean(deaths_daily_smooth)
    print(rmse_no_vaccination[count])
    print(rmse_age_agnostic[count])
    print(bar_standardSEIR[count])
    count += 1
bar_agnostic = rmse_age_agnostic
bar_aware = rmse_no_vaccination

improve_agnostic = []
improve_standardSEIR = []
for i in range(len(bar_aware)):
    improve_agnostic.append(-(bar_aware[i]-bar_agnostic[i])/bar_agnostic[i])
    improve_standardSEIR.append(-(bar_aware[i]-bar_standardSEIR[i])/bar_standardSEIR[i])
print('NRMSE reduction compared to agnostic:', improve_agnostic, np.max(np.array(improve_agnostic)))
print('NRMSE reduction compared to standardSEIR:', improve_standardSEIR, np.max(np.array(improve_standardSEIR)))   

error_agnostic = np.zeros(9)
error_aware = np.zeros(9)
count=0
for i in range(len(constants.MSA_NAME_LIST)):
    if(constants.MSA_NAME_LIST[i]=='NewYorkCity'):continue
    print(count)
    error_agnostic[count] = np.std(deaths_daily_rmse_seed_age_agnostic_msa[constants.MSA_NAME_LIST[i]][1])
    error_aware[count] = np.std(deaths_daily_rmse_seed_no_vaccination_msa[constants.MSA_NAME_LIST[i]][1])
    count += 1

bar_agnostic = np.around(bar_agnostic,2)
bar_aware = np.around(bar_aware,2)
error_agnostic = np.around(error_agnostic,2)
error_aware = np.around(error_aware,2)
print('bar_agnostic:',bar_agnostic)
print('bar_aware',bar_aware)

# Re-order
bar_standardSEIR_reorder = np.sort(bar_standardSEIR)[::-1]
ref = np.argsort(bar_standardSEIR)
print('bar_standardSEIR_reorder:',bar_standardSEIR_reorder)
print('ref:',ref)

bar_agnostic_reorder = bar_agnostic[ref[::-1]]
bar_aware_reorder = bar_aware[ref[::-1]]
error_agnostic_reorder = error_agnostic[ref[::-1]]
error_aware_reorder = error_aware[ref[::-1]]
print('bar_aware_reorder:',bar_aware_reorder)

bar_standardSEIR = np.around(bar_standardSEIR,2)
bar_standardSEIR_reorder = bar_standardSEIR[ref[::-1]]

anno_list = ['Atlanta','Chicago','Dallas','Houston', 'L.A.','Miami','Phila.','S.F.','D.C.']


# Horizontal bars
height = 1
distance = height*4
fig,ax = plt.subplots(figsize=(4.3,7))
ax.barh(np.arange(9)*distance+height*1.1,bar_standardSEIR_reorder,label='SEIR model',color='grey',alpha=0.6,height=height)
ax.barh(np.arange(9)*distance,bar_agnostic_reorder,label='Meta-population model',xerr=error_agnostic_reorder,height=height)
ax.barh(np.arange(9)*distance-height*1.1,bar_aware_reorder,label='BD model',xerr=error_aware_reorder,alpha=1,height=height)
ax.legend(fontsize=14.8, ncol=1,loc='upper center',bbox_to_anchor=(0.5,1.22))

anno_list_reorder = np.array(anno_list)[ref[::-1]]
plt.yticks(np.arange(9)*distance, anno_list_reorder,fontsize=18)#,rotation=20
plt.xlabel('NRMSE of daily deaths',fontsize=19)
plt.xticks(fontsize=13)
savepath = os.path.join(fig_save_root, 'fig1c.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Fig1c, saved at {savepath}')

####################################################################################################
# Fig.S1

color_list = ['C0','C1','grey','C2']
alpha_list = [1,1,0.6,1]

for msa_idx in range(len(msa_name_list)):
    this_msa = msa_name_list[msa_idx]

    # No_Vaccination, accumulated results
    history_D2_no_vac = np.fromfile(os.path.join(result_root, 'vaccination_results_adaptive_31d_0.1_0.01', f'20210206_history_D2_no_vaccination_adaptive_{args.vaccination_ratio}_{this_recheck_interval}_30seeds_{this_msa}')) 
    history_D2_no_vac = np.reshape(history_D2_no_vac,(63, 30, -1))
    mean_no_vac,max_no_vac,min_no_vac = get_mean_max_min(history_D2_no_vac)
    deaths_daily_total_no_vaccination = mean_no_vac

    # Age_Agnostic, accumulated results
    history_D2_age_agnostic = np.fromfile(os.path.join(result_root, 'vaccination_results_adaptive_31d_0.1_0.01', f'20210206_history_D2_age_agnostic_adaptive_{args.vaccination_ratio}_{this_recheck_interval}_30seeds_{this_msa}')) 
    history_D2_age_agnostic = np.reshape(history_D2_age_agnostic,(63, 30, -1))
    mean_age_agnostic,max_age_agnostic,min_age_agnostic = get_mean_max_min(history_D2_age_agnostic)
    deaths_daily_total_age_agnostic = mean_age_agnostic

    # No_Vaccination, upper & lower bound, accumulated results
    upperbound = np.load(os.path.join(result_root,'age_aware_1.5_upperbound_%s_%s.npy'%(constants.upper_lower_death_scales[1.5][this_msa][1],this_msa)))
    upperbound_daily = [0]
    for i in range(1,len(upperbound)):
        upperbound_daily.append(upperbound[i]-upperbound[i-1])
    lowerbound = np.load(os.path.join(result_root,'age_aware_1.5_lowerbound_%s_%s.npy'%(constants.upper_lower_death_scales[1.5][this_msa][0],this_msa)))
    lowerbound_daily = [0]
    for i in range(1,len(lowerbound)):
        lowerbound_daily.append(lowerbound[i]-lowerbound[i-1])
        
    # Standard SEIR
    predicted_deaths_daily = np.fromfile(os.path.join(result_root, f'seir_daily_deaths_{this_msa}'))

    # NYT ground truth, daily & daily_smooth
    deaths_daily = np.load(os.path.join(result_root,'20210206_deaths_daily_nyt_%s.npy'%this_msa))
    deaths_daily_smooth = np.load(os.path.join(result_root,'20210206_deaths_daily_smooth_nyt_%s.npy'%this_msa))


    plt.figure(figsize=(6,5.5))
    plt.title(anno_list[msa_idx],fontsize=30)
    plt.plot(upperbound_daily,color='peachpuff')
    plt.plot(lowerbound_daily,color='peachpuff')
    t = np.arange(63)
    plt.fill_between(t,upperbound_daily,lowerbound_daily,color='peachpuff')

    markersize= 7.5 #6
    alpha = 0.1
    
    plt.plot(deaths_daily_total_age_agnostic,label='Meta-population',marker='o',markersize=markersize,color=color_list[0],alpha=alpha_list[0]) 
    plt.plot(deaths_daily_total_no_vaccination,label='BD',marker='o',markersize=markersize,color=color_list[1],alpha=alpha_list[1]) 
    plt.plot(predicted_deaths_daily,label='Standard SEIR',marker='o',markersize=markersize,color=color_list[2],alpha=alpha_list[2])
    #plt.scatter(np.arange(len(deaths_daily)),deaths_daily,color='grey',marker='x') 
    plt.plot(deaths_daily_smooth,label='Ground Truth',marker='o',markersize=markersize,color=color_list[3], alpha=alpha_list[3])

    plt.yticks(fontsize=14) 
    plt.ylabel('Daily deaths', fontsize=25)

    plt.xlim(-1,63)
    plt.xticks(np.arange(13)*5,fontsize=12)
    plt.xlabel('Days',fontsize=25)
    #plt.legend(loc='upper left',fontsize=17.5)

    # Save the figure
    savepath = os.path.join(fig_save_root, 'sup', f'sup_model_curve_{this_msa}.pdf')
    plt.savefig(savepath,bbox_inches = 'tight')
    print(f'sup_curve_{this_msa}, figure saved at: {savepath}.')

# legend
plt.figure()
label_list = ['Meta-population model', 'Our proposed BD model', 'SEIR model', 'Ground truth']
patches = [plt.scatter([],[],marker='o',s=500,color=color_list[i], label="{:s}".format(label_list[i])) for i in range(len(label_list))]
plt.legend(handles=patches,ncol=2,fontsize=20,bbox_to_anchor=(0.8,-0.1)) 
# Save figure
savepath = os.path.join(fig_save_root, 'sup', f'sup_model_curve_legend.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Supplementary legend, saved at {savepath}')



####################################################################################################
# Fig.S18

if(args.with_vac):
    color_list = ['k','#FE2E2E','blue']

    for msa_idx in range(len(msa_name_list)):
        this_msa = msa_name_list[msa_idx]

        # No_Vaccination, accumulated results
        history_D2_no_vac = np.fromfile(os.path.join(result_root, 'vaccination_results_adaptive_31d_0.1_0.01', f'20210206_history_D2_no_vaccination_adaptive_{args.vaccination_ratio}_{this_recheck_interval}_30seeds_{this_msa}')) 
        history_D2_no_vac = np.reshape(history_D2_no_vac,(63, 30, -1))
        mean_no_vac,max_no_vac,min_no_vac = get_mean_max_min(history_D2_no_vac)

        # Baseline, accumulated results
        history_D2_baseline = np.fromfile(os.path.join(result_root, 'vaccination_results_adaptive_31d_0.1_0.01', f'test_history_D2_baseline_adaptive_31d_{args.vaccination_ratio}_{this_recheck_interval}_30seeds_{this_msa}')) 
        history_D2_baseline = np.reshape(history_D2_baseline,(63, 30, -1))
        mean_baseline,max_baseline,min_baseline = get_mean_max_min(history_D2_baseline)

        # Comprehensive, accumulated results
        policy = 'hybrid'
        list_glob = glob.glob(os.path.join(result_root, f'comprehensive/vac_results_{args.vaccination_time}d_{args.vaccination_ratio}_{this_recheck_interval}', f'history_D2_{policy}_{args.vaccination_time}d_{args.vaccination_ratio}_{this_recheck_interval}_*_30seeds_{this_msa}'))
        result_path = list_glob[0]
        history_D2_hybrid = np.fromfile(result_path)
        history_D2_hybrid = np.reshape(history_D2_hybrid ,(63, 30, -1)) #(63,30,3130)
        mean_hybrid,max_hybrid,min_hybrid = get_mean_max_min(history_D2_hybrid)
        
        plt.figure(figsize=(6,5.5))
        plt.title(anno_list[msa_idx],fontsize=30)
        t = np.arange(63)
        markersize=6
        alpha = 0.1

        plt.plot(mean_no_vac,label='No Vaccination',marker='o',markersize=markersize,color=color_list[0]) 
        plt.plot(max_no_vac,marker='o',markersize=markersize,color=color_list[0],alpha=alpha) 
        plt.plot(min_no_vac,marker='o',markersize=markersize,color=color_list[0],alpha=alpha) 
        plt.fill_between(t,max_no_vac,min_no_vac,color=color_list[0],alpha=alpha)

        plt.plot(mean_baseline,label='Baseline',marker='o',markersize=markersize,color=color_list[2]) 
        plt.plot(max_baseline,marker='o',markersize=markersize,color=color_list[2],alpha=alpha) 
        plt.plot(min_baseline,marker='o',markersize=markersize,color=color_list[2],alpha=alpha) 
        plt.fill_between(t,max_baseline,min_baseline,color=color_list[2],alpha=alpha)

        plt.plot(mean_hybrid,label='Comprehensive',marker='o',markersize=markersize,color=color_list[1]) 
        plt.plot(max_hybrid,marker='o',markersize=markersize,color=color_list[1],alpha=alpha)
        plt.plot(min_hybrid,marker='o',markersize=markersize,color=color_list[1],alpha=alpha)
        plt.fill_between(t,max_hybrid,min_hybrid,color=color_list[1],alpha=alpha)

        y_max = np.max(np.array([np.max(mean_no_vac),np.max(max_no_vac),np.max(min_no_vac), 
                                np.max(mean_baseline),np.max(max_baseline),np.max(min_baseline),
                                np.max(mean_hybrid),np.max(max_hybrid),np.max(min_hybrid)])) #20220316
        plt.vlines(31, 0, y_max, colors='black',linestyles ="dashed") #20220316

        plt.yticks(fontsize=14) 
        plt.ylabel('Daily deaths', fontsize=25)

        plt.xlim(-1,63)
        plt.xticks(np.arange(13)*5+1,fontsize=12)
        plt.xlabel('Days',fontsize=25)

        # Save the figure
        savepath = os.path.join(fig_save_root, 'sup', f'sup_withvac_curve_{this_msa}.pdf')
        plt.savefig(savepath,bbox_inches = 'tight')
        print(f'sup_curve_{this_msa}, figure saved at: {savepath}.')

    # legend
    plt.figure()
    label_list = ['No Vaccination', 'Comprehensive', 'Homogeneous']
    color_list = color_list
    patches = [plt.scatter([],[],marker='o',s=500,color=color_list[i], label="{:s}".format(label_list[i])) for i in range(len(label_list))]
    plt.legend(handles=patches,ncol=3,fontsize=20,bbox_to_anchor=(0.8,-0.1)) 
    # Save figure
    savepath = os.path.join(fig_save_root, 'sup', f'sup_withvac_curve_legend.pdf')
    plt.savefig(savepath,bbox_inches = 'tight')
    print(f'Supplementary legend, saved at {savepath}')

