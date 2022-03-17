# python plot_curves.py

# pylint: disable=invalid-name,trailing-whitespace,superfluous-parens,line-too-long,multiple-statements, unnecessary-semicolon, redefined-outer-name, consider-using-enumerate

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import socket
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob

import constants

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--vaccination_time', type=int, default=31,
                    help='Time to distribute vaccines.')
parser.add_argument('--vaccination_ratio' , type=float, default=0.1,
                    help='Vaccination ratio relative to MSA population.')
args = parser.parse_args()

# root
hostname = socket.gethostname()
print('hostname: ', hostname)
if(hostname=='fib-dl3'): 
    root = '/data/chenlin/COVID-19/Data'
    saveroot = '/data/chenlin/utility-equity-covid-vac'
elif(hostname=='rl4'):
    root = '/home/chenlin/COVID-19/Data'
    saveroot = '/home/chenlin/utility-equity-covid-vac'

# subroot
subroot = 'results/figures'
if not os.path.exists(os.path.join(root, subroot)): # if folder does not exist, create one. #2022032
    os.makedirs(os.path.join(root, subroot))


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
# Fig.S1

color_list = ['C0','C1','grey','C2']
alpha_list = [1,1,0.6,1]

for msa_idx in range(len(msa_name_list)):
    this_msa = msa_name_list[msa_idx]

    # No_Vaccination, accumulated results
    deaths_total_no_vaccination = np.load(os.path.join(root,this_msa,f'20210206_deaths_total_no_vaccination_{this_msa}.npy'))
    #history_D2_no_vac = np.fromfile(os.path.join(root, this_msa, 'vaccination_results_adaptive_31d_0.1_0.01', f'20210206_history_D2_no_vaccination_adaptive_{args.vaccination_ratio}_{this_recheck_interval}_30seeds_{this_msa}')) #20220313
    #history_D2_no_vac = np.reshape(history_D2_no_vac,(63, 30, -1))
    #mean_no_vac,max_no_vac,min_no_vac = get_mean_max_min(history_D2_no_vac)
    #deaths_total_no_vaccination = np.mean(np.sum(history_D2_no_vac, axis=2),axis=1)
    #print(deaths_total_no_vaccination.shape)
    #pdb.set_trace()
    deaths_daily_total_no_vaccination = [0]
    for i in range(1,len(deaths_total_no_vaccination)):
        deaths_daily_total_no_vaccination.append(deaths_total_no_vaccination[i]-deaths_total_no_vaccination[i-1])

    
    # Age_Agnostic, accumulated results
    deaths_total_age_agnostic = np.load(os.path.join(root,this_msa,f'20210206_deaths_total_age_agnostic_{this_msa}.npy'))
    # Transform into daily results
    deaths_daily_total_age_agnostic = [0]
    for i in range(1,len(deaths_total_age_agnostic)):
        deaths_daily_total_age_agnostic.append(deaths_total_age_agnostic[i]-deaths_total_age_agnostic[i-1])

    # No_Vaccination, upper & lower bound, accumulated results
    upperbound = np.load(os.path.join(root,this_msa,'age_aware_1.5_upperbound_%s_%s.npy'%(constants.upper_lower_death_scales[1.5][this_msa][1],this_msa)))
    upperbound_daily = [0]
    for i in range(1,len(upperbound)):
        upperbound_daily.append(upperbound[i]-upperbound[i-1])
    lowerbound = np.load(os.path.join(root,this_msa,'age_aware_1.5_lowerbound_%s_%s.npy'%(constants.upper_lower_death_scales[1.5][this_msa][0],this_msa)))
    lowerbound_daily = [0]
    for i in range(1,len(lowerbound)):
        lowerbound_daily.append(lowerbound[i]-lowerbound[i-1])
        
    # Standard SEIR
    predicted_deaths_daily = np.fromfile(os.path.join(saveroot, 'results/seir', f'seir_daily_deaths_{this_msa}'))

    # NYT ground truth, daily & daily_smooth
    deaths_daily = np.load(os.path.join(root,this_msa,'20210206_deaths_daily_nyt_%s.npy'%this_msa))
    deaths_daily_smooth = np.load(os.path.join(root,this_msa,'20210206_deaths_daily_smooth_nyt_%s.npy'%this_msa))


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

    #plt.ylim(0,11)
    #plt.yticks(np.arange(6)*2,fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.ylabel('Daily deaths', fontsize=25)

    plt.xlim(-1,63)
    plt.xticks(np.arange(13)*5,fontsize=12)
    plt.xlabel('Days',fontsize=25)
    #plt.legend(loc='upper left',fontsize=17.5)

    # Save the figure
    savepath = os.path.join(saveroot, subroot , 'sup', f'sup_model_curve_{this_msa}.pdf')
    plt.savefig(savepath,bbox_inches = 'tight')
    print(f'sup_curve_{this_msa}, figure saved at: {savepath}.')

# legend
plt.figure()
label_list = ['Meta-population model', 'Our proposed BD model', 'SEIR model', 'Ground truth']
patches = [plt.scatter([],[],marker='o',s=500,color=color_list[i], label="{:s}".format(label_list[i])) for i in range(len(label_list))]
plt.legend(handles=patches,ncol=2,fontsize=20,bbox_to_anchor=(0.8,-0.1)) 
# Save figure
savepath = os.path.join(saveroot, subroot , 'sup', f'sup_model_curve_legend.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Supplementary legend, saved at {savepath}')

#pdb.set_trace()

####################################################################################################
# Fig.S18

color_list = ['k','#FE2E2E','blue']

for msa_idx in range(len(msa_name_list)):
    this_msa = msa_name_list[msa_idx]

    # No_Vaccination, accumulated results
    #deaths_total_no_vac = np.load(os.path.join(root,this_msa,f'20210206_deaths_total_no_vaccination_{this_msa}.npy'))
    history_D2_no_vac = np.fromfile(os.path.join(root, this_msa, 'vaccination_results_adaptive_31d_0.1_0.01', f'20210206_history_D2_no_vaccination_adaptive_{args.vaccination_ratio}_{this_recheck_interval}_30seeds_{this_msa}')) #20220313
    history_D2_no_vac = np.reshape(history_D2_no_vac,(63, 30, -1))
    mean_no_vac,max_no_vac,min_no_vac = get_mean_max_min(history_D2_no_vac)
    '''
    deaths_total_mean_no_vac = np.mean(np.sum(history_D2_no_vac, axis=2), axis=1) 
    deaths_total_max_no_vac = np.max(np.sum(history_D2_no_vac, axis=2), axis=1) 
    deaths_total_min_no_vac = np.min(np.sum(history_D2_no_vac, axis=2), axis=1) 
    # Transform into daily results
    mean_no_vac = [0]
    max_no_vac = [0]
    min_no_vac = [0]
    for i in range(1,len(deaths_total_mean_no_vac)):
        mean_no_vac.append(deaths_total_mean_no_vac[i]-deaths_total_mean_no_vac[i-1])
        max_no_vac.append(deaths_total_max_no_vac[i]-deaths_total_max_no_vac[i-1])
        min_no_vac.append(deaths_total_min_no_vac[i]-deaths_total_min_no_vac[i-1])
    '''

    # Baseline, accumulated results
    history_D2_baseline = np.fromfile(os.path.join(root, this_msa, 'vaccination_results_adaptive_31d_0.1_0.01', f'test_history_D2_baseline_adaptive_31d_{args.vaccination_ratio}_{this_recheck_interval}_30seeds_{this_msa}')) #20220313
    history_D2_baseline = np.reshape(history_D2_baseline,(63, 30, -1))
    mean_baseline,max_baseline,min_baseline = get_mean_max_min(history_D2_baseline)

    # Comprehensive, accumulated results
    policy = 'hybrid'
    list_glob = glob.glob(os.path.join(saveroot, f'results/comprehensive/vac_results_{args.vaccination_time}d_{args.vaccination_ratio}_{this_recheck_interval}', f'history_D2_{policy}_{args.vaccination_time}d_{args.vaccination_ratio}_{this_recheck_interval}_*_30seeds_{this_msa}'))
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

    #plt.ylim(0,11)
    #plt.yticks(np.arange(6)*2,fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.ylabel('Daily deaths', fontsize=25)

    plt.xlim(-1,63)
    plt.xticks(np.arange(13)*5+1,fontsize=12)
    plt.xlabel('Days',fontsize=25)
    #plt.legend(loc='upper left',fontsize=17.5)

    # Save the figure
    savepath = os.path.join(saveroot, subroot , 'sup', f'sup_withvac_curve_{this_msa}.pdf')
    plt.savefig(savepath,bbox_inches = 'tight')
    print(f'sup_curve_{this_msa}, figure saved at: {savepath}.')

# legend
plt.figure()
label_list = ['No Vaccination', 'Comprehensive', 'Homogeneous']
color_list = color_list
patches = [plt.scatter([],[],marker='o',s=500,color=color_list[i], label="{:s}".format(label_list[i])) for i in range(len(label_list))]
plt.legend(handles=patches,ncol=3,fontsize=20,bbox_to_anchor=(0.8,-0.1)) 
# Save figure
savepath = os.path.join(saveroot, subroot , 'sup', f'sup_withvac__curve_legend.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Supplementary legend, saved at {savepath}')
