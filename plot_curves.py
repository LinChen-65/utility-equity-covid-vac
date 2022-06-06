# python plot_curves.py (for Fig. 1(b)(c), Supplementary Fig.1)
# python plot_curves.py --with_vac (for Supplementary Fig.18)


import socket
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import stats
from math import sqrt
from sklearn.metrics import mean_squared_error
import functions
import constants
import time
import seaborn as sns

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
#anno_list = ['Atlanta', 'Chicago', 'Dallas', 'Houston', 'Los Angeles', 'Miami', 'Philadelphia', 'San Francisco', 'Washington D.C.']
anno_list = ['Atlanta','Chicago','Dallas','Houston', 'L.A.','Miami','Phila.','S.F.','D.C.']
this_recheck_interval = 0.01
num_seeds = 60 
num_days = 63

def get_mean_max_min(history_D2):
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

def get_ci(data):
    bounds = list(stats.t.interval(alpha=0.99, df=len(data)-1, loc=np.mean(data), scale=stats.sem(data)))
    lower = bounds[0]
    upper = bounds[1]
    return lower, upper

####################################################################################################
# Fig. 1(c)

# Load cbg dict
dict_savepath = os.path.join(dataroot, 'cbg_nyt_dict.npy')
cbg_dict = np.load(dict_savepath, allow_pickle=True).item()

bar_standardSEIR = []
for msa_idx in range(len(msa_name_list)):
    this_msa = msa_name_list[msa_idx]
    predicted_deaths_daily = np.fromfile(os.path.join(result_root, f'new_seir_daily_deaths_{this_msa}'))
    # NYT ground truth, daily & daily_smooth
    deaths_daily = np.load(os.path.join(result_root,'20210206_deaths_daily_nyt_%s.npy'%this_msa))
    deaths_daily_smooth = np.load(os.path.join(result_root,'20210206_deaths_daily_smooth_nyt_%s.npy'%this_msa))
    RMSE = sqrt(mean_squared_error(deaths_daily_smooth, predicted_deaths_daily))
    normalizer = deaths_daily_smooth.mean()
    RMSE_norm = RMSE / normalizer
    bar_standardSEIR.append(np.round(RMSE_norm, 2))
    #print(predicted_deaths_daily)
bar_standardSEIR = np.array(bar_standardSEIR)

scatter_age_agnostic_path = os.path.join(result_root, 'scatter_age_agnostic.npy')
scatter_no_vac_path = os.path.join(result_root, 'scatter_no_vac.npy')
if(os.path.exists(scatter_age_agnostic_path)):
    scatter_age_agnostic = np.load(scatter_age_agnostic_path, allow_pickle=True).item()
    scatter_no_vac = np.load(scatter_no_vac_path, allow_pickle=True).item()
    bar_agnostic = []
    error_agnostic = []
    bar_aware = []
    error_aware = []
    for msa_idx in range(len(msa_name_list)):
        this_msa = msa_name_list[msa_idx]
        bar_agnostic.append(np.array(scatter_age_agnostic[this_msa]).mean())
        error_agnostic.append(np.array(scatter_age_agnostic[this_msa]).std())
        bar_aware.append(np.array(scatter_no_vac[this_msa]).mean())
        error_aware.append(np.array(scatter_no_vac[this_msa]).std())
    bar_agnostic = np.array(bar_agnostic)
    error_agnostic = np.array(error_agnostic)
    bar_aware = np.array(bar_aware)
    error_aware = np.array(error_aware)
else:
    np.random.seed(42)
    num_samples = args.num_samples #150 #200 #150 #100
    per_sample_size = args.per_sample_size #5
    print('num_samples:', num_samples,'per_sample_size:', per_sample_size)
    scatter_age_agnostic = {}
    scatter_no_vac = {}
    bar_agnostic = []
    error_agnostic = []
    error_aware = []
    bar_aware = []

    start = time.time()
    for msa_idx in range(len(constants.MSA_NAME_LIST)):
        this_msa = constants.MSA_NAME_LIST[msa_idx]
        scatter_age_agnostic[this_msa] = []
        scatter_no_vac[this_msa] = []

        # NYT ground truth, daily & daily_smooth
        deaths_daily = np.load(os.path.join(result_root,'20210206_deaths_daily_nyt_%s.npy'%this_msa))
        deaths_daily_smooth = np.load(os.path.join(result_root,'20210206_deaths_daily_smooth_nyt_%s.npy'%this_msa))
        policy = 'Age_Agnostic'.lower()
        age_agnostic_filepath = os.path.join(result_root, 'vaccination_results_adaptive_31d_0.1_0.01', r'20210206_history_D2_%s_adaptive_%s_0.01_%sseeds_%s') % (policy,args.vaccination_ratio,num_seeds,this_msa)
        history_D2_age_agnostic = np.fromfile(age_agnostic_filepath)
        history_D2_age_agnostic = np.reshape(history_D2_age_agnostic,(num_days, num_seeds, -1))
        policy = 'No_Vaccination'.lower()
        no_vac_filepath = os.path.join(result_root, 'vaccination_results_adaptive_31d_0.1_0.01', r'20210206_history_D2_%s_adaptive_%s_0.01_%sseeds_%s') % (policy,args.vaccination_ratio,num_seeds,this_msa)
        history_D2_no_vac = np.fromfile(no_vac_filepath)
        history_D2_no_vac = np.reshape(history_D2_no_vac,(num_days, num_seeds, -1))

        # Compare 
        rng = np.random.default_rng(42)
        for sample_idx in range(num_samples):
            subset = rng.choice(num_seeds, size=per_sample_size, replace=False)
            _, deaths_total_age_agnostic = functions.average_across_random_seeds_only_death(history_D2_age_agnostic[:,subset,:], history_D2_age_agnostic.shape[-1], cbg_dict[this_msa], print_results=False)
            deaths_daily_total_age_agnostic = [0]
            for i in range(1,len(deaths_total_age_agnostic)):
                deaths_daily_total_age_agnostic.append(deaths_total_age_agnostic[i]-deaths_total_age_agnostic[i-1])
            RMSE = sqrt(mean_squared_error(deaths_daily_smooth, deaths_daily_total_age_agnostic))
            normalizer = deaths_daily_smooth.mean()
            RMSE_norm = RMSE / normalizer
            scatter_age_agnostic[this_msa].append(np.round(RMSE_norm,2))
        bar_agnostic.append(np.array(scatter_age_agnostic[this_msa]).mean())
        error_agnostic.append(np.array(scatter_age_agnostic[this_msa]).std())
        
        for sample_idx in range(num_samples):
            subset = rng.choice(num_seeds, size=per_sample_size, replace=False)
            _, deaths_total_no_vac = functions.average_across_random_seeds_only_death(history_D2_no_vac[:,subset,:], history_D2_no_vac.shape[-1], cbg_dict[this_msa], print_results=False)
            deaths_daily_total_no_vac = [0]
            for i in range(1,len(deaths_total_no_vac)):
                deaths_daily_total_no_vac.append(deaths_total_no_vac[i]-deaths_total_no_vac[i-1])
            RMSE = sqrt(mean_squared_error(deaths_daily_smooth, deaths_daily_total_no_vac))
            normalizer = deaths_daily_smooth.mean()
            RMSE_norm = RMSE / normalizer
            scatter_no_vac[this_msa].append(np.round(RMSE_norm,2))
        bar_aware.append(np.array(scatter_no_vac[this_msa]).mean())
        error_aware.append(np.array(scatter_no_vac[this_msa]).std())

    bar_agnostic = np.round(bar_agnostic,2)
    bar_aware = np.round(bar_aware,2)
    error_agnostic = np.round(error_agnostic,2)
    error_aware = np.round(error_aware,2)
    print('bar_agnostic:',bar_agnostic)
    print('bar_aware',bar_aware)
    print('error_agnostic:',error_agnostic)
    print('error_aware',error_aware)
    print('Time: ', time.time()-start)

    np.save(scatter_age_agnostic_path, scatter_age_agnostic)
    np.save(scatter_no_vac_path, scatter_no_vac)
    print('File saved. Example path: ', scatter_no_vac_path)


all_agnostic = []
all_aware = []
all_standardSEIR = []
for i in range(len(msa_name_list)):
    this_msa = msa_name_list[i]
    all_agnostic.extend(scatter_age_agnostic[this_msa])
    all_aware.extend(scatter_no_vac[this_msa])
    all_standardSEIR.extend([bar_standardSEIR[i]]*len(scatter_age_agnostic[this_msa]))

improve_agnostic = []
improve_standardSEIR = []
for i in range(len(all_aware)):
    improve_agnostic.append(-(all_aware[i]-all_agnostic[i])/all_agnostic[i])
    improve_standardSEIR.append(-(all_aware[i]-all_standardSEIR[i])/all_standardSEIR[i])
print('NRMSE reduction compared to agnostic:', #improve_agnostic, np.max(np.array(improve_agnostic)), 
    np.mean(np.array(improve_agnostic)), np.std(np.array(improve_agnostic)))
print(get_ci(improve_agnostic))
print('NRMSE reduction compared to standardSEIR:', #improve_standardSEIR, np.max(np.array(improve_standardSEIR)),
    np.mean(np.array(improve_standardSEIR)), np.std(np.array(improve_standardSEIR)))   
print(get_ci(improve_standardSEIR))


# Re-order
bar_standardSEIR_reorder = np.sort(bar_standardSEIR)
ref = np.argsort(bar_standardSEIR)
bar_agnostic_reorder = bar_agnostic[ref]
bar_aware_reorder = bar_aware[ref]
error_agnostic_reorder = error_agnostic[ref]
error_aware_reorder = error_aware[ref]
bar_standardSEIR_reorder = bar_standardSEIR[ref]
anno_list = ['Atlanta','Chicago','Dallas','Houston', 'L.A.','Miami','Phila.','S.F.','D.C.']
anno_list_reorder = np.array(anno_list)[ref]
msa_name_list_reorder = np.array(msa_name_list)[ref] 


# distribution # 20220604
num_msas = len(msa_name_list)
my_pal = {'Metapopulation model': 'C0', 'BD model': 'C1', 'dummy': 'k'}
alpha = 0.8
width = 1.6
plt.subplots(figsize=(4.3,7))

# Draw standard SEIR
for i in range(len(msa_name_list)):
    plt.vlines(bar_standardSEIR_reorder[i], i-0.8, i, color='green', linestyles='dashed', linewidth=1)

# Organize data into a dataframe
data_df = pd.DataFrame(columns=['msa', 'model', 'result'])
for i in range(num_msas):
    msa_name = msa_name_list_reorder[i]
    for j in range(len(scatter_age_agnostic[msa_name])):
        data_df.loc[len(data_df)] = [msa_name, 'Metapopulation model', scatter_age_agnostic[msa_name][j]]
        #data_df.loc[len(data_df)] = [msa_name, 'BD model', scatter_no_vac[msa_name][j]]
 # Dummy data point
data_df.loc[len(data_df)] = ['dummy', 'dummy', -999]
# Plot violin
g1 = sns.violinplot(y="msa", x="result", hue="model",
        data=data_df, palette=my_pal, saturation=0.8, linewidth=0.4, split=True, scale='width', inner=None, width=width)
# Adjust transparancy
for violin in g1.collections:
    violin.set_alpha(alpha)
# Add median lines
for i in range(num_msas):
    plt.vlines(data_df.loc[(data_df['msa']==msa_name_list_reorder[i]) & (data_df['model']=='Metapopulation model')].median(), i-0.8, i, color='k', linewidth=0.8)
    #plt.vlines(data_df.loc[(data_df['msa']==msa_name_list_reorder[i]) & (data_df['model']=='BD model')].median(), i, i+0.4, color='k', linewidth=0.6)

# Organize data into a dataframe
data_df = pd.DataFrame(columns=['msa', 'model', 'result'])
for i in range(num_msas):
    msa_name = msa_name_list_reorder[i]
    for j in range(len(scatter_age_agnostic[msa_name])):
        #data_df.loc[len(data_df)] = [msa_name, 'Metapopulation model', scatter_age_agnostic[msa_name][j]]
        data_df.loc[len(data_df)] = [msa_name, 'BD model', scatter_no_vac[msa_name][j]]
 # Dummy data point
data_df.loc[len(data_df)] = ['dummy', 'dummy', -999]
# Plot violin
g2 = sns.violinplot(y="msa", x="result", hue="model",
        data=data_df, palette=my_pal, saturation=0.8, linewidth=0.4, split=True, scale='width', inner=None, width=width)
# Adjust transparancy
for violin in g2.collections:
    violin.set_alpha(alpha)
# Add median lines
for i in range(num_msas):
    #plt.vlines(data_df.loc[(data_df['msa']==msa_name_list_reorder[i]) & (data_df['model']=='Metapopulation model')].median(), i-0.4, i, color='k', linewidth=0.6)
    plt.vlines(data_df.loc[(data_df['msa']==msa_name_list_reorder[i]) & (data_df['model']=='BD model')].median(), i-0.8, i, color='k', linewidth=0.8)

# Add horizontal lines
for i in range(num_msas):
    plt.hlines(i, 0,1.65, color='k',linewidth=0.5)
plt.xlim(0,1.65)
plt.ylim(num_msas-1-0.4+width/2, -0.9) 

# Background color
for i in range(num_msas):
    if(i%2==0):
        plt.axhspan(i-0.8, i, color='grey',alpha=0.15)
    else:
        plt.axhspan(i-0.8, i, color='silver',alpha=0.15)

# Remove hue legend
leg = plt.gca().legend()
leg.remove()
g1.tick_params(left=False)  # remove the ticks
g1.set(ylabel=None)
plt.yticks(np.arange(9)-width/4, anno_list_reorder,fontsize=18)#,rotation=20
plt.xlabel('NRMSE of daily deaths',fontsize=19)
plt.xticks(fontsize=13)
savepath = os.path.join(fig_save_root, 'fig1c.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'test, saved at {savepath}')


# Fig1c, legend
plt.figure()
label_list = ['Meta-population model', 'BD model'] #'SEIR model',
color_list = ['C0','C1'] #'grey',
alpha_list = [0.8, 0.8] #[1,1] #0.6,

plt.vlines(bar_standardSEIR_reorder[0], 0, 1, color='green', linestyles='dashed', label='SEIR model')
plt.bar(1,1,label=label_list[0],color=color_list[0], alpha=alpha_list[0])
plt.bar(2,1,label=label_list[1],color=color_list[1], alpha=alpha_list[1])
plt.legend(ncol=1,fontsize=20,bbox_to_anchor=(0.8,-0.1)) 
#patches = [mpatches.Patch(color=color_list[i], alpha=alpha_list[i], label="{:s}".format(label_list[i])) for i in range(len(label_list)) ]
#plt.legend(handles=patches,ncol=1,fontsize=20,bbox_to_anchor=(0.8,-0.1)) 
# Save figure
savepath = os.path.join(fig_save_root, f'fig1c_legend.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Fig1c_legend, saved at {savepath}')



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
    plt.plot(deaths_daily_smooth,label='Ground Truth',marker='o',markersize=markersize,color=color_list[3], alpha=alpha_list[3])

    plt.yticks(fontsize=14) 
    plt.ylabel('Daily deaths', fontsize=25)

    plt.xlim(-1,63)
    plt.xticks(np.arange(13)*5,fontsize=12)
    plt.xlabel('Days',fontsize=25)

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
                                np.max(mean_hybrid),np.max(max_hybrid),np.max(min_hybrid)]))
        plt.vlines(31, 0, y_max, colors='black',linestyles ="dashed")

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

