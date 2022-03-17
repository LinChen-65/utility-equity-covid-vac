# plot_hesitancy_scenarios.py

# pylint: disable=invalid-name,trailing-whitespace,superfluous-parens,line-too-long,multiple-statements, unnecessary-semicolon, redefined-outer-name, consider-using-enumerate

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import socket
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import constants
import functions

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--vaccination_time', type=int, default=31,
                    help='Time to distribute vaccines.')
parser.add_argument('--vaccination_ratio' , type=float, default=0.1,
                    help='Vaccination ratio relative to MSA population.')
parser.add_argument('--num_seeds', type=int, default=30,
                    help='Num of seeds. Used to identify which files to load.')
parser.add_argument('--num_groups', type=int, default=5,
                    help='Num of groups to divide CBGs into.') 
parser.add_argument('--recheck_interval', type=float, default = 0.01,
                    help='Recheck interval (After distributing some portion of vaccines, recheck the most vulnerable demographic group).')                             
parser.add_argument('--rel_to', default='Baseline',
                    help='Relative to which strategy (either No_Vaccination or Baseline).')
args = parser.parse_args()  

# root
hostname = socket.gethostname()
print('hostname: ', hostname)
if(hostname in ['fib-dl3','rl3','rl2']): 
    root = '/data/chenlin/utility-equity-covid-vac/results'
elif(hostname=='rl4'):
    root = '/home/chenlin/utility-equity-covid-vac/results'
# subroot
subroot = 'figures'
if not os.path.exists(os.path.join(root, subroot)): # if folder does not exist, create one. #2022032
    os.makedirs(os.path.join(root, subroot))


notation_string_list = ['','acceptance_real_', 'access_acceptance_real_','acceptance_cf18_','acceptance_cf13_','acceptance_cf17_']
anno_list = ['Fully-Accepted','Estimated Hesitancy', 'Hesitancy+Capability', 'Hypothetical-1', 'Hypothetical-2', 'Hypothetical-3']
'''
notation_string_list = ['','acceptance_new1_', 'access_acceptance_new1_']
anno_list = ['Fully-Accepted','Estimated Hesitancy', 'Hesitancy+Accessibility']
'''
msa_name_list = ['Atlanta', 'Chicago', 'Dallas', 'Houston', 'LosAngeles', 'Miami', 'Philadelphia', 'SanFrancisco', 'WashingtonDC']
msa_pop_list = [7191638, 10140946,8895355,7263757,15859681,6635035,6727280,5018570,7536060]
num_msas = len(msa_name_list)
color_list = ['#e63946', '#fa7921', '#f4a261', '#348aa7', '#2a9d8f', '#4ecdc4']


def get_gini_dict(vaccination_time, vaccination_ratio,notation_string,rel_to,msa_list,root): #20220308
    '''Load gini tables to construct gini_df_dict'''
    gini_df_dict = dict()
    for this_msa in msa_list:
        filepath = os.path.join(root, 'gini_table', f'gini_table_{notation_string}{str(vaccination_time)}_{vaccination_ratio}_{args.recheck_interval}_{args.num_groups}_{this_msa}_rel2{rel_to}.csv')
        gini_df = pd.read_csv(filepath)
        gini_df.rename(columns={'Unnamed: 0':'Dimension','Unnamed: 1':'Metric'},inplace=True)
        gini_df_dict[this_msa] = gini_df
    return gini_df_dict


def get_results(gini_df_dict): #20220309
    age_util_list = [];age_equi_list = []
    income_util_list = [];income_equi_list = []
    occupation_util_list = [];occupation_equi_list = []
    minority_util_list = [];minority_equi_list = []

    age_util = np.zeros(num_msas)
    income_util = np.zeros(num_msas)
    occupation_util = np.zeros(num_msas)
    minority_util = np.zeros(num_msas)
    for i in range(num_msas):
        age_util[i] = -gini_df_dict[msa_name_list[i]]['Age'].iloc[1] 
        income_util[i] = -gini_df_dict[msa_name_list[i]]['Income'].iloc[1] 
        occupation_util[i] = -gini_df_dict[msa_name_list[i]]['Occupation'].iloc[1] 
        minority_util[i] = -gini_df_dict[msa_name_list[i]]['Minority'].iloc[1] 
    
    age_equi = np.zeros(num_msas)
    income_equi = np.zeros(num_msas)
    occupation_equi = np.zeros(num_msas)
    minority_equi = np.zeros(num_msas)
    for i in range(num_msas):
        age_equi[i] = -gini_df_dict[msa_name_list[i]]['Age'].iloc[3] 
        income_equi[i] = -gini_df_dict[msa_name_list[i]]['Income'].iloc[5] 
        occupation_equi[i] = -gini_df_dict[msa_name_list[i]]['Occupation'].iloc[7]
        minority_equi[i] = -gini_df_dict[msa_name_list[i]]['Minority'].iloc[9]

    age_util_list.append(age_util); age_equi_list.append(age_equi)
    income_util_list.append(income_util);income_equi_list.append(income_equi)
    occupation_util_list.append(occupation_util);occupation_equi_list.append(occupation_equi)
    minority_util_list.append(minority_util); minority_equi_list.append(minority_equi)
    
    return age_util_list,income_util_list,occupation_util_list,minority_util_list
    
age_util_dict = dict()
income_util_dict = dict()
occupation_util_dict = dict()
minority_util_dict = dict()
for notation_string in notation_string_list:
    # load gini table to construct gini_df_dict
    gini_df_dict = get_gini_dict(args.vaccination_time, args.vaccination_ratio, notation_string, args.rel_to, msa_name_list, root)
    age_util_list,income_util_list,occupation_util_list,minority_util_list = get_results(gini_df_dict)
    age_util_dict[notation_string] = np.array(age_util_list).squeeze()
    income_util_dict[notation_string] = np.array(income_util_list).squeeze()
    occupation_util_dict[notation_string] = np.array(occupation_util_list).squeeze()
    minority_util_dict[notation_string] = np.array(minority_util_list).squeeze()


# Draw figure
#figsize = (7.5, 4.2)
figsize = (7.5, 3.8)

def draw_boxplot(results,medians,figsize,policy,boxplot=True,show_legend=True): #20220309
    policy = policy.lower()
    plt.figure(figsize=figsize) #(7,6)
    plt.axhline(0,color='k',linestyle='--')
    num_scenarios = len(results)
    if(boxplot):
        #plt.boxplot(results,widths=0.3) # showfliers=False      
        bp = plt.boxplot(results, widths=0.3, patch_artist=True) #labels=anno_list,
        
        [bp['boxes'][i].set(facecolor=color_list[i], alpha=0.7) for i in range(num_scenarios)]
        plt.plot(np.arange(num_scenarios)+1,medians,marker='o',markersize=10)
    else:
        for i in range(num_scenarios):
            for j in range(num_msas):
                plt.scatter(i,results[i][j])
    if(show_legend):
        plt.xticks(np.arange(num_scenarios)+1,anno_list,fontsize=16,rotation=-20,ha='left')
    else:
        plt.xticks(np.arange(num_scenarios)+1,['','','','','',''],fontsize=16,rotation=-20,ha='left')
        #plt.xticks(np.arange(num_scenarios)+1,['','',''],fontsize=16,rotation=-20,ha='left')
    if(policy=='minority'):
        plt.xlabel(f'Prioritize by race/ethnicity',fontsize=22)
    else:
        plt.xlabel(f'Prioritize by {policy}',fontsize=22)
    plt.ylabel('Change in social utility',fontsize=21)
    plt.yticks(fontsize=15)
    # Save the figure
    savepath = os.path.join(root, subroot , f'fig2b_{policy}.pdf')
    plt.savefig(savepath,bbox_inches = 'tight')
    print(f'fig2b_{policy}, figure saved at: {savepath}.')

# Age
results = []
medians = []
for notation_string in notation_string_list:
    results.append(age_util_dict[notation_string])
    medians.append(np.median(age_util_dict[notation_string]))
draw_boxplot(results,medians,figsize,policy='age',show_legend=False)

# Income
results = []
medians = []
for notation_string in notation_string_list:
    results.append(income_util_dict[notation_string])
    medians.append(np.median(income_util_dict[notation_string]))
draw_boxplot(results,medians,figsize,policy='income',show_legend=False)

# Occupation
results = []
medians = []
for notation_string in notation_string_list:
    results.append(occupation_util_dict[notation_string])
    medians.append(np.median(occupation_util_dict[notation_string]))
draw_boxplot(results,medians,figsize,policy='occupation',show_legend=False)

# Minority
results = []
medians = []
for notation_string in notation_string_list:
    results.append(minority_util_dict[notation_string])
    medians.append(np.median(minority_util_dict[notation_string]))
draw_boxplot(results,medians,figsize,policy='minority',show_legend=False)

# Legend
num_scenarios = len(results)
plt.figure()
patches = [mpatches.Patch(color=color_list[i], label="{:s}".format(anno_list[i])) for i in range(num_scenarios) ]
plt.legend(handles=patches,ncol=2,fontsize=20,bbox_to_anchor=(0.8,-0.1)) 
# Save figure
savepath = os.path.join(root, 'figures', 'fig2b_legend.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'fig2b_legend, saved at {savepath}')
