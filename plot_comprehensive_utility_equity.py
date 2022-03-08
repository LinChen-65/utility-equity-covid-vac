# python plot_comprehensive_utility_equity.py

# pylint: disable=invalid-name,trailing-whitespace,superfluous-parens,line-too-long,multiple-statements

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import socket
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
if(hostname=='fib-dl3'): 
    root = '/data/chenlin/utility-equity-covid-vac/results'
elif(hostname=='rl4'):
    root = '/home/chenlin/utility-equity-covid-vac/results'
# subroot
subroot = 'figures'
if not os.path.exists(os.path.join(root, subroot)): # if folder does not exist, create one. #2022032
    os.makedirs(os.path.join(root, subroot))

# Recheck interval for other strategies #20220306
recheck_interval_others = 0.01

policy_list = ['No_Vaccination', 'Baseline', 'Age', 'Income', 'Occupation', 'Minority', 'SVI', 'Hybrid', 'Hybrid_Ablation']
msa_name_list = ['Atlanta', 'Chicago', 'Dallas', 'Houston', 'LosAngeles', 'Miami', 'Philadelphia', 'SanFrancisco', 'WashingtonDC']

# Drawing settings
anno_list = ['Atlanta','Chicago','Dallas','Houston','L.A.','Miami','Phila.','S.F.','D.C.']
color_list = ['#FE2E2E','#FFBF00','#5FB404','#81BEF7','#29088A','grey','plum']

###########################################################################################################################
# Load gini tables

gini_df_dict = dict()
for this_msa in msa_name_list:
    print(this_msa)
    filepath = os.path.join(root, 'gini_table', f'gini_table_comprehensive_{str(args.vaccination_time)}_{args.recheck_interval}_{args.num_groups}_{this_msa}_rel2No_Vaccination.csv')
    gini_df = pd.read_csv(filepath)
    gini_df.rename(columns={'Unnamed: 0':'Dimension','Unnamed: 1':'Metric'},inplace=True)
    gini_df_dict[this_msa] = gini_df

num_msas = len(gini_df_dict)
msa_list = list(gini_df_dict.keys())    

baseline_util = np.zeros(num_msas); baseline_equi_age = np.zeros(num_msas); baseline_equi_income = np.zeros(num_msas); baseline_equi_occupation = np.zeros(num_msas); baseline_equi_minority = np.zeros(num_msas)
age_util = np.zeros(num_msas); age_equi_age = np.zeros(num_msas); age_equi_income = np.zeros(num_msas); age_equi_occupation = np.zeros(num_msas); age_equi_minority = np.zeros(num_msas)
income_util = np.zeros(num_msas); income_equi_age = np.zeros(num_msas); income_equi_income = np.zeros(num_msas); income_equi_occupation = np.zeros(num_msas); income_equi_minority = np.zeros(num_msas)
occupation_util = np.zeros(num_msas); occupation_equi_age = np.zeros(num_msas); occupation_equi_income = np.zeros(num_msas); occupation_equi_occupation = np.zeros(num_msas); occupation_equi_minority = np.zeros(num_msas)
minority_util = np.zeros(num_msas); minority_equi_age = np.zeros(num_msas); minority_equi_income = np.zeros(num_msas); minority_equi_occupation = np.zeros(num_msas); minority_equi_minority = np.zeros(num_msas)
hybrid_util = np.zeros(num_msas); hybrid_equi_age = np.zeros(num_msas); hybrid_equi_income = np.zeros(num_msas); hybrid_equi_occupation = np.zeros(num_msas); hybrid_equi_minority = np.zeros(num_msas)
hybrid_ablation_util = np.zeros(num_msas); hybrid_ablation_equi_age = np.zeros(num_msas); hybrid_ablation_equi_income = np.zeros(num_msas); hybrid_ablation_equi_occupation = np.zeros(num_msas); hybrid_ablation_equi_minority = np.zeros(num_msas)
for i in range(num_msas):
    # utility
    baseline_util[i] = -gini_df_dict[msa_list[i]]['Baseline'].iloc[1] 
    age_util[i] = -gini_df_dict[msa_list[i]]['Age'].iloc[1] 
    income_util[i] = -gini_df_dict[msa_list[i]]['Income'].iloc[1] 
    occupation_util[i] = -gini_df_dict[msa_list[i]]['Occupation'].iloc[1] 
    minority_util[i] = -gini_df_dict[msa_list[i]]['Minority'].iloc[1] 
    hybrid_util[i] = -gini_df_dict[msa_list[i]]['Hybrid'].iloc[1] 
    hybrid_ablation_util[i] = -gini_df_dict[msa_list[i]]['Hybrid_Ablation'].iloc[1] 
    
    # equity by age
    baseline_equi_age[i] = -gini_df_dict[msa_list[i]]['Baseline'].iloc[3] 
    age_equi_age[i] = -gini_df_dict[msa_list[i]]['Age'].iloc[3] 
    income_equi_age[i] = -gini_df_dict[msa_list[i]]['Income'].iloc[3] 
    occupation_equi_age[i] = -gini_df_dict[msa_list[i]]['Occupation'].iloc[3] 
    minority_equi_age[i] = -gini_df_dict[msa_list[i]]['Minority'].iloc[3] 
    hybrid_equi_age[i] = -gini_df_dict[msa_list[i]]['Hybrid'].iloc[3] 
    hybrid_ablation_equi_age[i] = -gini_df_dict[msa_list[i]]['Hybrid_Ablation'].iloc[3] 

    # equity by income
    baseline_equi_income[i] = -gini_df_dict[msa_list[i]]['Baseline'].iloc[5] 
    age_equi_income[i] = -gini_df_dict[msa_list[i]]['Age'].iloc[5] 
    income_equi_income[i] = -gini_df_dict[msa_list[i]]['Income'].iloc[5] 
    occupation_equi_income[i] = -gini_df_dict[msa_list[i]]['Occupation'].iloc[5] 
    minority_equi_income[i] = -gini_df_dict[msa_list[i]]['Minority'].iloc[5] 
    hybrid_equi_income[i] = -gini_df_dict[msa_list[i]]['Hybrid'].iloc[5] 
    hybrid_ablation_equi_income[i] = -gini_df_dict[msa_list[i]]['Hybrid_Ablation'].iloc[5] 

    # equity by occupation
    baseline_equi_occupation[i] = -gini_df_dict[msa_list[i]]['Baseline'].iloc[7] 
    age_equi_occupation[i] = -gini_df_dict[msa_list[i]]['Age'].iloc[7] 
    income_equi_occupation[i] = -gini_df_dict[msa_list[i]]['Income'].iloc[7] 
    occupation_equi_occupation[i] = -gini_df_dict[msa_list[i]]['Occupation'].iloc[7] 
    minority_equi_occupation[i] = -gini_df_dict[msa_list[i]]['Minority'].iloc[7] 
    hybrid_equi_occupation[i] = -gini_df_dict[msa_list[i]]['Hybrid'].iloc[7] 
    hybrid_ablation_equi_occupation[i] = -gini_df_dict[msa_list[i]]['Hybrid_Ablation'].iloc[7] 

    # equity by minority
    baseline_equi_minority[i] = -gini_df_dict[msa_list[i]]['Baseline'].iloc[9] 
    age_equi_minority[i] = -gini_df_dict[msa_list[i]]['Age'].iloc[9] 
    income_equi_minority[i] = -gini_df_dict[msa_list[i]]['Income'].iloc[9] 
    occupation_equi_minority[i] = -gini_df_dict[msa_list[i]]['Occupation'].iloc[9] 
    minority_equi_minority[i] = -gini_df_dict[msa_list[i]]['Minority'].iloc[9] 
    hybrid_equi_minority[i] = -gini_df_dict[msa_list[i]]['Hybrid'].iloc[9] 
    hybrid_ablation_equi_minority[i] = -gini_df_dict[msa_list[i]]['Hybrid_Ablation'].iloc[9] 

###########################################################################################################################
# Fig4a: main scenario, radar plot


###########################################################################################################################
# Fig4b: main scenario, utility change in each MSA

plt.figure(figsize=(14,7))

dist = 4 
alpha = 1 
plt.bar(np.arange(9)*dist,hybrid_util,label='Comprehensive',color=color_list[0],alpha=alpha)
plt.bar(np.arange(9)*dist+1,age_util,label='Prioritize by age',color=color_list[1],alpha=alpha) 
plt.bar(np.arange(9)*dist+2,baseline_util,label='Homogeneous',color=color_list[4],alpha=alpha)
#plt.bar(np.arange(9)*dist+3,income_util,label='Income-Prioritized')
#plt.bar(np.arange(9)*dist+4,occupation_util,label='Occupation-Prioritized')

plt.xticks(np.arange(9)*dist+1,anno_list,fontsize=24,rotation=20)
#plt.yticks(fontsize=12)
plt.ylabel('Change in social utility\n(w.r.t. No-vaccination)',fontsize=24)

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8]) #https://blog.csdn.net/yywan1314520/article/details/53740001
plt.legend(ncol=3,fontsize=21,bbox_to_anchor=(0.99,-0.2)) 

# Save the figure
savepath = os.path.join(root, subroot , '0307_fig4b.png')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Fig4b, figure saved at: {savepath}.')


###########################################################################################################################
# Fig4c: main scenario, equity change in each MSA

marker_list = ['^','o','*','P']
alpha = 1 
dist_1 = 13
dist_2 = 3

plt.figure(figsize=(14,6))

markersize=170
plt.scatter(np.arange(9)*dist_1,hybrid_equi_age,marker=marker_list[0],s=markersize,color=color_list[0],alpha=alpha) 
plt.scatter(np.arange(9)*dist_1,hybrid_equi_income,marker=marker_list[1],s=markersize,color=color_list[0],alpha=alpha)
plt.scatter(np.arange(9)*dist_1,hybrid_equi_occupation,marker=marker_list[2],s=markersize,color=color_list[0],alpha=alpha)
plt.scatter(np.arange(9)*dist_1,hybrid_equi_minority,marker=marker_list[3],s=markersize,color=color_list[0],alpha=alpha)

plt.scatter(np.arange(9)*dist_1+dist_2,age_equi_age,marker=marker_list[0],s=markersize,color=color_list[1],alpha=alpha) 
plt.scatter(np.arange(9)*dist_1+dist_2,age_equi_income,marker=marker_list[1],s=markersize,color=color_list[1],alpha=alpha)
plt.scatter(np.arange(9)*dist_1+dist_2,age_equi_occupation,marker=marker_list[2],s=markersize,color=color_list[1],alpha=alpha)
plt.scatter(np.arange(9)*dist_1+dist_2,age_equi_minority,marker=marker_list[3],s=markersize,color=color_list[1],alpha=alpha)

'''
plt.scatter(np.arange(9)*dist_1+dist_2*2,income_equi_income,marker='^',s=markersize,color=color_list[2],alpha=0.5) 
plt.scatter(np.arange(9)*dist_1+dist_2*2,income_equi_age,marker='o',s=markersize,color=color_list[2],alpha=0.5)
plt.scatter(np.arange(9)*dist_1+dist_2*2,income_equi_occupation,marker='*',s=markersize,color=color_list[2],alpha=0.5)

plt.scatter(np.arange(9)*dist_1+dist_2*3,occupation_equi_occupation,marker='^',s=markersize,color=color_list[3],alpha=0.5)
plt.scatter(np.arange(9)*dist_1+dist_2*3,occupation_equi_age,marker='o',s=markersize,color=color_list[3],alpha=0.5)
plt.scatter(np.arange(9)*dist_1+dist_2*3,occupation_equi_income,marker='*',s=markersize,color=color_list[3],alpha=0.5)
'''

plt.scatter(np.arange(9)*dist_1+dist_2*2,baseline_equi_occupation,marker=marker_list[0],s=markersize,color=color_list[4],alpha=alpha)
plt.scatter(np.arange(9)*dist_1+dist_2*2,baseline_equi_age,marker=marker_list[1],s=markersize,color=color_list[4],alpha=alpha)
plt.scatter(np.arange(9)*dist_1+dist_2*2,baseline_equi_income,marker=marker_list[2],s=markersize,color=color_list[4],alpha=alpha)
plt.scatter(np.arange(9)*dist_1+dist_2*2,baseline_equi_minority,marker=marker_list[3],s=markersize,color=color_list[4],alpha=alpha)
  
xmin = -2
xmax = 9*dist_1-2
ymax = 1
ymin = -0.5 #-0.4
plt.xlim(xmin-1,xmax)
plt.ylim(ymin,ymax)
plt.ylabel('Change in equity\n(w.r.t. No-vaccination)',fontsize=24)
anno_list = ['Atlanta','Chicago','Dallas','Houston', 'L.A.','Miami','Phila.','S.F.','D.C.']
plt.xticks(np.arange(9)*dist_1+dist_2+2,anno_list, fontsize=24,rotation=20)
plt.hlines(0, xmin, xmax, linewidth=1.2, linestyle='dashed')

# 背景填色
x = (np.arange(9*dist_1)-2)
for i in range(num_msas):
    if(i%2==0):
        plt.fill_between(x[i*dist_1:(i+1)*dist_1], ymin, ymax, facecolor='grey',alpha=0.15)
    else:
        plt.fill_between(x[i*dist_1:(i+1)*dist_1], ymin, ymax, facecolor='silver',alpha=0.15)

# 图例
ax = plt.gca()
box = ax.get_position()

label_list = ['Equity-by-age','Equity-by-income','Equity-by-occupation', 'Equity-by-minority']
patches = [plt.scatter([],[],marker=marker_list[i],s=markersize+10,c='none',edgecolors='k',label="{:s}".format(label_list[i]) ) for i in range(len(label_list)) ]
plt.legend(handles=patches, loc='lower center',ncol=2,fontsize=18)

# Save the figure
savepath = os.path.join(root, subroot , '0307_fig4c.png')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Fig4c, figure saved at: {savepath}.')

###########################################################################################################################
