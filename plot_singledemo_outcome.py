# python plot_singledemo_outcome.py

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import socket
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text

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
'''
hostname = socket.gethostname()
print('hostname: ', hostname)
if(hostname in ['fib-dl3','rl3','rl2']):
    root = '/data/chenlin/utility-equity-covid-vac/results'
elif(hostname=='rl4'):
    root = '/home/chenlin/utility-equity-covid-vac/results'
'''
root = os.getcwd()
resultroot = os.path.join(root, 'results')
# subroot
subroot = 'figures'
if not os.path.exists(os.path.join(root, subroot)): # if folder does not exist, create one. #2022032
    os.makedirs(os.path.join(root, subroot))

# demo_feats
#demo_feat_list = ['Age', 'Income', 'Occupation', 'Minority'] #20220302
#print('Demographic feature list: ', demo_feat_list)

# Initialize dict
age_util_dict = dict(); age_equi_dict = dict()
income_util_dict = dict(); income_equi_dict = dict()
occupation_util_dict = dict(); occupation_equi_dict = dict()
minority_util_dict = dict(); minority_equi_dict = dict()

gini_df_dict = dict()
for this_msa in constants.MSA_NAME_LIST:
    if(this_msa=='NewYorkCity'):continue
    filepath = os.path.join(resultroot, f'gini_table/gini_table_{str(args.vaccination_time)}_{args.vaccination_ratio}_{args.recheck_interval}_{args.num_groups}_{this_msa}_rel2{args.rel_to}.csv')
    gini_df = pd.read_csv(filepath)
    gini_df.rename(columns={'Unnamed: 0':'Dimension','Unnamed: 1':'Metric'},inplace=True)
    gini_df_dict[this_msa] = gini_df
    
num_msas = len(gini_df_dict)
msa_list = list(gini_df_dict.keys())


# Prioritize the most disadvantaged
age_util = np.zeros(num_msas); age_equi = np.zeros(num_msas)
income_util = np.zeros(num_msas); income_equi = np.zeros(num_msas)
occupation_util = np.zeros(num_msas); occupation_equi = np.zeros(num_msas)
minority_util = np.zeros(num_msas); minority_equi = np.zeros(num_msas)
for i in range(num_msas):
    age_util[i] = -gini_df_dict[msa_list[i]]['Age'].iloc[1] 
    income_util[i] = -gini_df_dict[msa_list[i]]['Income'].iloc[1] 
    occupation_util[i] = -gini_df_dict[msa_list[i]]['Occupation'].iloc[1] 
    minority_util[i] = -gini_df_dict[msa_list[i]]['Minority'].iloc[1] 

    age_equi[i] = -gini_df_dict[msa_list[i]]['Age'].iloc[3] 
    income_equi[i] = -gini_df_dict[msa_list[i]]['Income'].iloc[5] 
    occupation_equi[i] = -gini_df_dict[msa_list[i]]['Occupation'].iloc[7]
    minority_equi[i] = -gini_df_dict[msa_list[i]]['Minority'].iloc[9] 
age_util_dict[0] = age_util; age_equi_dict[0] = age_equi
income_util_dict[0] = income_util;income_equi_dict[0] = income_equi
occupation_util_dict[0] = occupation_util;occupation_equi_dict[0] = occupation_equi
minority_util_dict[0] = minority_util; minority_equi_dict[0] = age_equi

# Prioritize the least disadvantaged
age_util = np.zeros(num_msas); age_equi = np.zeros(num_msas)
income_util = np.zeros(num_msas); income_equi = np.zeros(num_msas)
occupation_util = np.zeros(num_msas); occupation_equi = np.zeros(num_msas)
minority_util = np.zeros(num_msas); minority_equi = np.zeros(num_msas)
for i in range(num_msas):
    age_util[i] = -gini_df_dict[msa_list[i]]['Age_Reverse'].iloc[1] 
    income_util[i] = -gini_df_dict[msa_list[i]]['Income_Reverse'].iloc[1] 
    occupation_util[i] = -gini_df_dict[msa_list[i]]['Occupation_Reverse'].iloc[1] 
    minority_util[i] = -gini_df_dict[msa_list[i]]['Minority_Reverse'].iloc[1] 

    age_equi[i] = -gini_df_dict[msa_list[i]]['Age_Reverse'].iloc[3] 
    income_equi[i] = -gini_df_dict[msa_list[i]]['Income_Reverse'].iloc[5] 
    occupation_equi[i] = -gini_df_dict[msa_list[i]]['Occupation_Reverse'].iloc[7]
    minority_equi[i] = -gini_df_dict[msa_list[i]]['Minority_Reverse'].iloc[9] 
age_util_dict[1] = age_util; age_equi_dict[1] = age_equi
income_util_dict[1] = income_util; income_equi_dict[1] = income_equi
occupation_util_dict[1] = occupation_util;occupation_equi_dict[1] = occupation_equi    
minority_util_dict[1] = minority_util; minority_equi_dict[1] = age_equi

# Drawing settings
marker_list = ['o','^','h','8','v',',','p','d','X'] # for different MSAs
color_list = ['red','blue','green'] # for different policies
anno_list=['Atlanta','Chicago','Dallas','Houston','L.A.','Miami','Phila.','S.F.','D.C.']

########################################################################################
# Age-Priortized

plt.figure(figsize=(8,7))

# Scatter the dots
for policy_idx in range(2): 
    for msa_idx in range(num_msas):
        plt.scatter(age_util_dict[policy_idx][msa_idx], age_equi_dict[policy_idx][msa_idx], 
                    s=70, marker = marker_list[policy_idx], color = color_list[policy_idx]) # marker = 'o'
                    
# Connect dots of the same MSA #20211106
for msa_idx in range(num_msas):
    plt.arrow(age_util_dict[1][msa_idx],age_equi_dict[1][msa_idx],
              (age_util_dict[0][msa_idx]-age_util_dict[1][msa_idx]),(age_equi_dict[0][msa_idx]-age_equi_dict[1][msa_idx]),
              width=0.0005, #width=0.0015,
              color='grey',
              alpha=0.2,
              linestyle='--'
             )

# Annotation of MSAs # Only annotate quadrant 1 
texts = [plt.text(age_util_dict[policy_idx][msa_idx], age_equi_dict[policy_idx][msa_idx],
                  anno_list[msa_idx],fontsize=19.5) for msa_idx in range(num_msas) for policy_idx in range(1)]           

adjust_text(texts,
            arrowprops=dict(arrowstyle="-", color='k', lw=1),
            expand_text=[1.1,1.1], #expand_text=[1.1,1.1],
            force_text=1.55,
            force_points=1.4,
            lim=1000,
            only_move={'text':'y','points':'x'} #only_move={'text':'y','points':'x'}
           )

# Ranges for simulteneously plotting 3 policies
xmin = -0.09 
xmax = 0.09
ymin = -0.5 #-0.8 
ymax = 0.5 #0.8

plt.hlines(0, xmin, xmax, linewidth=0.5, linestyle='dashed')
plt.vlines(0, ymin, ymax, linewidth=0.5, linestyle='dashed')
plt.xlim(xmin,xmax);plt.ylim(ymin,ymax)
plt.xticks(fontsize=14);plt.yticks(fontsize=14)

plt.xlabel('Prioritize by age',fontsize=25)

# 象限背景填色
x = (np.arange(22)-10)*0.01
plt.fill_between(x[:11], 0, ymax, facecolor='orange',alpha=0.2) #alpha=0.15
plt.fill_between(x[10:], ymin, 0, facecolor='green',alpha=0.2) #alpha=0.15
plt.fill_between(x[:11], ymin, 0, facecolor='blue',alpha=0.2) #alpha=0.15
plt.fill_between(x[10:], 0, ymax, facecolor='red',alpha=0.35) #alpha=0.15

#设置图片的边框为不显示
ax=plt.gca()  #gca:get current axis得到当前轴
ax.spines['right'].set_color('none');ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none');ax.spines['bottom'].set_color('none')

# 文字标明social utility/equity增长方向
ax.arrow(xmax-0.05,0,0.05, 0,length_includes_head=True,
         color='k',width=0.001,head_width=0.015,head_length=0.002)
t = ax.text(xmax, -0.045, "Social Utility", ha="right", va="center", rotation=0, size=22)
ax.arrow(0, ymax-0.25,0, 0.25,length_includes_head=True,
          color='k',width=0.0005,head_width=0.003,head_length=0.02)
t = ax.text(-0.006, ymax, "Equity", ha="center", va="top", rotation=90, size=22)

# Save the figure
savepath = os.path.join(root, subroot , 'fig2a_age.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Age, figure saved at: {savepath}.')


########################################################################################
# Income-Priortized

plt.figure(figsize=(8,7))

for policy_idx in range(2):
    for msa_idx in range(num_msas):
        plt.scatter(income_util_dict[policy_idx][msa_idx], income_equi_dict[policy_idx][msa_idx], 
                    s=70, marker = marker_list[policy_idx], color = color_list[policy_idx]) # marker = 'o'

# Connect dots of the same MSA #20211106
for msa_idx in range(num_msas):
    plt.arrow(income_util_dict[1][msa_idx],income_equi_dict[1][msa_idx],
              (income_util_dict[0][msa_idx]-income_util_dict[1][msa_idx]),(income_equi_dict[0][msa_idx]-income_equi_dict[1][msa_idx]),
              width=0.001,
              color='grey',
              alpha=0.2,
              linestyle='--'
             )
        
# Annotation of MSAs        
texts = [plt.text(income_util_dict[policy_idx][msa_idx], income_equi_dict[policy_idx][msa_idx],
                  anno_list[msa_idx],fontsize=19.5) for msa_idx in range(num_msas) for policy_idx in range(1)]
adjust_text(texts,arrowprops=dict(arrowstyle="-", color='k', lw=1),
            force_text=1.2,force_points=1.2,
            lim=1000,
            only_move={'text':'y','points':'x'})

# ranges for simulteneously plotting 3 policies
xmin = -0.15 
xmax = 0.15 
ymin = -1.3 
ymax = 1.3

plt.hlines(0, xmin, xmax, linewidth=0.5, linestyle='dashed')
plt.vlines(0, ymin, ymax, linewidth=0.5, linestyle='dashed')
plt.xlim(xmin,xmax);plt.ylim(ymin,ymax)
plt.xticks(fontsize=14);plt.yticks(fontsize=14)

plt.xlabel('Prioritize by income',fontsize=25)

# 象限背景填色
x = (np.arange(32)-15)*0.01
plt.fill_between(x[:16], 0, ymax, facecolor='orange',alpha=0.2) #alpha=0.15
plt.fill_between(x[15:], ymin, 0, facecolor='green',alpha=0.2) #alpha=0.15
plt.fill_between(x[:16], ymin, 0, facecolor='blue',alpha=0.2) #alpha=0.15
plt.fill_between(x[15:], 0, ymax, facecolor='red',alpha=0.35) #alpha=0.15

#设置图片的边框为不显示
ax=plt.gca()  #gca:get current axis得到当前轴
ax.spines['right'].set_color('none');ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none');ax.spines['bottom'].set_color('none')

# 文字标明social utility/equity增长方向
ax.arrow(xmax-0.1,0,0.1, 0,length_includes_head=True,
         color='k',width=0.002,head_width=0.025,head_length=0.006)
t = ax.text(xmax, -0.085, "Social Utility", ha="right", va="center", rotation=0, size=22)
ax.arrow(0, ymax-0.45,0, 0.45,length_includes_head=True,
          color='k',width=0.0005,head_width=0.005,head_length=0.045)
t = ax.text(-0.01, ymax, "Equity", ha="center", va="top", rotation=90, size=22)

# Save the figure
savepath = os.path.join(root, subroot , 'fig2a_income.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Income, figure saved at: {savepath}.')


########################################################################################
# Occupation-Priortized

plt.figure(figsize=(8,7))

for policy_idx in range(2): 
    for msa_idx in range(num_msas):
        plt.scatter(occupation_util_dict[policy_idx][msa_idx], occupation_equi_dict[policy_idx][msa_idx], 
                    s=70, marker = marker_list[policy_idx], color = color_list[policy_idx]) # marker = 'o'

# Connect dots of the same MSA #20211106
for msa_idx in range(num_msas):
    plt.arrow(occupation_util_dict[1][msa_idx],occupation_equi_dict[1][msa_idx],
              (occupation_util_dict[0][msa_idx]-occupation_util_dict[1][msa_idx]),(occupation_equi_dict[0][msa_idx]-occupation_equi_dict[1][msa_idx]),
              width=0.0004,
              color='grey',
              alpha=0.2,
              linestyle='--'
             )
    
# Annotation of MSAs        
texts = [plt.text(occupation_util_dict[policy_idx][msa_idx], occupation_equi_dict[policy_idx][msa_idx],
                  anno_list[msa_idx],fontsize=20) for msa_idx in range(num_msas) for policy_idx in range(1)]
adjust_text(texts,
            arrowprops=dict(arrowstyle="-", color='k', lw=1),
            expand_text=[1.1,1.1],force_text=1.5,force_points=1.2,lim=5000,only_move={'text':'y','points':'x'})

# Ranges for simulteneously plotting 3 policies
xmin = -0.06 #-0.08 
xmax = 0.06 #0.08 
ymin = -0.9 #-1.1
ymax = 0.9 #1.1

plt.hlines(0, xmin, xmax, linewidth=0.5, linestyle='dashed')
plt.vlines(0, ymin, ymax, linewidth=0.5, linestyle='dashed')
plt.xlim(xmin,xmax);plt.ylim(ymin,ymax)
plt.xticks(fontsize=14);plt.yticks(fontsize=14)

plt.xlabel('Prioritize by occupation',fontsize=25)

# 象限背景填色
x = (np.arange(20)-8)*0.01
plt.fill_between(x[:9], 0, ymax, facecolor='orange',alpha=0.2) #alpha=0.15
plt.fill_between(x[8:], ymin, 0, facecolor='green',alpha=0.2) #alpha=0.15
plt.fill_between(x[:9], ymin, 0, facecolor='blue',alpha=0.2) #alpha=0.15
plt.fill_between(x[8:], 0, ymax, facecolor='red',alpha=0.35)#alpha=0.15

#设置图片的边框为不显示
ax=plt.gca()  #gca:get current axis得到当前轴
ax.spines['right'].set_color('none');ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none');ax.spines['bottom'].set_color('none')

# 文字标明social utility/equity增长方向
ax.arrow(xmax-0.045,0,0.045, 0,length_includes_head=True,
         color='k',width=0.002,head_width=0.025,head_length=0.005)
t = ax.text(xmax, -0.07, "Social Utility", ha="right", va="center", rotation=0, size=22)
ax.arrow(-0.0002, ymax-0.35,0, 0.35,length_includes_head=True,
          color='k',width=0.0002,head_width=0.003,head_length=0.025)
t = ax.text(-0.006, ymax, "Equity", ha="center", va="top", rotation=90, size=22)

# Save the figure
savepath = os.path.join(root, subroot , 'fig2a_occupation.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Occupation, figure saved at: {savepath}.')


########################################################################################
# Minority-Priortized

plt.figure(figsize=(8,7))

for policy_idx in range(2): 
    for msa_idx in range(num_msas):
        plt.scatter(minority_util_dict[policy_idx][msa_idx], minority_equi_dict[policy_idx][msa_idx], 
                    s=70, marker = marker_list[policy_idx], color = color_list[policy_idx]) # marker = 'o'

# Connect dots of the same MSA #20211106
for msa_idx in range(num_msas):
    plt.arrow(minority_util_dict[1][msa_idx], minority_equi_dict[1][msa_idx],
              (minority_util_dict[0][msa_idx]-minority_util_dict[1][msa_idx]),(minority_equi_dict[0][msa_idx]-minority_equi_dict[1][msa_idx]),
              width=0.0004,
              color='grey', alpha=0.2,
              linestyle='--'
             )
    
# Annotation of MSAs  
'''      
texts = [plt.text(minority_util_dict[policy_idx][msa_idx], minority_equi_dict[policy_idx][msa_idx],
                  anno_list[msa_idx],fontsize=20) for msa_idx in range(num_msas) for policy_idx in range(1)]
adjust_text(texts,
            arrowprops=dict(arrowstyle="-", color='k', lw=1),
            expand_text=[1.1,1.1],force_text=1.5,force_points=1.2,lim=5000,only_move={'text':'y','points':'x'})
'''
# Ranges for simulteneously plotting 3 policies
xmin = -0.09 
xmax = 0.09 
ymin = -0.6 
ymax = 0.6 

plt.hlines(0, xmin, xmax, linewidth=0.5, linestyle='dashed')
plt.vlines(0, ymin, ymax, linewidth=0.5, linestyle='dashed')
plt.xlim(xmin,xmax);plt.ylim(ymin,ymax)
plt.xticks(fontsize=14);plt.yticks(fontsize=14)

plt.xlabel('Prioritize by race/ethnicity',fontsize=25) #'Prioritize by minority' #20220311

# 象限背景填色
x = (np.arange(20)-9)*0.01
plt.fill_between(x[:10], 0, ymax, facecolor='orange',alpha=0.2) #alpha=0.15
plt.fill_between(x[9:], ymin, 0, facecolor='green',alpha=0.2) #alpha=0.15
plt.fill_between(x[:10], ymin, 0, facecolor='blue',alpha=0.2) #alpha=0.15
plt.fill_between(x[9:], 0, ymax, facecolor='red',alpha=0.35)#alpha=0.15

#设置图片的边框为不显示
ax=plt.gca()  #gca:get current axis得到当前轴
ax.spines['right'].set_color('none');ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none');ax.spines['bottom'].set_color('none')

# 文字标明social utility/equity增长方向
ax.arrow(xmax-0.045,0,0.045, 0,length_includes_head=True,
         color='k',width=0.002,head_width=0.025,head_length=0.005)
t = ax.text(xmax, -0.07, "Social Utility", ha="right", va="center", rotation=0, size=22)
ax.arrow(-0.0002, ymax-0.35,0, 0.35,length_includes_head=True,
          color='k',width=0.0002,head_width=0.003,head_length=0.025)
t = ax.text(-0.006, ymax, "Equity", ha="center", va="top", rotation=90, size=22)

# Save the figure
savepath = os.path.join(root, subroot , 'fig2a_minority.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Minority, figure saved at: {savepath}.')


########################################################################################
# Legend
label_list = ['Prioritize the most disadvantaged','Prioritize the least disadvantaged']
plt.figure()
#patches = [mpatches.Patch(color=color_list[i], label="{:s}".format(label_list[i]) ) for i in range(2) ]
patches = [plt.scatter([],[],marker=marker_list[i],s=500,color=color_list[i], label="{:s}".format(label_list[i]) ) for i in range(len(label_list)) ]
#plt.set_position([box.x0, box.y0, box.width , box.height* 0.8]) #https://blog.csdn.net/yywan1314520/article/details/53740001
plt.legend(handles=patches,ncol=2,fontsize=30,bbox_to_anchor=(0.8,-0.1)) #,title='Policy',title_fontsize='x-large')
# Save the figure
savepath = os.path.join(root, subroot , 'fig2a_legend.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Legend, figure saved at: {savepath}.')