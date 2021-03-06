# python plot_comprehensive_utility_equity.py --with_supplementary

# pylint: disable=invalid-name,trailing-whitespace,superfluous-parens,line-too-long,multiple-statements, unnecessary-semicolon, redefined-outer-name, consider-using-enumerate


import socket
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import constants
import functions


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
parser.add_argument('--with_supplementary', default=False, action='store_true', #20220312
                    help='If true, plot all supplementary graphs.')
args = parser.parse_args()  


# root
root = os.getcwd()
resultroot = os.path.join(root, 'results')
# subroot
subroot = 'figures'
if not os.path.exists(os.path.join(root, subroot)): # if folder does not exist, create one. 
    os.makedirs(os.path.join(root, subroot))

recheck_interval_others = 0.01
msa_name_list = ['Atlanta', 'Chicago', 'Dallas', 'Houston', 'LosAngeles', 'Miami', 'Philadelphia', 'SanFrancisco', 'WashingtonDC']
msa_pop_list = [7191638, 10140946,8895355,7263757,15859681,6635035,6727280,5018570,7536060]
num_msas = len(msa_name_list)

# Drawing settings
anno_list = ['Atlanta','Chicago','Dallas','Houston','L.A.','Miami','Phila.','S.F.','D.C.']
msa_name_anno_list = anno_list #20220315
color_list = ['#FE2E2E','#FFBF00','#5FB404','#81BEF7','#29088A','grey','plum', '#FF8C00']

###########################################################################################################################
# Functions

def get_gini_dict(vaccination_time, vaccination_ratio,rel_to,msa_list,root, is_seed=False): 
    '''Load gini tables to construct gini_df_dict'''
    gini_df_dict = dict()
    for this_msa in msa_list:
        if(is_seed):
            print('Load seed results.')
            filepath = os.path.join(resultroot, 'gini_table', f'seed_gini_table_comprehensive_{str(vaccination_time)}_{vaccination_ratio}_{args.recheck_interval}_{args.num_groups}_{this_msa}_rel2{rel_to}.csv')
        else:
            filepath = os.path.join(resultroot, 'gini_table', f'gini_table_comprehensive_{str(vaccination_time)}_{vaccination_ratio}_{args.recheck_interval}_{args.num_groups}_{this_msa}_rel2{rel_to}.csv')
            
        gini_df = pd.read_csv(filepath)
        gini_df.rename(columns={'Unnamed: 0':'Dimension','Unnamed: 1':'Metric'},inplace=True)
        gini_df_dict[this_msa] = gini_df
    return gini_df_dict


def get_l1_norm(data_column):
    return -(data_column.iloc[1]+data_column.iloc[3]+data_column.iloc[5]+data_column.iloc[7]+data_column.iloc[9])


def get_overall_performance(gini_df_dict,  with_real_scaled=False):
    baseline_l1_norm = np.zeros(num_msas); 
    age_l1_norm = np.zeros(num_msas); income_l1_norm = np.zeros(num_msas); occupation_l1_norm = np.zeros(num_msas); minority_l1_norm = np.zeros(num_msas); 
    hybrid_l1_norm = np.zeros(num_msas); hybrid_ablation_l1_norm = np.zeros(num_msas);svi_l1_norm = np.zeros(num_msas);real_scaled_l1_norm = np.zeros(num_msas);

    for i in range(num_msas):
        baseline_l1_norm[i] = get_l1_norm(gini_df_dict[msa_name_list[i]]['Baseline'])
        age_l1_norm[i] =  get_l1_norm(gini_df_dict[msa_name_list[i]]['Age'])
        income_l1_norm[i] = get_l1_norm(gini_df_dict[msa_name_list[i]]['Income'])
        occupation_l1_norm[i] = get_l1_norm(gini_df_dict[msa_name_list[i]]['Occupation'])
        minority_l1_norm[i] = get_l1_norm(gini_df_dict[msa_name_list[i]]['Minority'])
        hybrid_l1_norm[i] = get_l1_norm(gini_df_dict[msa_name_list[i]]['Hybrid'])
        hybrid_ablation_l1_norm[i] = get_l1_norm(gini_df_dict[msa_name_list[i]]['Hybrid_Ablation'])  
        svi_l1_norm[i] = get_l1_norm(gini_df_dict[msa_name_list[i]]['SVI_new'])
        if(with_real_scaled):
            real_scaled_l1_norm[i] = get_l1_norm(gini_df_dict[msa_name_list[i]]['Real_Scaled'])

    # Weighted average results for all MSAs
    avg_baseline_l1_norm = np.average(baseline_l1_norm,weights=msa_pop_list)
    avg_age_l1_norm = np.average(age_l1_norm,weights=msa_pop_list)
    avg_income_l1_norm = np.average(income_l1_norm,weights=msa_pop_list)
    avg_occupation_l1_norm = np.average(occupation_l1_norm,weights=msa_pop_list)
    avg_minority_l1_norm = np.average(minority_l1_norm,weights=msa_pop_list) #20220308
    avg_hybrid_l1_norm = np.average(hybrid_l1_norm,weights=msa_pop_list)
    avg_hybrid_ablation_l1_norm = np.average(hybrid_ablation_l1_norm,weights=msa_pop_list)
    avg_svi_l1_norm = np.average(svi_l1_norm,weights=msa_pop_list)
    results = [avg_baseline_l1_norm,avg_age_l1_norm,avg_income_l1_norm,avg_occupation_l1_norm,avg_minority_l1_norm,
                avg_svi_l1_norm,avg_hybrid_ablation_l1_norm,avg_hybrid_l1_norm]
    anno_list = ['Homogeneous','Prioritize by age','Prioritize by income','Prioritize by occupation','Prioritize by race/ethnicity',
                 'SVI-informed','Comprehensive-ablation', 'Comprehensive']
    color_list = ['#29088A','#FFBF00','#5FB404','#81BEF7', '#FF8C00',
                  'grey','plum','#FE2E2E']
    if(with_real_scaled):
        avg_real_scaled_l1_norm = np.average(real_scaled_l1_norm,weights=msa_pop_list)
        results.append(avg_real_scaled_l1_norm)
        anno_list.append('Real-world')
        color_list.append('fuchsia')

    return results,anno_list,color_list

###########################################################################################################################
# Load gini tables to construct gini_df_dict

#gini_df_dict = get_gini_dict(args.vaccination_time, args.vaccination_ratio, args.rel_to, msa_name_list, root)

def get_util_equi_for_each_MSA(gini_df_dict): #20220312
    # Get util and equi change in each dimension, for each MSA
    baseline_util = np.zeros(num_msas); baseline_equi_age = np.zeros(num_msas); baseline_equi_income = np.zeros(num_msas); baseline_equi_occupation = np.zeros(num_msas); baseline_equi_minority = np.zeros(num_msas)
    age_util = np.zeros(num_msas); age_equi_age = np.zeros(num_msas); age_equi_income = np.zeros(num_msas); age_equi_occupation = np.zeros(num_msas); age_equi_minority = np.zeros(num_msas)
    income_util = np.zeros(num_msas); income_equi_age = np.zeros(num_msas); income_equi_income = np.zeros(num_msas); income_equi_occupation = np.zeros(num_msas); income_equi_minority = np.zeros(num_msas)
    occupation_util = np.zeros(num_msas); occupation_equi_age = np.zeros(num_msas); occupation_equi_income = np.zeros(num_msas); occupation_equi_occupation = np.zeros(num_msas); occupation_equi_minority = np.zeros(num_msas)
    minority_util = np.zeros(num_msas); minority_equi_age = np.zeros(num_msas); minority_equi_income = np.zeros(num_msas); minority_equi_occupation = np.zeros(num_msas); minority_equi_minority = np.zeros(num_msas)
    hybrid_util = np.zeros(num_msas); hybrid_equi_age = np.zeros(num_msas); hybrid_equi_income = np.zeros(num_msas); hybrid_equi_occupation = np.zeros(num_msas); hybrid_equi_minority = np.zeros(num_msas)
    hybrid_ablation_util = np.zeros(num_msas); hybrid_ablation_equi_age = np.zeros(num_msas); hybrid_ablation_equi_income = np.zeros(num_msas); hybrid_ablation_equi_occupation = np.zeros(num_msas); hybrid_ablation_equi_minority = np.zeros(num_msas)
    svi_util = np.zeros(num_msas); svi_equi_age = np.zeros(num_msas); svi_equi_income = np.zeros(num_msas); svi_equi_occupation = np.zeros(num_msas); svi_equi_minority = np.zeros(num_msas)

    for i in range(num_msas):
        # utility
        baseline_util[i] = -gini_df_dict[msa_name_list[i]]['Baseline'].iloc[1] 
        age_util[i] = -gini_df_dict[msa_name_list[i]]['Age'].iloc[1] 
        income_util[i] = -gini_df_dict[msa_name_list[i]]['Income'].iloc[1] 
        occupation_util[i] = -gini_df_dict[msa_name_list[i]]['Occupation'].iloc[1] 
        minority_util[i] = -gini_df_dict[msa_name_list[i]]['Minority'].iloc[1] 
        hybrid_util[i] = -gini_df_dict[msa_name_list[i]]['Hybrid'].iloc[1] 
        hybrid_ablation_util[i] = -gini_df_dict[msa_name_list[i]]['Hybrid_Ablation'].iloc[1] 
        svi_util[i] = -gini_df_dict[msa_name_list[i]]['SVI_new'].iloc[1] 
        
        # equity by age
        baseline_equi_age[i] = -gini_df_dict[msa_name_list[i]]['Baseline'].iloc[3] 
        age_equi_age[i] = -gini_df_dict[msa_name_list[i]]['Age'].iloc[3] 
        income_equi_age[i] = -gini_df_dict[msa_name_list[i]]['Income'].iloc[3] 
        occupation_equi_age[i] = -gini_df_dict[msa_name_list[i]]['Occupation'].iloc[3] 
        minority_equi_age[i] = -gini_df_dict[msa_name_list[i]]['Minority'].iloc[3] 
        hybrid_equi_age[i] = -gini_df_dict[msa_name_list[i]]['Hybrid'].iloc[3] 
        hybrid_ablation_equi_age[i] = -gini_df_dict[msa_name_list[i]]['Hybrid_Ablation'].iloc[3]
        svi_equi_age[i] = -gini_df_dict[msa_name_list[i]]['SVI_new'].iloc[3]

        # equity by income
        baseline_equi_income[i] = -gini_df_dict[msa_name_list[i]]['Baseline'].iloc[5] 
        age_equi_income[i] = -gini_df_dict[msa_name_list[i]]['Age'].iloc[5] 
        income_equi_income[i] = -gini_df_dict[msa_name_list[i]]['Income'].iloc[5] 
        occupation_equi_income[i] = -gini_df_dict[msa_name_list[i]]['Occupation'].iloc[5] 
        minority_equi_income[i] = -gini_df_dict[msa_name_list[i]]['Minority'].iloc[5] 
        hybrid_equi_income[i] = -gini_df_dict[msa_name_list[i]]['Hybrid'].iloc[5] 
        hybrid_ablation_equi_income[i] = -gini_df_dict[msa_name_list[i]]['Hybrid_Ablation'].iloc[5] 
        svi_equi_income[i] = -gini_df_dict[msa_name_list[i]]['SVI_new'].iloc[5] 

        # equity by occupation
        baseline_equi_occupation[i] = -gini_df_dict[msa_name_list[i]]['Baseline'].iloc[7] 
        age_equi_occupation[i] = -gini_df_dict[msa_name_list[i]]['Age'].iloc[7] 
        income_equi_occupation[i] = -gini_df_dict[msa_name_list[i]]['Income'].iloc[7] 
        occupation_equi_occupation[i] = -gini_df_dict[msa_name_list[i]]['Occupation'].iloc[7] 
        minority_equi_occupation[i] = -gini_df_dict[msa_name_list[i]]['Minority'].iloc[7] 
        hybrid_equi_occupation[i] = -gini_df_dict[msa_name_list[i]]['Hybrid'].iloc[7] 
        hybrid_ablation_equi_occupation[i] = -gini_df_dict[msa_name_list[i]]['Hybrid_Ablation'].iloc[7] 
        svi_equi_occupation[i] = -gini_df_dict[msa_name_list[i]]['SVI_new'].iloc[7]

        # equity by minority
        baseline_equi_minority[i] = -gini_df_dict[msa_name_list[i]]['Baseline'].iloc[9] 
        age_equi_minority[i] = -gini_df_dict[msa_name_list[i]]['Age'].iloc[9] 
        income_equi_minority[i] = -gini_df_dict[msa_name_list[i]]['Income'].iloc[9] 
        occupation_equi_minority[i] = -gini_df_dict[msa_name_list[i]]['Occupation'].iloc[9] 
        minority_equi_minority[i] = -gini_df_dict[msa_name_list[i]]['Minority'].iloc[9] 
        hybrid_equi_minority[i] = -gini_df_dict[msa_name_list[i]]['Hybrid'].iloc[9] 
        hybrid_ablation_equi_minority[i] = -gini_df_dict[msa_name_list[i]]['Hybrid_Ablation'].iloc[9] 
        svi_equi_minority[i] = -gini_df_dict[msa_name_list[i]]['SVI_new'].iloc[9] 


    return baseline_util, age_util, income_util, occupation_util, minority_util, hybrid_util, hybrid_ablation_util, svi_util, baseline_equi_age, age_equi_age, income_equi_age, occupation_equi_age, minority_equi_age, hybrid_equi_age, hybrid_ablation_equi_age, svi_equi_age, baseline_equi_income, age_equi_income, income_equi_income, occupation_equi_income, minority_equi_income, hybrid_equi_income, hybrid_ablation_equi_income, svi_equi_income, baseline_equi_occupation, age_equi_occupation, income_equi_occupation, occupation_equi_occupation, minority_equi_occupation, hybrid_equi_occupation, hybrid_ablation_equi_occupation, svi_equi_occupation, baseline_equi_minority, age_equi_minority, income_equi_minority, occupation_equi_minority, minority_equi_minority, hybrid_equi_minority, hybrid_ablation_equi_minority, svi_equi_minority

def get_seed_util_equi_for_each_MSA(gini_df_dict): #20220525
    #gini_df_dict[msa_name_list[0]][['Dimension','Metric']]
    #            Dimension            Metric
    #0                    All  deaths_total_abs
    #1                    All  deaths_total_rel
    #2            Elder_Ratio   deaths_gini_abs
    #3  Mean_Household_Income   deaths_gini_abs
    #4               EW_Ratio   deaths_gini_abs
    #5         Minority_Ratio   deaths_gini_abs
    #6            Elder_Ratio   deaths_gini_rel
    #7  Mean_Household_Income   deaths_gini_rel
    #8               EW_Ratio   deaths_gini_rel
    #9         Minority_Ratio   deaths_gini_rel
    # Get util and equi change in each dimension, for each MSA
    baseline_util = [np.zeros(args.num_seeds)]*num_msas; baseline_equi_age = [np.zeros(args.num_seeds)]*num_msas; baseline_equi_income = [np.zeros(args.num_seeds)]*num_msas; baseline_equi_occupation = [np.zeros(args.num_seeds)]*num_msas; baseline_equi_minority = [np.zeros(args.num_seeds)]*num_msas
    age_util = [np.zeros(args.num_seeds)]*num_msas; age_equi_age = [np.zeros(args.num_seeds)]*num_msas; age_equi_income = [np.zeros(args.num_seeds)]*num_msas; age_equi_occupation = [np.zeros(args.num_seeds)]*num_msas; age_equi_minority = [np.zeros(args.num_seeds)]*num_msas
    income_util = [np.zeros(args.num_seeds)]*num_msas; income_equi_age = [np.zeros(args.num_seeds)]*num_msas; income_equi_income = [np.zeros(args.num_seeds)]*num_msas; income_equi_occupation = [np.zeros(args.num_seeds)]*num_msas; income_equi_minority = [np.zeros(args.num_seeds)]*num_msas
    occupation_util = [np.zeros(args.num_seeds)]*num_msas; occupation_equi_age = [np.zeros(args.num_seeds)]*num_msas; occupation_equi_income = [np.zeros(args.num_seeds)]*num_msas; occupation_equi_occupation = [np.zeros(args.num_seeds)]*num_msas; occupation_equi_minority = [np.zeros(args.num_seeds)]*num_msas
    minority_util = [np.zeros(args.num_seeds)]*num_msas; minority_equi_age = [np.zeros(args.num_seeds)]*num_msas; minority_equi_income = [np.zeros(args.num_seeds)]*num_msas; minority_equi_occupation = [np.zeros(args.num_seeds)]*num_msas; minority_equi_minority = [np.zeros(args.num_seeds)]*num_msas
    hybrid_util = [np.zeros(args.num_seeds)]*num_msas; hybrid_equi_age = [np.zeros(args.num_seeds)]*num_msas; hybrid_equi_income = [np.zeros(args.num_seeds)]*num_msas; hybrid_equi_occupation = [np.zeros(args.num_seeds)]*num_msas; hybrid_equi_minority = [np.zeros(args.num_seeds)]*num_msas
    hybrid_ablation_util = [np.zeros(args.num_seeds)]*num_msas; hybrid_ablation_equi_age = [np.zeros(args.num_seeds)]*num_msas; hybrid_ablation_equi_income = [np.zeros(args.num_seeds)]*num_msas; hybrid_ablation_equi_occupation = [np.zeros(args.num_seeds)]*num_msas; hybrid_ablation_equi_minority = [np.zeros(args.num_seeds)]*num_msas
    svi_util = [np.zeros(args.num_seeds)]*num_msas; svi_equi_age = [np.zeros(args.num_seeds)]*num_msas; svi_equi_income = [np.zeros(args.num_seeds)]*num_msas; svi_equi_occupation = [np.zeros(args.num_seeds)]*num_msas; svi_equi_minority = [np.zeros(args.num_seeds)]*num_msas

    for i in range(num_msas):
        # utility
        baseline_util[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Baseline'].iloc[1])]
        age_util[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Age'].iloc[1])]
        income_util[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Income'].iloc[1])]
        occupation_util[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Occupation'].iloc[1])]
        minority_util[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Minority'].iloc[1])]
        hybrid_util[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Hybrid'].iloc[1])]
        hybrid_ablation_util[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Hybrid_Ablation'].iloc[1])]
        svi_util[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['SVI_new'].iloc[1])]
        
        # equity by age
        baseline_equi_age[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Baseline'].iloc[6])]
        age_equi_age[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Age'].iloc[6])]
        income_equi_age[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Income'].iloc[6])]
        occupation_equi_age[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Occupation'].iloc[6])]
        minority_equi_age[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Minority'].iloc[6])]
        hybrid_equi_age[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Hybrid'].iloc[6])]
        hybrid_ablation_equi_age[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Hybrid_Ablation'].iloc[6])]
        svi_equi_age[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['SVI_new'].iloc[6])]

        # equity by income
        baseline_equi_income[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Baseline'].iloc[7])]
        age_equi_income[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Age'].iloc[7])]
        income_equi_income[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Income'].iloc[7])]
        occupation_equi_income[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Occupation'].iloc[7])]
        minority_equi_income[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Minority'].iloc[7])]
        hybrid_equi_income[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Hybrid'].iloc[7])]
        hybrid_ablation_equi_income[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Hybrid_Ablation'].iloc[7])]
        svi_equi_income[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['SVI_new'].iloc[7])]

        # equity by occupation
        baseline_equi_occupation[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Baseline'].iloc[8])]
        age_equi_occupation[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Age'].iloc[8])]
        income_equi_occupation[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Income'].iloc[8])]
        occupation_equi_occupation[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Occupation'].iloc[8])]
        minority_equi_occupation[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Minority'].iloc[8])]
        hybrid_equi_occupation[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Hybrid'].iloc[8])]
        hybrid_ablation_equi_occupation[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Hybrid_Ablation'].iloc[8])]
        svi_equi_occupation[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['SVI_new'].iloc[8])]

        # equity by minority
        baseline_equi_minority[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Baseline'].iloc[9] )]
        age_equi_minority[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Age'].iloc[9] )]
        income_equi_minority[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Income'].iloc[9] )]
        occupation_equi_minority[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Occupation'].iloc[9])]
        minority_equi_minority[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Minority'].iloc[9])]
        hybrid_equi_minority[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Hybrid'].iloc[9])]
        hybrid_ablation_equi_minority[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['Hybrid_Ablation'].iloc[9])]
        svi_equi_minority[i] = [-np.array(value) for value in eval(gini_df_dict[msa_name_list[i]]['SVI_new'].iloc[9])] 

    return baseline_util, age_util, income_util, occupation_util, minority_util, hybrid_util, hybrid_ablation_util, svi_util, baseline_equi_age, age_equi_age, income_equi_age, occupation_equi_age, minority_equi_age, hybrid_equi_age, hybrid_ablation_equi_age, svi_equi_age, baseline_equi_income, age_equi_income, income_equi_income, occupation_equi_income, minority_equi_income, hybrid_equi_income, hybrid_ablation_equi_income, svi_equi_income, baseline_equi_occupation, age_equi_occupation, income_equi_occupation, occupation_equi_occupation, minority_equi_occupation, hybrid_equi_occupation, hybrid_ablation_equi_occupation, svi_equi_occupation, baseline_equi_minority, age_equi_minority, income_equi_minority, occupation_equi_minority, minority_equi_minority, hybrid_equi_minority, hybrid_ablation_equi_minority, svi_equi_minority


###########################################################################################################################
# Fig4a: main scenario, radar plot

def get_avg(inputs, weights): #20220308
    output = []
    for i in range(len(inputs)):
        output.append(np.average(inputs[i],weights=weights))
    return output



def draw_hybrid_policy(vac_ratio, vac_time, savepath, overall=True, show_axis_label=False, show_legend=False):
    print('%s%% Vaccinated on Day %s' % (int(vac_ratio*100), vac_time))
    if(overall): # Weighted average results for all MSAs
        avg_hybrid_util, avg_hybrid_equi_age, avg_hybrid_equi_income, avg_hybrid_equi_occupation, avg_hybrid_equi_minority = get_avg([hybrid_util, hybrid_equi_age, hybrid_equi_income,hybrid_equi_occupation,hybrid_equi_minority], weights=msa_pop_list)
        avg_baseline_util, avg_baseline_equi_age, avg_baseline_equi_income, avg_baseline_equi_occupation, avg_baseline_equi_minority = get_avg([baseline_util, baseline_equi_age, baseline_equi_income, baseline_equi_occupation, baseline_equi_minority], weights=msa_pop_list)
        avg_age_util, avg_age_equi_age, avg_age_equi_income, avg_age_equi_occupation, avg_age_equi_minority = get_avg([age_util, age_equi_age, age_equi_income, age_equi_occupation, age_equi_minority], weights=msa_pop_list)
        avg_income_util, avg_income_equi_age, avg_income_equi_income, avg_income_equi_occupation, avg_income_equi_minority = get_avg([income_util, income_equi_age, income_equi_income, income_equi_occupation, income_equi_minority], weights=msa_pop_list)
        avg_occupation_util, avg_occupation_equi_age, avg_occupation_equi_income, avg_occupation_equi_occupation, avg_occupation_equi_minority = get_avg([occupation_util, occupation_equi_age, occupation_equi_income, occupation_equi_occupation, occupation_equi_minority], weights=msa_pop_list)
        avg_minority_util, avg_minority_equi_age, avg_minority_equi_income, avg_minority_equi_occupation, avg_minority_equi_minority = get_avg([minority_util, minority_equi_age, minority_equi_income, minority_equi_occupation, minority_equi_minority], weights=msa_pop_list)
        avg_hybrid_ablation_util, avg_hybrid_ablation_equi_age, avg_hybrid_ablation_equi_income, avg_hybrid_ablation_equi_occupation, avg_hybrid_ablation_equi_minority = get_avg([hybrid_ablation_util, hybrid_ablation_equi_age, hybrid_ablation_equi_income, hybrid_ablation_equi_occupation,hybrid_ablation_equi_minority], weights=msa_pop_list)
        avg_svi_util, avg_svi_equi_age, avg_svi_equi_income, avg_svi_equi_occupation, avg_svi_equi_minority = get_avg([svi_util, svi_equi_age, svi_equi_income, svi_equi_occupation, svi_equi_minority], weights=msa_pop_list)
        
        radar_df = pd.DataFrame([[avg_hybrid_util,avg_hybrid_equi_age,avg_hybrid_equi_income,avg_hybrid_equi_occupation,avg_hybrid_equi_minority],
                                [avg_baseline_util,avg_baseline_equi_age,avg_baseline_equi_income,avg_baseline_equi_occupation,avg_baseline_equi_minority],
                                [avg_age_util,avg_age_equi_age,avg_age_equi_income,avg_age_equi_occupation,avg_age_equi_minority],
                                [avg_income_util,avg_income_equi_age,avg_income_equi_income,avg_income_equi_occupation,avg_income_equi_minority],
                                [avg_occupation_util,avg_occupation_equi_age,avg_occupation_equi_income,avg_occupation_equi_occupation,avg_occupation_equi_minority],
                                [avg_hybrid_ablation_util,avg_hybrid_ablation_equi_age,avg_hybrid_ablation_equi_income,avg_hybrid_ablation_equi_occupation,avg_hybrid_ablation_equi_minority],
                                [avg_minority_util,avg_minority_equi_age,avg_minority_equi_income,avg_minority_equi_occupation,avg_minority_equi_minority], #20220308
                                [avg_svi_util,avg_svi_equi_age,avg_svi_equi_income,avg_svi_equi_occupation,avg_svi_equi_minority],
                                ],
                                columns=list(['Utility','Equity-by-age','Equity-by-income','Equity-by-occupation','Equity-by-minority']))
        draw_radar(radar_df, savepath, show_axis_label, show_legend)

    else: # Draw figures for each MSA
        assert len(savepath)==len(msa_name_list)
        for i in range(len(msa_name_list)):
            radar_df = pd.DataFrame([[hybrid_util[i],hybrid_equi_age[i],hybrid_equi_income[i],hybrid_equi_occupation[i],hybrid_equi_minority[i]],
                                [baseline_util[i],baseline_equi_age[i],baseline_equi_income[i],baseline_equi_occupation[i],baseline_equi_minority[i]],
                                [age_util[i],age_equi_age[i],age_equi_income[i],age_equi_occupation[i],age_equi_minority[i]],
                                [income_util[i],income_equi_age[i],income_equi_income[i],income_equi_occupation[i],income_equi_minority[i]],
                                [occupation_util[i],occupation_equi_age[i],occupation_equi_income[i],occupation_equi_occupation[i],occupation_equi_minority[i]],
                                [hybrid_ablation_util[i],hybrid_ablation_equi_age[i],hybrid_ablation_equi_income[i],hybrid_ablation_equi_occupation[i],hybrid_ablation_equi_minority[i]],
                                [minority_util[i],minority_equi_age[i],minority_equi_income[i],minority_equi_occupation[i],minority_equi_minority[i]], #20220308
                                [svi_util[i],svi_equi_age[i],svi_equi_income[i],svi_equi_occupation[i],svi_equi_minority[i]],
                                ],
                                columns=list(['Utility','Equity-by-age','Equity-by-income','Equity-by-occupation','Equity-by-minority'])) 
            draw_radar(radar_df, savepath[i], title=msa_name_anno_list[i])        


def draw_radar(radar_df, savepath, show_axis_label=False, show_legend=False, title=None): #20220309
    # Normalization:
    for column in radar_df.columns:
        radar_df[column] /= radar_df[column].iloc[0]#this is original
        #radar_df[column] /= radar_df[column].iloc[1]

    # Start plotting
    # Number of variables we're plotting.
    num_vars = 5 #4
    # Split the circle into even parts and save the angles, so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # The plot is a circle, so we need to "complete the loop", and append the start value to the end.
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    linewidth = 5 #4 #2.5
    markersize = 15 #10
    value_max_list = [] #20220312
    # Retreive the values
    # Age-Prioritized
    values = radar_df.loc[2].tolist();value_max_list.append(max(values))
    values += values[:1]
    ax.plot(angles, values, color=color_list[1], linewidth=linewidth,label='Prioritize by age',marker='o',markersize=markersize, zorder=1)
    # Income-Prioritized
    values = radar_df.loc[3].tolist();value_max_list.append(max(values))
    values += values[:1]
    ax.plot(angles, values, color=color_list[2], linewidth=linewidth,label='Prioritize by income',marker='o',markersize=markersize, zorder=1)
    # Occupation-Prioritized
    values = radar_df.loc[4].tolist();value_max_list.append(max(values))
    values += values[:1]
    ax.plot(angles, values, color=color_list[3], linewidth=linewidth,label='Prioritize by occupation',marker='o',markersize=markersize, zorder=1)
    # Minority-Priotized 
    values = radar_df.loc[6].tolist();value_max_list.append(max(values))
    values += values[:1]
    ax.plot(angles, values, color=color_list[7], linewidth=linewidth,label='Prioritize by race/ethnicity',marker='o',markersize=markersize, zorder=1)
    
    # SVI
    values = radar_df.loc[7].tolist();value_max_list.append(max(values))
    values += values[:1]
    ax.plot(angles, values, color=color_list[5], linewidth=linewidth,label='SVI-informed',marker='o',markersize=markersize, zorder=1)

    # Hybrid_Ablation
    values = radar_df.loc[5].tolist();value_max_list.append(max(values))
    values += values[:1]
    ax.plot(angles, values, color=color_list[6], linewidth=linewidth,label='Comprehensive-ablation',marker='o',markersize=markersize, zorder=1)
    
    # Random Baseline
    values = radar_df.loc[1].tolist();value_max_list.append(max(values))
    values += values[:1]
    ax.plot(angles, values, color=color_list[4], linewidth=linewidth,label='Homogeneous',marker='o',markersize=markersize, zorder=1)

    # Hybrid
    values = radar_df.loc[0].tolist();value_max_list.append(max(values))
    # The plot is a circle, so we need to "complete the loop", and append the start value to the end.
    values += values[:1]
    # Draw the outline of our data.
    ax.plot(angles, values, color=color_list[0], linewidth=linewidth,label='Comprehensive',marker='o',markersize=markersize, zorder=2)
    
    ############################################################
    # Refinement of the graph

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    if(show_axis_label):
        show_labels = ['Change in\nsocial utility','Change in\nequity-by-\nage','Change in\nequity-by-\nincome','Change in\nequity-by-\noccupation','Change in\nequity-by-\nrace/ethnicity']
    else:
        show_labels = []
    ax.set_thetagrids((np.array(angles) * 180/np.pi)[0:5], show_labels, fontsize=26)

    # Go through labels and adjust alignment based on where it is in the circle.
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle==np.pi: label.set_verticalalignment('top')
        if angle==0: label.set_verticalalignment('bottom')
        if angle in (0, np.pi): label.set_horizontalalignment('center')
        elif 0 < angle < np.pi: label.set_horizontalalignment('left')
        else: label.set_horizontalalignment('right')

    # Set position of y-labels (0-100) to be in the middle of the first two axes.
    ax.set_rlabel_position(180 / num_vars)
    #ax.set_rgrids([0,0.2,0.4,0.6,0.8,1.0,1.2], fontsize=18)
    all_max = max(value_max_list) #20220312
    ax.set_rgrids(np.round(np.arange(7) *all_max/7, 2), fontsize=14, zorder=3)
       
    # Legend
    if(show_legend):
        ax.legend(loc='lower center',ncol=2,bbox_to_anchor=(0.5,-1),fontsize=20.5) 

    # Title (MSA name) 
    if(title):
        #print('title: ', title)
        ax.set_title(title, fontsize=28)

    # Save the figure
    plt.savefig(savepath,bbox_inches = 'tight')
    plt.close()


vaccination_time = 31
vaccination_ratio = 0.1    
rel_to = 'No_Vaccination'
gini_df_dict = get_gini_dict(vaccination_time, vaccination_ratio,rel_to,msa_name_list,root)
result = get_util_equi_for_each_MSA(gini_df_dict)
baseline_util, age_util, income_util, occupation_util, minority_util, hybrid_util, hybrid_ablation_util, svi_util, baseline_equi_age, age_equi_age, income_equi_age, occupation_equi_age, minority_equi_age, hybrid_equi_age, hybrid_ablation_equi_age, svi_equi_age, baseline_equi_income, age_equi_income, income_equi_income, occupation_equi_income, minority_equi_income, hybrid_equi_income, hybrid_ablation_equi_income, svi_equi_income, baseline_equi_occupation, age_equi_occupation, income_equi_occupation, occupation_equi_occupation, minority_equi_occupation, hybrid_equi_occupation, hybrid_ablation_equi_occupation, svi_equi_occupation, baseline_equi_minority, age_equi_minority, income_equi_minority, occupation_equi_minority, minority_equi_minority, hybrid_equi_minority, hybrid_ablation_equi_minority, svi_equi_minority = result
# Save the figure
savepath = os.path.join(root, subroot , 'fig4a.pdf')
draw_hybrid_policy(vaccination_ratio, vaccination_time, overall=True, show_axis_label=True, show_legend=True, savepath=savepath)
print(f'Fig4a, figure saved at: {savepath}.')

###########################################################################################################################
# Fig4b: main scenario, utility change in each MSA

vaccination_time = 31
vaccination_ratio = 0.1    
rel_to = 'No_Vaccination'
gini_df_dict = get_gini_dict(vaccination_time, vaccination_ratio,rel_to,msa_name_list,root,is_seed=True)
result = get_seed_util_equi_for_each_MSA(gini_df_dict) #20220525
baseline_util, age_util, income_util, occupation_util, minority_util, hybrid_util, hybrid_ablation_util, svi_util, baseline_equi_age, age_equi_age, income_equi_age, occupation_equi_age, minority_equi_age, hybrid_equi_age, hybrid_ablation_equi_age, svi_equi_age, baseline_equi_income, age_equi_income, income_equi_income, occupation_equi_income, minority_equi_income, hybrid_equi_income, hybrid_ablation_equi_income, svi_equi_income, baseline_equi_occupation, age_equi_occupation, income_equi_occupation, occupation_equi_occupation, minority_equi_occupation, hybrid_equi_occupation, hybrid_ablation_equi_occupation, svi_equi_occupation, baseline_equi_minority, age_equi_minority, income_equi_minority, occupation_equi_minority, minority_equi_minority, hybrid_equi_minority, hybrid_ablation_equi_minority, svi_equi_minority = result

plt.figure(figsize=(14,6))

dist = 3.3 #4 
alpha = 1 
widths = 0.9
bp = plt.boxplot(hybrid_util, positions=np.arange(9)*dist,patch_artist=True,widths=widths) #labels='Comprehensive',color=color_list[0],alpha=alpha
[bp['boxes'][i].set(facecolor=color_list[0], alpha=alpha) for i in range(num_msas)]
bp = plt.boxplot(age_util, positions=np.arange(9)*dist+1,patch_artist=True,widths=widths) #labels='Prioritize by age',color=color_list[1],alpha=alpha
[bp['boxes'][i].set(facecolor=color_list[1], alpha=alpha) for i in range(num_msas)]
bp = plt.boxplot(baseline_util, positions=np.arange(9)*dist+2,patch_artist=True,widths=widths) #labels='Homogeneous',color=color_list[4],alpha=alpha
[bp['boxes'][i].set(facecolor=color_list[4], alpha=alpha) for i in range(num_msas)]

plt.xlim(-widths/2-0.2, 8*dist+2+widths/2+0.2)
plt.xticks(np.arange(9)*dist+1,anno_list,fontsize=24,rotation=20)
plt.ylabel('Change in social utility\n(w.r.t. No-vaccination)',fontsize=24)

# Background color
for i in range(num_msas):
    if(i%2==0):
        plt.axvspan(i*dist-widths/2, i*dist-widths/2+3, color='grey',alpha=0.15)
    else:
        plt.axvspan(i*dist-widths/2, i*dist-widths/2+3, color='silver',alpha=0.15)


# Save the figure
savepath = os.path.join(root, subroot , 'fig4b.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Fig4b, figure saved at: {savepath}.')
plt.close()

# Fig4b, legend
plt.figure()
label_list = ['Comprehensive','Prioritize by age','Homogeneous']
legend_color_list = [color_list[0],color_list[1],color_list[4]]
alpha_list = [alpha]*3
patches = [mpatches.Patch(color=legend_color_list[i], alpha=alpha_list[i], label="{:s}".format(label_list[i])) for i in range(len(label_list)) ]
plt.legend(handles=patches,ncol=3,fontsize=20,bbox_to_anchor=(0.8,-0.1)) 
# Save figure
savepath = os.path.join(root, subroot, f'fig4b_legend_0525.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Fig4b_legend, saved at {savepath}')

###########################################################################################################################
# Fig4c: main scenario, equity change in each MSA

vaccination_time = 31
vaccination_ratio = 0.1    
rel_to = 'No_Vaccination'
gini_df_dict = get_gini_dict(vaccination_time, vaccination_ratio,rel_to,msa_name_list,root)
result = get_util_equi_for_each_MSA(gini_df_dict)
baseline_util, age_util, income_util, occupation_util, minority_util, hybrid_util, hybrid_ablation_util, svi_util, baseline_equi_age, age_equi_age, income_equi_age, occupation_equi_age, minority_equi_age, hybrid_equi_age, hybrid_ablation_equi_age, svi_equi_age, baseline_equi_income, age_equi_income, income_equi_income, occupation_equi_income, minority_equi_income, hybrid_equi_income, hybrid_ablation_equi_income, svi_equi_income, baseline_equi_occupation, age_equi_occupation, income_equi_occupation, occupation_equi_occupation, minority_equi_occupation, hybrid_equi_occupation, hybrid_ablation_equi_occupation, svi_equi_occupation, baseline_equi_minority, age_equi_minority, income_equi_minority, occupation_equi_minority, minority_equi_minority, hybrid_equi_minority, hybrid_ablation_equi_minority, svi_equi_minority = result

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

# ????????????
x = (np.arange(9*dist_1)-2)
for i in range(num_msas):
    if(i%2==0):
        plt.fill_between(x[i*dist_1:(i+1)*dist_1], ymin, ymax, facecolor='grey',alpha=0.15)
    else:
        plt.fill_between(x[i*dist_1:(i+1)*dist_1], ymin, ymax, facecolor='silver',alpha=0.15)

# ??????
ax = plt.gca()
box = ax.get_position()

label_list = ['Equity-by-age','Equity-by-income','Equity-by-occupation', 'Equity-by-race/ethnicity']
patches = [plt.scatter([],[],marker=marker_list[i],s=markersize+10,c='none',edgecolors='k',label="{:s}".format(label_list[i]) ) for i in range(len(label_list)) ]
plt.legend(handles=patches, loc='lower center',ncol=2,fontsize=18)

# Save the figure
savepath = os.path.join(root, subroot , 'fig4c.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Fig4c, figure saved at: {savepath}.')
plt.close()

###########################################################################################################################
# Fig4d

rel_to = 'Baseline'
opt = 'rate' 

ratio_time_list = [[0.05,31],[0.1,31],[0.15,31],[0.2,31],[0.4,31],[0.56,31]]
x_positions = [5,10,15,20,40,56] #x_positions=np.arange(num_scenarios)
x_ticks_list = []
for i in range(len(x_positions)):
    x_ticks_list.append(str(x_positions[i])+'%')

results_dict = dict()
for i in range(len(ratio_time_list)):
    vaccination_ratio = ratio_time_list[i][0]
    vaccination_time = ratio_time_list[i][1]
    print('%s%% Vaccinated on Day %s' % (int(vaccination_ratio*100), vaccination_time))
    
    if(vaccination_ratio==0.56):
        with_real_scaled = True
    else:
        with_real_scaled = False
    gini_df_dict = get_gini_dict(vaccination_time,vaccination_ratio, rel_to, msa_name_list, root)
    if(vaccination_ratio==0.56):
        results, anno_list, color_list = get_overall_performance(gini_df_dict,with_real_scaled=True)
    else:
        results, anno_list, color_list = get_overall_performance(gini_df_dict,with_real_scaled=False)
    results_dict['%s, %s'%(ratio_time_list[i][0],ratio_time_list[i][1])] = results

plt.figure(figsize=(12,3))
num_scenarios = len(results_dict.keys())
num_strategies = len(list(results_dict.values())[0])
for strategy_idx in range(1,num_strategies): #0 is Baseline, we don't plot it
    strategy_results = []
    for i in range(num_scenarios):
        strategy_results.append(results_dict['%s, %s'%(ratio_time_list[i][0],ratio_time_list[i][1])][strategy_idx])
    plt.plot(x_positions,strategy_results,
             color=color_list[strategy_idx],label=anno_list[strategy_idx],marker='^',markersize=14,zorder=1+strategy_idx)
    
if('0.56, 31' in results_dict.keys()):
    plt.scatter(x_positions[-1],results_dict['0.56, 31'][num_strategies],
                label='Real-world',color=color_list[num_strategies],marker='*',s=220,zorder=50)
        
plt.xticks(x_positions,x_ticks_list,fontsize=15)
plt.legend(bbox_to_anchor=(1.01, 1),fontsize=13)
plt.ylim(-0.55,1.85)
plt.xlabel('Scenarios with different vaccination %ss'%opt,fontsize=18)
plt.ylabel('Overall performance\n(w.r.t. Homogeneous)',fontsize=17)

# Save the figure
savepath = os.path.join(root, subroot , 'fig4d.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Fig4d, figure saved at: {savepath}.')
plt.close()

###########################################################################################################################
# Fig4e

rel_to = 'Baseline'
ratio_time_list = [[0.1,26],[0.1,31],[0.1,36],[0.1,41]]
results_dict = dict()
for i in range(len(ratio_time_list)):
    vaccination_ratio = ratio_time_list[i][0]
    vaccination_time = ratio_time_list[i][1]
    print('%s%% Vaccinated on Day %s' % (int(vaccination_ratio*100), vaccination_time))
    gini_df_dict = get_gini_dict(vaccination_time,vaccination_ratio, rel_to, msa_name_list, root)
    results, anno_list, color_list = get_overall_performance(gini_df_dict,with_real_scaled=False)
    results_dict['%s, %s'%(ratio_time_list[i][0],ratio_time_list[i][1])] = results
#print(results_dict)
    
plt.figure(figsize=(12,3))
num_scenarios = len(results_dict.keys())
num_strategies = len(results)
for strategy_idx in range(1,num_strategies):
    strategy_results = []
    for i in range(num_scenarios):
        strategy_results.append(results_dict['%s, %s'%(ratio_time_list[i][0],ratio_time_list[i][1])][strategy_idx])
    plt.plot(np.arange(num_scenarios),strategy_results,
             color=color_list[strategy_idx],label=anno_list[strategy_idx], marker='^',markersize=15)   
    plt.xticks(np.arange(num_scenarios),results_dict.keys(),fontsize=15)
    plt.legend(bbox_to_anchor=(1.01, 1),fontsize=14)
    plt.ylabel('Overall performance\n(w.r.t. Homogeneous)',fontsize=18)
    plt.xlabel('Scenarios with different vaccination timings (rate, timing)',fontsize=18)

# Save the figure
savepath = os.path.join(root, subroot , 'fig4e.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Fig4e, figure saved at: {savepath}.')
plt.close()

###########################################################################################################################
# Supplementary
color_list = ['#FE2E2E','#FFBF00','#5FB404','#81BEF7','#29088A','grey','plum', '#FF8C00']
#color_list = ['#29088A','#FFBF00','#5FB404','#81BEF7', '#FF8C00','grey','plum','#FE2E2E']
rel_to = 'No_Vaccination'
# Subsubroot for supplementary figures
subsubroot = 'sup'
if not os.path.exists(os.path.join(root, subroot, subsubroot)): # if folder does not exist, create one. #20220309
    os.makedirs(os.path.join(root, subroot, subsubroot))
#ratio_time_list = [[0.1,24],[0.1,29],[0.1,34],[0.1,39],
#                   [0.4,31],[0.56,31],[0.03,31],[0.08,31],[0.13,31],[0.18,31]]
ratio_time_list = [[0.1,26],[0.1,31],[0.1,36],[0.1,41],
                   [0.05,31],[0.1,31],[0.15,31],[0.2,31],[0.4,31],[0.56,31]]

if(args.with_supplementary):
    for i in range(len(ratio_time_list)):
        print(ratio_time_list[i])
        vaccination_ratio = ratio_time_list[i][0]
        vaccination_time = ratio_time_list[i][1] 
        gini_df_dict = get_gini_dict(vaccination_time, vaccination_ratio,rel_to,msa_name_list,root)
        result = get_util_equi_for_each_MSA(gini_df_dict)
        baseline_util, age_util, income_util, occupation_util, minority_util, hybrid_util, hybrid_ablation_util, svi_util, baseline_equi_age, age_equi_age, income_equi_age, occupation_equi_age, minority_equi_age, hybrid_equi_age, hybrid_ablation_equi_age, svi_equi_age, baseline_equi_income, age_equi_income, income_equi_income, occupation_equi_income, minority_equi_income, hybrid_equi_income, hybrid_ablation_equi_income, svi_equi_income, baseline_equi_occupation, age_equi_occupation, income_equi_occupation, occupation_equi_occupation, minority_equi_occupation, hybrid_equi_occupation, hybrid_ablation_equi_occupation, svi_equi_occupation, baseline_equi_minority, age_equi_minority, income_equi_minority, occupation_equi_minority, minority_equi_minority, hybrid_equi_minority, hybrid_ablation_equi_minority, svi_equi_minority = result

        savepath_list = []
        for this_msa in msa_name_list:
            savepath = os.path.join(root, subroot , subsubroot, f'sup_{vaccination_time}_{vaccination_ratio}_{this_msa}.pdf')
            savepath_list.append(savepath)
        draw_hybrid_policy(vaccination_ratio, vaccination_time, overall=False, show_axis_label=True, show_legend=True, savepath=savepath_list)
        print(f'Supplementary, figure saved at: {savepath_list}.')

    # Supplementary, legend
    plt.figure()
    label_list = anno_list
    color_list = color_list
    patches = [plt.scatter([],[],marker='o',s=500,color=color_list[i], label="{:s}".format(label_list[i]) ) for i in range(len(label_list))]
    plt.legend(handles=patches,ncol=1,fontsize=20,bbox_to_anchor=(0.8,-0.1)) 
    # Save figure
    savepath = os.path.join(root, subroot , subsubroot, f'supplementary_legend.pdf')
    plt.savefig(savepath,bbox_inches = 'tight')
    print(f'Supplementary legend, saved at {savepath}')