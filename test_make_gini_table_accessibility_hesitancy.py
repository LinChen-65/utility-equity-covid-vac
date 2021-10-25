# python test_make_gini_table_accessibility_hesitancy.py MSA_NAME vaccination_time vaccination_ratio recheck_interval rel_to NUM_GROUPS consider_hesitancy ACCEPTANCE_SCENARIO consider_accessibility
# python test_make_gini_table_accessibility_hesitancy.py Atlanta 31 0.1 recheck_interval Baseline 5 True real True

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import sys
import socket
import os
import glob
import re
import pandas as pd
import numpy as np
import constants
import functions
import pdb

############################################################
# Main variable settings

# root
hostname = socket.gethostname()
print('hostname: ', hostname)
if(hostname=='fib-dl3'):
    root = '/data/chenlin/COVID-19/Data' #dl3
elif(hostname=='rl4'):
    root = '/home/chenlin/COVID-19/Data' #rl4

# Simulation times and random seeds
NUM_SEEDS = 30

MSA_NAME = sys.argv[1]; print('MSA_NAME:',MSA_NAME)
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME]; #print('MSA_NAME_FULL:',MSA_NAME_FULL)

# Vaccination Time
VACCINATION_TIME_STR = sys.argv[2]; print('VACCINATION_TIME:',VACCINATION_TIME_STR)
VACCINATION_TIME = float(VACCINATION_TIME_STR)

# Vaccination_Ratio
VACCINATION_RATIO = float(sys.argv[3]); print('VACCINATION_RATIO: ', VACCINATION_RATIO)

# Recheck interval for comprehensive & comprehensive_ablation
RECHECK_INTERVAL = float(sys.argv[4]); print('RECHECK_INTERVAL: ', RECHECK_INTERVAL)
# Recheck interval for other strategies
RECHECK_INTERVAL_OTHERS = 0.01

# Relative to which variable
REL_TO = sys.argv[5]; print('Relative to: ', REL_TO)

demo_policy_list = ['Age_Flood', 'Income_Flood', 'JUE_EW_Flood'] 

# Number of groups
NUM_GROUPS = int(sys.argv[6]); print('NUM_GROUPS: ',NUM_GROUPS)

# 由于Comprehensive的weight_string要由Comprehensive_Ablation得到，故后者应放在前面。
# 仅用于vaccination_rate=0.56的policy: 'Real_Scaled','Real_Scaled_Flood'
if(REL_TO=='No_Vaccination'):
    if(VACCINATION_RATIO==0.56):
        policy_list = ['No_Vaccination','Baseline', 'Age_Flood', 'Income_Flood', 'JUE_EW_Flood',
                       'SVI', 'Real_Scaled','Comprehensive_Ablation','Comprehensive'] 
    else:
        policy_list = ['No_Vaccination','Baseline', 'Age_Flood', 'Income_Flood', 'JUE_EW_Flood',
                       'SVI', 'Comprehensive_Ablation','Comprehensive'] 
    #policy_list = ['No_Vaccination','Baseline',
    #                'Age_Flood','Age_Flood_Reverse',
    #                'Income_Flood', 'Income_Flood_Reverse',
    #                'JUE_EW_Flood','JUE_EW_Flood_Reverse'] 

elif(REL_TO=='Baseline'):
    if(VACCINATION_RATIO==0.56):
        policy_list = ['Baseline', 'No_Vaccination','Age_Flood', 'Income_Flood', 'JUE_EW_Flood',
                        'SVI','Real_Scaled','Comprehensive_Ablation','Comprehensive'] 
    else:         
        policy_list = ['Baseline', 'No_Vaccination','Age_Flood', 'Income_Flood', 'JUE_EW_Flood',
                        'SVI','Comprehensive_Ablation','Comprehensive']            
    # #policy_list = ['Baseline','No_Vaccination', 
    #                'Age_Flood','Age_Flood_Reverse',
    #                'Income_Flood', 'Income_Flood_Reverse',
    #                'JUE_EW_Flood','JUE_EW_Flood_Reverse'] 
else:
    print('Invalid REL_TO. Please check.')
    pdb.set_trace()
print('policy list:', policy_list)

# Consider hesitancy or not
consider_hesitancy = sys.argv[7]
print('Consider hesitancy? ', consider_hesitancy)
if(consider_hesitancy not in ['True','False']): 
    print('Invalid value for consider_hesitancy. Please check.')
    pdb.set_trace()

# Vaccine acceptance scenario
ACCEPTANCE_SCENARIO = sys.argv[8]; print('Vaccine acceptance scenario: ', ACCEPTANCE_SCENARIO)

# Consider accessibility or not
consider_accessibility = sys.argv[9]; print('Consider accessibility?', consider_accessibility)


############################################################
# Functions

# Analyze results and produce graphs: All policies
def output_result(cbg_table, demo_feat, policy_list, num_groups, print_result=True, rel_to=REL_TO):
    results = {}
    
    for policy in policy_list:
        exec("final_deaths_rate_%s_total = cbg_table['Final_Deaths_%s'].sum()/cbg_table['Sum'].sum()" % (policy.lower(),policy))
        cbg_table['Final_Deaths_' + policy] = eval('avg_final_deaths_' + policy.lower())
        exec("%s = np.zeros(num_groups)" % ('final_deaths_rate_'+ policy.lower()))
        deaths_total_abs = eval('final_deaths_rate_%s_total'%(policy.lower()))
        deaths_total_abs = np.round(deaths_total_abs,6)
        
        for i in range(num_groups):
            eval('final_deaths_rate_'+ policy.lower())[i] = cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Final_Deaths_' + policy].sum()
            eval('final_deaths_rate_'+ policy.lower())[i] /= cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Sum'].sum()
        deaths_gini_abs = functions.gini(eval('final_deaths_rate_'+ policy.lower()))
        deaths_gini_abs = np.round(deaths_gini_abs,6)
       
        if(rel_to=='No_Vaccination'): # compared to No_Vaccination
            if(policy=='No_Vaccination'):
                deaths_total_no_vaccination = deaths_total_abs
                deaths_gini_no_vaccination = deaths_gini_abs
                deaths_total_rel = 0
                deaths_gini_rel = 0
                results[policy] = {
                                   'deaths_total_abs':'%.6f'% deaths_total_abs, #.6f
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.6f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.6f'% deaths_gini_rel}   
            else:
                deaths_total_rel = (np.round(eval('final_deaths_rate_%s_total'%(policy.lower())),6) - deaths_total_no_vaccination) / deaths_total_no_vaccination
                deaths_gini_rel = (np.round(functions.gini(eval('final_deaths_rate_'+ policy.lower())),6) - deaths_gini_no_vaccination) / deaths_gini_no_vaccination
                results[policy] = {
                                   'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.6f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.6f'% deaths_gini_rel}    
        
        elif(rel_to=='Baseline'): # compared to Baseline
            if(policy=='Baseline'):
                deaths_total_baseline = deaths_total_abs
                deaths_gini_baseline = deaths_gini_abs
                deaths_total_rel = 0
                deaths_gini_rel = 0
                results[policy] = {
                                   'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.6f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.6f'% deaths_gini_rel}   
            else:
                deaths_total_rel = (np.round(eval('final_deaths_rate_%s_total'%(policy.lower())),6) - deaths_total_baseline) / deaths_total_baseline
                deaths_gini_rel = (np.round(functions.gini(eval('final_deaths_rate_'+ policy.lower())),6) - deaths_gini_baseline) / deaths_gini_baseline
                results[policy] = {
                                   'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.6f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.6f'% deaths_gini_rel}                        
    return results


def make_gini_table(policy_list, demo_feat_list, parallel, num_groups, save_result=False, save_path=None):
    cbg_table_name_dict=dict()
    cbg_table_name_dict['Age'] = cbg_age_msa
    cbg_table_name_dict['Mean_Household_Income'] = cbg_income_msa
    cbg_table_name_dict['Essential_Worker'] = cbg_occupation_msa
    
    #print('Policy list: ', policy_list)
    #print('Demographic feature list: ', demo_feat_list)

    gini_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('All','deaths_total_abs'),('All','deaths_total_rel')]))
    if(parallel==True):
        display_list = [policy_list[0]]
        for i in range(target_num):
            display_list.append('Target'+str(i)) # 20210627
        gini_df['Policy'] = display_list
    else:
        gini_df['Policy'] = policy_list
        
    for demo_feat in demo_feat_list:
        results = output_result(cbg_table_name_dict[demo_feat], 
                                demo_feat, policy_list, num_groups=NUM_GROUPS,
                                print_result=False, rel_to=REL_TO)
       
        for i in range(len(policy_list)):
            policy = policy_list[i]
            gini_df.loc[i,('All','deaths_total_abs')] = results[policy]['deaths_total_abs']
            gini_df.loc[i,('All','deaths_total_rel')] = results[policy]['deaths_total_rel'] 
            gini_df.loc[i,(demo_feat,'deaths_gini_abs')] = results[policy]['deaths_gini_abs']
            gini_df.loc[i,(demo_feat,'deaths_gini_rel')] = results[policy]['deaths_gini_rel'] 

    gini_df.set_index(['Policy'],inplace=True)
    # Transpose
    gini_df_trans = pd.DataFrame(gini_df.values.T, index=gini_df.columns, columns=gini_df.index)#转置
    # Save .csv
    if(save_result==True):
        gini_df_trans.to_csv(save_path)
        
    return gini_df_trans

def get_overall_performance(data_column): #20211020
    return -(float(data_column.iloc[1])+float(data_column.iloc[3])+float(data_column.iloc[5])+float(data_column.iloc[7]))

############################################################
# Load Data

# Load ACS Data for matching with NYT Data
acs_data = pd.read_csv(os.path.join(root,'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]

# Extract data specific to one msa, according to ACS data
# MSA list
msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
#print('\nMatching MSA_NAME_FULL to MSAs in ACS Data: ',msa_match)
msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
#print('Number of counties matched: ',len(msa_data))
msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
good_list = list(msa_data['FIPS Code'].values);#print('Indices of counties matched: ',good_list)

# Load CBG ids belonging to a specific metro area
cbg_ids_msa = pd.read_csv(os.path.join(root,MSA_NAME,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
M = len(cbg_ids_msa);#print('Number of CBGs in this metro area:', M)
# Mapping from cbg_ids to columns in hourly visiting matrices
cbgs_to_idxs = dict(zip(cbg_ids_msa['census_block_group'].values, range(M)))
x = {}
for i in cbgs_to_idxs:
    x[str(i)] = cbgs_to_idxs[i]

# Select counties belonging to the MSA
y = []
for i in x:
    if((len(i)==12) & (int(i[0:5])in good_list)):
        y.append(x[i])
    if((len(i)==11) & (int(i[0:4])in good_list)):
        y.append(x[i])
        
idxs_msa_all = list(x.values());#print('Number of CBGs in this metro area:', len(idxs_msa_all))
idxs_msa_nyt = y; #print('Number of CBGs in to compare with NYT data:', len(idxs_msa_nyt))

# Load ACS Data for MSA-county matching
acs_data = pd.read_csv(os.path.join(root,'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
good_list = list(msa_data['FIPS Code'].values)
#print('Counties included: ', good_list)
del acs_data

# Load SafeGraph data to obtain CBG sizes (i.e., populations)
filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_b01.csv")
cbg_agesex = pd.read_csv(filepath)
# Extract CBGs belonging to the MSA - https://covid-mobility.stanford.edu//datasets/
cbg_age_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
del cbg_agesex
# Add up males and females of the same age, according to the detailed age list (DETAILED_AGE_LIST)
# which is defined in Constants.py
for i in range(3,25+1): # 'B01001e3'~'B01001e25'
    male_column = 'B01001e'+str(i)
    female_column = 'B01001e'+str(i+24)
    cbg_age_msa[constants.DETAILED_AGE_LIST[i-3]] = cbg_age_msa.apply(lambda x : x[male_column]+x[female_column],axis=1)
# Rename
cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
# Extract columns of interest
columns_of_interest = ['census_block_group','Sum'] + constants.DETAILED_AGE_LIST
cbg_age_msa = cbg_age_msa[columns_of_interest].copy()
# Deal with CBGs with 0 populations
cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
# Calculate elder ratios
cbg_age_msa['Elder_Absolute'] = cbg_age_msa.apply(lambda x : x['70 To 74 Years']+x['75 To 79 Years']+x['80 To 84 Years']+x['85 Years And Over'],axis=1)
cbg_age_msa['Elder_Ratio'] = cbg_age_msa['Elder_Absolute'] / cbg_age_msa['Sum']

# Obtain cbg sizes (populations)
cbg_sizes = cbg_age_msa['Sum'].values
cbg_sizes = np.array(cbg_sizes,dtype='int32')
print('Total population: ',np.sum(cbg_sizes))

# Load other Safegraph demographic data
# cbg_c24.csv: Occupation
filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_c24.csv")
cbg_occupation = pd.read_csv(filepath)
# Extract pois corresponding to the metro area, by merging dataframes
cbg_occupation_msa = pd.merge(cbg_ids_msa, cbg_occupation, on='census_block_group', how='left')
del cbg_occupation

columns_of_essential_workers = list(constants.ew_rate_dict.keys())
for column in columns_of_essential_workers:
    cbg_occupation_msa[column] = cbg_occupation_msa[column].apply(lambda x : x*constants.ew_rate_dict[column])
cbg_occupation_msa['Essential_Worker_Absolute'] = cbg_occupation_msa.apply(lambda x : x[columns_of_essential_workers].sum(), axis=1)
cbg_occupation_msa['Sum'] = cbg_age_msa['Sum']
cbg_occupation_msa['Essential_Worker_Ratio'] = cbg_occupation_msa['Essential_Worker_Absolute'] / cbg_occupation_msa['Sum']
columns_of_interest = ['census_block_group','Sum','Essential_Worker_Absolute','Essential_Worker_Ratio']
cbg_occupation_msa = cbg_occupation_msa[columns_of_interest].copy()

# Load ACS 5-year (2013-2017) Data: Mean Household Income
filepath = os.path.join(root,"ACS_5years_Income_Filtered_Summary.csv")
cbg_income = pd.read_csv(filepath)
# Drop duplicate column 'Unnamed:0'
cbg_income.drop(['Unnamed: 0'],axis=1, inplace=True)
# Extract pois corresponding to the metro area, by merging dataframes
cbg_income_msa = pd.merge(cbg_ids_msa, cbg_income, on='census_block_group', how='left')
del cbg_income
# Add information of cbg populations, from cbg_age_msa(cbg_b01.csv)
cbg_income_msa['Sum'] = cbg_age_msa['Sum'].copy()
# Rename
cbg_income_msa.rename(columns = {'total_households':'Total_Households',
                                 'mean_household_income':'Mean_Household_Income'},inplace=True)

# Deal with NaN values
cbg_age_msa.fillna(0,inplace=True)
cbg_income_msa.fillna(0,inplace=True)
cbg_occupation_msa.fillna(0,inplace=True)

###############################################################################
# Grouping: 按NUM_GROUPS分位数，将全体CBG分为NUM_GROUPS个组，将分割点存储在separators中

separators = functions.get_separators(cbg_age_msa, NUM_GROUPS, 'Elder_Ratio','Sum', normalized=True)
cbg_age_msa['Age_Quantile'] =  cbg_age_msa['Elder_Ratio'].apply(lambda x : functions.assign_group(x, separators))

separators = functions.get_separators(cbg_occupation_msa, NUM_GROUPS, 'Essential_Worker_Ratio','Sum', normalized=True)
cbg_occupation_msa['Essential_Worker_Quantile'] =  cbg_occupation_msa['Essential_Worker_Ratio'].apply(lambda x : functions.assign_group(x, separators))

separators = functions.get_separators(cbg_income_msa, NUM_GROUPS, 'Mean_Household_Income','Sum', normalized=False)
cbg_income_msa['Mean_Household_Income_Quantile'] =  cbg_income_msa['Mean_Household_Income'].apply(lambda x : functions.assign_group(x, separators))

###############################################################################

subroot = 'vaccination_results_adaptive_%sd_%s_0.01' %(VACCINATION_TIME_STR,VACCINATION_RATIO) 
subroot_reverse = 'vaccination_results_adaptive_reverse_%sd_%s_0.01' %(VACCINATION_TIME_STR,VACCINATION_RATIO)    
if(consider_accessibility=='False'):
    if(consider_hesitancy=='True'):
        notation_string = 'acceptance_%s_'%ACCEPTANCE_SCENARIO
    else:
        notation_string = ''
else:
    if(consider_hesitancy=='True'):
        notation_string = 'access_acceptance_%s_'%ACCEPTANCE_SCENARIO
    else:
        notation_string = 'access_'

for policy in policy_list:
    policy = policy.lower()
    if(policy=='no_vaccination'):
        history_D2_no_vaccination = np.fromfile(os.path.join(root,MSA_NAME,'vaccination_results_adaptive_31d_0.1_0.01','20210206_history_D2_no_vaccination_adaptive_0.1_0.01_30seeds_%s'%(MSA_NAME)))
        #history_D2_no_vaccination = np.reshape(history_D2_no_vaccination,(63,NUM_SEEDS,M))
    #elif(policy=='baseline'):
    #    history_D2_baseline = np.fromfile(os.path.join(root,MSA_NAME,subroot,
    #                                                   'test_history_D2_baseline_adaptive_%sd_%s_0.01_30seeds_%s%s'%(VACCINATION_TIME_STR,VACCINATION_RATIO,notation_string,MSA_NAME)))
    #    history_D2_baseline = np.reshape(history_D2_baseline,(63,NUM_SEEDS,M))
    elif(policy =='comprehensive_ablation'):
        # 通配匹配，绕过weight string
        filepath_list = glob.glob(os.path.join(root,MSA_NAME,subroot, 'test_history_D2_adaptive_hybrid_ablation_%sd_%s_%s*_30seeds_%s%s' %(VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,notation_string,MSA_NAME)))
        if(len(filepath_list)>1):
            print('When loading %s, multiple files are detected. Please check.' % policy)
            pdb.set_trace()
        ablation_path = filepath_list[0]
        history_D2_comprehensive_ablation = np.fromfile(ablation_path)
        prefix ='test_history_D2_adaptive_hybrid_ablation_%sd_%s_%s_'%(VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL)
        suffix = '_30seeds'
        pattern = re.compile(r'%s(.+?)%s'%(prefix,suffix))
        weight_string = re.findall(pattern, ablation_path)[0]
        print('weight_string: ',weight_string)
    elif(policy =='comprehensive'):
        # 通配匹配，绕过weight string
        #filepath_list = glob.glob(os.path.join(root,MSA_NAME,subroot, 'test_history_D2_adaptive_hybrid_%sd_%s_%s*_30seeds_%s%s'%(VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,notation_string,MSA_NAME)))
        filepath_list = glob.glob(os.path.join(root,MSA_NAME,subroot, 'test_history_D2_adaptive_hybrid_%sd_%s_%s_%s*_30seeds_%s%s'
                                                                   %(VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,weight_string,notation_string,MSA_NAME)))
        for filepath in filepath_list:
            if((filepath == ablation_path) or (weight_string not in filepath)):
                continue
            print(filepath)
            history_D2_comprehensive = np.fromfile(filepath)
    else:
        if('reverse' in policy): #['Age_Flood_Reverse','Income_Flood_Reverse','JUE_EW_Flood_Reverse']
            exec('history_D2_%s_reverse = np.fromfile(os.path.join(root,MSA_NAME,subroot_reverse, \'test_history_D2_%s_adaptive_reverse_%sd_VACCINATION_RATIO_0.01_30seeds_%s%s\'))' 
                % (policy,policy,VACCINATION_TIME_STR,notation_string,MSA_NAME))    
        else:  #['Age_Flood','Income_Flood','JUE_EW_Flood']
            exec('history_D2_%s = np.fromfile(os.path.join(root,MSA_NAME,subroot, \'test_history_D2_%s_adaptive_%sd_%s_0.01_30seeds_%s%s\'))' 
                 % (policy,policy,VACCINATION_TIME_STR,VACCINATION_RATIO,notation_string,MSA_NAME))
    exec('history_D2_%s = np.reshape(history_D2_%s,(63,NUM_SEEDS,M))' %(policy,policy))    
    
#print('history_D2_no_vaccination.shape:', history_D2_no_vaccination.shape) # (63, 30, 2943)

###############################################################################
# Add simulation results to grouping tables

#for policy in ['No_Vaccination', 'Baseline']:
for policy in policy_list:
    exec("history_D2_%s = np.array(history_D2_%s)" % (policy.lower(),policy.lower()))
    exec("avg_history_D2_%s = np.mean(history_D2_%s,axis=1)" % (policy.lower(),policy.lower()))
    exec("avg_final_deaths_%s = avg_history_D2_%s[-1,:]" % (policy.lower(),policy.lower()))
    
    exec("cbg_age_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
    exec("cbg_income_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
    exec("cbg_occupation_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))

'''
for policy in demo_policy_list:
    # Prioritize the most disadvantaged
    exec('history_D2_%s = np.array(history_D2_%s)'%(policy.lower(),policy.lower()))
    exec("avg_history_D2_%s = np.mean(history_D2_%s,axis=1)" % (policy.lower(),policy.lower()))
    exec("avg_final_deaths_%s = avg_history_D2_%s[-1,:]" % (policy.lower(),policy.lower()))
    exec("cbg_age_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
    exec("cbg_income_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
    exec("cbg_occupation_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
    # Prioritize the least disadvantaged
    exec('history_D2_%s_reverse = np.array(history_D2_%s_reverse)'%(policy.lower(),policy.lower()))
    exec("avg_history_D2_%s_reverse = np.mean(history_D2_%s_reverse,axis=1)" % (policy.lower(),policy.lower()))
    exec("avg_final_deaths_%s_reverse = avg_history_D2_%s_reverse[-1,:]" % (policy.lower(),policy.lower()))
    exec("cbg_age_msa['Final_Deaths_%s_Reverse'] = avg_final_deaths_%s_reverse" % (policy,policy.lower()))
    exec("cbg_income_msa['Final_Deaths_%s_Reverse'] = avg_final_deaths_%s_reverse" % (policy,policy.lower()))
    exec("cbg_occupation_msa['Final_Deaths_%s_Reverse'] = avg_final_deaths_%s_reverse" % (policy,policy.lower()))
'''

###############################################################################
# Check whether there is NaN in cbg_tables

if((cbg_age_msa.isnull().any().any()) or (cbg_income_msa.isnull().any().any()) or (cbg_occupation_msa.isnull().any().any())):
    print('There are nan values in cbg_tables. Please check.')
    pdb.set_trace()

###############################################################################
# Gini table: efficiency & equity change under different policies

demo_feat_list = ['Age', 'Mean_Household_Income', 'Essential_Worker']
#print('Demographic feature list: ', demo_feat_list)

#save_path = os.path.join(root, 'adaptive_results_hesitancy_by_income_%s_0.1_0.01_%s_%s_acceptance_%s_rel2%s.csv'%(VACCINATION_TIME_STR,NUM_GROUPS, MSA_NAME,ACCEPTANCE_SCENARIO,REL_TO))
save_path = os.path.join(root, 'test_gini_table_%s_%s_%s_%srel2%s.csv'%(VACCINATION_TIME_STR,VACCINATION_RATIO, MSA_NAME,notation_string,REL_TO))
print('save_path: ',save_path)
#gini_df = make_gini_table(policy_list=all_policy_list, parallel=False, demo_feat_list=demo_feat_list, num_groups=NUM_GROUPS, save_result=True, save_path=save_path)
gini_df = make_gini_table(policy_list=policy_list, parallel=False, demo_feat_list=demo_feat_list, num_groups=NUM_GROUPS, save_result=True, save_path=save_path)
print(gini_df)
