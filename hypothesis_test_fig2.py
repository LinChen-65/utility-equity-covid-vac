# python hypothesis_test_fig2.py MSA_NAME 
# python hypothesis_test_fig2.py Atlanta 

#import setproctitle
#setproctitle.setproctitle("covid-19-vac@chenlin")

import sys
import pdb

import os
import datetime
import pandas as pd
import numpy as np

import constants
import helper
import functions

import gini
import time

from scipy import stats

###############################################################################
# Constants

root = '/data/chenlin/COVID-19/Data'

timestring = '20210206'
MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
NUM_DAYS = 63

NUM_GROUPS = 5
print('NUM_GROUPS:',NUM_GROUPS)

NUM_SEEDS = 30
print('NUM_SEEDS: ', NUM_SEEDS)
STARTING_SEED = range(NUM_SEEDS)

VACCINATION_RATIO = 0.1
RECHECK_INTERVAL = 0.01
NUM_GROUPS = 5

# Vaccination protection rate
PROTECTION_RATE = 1

# Policy execution ratio
EXECUTION_RATIO = 1

#############################################################################
# Main variable settings

MSA_NAME = sys.argv[1]; print('MSA_NAME: ',MSA_NAME)
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME] 

policy_list = ['Baseline', 'No_Vaccination','Age_Flood', 'Income_Flood','JUE_EW_Flood']
print('policy list:', policy_list)

#demo_feat_list = ['Age', 'Mean_Household_Income', 'Essential_Worker'] #, 'Hybrid'
demo_feat_list = ['Age', 'Income', 'Occupation'] 
print('Demographic feature list: ', demo_feat_list)

###############################################################################
# Functions
'''
def output_result(cbg_table, demo_feat, policy_list, num_groups, print_result=True,draw_result=True, rel_to=REL_TO):
    #demo_feat: demographic feature of interest: 'Age'
    #num_groups: number of quantiles that all CBGs are divided into

    print('Observation dimension: ', demo_feat)

    results = {}
    for policy in policy_list:
        exec("final_deaths_rate_%s_total = cbg_table['Final_Deaths_%s'].sum()/cbg_table['Sum'].sum()" % (policy.lower(),policy))
        cbg_table['Final_Deaths_' + policy] = eval('avg_final_deaths_' + policy.lower())
        exec("%s = np.zeros(num_groups)" % ('final_deaths_rate_'+ policy.lower()))
        deaths_total_abs = eval('final_deaths_rate_%s_total'%(policy.lower()))
        
        if(demo_feat!='Hybrid'):
            for i in range(num_groups):
                eval('final_deaths_rate_'+ policy.lower())[i] = cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Final_Deaths_' + policy].sum()
                eval('final_deaths_rate_'+ policy.lower())[i] /= cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Sum'].sum()
            
            deaths_gini_abs = gini.gini(eval('final_deaths_rate_'+ policy.lower()))
        
        if(rel_to=='No_Vaccination'):
            # rel is compared to No_Vaccination
            if(policy=='No_Vaccination'):
                deaths_total_no_vaccination = deaths_total_abs;print('deaths_total_no_vaccination:',deaths_total_no_vaccination)
                deaths_gini_no_vaccination = deaths_gini_abs
                    
                deaths_total_rel = 0
                deaths_gini_rel = 0
                                             
                results[policy] = {'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.6f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.6f'% deaths_gini_rel}   
            else:
                deaths_total_rel = (eval('final_deaths_rate_%s_total'%(policy.lower())) - deaths_total_no_vaccination) / deaths_total_no_vaccination
                deaths_gini_rel = (gini.gini(eval('final_deaths_rate_'+ policy.lower())) - deaths_gini_no_vaccination) / deaths_gini_no_vaccination
                results[policy] = {'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.6f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.6f'% deaths_gini_rel}    
        
        elif(rel_to=='Baseline'):
            # rel is compared to Baseline
            if(policy=='Baseline'):
                deaths_total_baseline = deaths_total_abs;print('deaths_total_baseline:',deaths_total_baseline)
                deaths_gini_baseline = deaths_gini_abs
                    
                deaths_total_rel = 0
                deaths_gini_rel = 0
                                             
                results[policy] = {'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.6f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.6f'% deaths_gini_rel}   
            else:
                deaths_total_rel = (eval('final_deaths_rate_%s_total'%(policy.lower())) - deaths_total_baseline) / deaths_total_baseline
                deaths_gini_rel = (gini.gini(eval('final_deaths_rate_'+ policy.lower())) - deaths_gini_baseline) / deaths_gini_baseline
                results[policy] = {'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.6f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.6f'% deaths_gini_rel}                        
        
        if(print_result==True):
            print('Policy: ', policy)
            print('Deaths, Gini Index: ',gini.gini(eval('final_deaths_rate_'+ policy.lower())))
            
            if(policy=='Baseline'):
                deaths_total_baseline = eval('final_deaths_rate_%s_total'%(policy.lower()))
                deaths_gini_baseline = gini.gini(eval('final_deaths_rate_'+ policy.lower()))
                
            if(policy!='Baseline' and policy!='No_Vaccination'):
                print('Compared to baseline:')
                print('Deaths total: ', (eval('final_deaths_rate_%s_total'%(policy.lower())) - deaths_total_baseline) / deaths_total_baseline)
                print('Deaths gini: ', (gini.gini(eval('final_deaths_rate_'+ policy.lower())) - deaths_gini_baseline) / deaths_gini_baseline)

    return results
'''

###############################################################################
# Load Data

# Load ACS Data for MSA-county matching
acs_data = pd.read_csv(os.path.join(root,'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
good_list = list(msa_data['FIPS Code'].values)
print('CBG included: ', good_list)
del acs_data

# Load CBG ids for the MSA
cbg_ids_msa = pd.read_csv(os.path.join(root,MSA_NAME,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
M = len(cbg_ids_msa)

# Mapping from cbg_ids to columns in hourly visiting matrices
cbgs_to_idxs = dict(zip(cbg_ids_msa['census_block_group'].values, range(M)))
x = {}
for i in cbgs_to_idxs:
    x[str(i)] = cbgs_to_idxs[i]
#print('Number of CBGs in this metro area:', M)

# Load SafeGraph data to obtain CBG sizes (i.e., populations)
filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_b01.csv")
cbg_agesex = pd.read_csv(filepath)
# Extract CBGs belonging to the MSA - https://covid-mobility.stanford.edu//datasets/
cbg_age_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
del cbg_agesex
# Add up males and females of the same age, according to the detailed age list (DETAILED_AGE_LIST)
# which is defined in constants.py
for i in range(3,25+1): # 'B01001e3'~'B01001e25'
    male_column = 'B01001e'+str(i)
    female_column = 'B01001e'+str(i+24)
    cbg_age_msa[constants.DETAILED_AGE_LIST[i-3]] = cbg_age_msa.apply(lambda x : x[male_column]+x[female_column],axis=1)
# Rename
cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
# Extract columns of interest
columns_of_interest = ['census_block_group','Sum'] + constants.DETAILED_AGE_LIST
cbg_age_msa = cbg_age_msa[columns_of_interest].copy()
# Deal with NaN values
cbg_age_msa.fillna(0,inplace=True)
# Deal with CBGs with 0 populations
cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)

# Obtain cbg sizes (populations)
cbg_sizes = cbg_age_msa['Sum'].values
cbg_sizes = np.array(cbg_sizes,dtype='int32')
print('Total population: ',np.sum(cbg_sizes))

# Select counties belonging to the MSA
y = []
for i in x:
    if((len(i)==12) & (int(i[0:5])in good_list)):
        y.append(x[i])
    if((len(i)==11) & (int(i[0:4])in good_list)):
        y.append(x[i])
        
idxs_msa_all = list(x.values())
idxs_msa_nyt = y
print('Number of CBGs in this metro area:', len(idxs_msa_all))
print('Number of CBGs in to compare with NYT data:', len(idxs_msa_nyt))


# Calculate elder ratios
cbg_age_msa['Elder_Absolute'] = cbg_age_msa.apply(lambda x : x['70 To 74 Years']+x['75 To 79 Years']+x['80 To 84 Years']+x['85 Years And Over'],axis=1)
cbg_age_msa['Elder_Ratio'] = cbg_age_msa['Elder_Absolute'] / cbg_age_msa['Sum']

# cbg_c24.csv: Occupation
filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_c24.csv")
cbg_occupation = pd.read_csv(filepath)
# Extract pois corresponding to the metro area, by merging dataframes
cbg_occupation_msa = pd.merge(cbg_ids_msa, cbg_occupation, on='census_block_group', how='left')
del cbg_occupation

ew_rate_dict = {
    'C24030e4' : 1,
    'C24030e31': 1,
    'C24030e5': 1,  
    'C24030e32': 1,
    'C24030e12': 1,
    'C24030e39': 1,
    'C24030e6': 1,
    'C24030e33': 1,
    'C24030e7': 1,
    'C24030e34': 1,
    'C24030e8': 0.842,
    'C24030e35': 0.842,
    'C24030e9': 0.444,
    'C24030e36': 0.444,
    'C24030e11': 0.821,
    'C24030e38': 0.821,
    'C24030e13': 0.545,
    'C24030e40': 0.545,
    'C24030e15': 1,
    'C24030e42': 1,
    'C24030e16': 0.5,
    'C24030e43': 0.5,
    'C24030e18': 0.778,
    'C24030e45': 0.778,
    'C24030e19': 1,
    'C24030e46': 1,
    'C24030e20': 0.636,
    'C24030e47': 0.636,
    'C24030e22': 0,
    'C24030e49': 0,
    'C24030e23': 1,
    'C24030e50': 1,
    'C24030e25': 0,
    'C24030e52': 0,
    'C24030e26': 0.667,
    'C24030e53': 0.667,
    'C24030e27': 0.643,
    'C24030e54': 0.643
}

columns_of_essential_workers = list(ew_rate_dict.keys())
for column in columns_of_essential_workers:
    cbg_occupation_msa[column] = cbg_occupation_msa[column].apply(lambda x : x*ew_rate_dict[column])
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
# Extract pois corresponding to the metro area (Philadelphia), by merging dataframes
cbg_income_msa = pd.merge(cbg_ids_msa, cbg_income, on='census_block_group', how='left')
del cbg_income
# Add information of cbg populations, from cbg_age_Phi(cbg_b01.csv)
cbg_income_msa['Sum'] = cbg_age_msa['Sum'].copy()
# Rename
cbg_income_msa.rename(columns = {'total_households':'Total_Households',
                                 'mean_household_income':'Mean_Household_Income'},inplace=True)

# Deal with NaN values
cbg_age_msa.fillna(0,inplace=True)
cbg_income_msa.fillna(0,inplace=True)
cbg_occupation_msa.fillna(0,inplace=True)

###############################################################################
# Grouping
# 按NUM_GROUPS分位数，将全体CBG分为NUM_GROUPS个组，将分割点存储在separators中

separators = functions.get_separators(cbg_age_msa, NUM_GROUPS, 'Elder_Ratio','Sum', normalized=True)
cbg_age_msa['Age_Quantile'] =  cbg_age_msa['Elder_Ratio'].apply(lambda x : functions.assign_group(x, separators))

separators = functions.get_separators(cbg_occupation_msa, NUM_GROUPS, 'Essential_Worker_Ratio','Sum', normalized=True)
#cbg_occupation_msa['Essential_Worker_Quantile'] =  cbg_occupation_msa['Essential_Worker_Ratio'].apply(lambda x : functions.assign_group(x, separators))
cbg_occupation_msa['Occupation_Quantile'] =  cbg_occupation_msa['Essential_Worker_Ratio'].apply(lambda x : functions.assign_group(x, separators))

separators = functions.get_separators(cbg_income_msa, NUM_GROUPS, 'Mean_Household_Income','Sum', normalized=False)
#cbg_income_msa['Mean_Household_Income_Quantile'] =  cbg_income_msa['Mean_Household_Income'].apply(lambda x : functions.assign_group(x, separators))
cbg_income_msa['Income_Quantile'] =  cbg_income_msa['Mean_Household_Income'].apply(lambda x : functions.assign_group(x, separators))

###############################################################################
# Load vaccination results

for policy in policy_list:
    policy_lower = policy.lower()
    exec('gini_dict_%s = dict()' % policy_lower)

    exec('history_D2_%s = np.fromfile(os.path.join(root,MSA_NAME,\'vaccination_results_adaptive_%s_%s\',\'%s_history_D2_%s_adaptive_%s_%s_%sseeds_%s\'))' 
          %(policy_lower,VACCINATION_RATIO,RECHECK_INTERVAL,timestring,policy_lower,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS,MSA_NAME))
    #print('history_D2_no_vaccination.shape:', history_D2_no_vaccination.shape) # (63, 30, 2943)
        
    exec('history_D2_%s = np.reshape(history_D2_%s,(63,NUM_SEEDS,M))'%(policy_lower,policy_lower))
    exec('deaths_cbg_seed_%s = np.squeeze(np.array(history_D2_%s)[-1,:,:])' % (policy_lower,policy_lower)) # (30, 2943)

    # Compute gini index under certain demo_feat
    for demo_feat in demo_feat_list:
        demo_feat_lower = demo_feat.lower()
        exec('%s_gini_%s = np.zeros(NUM_SEEDS)' % (demo_feat,policy_lower))
        for seed_idx in range(NUM_SEEDS):
            exec('deaths_cbg = deaths_cbg_seed_%s[seed_idx]' % policy_lower)
            
            # Add results to cbg tables
            exec('cbg_%s_msa[\'Final_Deaths_%s\'] = deaths_cbg' % (demo_feat_lower,policy))
            '''
            exec('cbg_age_msa[\'Final_Deaths_%s\'] = deaths_cbg' % (policy_original))
            exec('cbg_income_msa[\'Final_Deaths_%s\'] = deaths_cbg' % (policy_original))
            exec('cbg_occupation_msa[\'Final_Deaths_%s\'] = deaths_cbg' % (policy_original))
            '''
            
            group_death_rate = np.zeros(NUM_GROUPS)
            for group_idx in range(NUM_GROUPS):
                exec('group_death_rate[group_idx] = cbg_%s_msa[cbg_%s_msa[\'%s_Quantile\']==group_idx][\'Final_Deaths_%s\'].sum()' 
                    % (demo_feat_lower,demo_feat_lower,demo_feat,policy))
                exec('group_death_rate[group_idx] /= cbg_%s_msa[cbg_%s_msa[\'%s_Quantile\']==group_idx][\'Sum\'].sum()'
                    % (demo_feat_lower,demo_feat_lower,demo_feat))
                #group_death_rate[group_idx] = cbg_age_msa[cbg_age_msa['Age_Quantile']==group_idx]['Final_Deaths_Most_Vulner'].sum()
                #group_death_rate[group_idx] /= cbg_age_msa[cbg_age_msa['Age_Quantile']==group_idx]['Sum'].sum()
        
            gini_death_rate = gini.gini(group_death_rate)
            exec('%s_gini_%s[seed_idx] = gini_death_rate' % (demo_feat,policy_lower))
            
        exec('gini_dict_%s[\'%s\'] = %s_gini_%s' % (policy_lower,demo_feat,demo_feat,policy_lower))

# Paired t-test, two-sided
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html#scipy.stats.ttest_rel
print('\npaired t-test, two-sided:')
print('Age-Prioritized, Age Equity:', stats.ttest_rel(gini_dict_age_flood['Age'],gini_dict_baseline['Age'])) 
print('Age-Prioritized, Income Equity:', stats.ttest_rel(gini_dict_age_flood['Income'],gini_dict_baseline['Income'])) #,alternative='two-sided'
print('Age-Prioritized, Occupation Equity:', stats.ttest_rel(gini_dict_age_flood['Occupation'],gini_dict_baseline['Occupation']))

print('Income-Prioritized, Age Equity:', stats.ttest_rel(gini_dict_income_flood['Age'],gini_dict_baseline['Age']))
print('Income-Prioritized, Income Equity:', stats.ttest_rel(gini_dict_income_flood['Income'],gini_dict_baseline['Income'])) 
print('Income-Prioritized, Occupation Equity:', stats.ttest_rel(gini_dict_income_flood['Occupation'],gini_dict_baseline['Occupation']))

print('Occupation-Prioritized, Age Equity:', stats.ttest_rel(gini_dict_jue_ew_flood['Age'],gini_dict_baseline['Age']))
print('Occupation-Prioritized, Income Equity:', stats.ttest_rel(gini_dict_jue_ew_flood['Income'],gini_dict_baseline['Income']))
print('Occupation-Prioritized, Occupation Equity:', stats.ttest_rel(gini_dict_jue_ew_flood['Occupation'],gini_dict_baseline['Occupation'])) 


# Paired t-test, single-sided, alternative='greater'
print('\npaired t-test, single-sided, alternative=\'greater\':')
print('Age-Prioritized, Income Equity:', stats.ttest_rel(gini_dict_age_flood['Income'],gini_dict_baseline['Income'], alternative='greater')) 
print('Age-Prioritized, Occupation Equity:', stats.ttest_rel(gini_dict_age_flood['Occupation'],gini_dict_baseline['Occupation'], alternative='greater'))

print('Income-Prioritized, Age Equity:', stats.ttest_rel(gini_dict_income_flood['Age'],gini_dict_baseline['Age'], alternative='greater'))
print('Income-Prioritized, Occupation Equity:', stats.ttest_rel(gini_dict_income_flood['Occupation'],gini_dict_baseline['Occupation'], alternative='greater'))

print('Occupation-Prioritized, Age Equity:', stats.ttest_rel(gini_dict_jue_ew_flood['Age'],gini_dict_baseline['Age'], alternative='greater'))
print('Occupation-Prioritized, Income Equity:', stats.ttest_rel(gini_dict_jue_ew_flood['Income'],gini_dict_baseline['Income'], alternative='greater'))

# Paired t-test, single-sided, alternative='less'
print('\npaired t-test, single-sided, alternative=\'less\':')
print('Age-Prioritized, Age Equity:', stats.ttest_rel(gini_dict_age_flood['Age'],gini_dict_baseline['Age'], alternative='less')) 
print('Age-Prioritized, Income Equity:', stats.ttest_rel(gini_dict_age_flood['Income'],gini_dict_baseline['Income'], alternative='less')) 
print('Age-Prioritized, Occupation Equity:', stats.ttest_rel(gini_dict_age_flood['Occupation'],gini_dict_baseline['Occupation'], alternative='less'))

print('Income-Prioritized, Age Equity:', stats.ttest_rel(gini_dict_income_flood['Age'],gini_dict_baseline['Age'], alternative='less'))
print('Income-Prioritized, Income Equity:', stats.ttest_rel(gini_dict_income_flood['Income'],gini_dict_baseline['Income'], alternative='less')) 
print('Income-Prioritized, Occupation Equity:', stats.ttest_rel(gini_dict_income_flood['Occupation'],gini_dict_baseline['Occupation'], alternative='less'))

print('Occupation-Prioritized, Age Equity:', stats.ttest_rel(gini_dict_jue_ew_flood['Age'],gini_dict_baseline['Age'], alternative='less'))
print('Occupation-Prioritized, Income Equity:', stats.ttest_rel(gini_dict_jue_ew_flood['Income'],gini_dict_baseline['Income'], alternative='less'))
print('Occupation-Prioritized, Occupation Equity:', stats.ttest_rel(gini_dict_jue_ew_flood['Occupation'],gini_dict_baseline['Occupation'], alternative='less')) 

'''
print('Independent t-test:')
print('Age-Prioritized, Income Equity:', stats.ttest_ind(gini_dict_age_flood['Income'],gini_dict_baseline['Income'])) #,alternative='two-sided'
print('Age-Prioritized, Occupation Equity:', stats.ttest_ind(gini_dict_age_flood['Occupation'],gini_dict_baseline['Occupation']))

print('Income-Prioritized, Age Equity:', stats.ttest_ind(gini_dict_income_flood['Age'],gini_dict_baseline['Age']))
print('Income-Prioritized, Occupation Equity:', stats.ttest_ind(gini_dict_income_flood['Occupation'],gini_dict_baseline['Occupation']))

print('Occupation-Prioritized, Age Equity:', stats.ttest_ind(gini_dict_jue_ew_flood['Age'],gini_dict_baseline['Age']))
print('Occupation-Prioritized, Income Equity:', stats.ttest_ind(gini_dict_jue_ew_flood['Income'],gini_dict_baseline['Income']))
'''

# test
'''
print('\ntest:')
a = np.array([0,0,0,0,1,0,0,0])
b = np.array([1,1,1,1,0,1,1,1])
# Paired t-test, two-sided
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html#scipy.stats.ttest_rel
print('\npaired t-test, two-sided:')
print(stats.ttest_rel(a,b)) #,alternative='two-sided'

# Paired t-test, single-sided, alternative='greater'
print('\npaired t-test, single-sided, alternative=\'greater\':')
print(stats.ttest_rel(a,b, alternative='greater')) 

# Paired t-test, single-sided, alternative='less'
print('\npaired t-test, single-sided, alternative=\'less\':')
print(stats.ttest_rel(a,b, alternative='less')) 
'''