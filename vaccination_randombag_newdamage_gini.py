# python vaccination_randombag_newdamage_gini.py MSA_NAME RANDOM_SEED quick_test
# python vaccination_randombag_newdamage_gini.py Atlanta 66 False

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import sys

import os
import datetime
import pandas as pd
import numpy as np
import pickle

import constants
import helper
import functions
import disease_model #disease_model_only_modify_attack_rates

import time
import pdb

###############################################################################
# Constants

root = '/data/chenlin/COVID-19/Data'

MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
NUM_DAYS = 63
NUM_GROUPS = 5

# Vaccination ratio
VACCINATION_RATIO = 0.02
print('VACCINATION_RATIO: ', VACCINATION_RATIO)

# Vaccination protection rate
PROTECTION_RATE = 1
# Policy execution ratio
EXECUTION_RATIO = 1
# Vaccination time
VACCINATION_TIME = 0 #31
print('VACCINATION_TIME: ', VACCINATION_TIME)

NUM_GROUPS_FOR_GINI = 5
NUM_GROUPS_FOR_RANDOMBAG = 3
#NUM_DIM = 3 # Age, Income, EW
NUM_DIM = 5 # Age, Income, EW, Vulner, Damage

###############################################################################
# Main variable settings

MSA_NAME = sys.argv[1]; print('MSA_NAME: ',MSA_NAME) #MSA_NAME = 'SanFrancisco'
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME] #MSA_NAME_FULL = 'San_Francisco_Oakland_Hayward_CA'

# Random Seed:
#RANDOM_SEED = 66
#RANDOM_SEED = 42 
#RANDOM_SEED = 5
RANDOM_SEED = int(sys.argv[2])
print('RANDOM_SEED:',RANDOM_SEED)


# Quick Test: prototyping
quick_test = sys.argv[3]; print('Quick testing?', quick_test)
if(quick_test == 'True'):
    NUM_SEEDS = 2
else:
    NUM_SEEDS = 60 #30
print('NUM_SEEDS: ', NUM_SEEDS)
STARTING_SEED = range(NUM_SEEDS)


# 在每个非0的hybrid_group里采样num_groupwise个样本。
NUM_GROUPWISE = 5 
print('NUM_GROUPWISE: ',NUM_GROUPWISE)

# Store filename
filename = 'newdamage_group_randombag_vaccination_results_withgini_%s_%s_%s_%s_%sseeds.csv'%(VACCINATION_RATIO,MSA_NAME,RANDOM_SEED, NUM_GROUPWISE, NUM_SEEDS)
print('File name: ', filename)

policy_list=['No_Vaccination','Randombag']
demo_feat_list = ['Age', 'Mean_Household_Income', 'Essential_Worker']

REL_TO = 'No_Vaccination'

###############################################################################
# Functions

def run_simulation(starting_seed, num_seeds, vaccination_vector, protection_rate=1):
    m = disease_model_only_modify_attack_rates.Model(starting_seed=starting_seed,
                                                     num_seeds=num_seeds,
                                                     debug=False,clip_poisson_approximation=True,ipf_final_match='poi',ipf_num_iter=100)

    m.init_exogenous_variables(poi_areas=poi_areas,
                               poi_dwell_time_correction_factors=poi_dwell_time_correction_factors,
                               cbg_sizes=cbg_sizes,
                               poi_cbg_visits_list=poi_cbg_visits_list,
                               all_hours=all_hours,
                               p_sick_at_t0=constants.parameters_dict[MSA_NAME][0],
                               #vaccination_time=24*31, # when to apply vaccination (which hour)
                               vaccination_time=24*VACCINATION_TIME, # when to apply vaccination (which hour)
                               vaccination_vector = vaccination_vector,
                               protection_rate = protection_rate,
                               home_beta=constants.parameters_dict[MSA_NAME][1],
                               cbg_attack_rates_original = cbg_attack_rates_scaled,
                               cbg_death_rates_original = cbg_death_rates_scaled,
                               poi_psi=constants.parameters_dict[MSA_NAME][2],
                               just_compute_r0=False,
                               latency_period=96,  # 4 days
                               infectious_period=84,  # 3.5 days
                               confirmation_rate=.1,
                               confirmation_lag=168,  # 7 days
                               death_lag=432
                               )

    m.init_endogenous_variables()

    T1,L_1,I_1,R_1,C2,D2,total_affected, history_C2, history_D2, total_affected_each_cbg = m.simulate_disease_spread(no_print=True)    
    del T1
    del L_1
    del I_1
    del C2
    del D2
    #return total_affected, history_C2, history_D2, total_affected_each_cbg
    return history_C2, history_D2

# Analyze results and produce graphs: All policies
def output_result(cbg_table, demo_feat, policy_list, num_groups, rel_to, print_result=True):
    #print('Observation dimension: ', demo_feat)
    results = {}
    
    for policy in policy_list:
        exec("final_deaths_rate_%s_total = cbg_table['Final_Deaths_%s'].sum()/cbg_table['Sum'].sum()" % (policy.lower(),policy))
        cbg_table['Final_Deaths_' + policy] = eval('avg_final_deaths_' + policy.lower())
        exec("%s = np.zeros(num_groups)" % ('final_deaths_rate_'+ policy.lower()))
        deaths_total_abs = eval('final_deaths_rate_%s_total'%(policy.lower()))
        
        for i in range(num_groups):
            #eval('final_deaths_rate_'+ policy.lower())[i] = cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Final_Deaths_' + policy].sum()
            #eval('final_deaths_rate_'+ policy.lower())[i] /= cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Sum'].sum()
            eval('final_deaths_rate_'+ policy.lower())[i] = cbg_table[cbg_table[demo_feat + '_Quantile_FOR_GINI']==i]['Final_Deaths_' + policy].sum()
            eval('final_deaths_rate_'+ policy.lower())[i] /= cbg_table[cbg_table[demo_feat + '_Quantile_FOR_GINI']==i]['Sum'].sum()
        deaths_gini_abs = functions.gini(eval('final_deaths_rate_'+ policy.lower()))
       
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
                deaths_total_rel = (eval('final_deaths_rate_%s_total'%(policy.lower())) - deaths_total_no_vaccination) / deaths_total_no_vaccination
                deaths_gini_rel = (functions.gini(eval('final_deaths_rate_'+ policy.lower())) - deaths_gini_no_vaccination) / deaths_gini_no_vaccination
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
                deaths_total_rel = (eval('final_deaths_rate_%s_total'%(policy.lower())) - deaths_total_baseline) / deaths_total_baseline
                deaths_gini_rel = (functions.gini(eval('final_deaths_rate_'+ policy.lower())) - deaths_gini_baseline) / deaths_gini_baseline
                results[policy] = {
                                   'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.6f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.6f'% deaths_gini_rel}                        
                                    
        if(print_result==True):
            print('Policy: ', policy)
            print('Deaths, Gini Index: ',functions.gini(eval('final_deaths_rate_'+ policy.lower())))
            
            if(policy=='Baseline'):
                deaths_total_baseline = eval('final_deaths_rate_%s_total'%(policy.lower()))
                deaths_gini_baseline = functions.gini(eval('final_deaths_rate_'+ policy.lower()))
                
            if(policy!='Baseline' and policy!='No_Vaccination'):
                print('Compared to baseline:')
                print('Deaths total: ', (eval('final_deaths_rate_%s_total'%(policy.lower())) - deaths_total_baseline) / deaths_total_baseline)
                print('Deaths gini: ', (functions.gini(eval('final_deaths_rate_'+ policy.lower())) - deaths_gini_baseline) / deaths_gini_baseline)

    return results

def make_gini_table(policy_list, demo_feat_list, rel_to, num_groups, save_result=False, save_path=None):
    
    cbg_table_name_dict=dict()
    cbg_table_name_dict['Age'] = cbg_age_msa
    cbg_table_name_dict['Mean_Household_Income'] = cbg_income_msa
    cbg_table_name_dict['Essential_Worker'] = cbg_occupation_msa
    
    print('Policy list: ', policy_list)
    print('Demographic feature list: ', demo_feat_list)

    gini_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('All','deaths_total_abs'),('All','deaths_total_rel')]))
    gini_df['Policy'] = policy_list
        
    for demo_feat in demo_feat_list:
        results = output_result(cbg_table_name_dict[demo_feat], 
                                demo_feat, policy_list, num_groups=NUM_GROUPS,
                                print_result=False, rel_to=rel_to)
       
        for i in range(len(policy_list)):
            policy = policy_list[i]
            gini_df.loc[i,('All','deaths_total_abs')] = results[policy]['deaths_total_abs']
            gini_df.loc[i,('All','deaths_total_rel')] = results[policy]['deaths_total_rel'] #if abs(float(results[policy]['deaths_total_rel']))>=0.01 else 0
            gini_df.loc[i,(demo_feat,'deaths_gini_abs')] = results[policy]['deaths_gini_abs']
            gini_df.loc[i,(demo_feat,'deaths_gini_rel')] = results[policy]['deaths_gini_rel'] #if abs(float(results[policy]['deaths_gini_rel']))>=0.01 else 0

    gini_df.set_index(['Policy'],inplace=True)
    # Transpose
    gini_df_trans = pd.DataFrame(gini_df.values.T, index=gini_df.columns, columns=gini_df.index)#转置
    # Save .csv
    if(save_result==True):
        gini_df_trans.to_csv(save_path)
        
    return gini_df_trans
    
###############################################################################
# Load Data

# Load POI-CBG visiting matrices
f = open(os.path.join(root, MSA_NAME, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()

# Load precomputed parameters to adjust(clip) POI dwell times
d = pd.read_csv(os.path.join(root,MSA_NAME, 'parameters_%s.csv' % MSA_NAME)) 

# No clipping
new_d = d

all_hours = helper.list_hours_in_range(MIN_DATETIME, MAX_DATETIME)
poi_areas = new_d['feet'].values#面积
poi_dwell_times = new_d['median'].values#平均逗留时间
poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
del new_d
del d

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
print('Number of CBGs in this metro area:', M)

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
cbg_age_msa['Elder_Absolute'] = cbg_age_msa.apply(lambda x : x['70 To 74 Years']+x['75 To 79 Years']+x['80 To 84 Years']+x['85 Years And Over'],axis=1)
cbg_age_msa['Elder_Ratio'] = cbg_age_msa['Elder_Absolute'] / cbg_age_msa['Sum']

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

nyt_included = np.zeros(len(idxs_msa_all))
for i in range(len(nyt_included)):
    if(i in idxs_msa_nyt):
        nyt_included[i] = 1
cbg_age_msa['NYT_Included'] = nyt_included.copy()

# Load other demographic data
filepath = os.path.join(root,"ACS_5years_Income_Filtered_Summary.csv")
cbg_income = pd.read_csv(filepath)
# Drop duplicate column 'Unnamed:0'
cbg_income.drop(['Unnamed: 0'],axis=1, inplace=True)
# Income Data Resource 1: ACS 5-year (2013-2017) Data
# Extract pois corresponding to the metro area (Philadelphia), by merging dataframes
cbg_income_msa = pd.merge(cbg_ids_msa, cbg_income, on='census_block_group', how='left')
del cbg_income
# Deal with NaN values
cbg_income_msa.fillna(0,inplace=True)
# Add information of cbg populations, from cbg_age_Phi(cbg_b01.csv)
cbg_income_msa['Sum'] = cbg_age_msa['Sum'].copy()
# Rename
cbg_income_msa.rename(columns = {'total_household_income':'Total_Household_Income', 
                                 'total_households':'Total_Households',
                                 'mean_household_income':'Mean_Household_Income',
                                 'median_household_income':'Median_Household_Income'},inplace=True)

                                 
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
# Deal with NaN values
cbg_occupation_msa.fillna(0,inplace=True)

###############################################################################
# Grouping: 按NUM_GROUPS_FOR_GINI分位数，将全体CBG分为NUM_GROUPS_FOR_GINI个组，将分割点存储在separators中

separators = functions.get_separators(cbg_age_msa, NUM_GROUPS_FOR_GINI, 'Elder_Ratio','Sum', normalized=True)
cbg_age_msa['Age_Quantile_FOR_GINI'] =  cbg_age_msa['Elder_Ratio'].apply(lambda x : functions.assign_group(x, separators))

separators = functions.get_separators(cbg_occupation_msa, NUM_GROUPS_FOR_GINI, 'Essential_Worker_Ratio','Sum', normalized=True)
cbg_occupation_msa['Essential_Worker_Quantile_FOR_GINI'] =  cbg_occupation_msa['Essential_Worker_Ratio'].apply(lambda x : functions.assign_group(x, separators))

separators = functions.get_separators(cbg_income_msa, NUM_GROUPS_FOR_GINI, 'Mean_Household_Income','Sum', normalized=False)
cbg_income_msa['Mean_Household_Income_Quantile_FOR_GINI'] =  cbg_income_msa['Mean_Household_Income'].apply(lambda x : functions.assign_group(x, separators))


##############################################################################
# Load and scale age-aware CBG-specific attack/death rates (original)

cbg_death_rates_original = np.loadtxt(os.path.join(root, MSA_NAME, 'cbg_death_rates_original_'+MSA_NAME))
cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)
print('Age-aware CBG-specific death rates loaded. Attack rates are irrelevant to age.')

# The scaling factors are set according to a grid search
# Fix attack_scale
attack_scale = 1
cbg_attack_rates_scaled = cbg_attack_rates_original * attack_scale
cbg_death_rates_scaled = cbg_death_rates_original * constants.death_scale_dict[MSA_NAME]
print('Age-aware CBG-specific death rates scaled.')

###############################################################################
# Compute vulnerability and damage

# Retrieve cbg_avg_infect_same, cbg_avg_infect_diff
# As planned, they have been computed in 'tradeoff_md_mv_theory.py'.
# Use them to get data['Vulnerability'] and data['Damage']
if(os.path.exists(os.path.join(root, '3cbg_avg_infect_same_%s.npy'%MSA_NAME))):
    print('cbg_avg_infect_same, cbg_avg_infect_diff: Load existing file.')
    cbg_avg_infect_same = np.load(os.path.join(root, '3cbg_avg_infect_same_%s.npy'%MSA_NAME))
    cbg_avg_infect_diff = np.load(os.path.join(root, '3cbg_avg_infect_diff_%s.npy'%MSA_NAME))
else:
    print('cbg_avg_infect_same, cbg_avg_infect_diff: File not found. Please check.')
    pdb.set_trace()
    
SEIR_at_30d = np.load(os.path.join(root, 'SEIR_at_30d.npy'),allow_pickle=True).item()
S_ratio = SEIR_at_30d[MSA_NAME]['S'] / (cbg_sizes.sum())
I_ratio = SEIR_at_30d[MSA_NAME]['I'] / (cbg_sizes.sum())
print('S_ratio:',S_ratio,'I_ratio:',I_ratio)

# Compute the average death rate for the whole MSA: perform another weighted average over all CBGs
avg_death_rates_scaled = np.matmul(cbg_sizes.T, cbg_death_rates_scaled) / np.sum(cbg_sizes)
print('avg_death_rates_scaled.shape:',avg_death_rates_scaled.shape) # shape: (), because it is a scalar

# Normalize by cbg population
cbg_avg_infect_same_norm = cbg_avg_infect_same / cbg_sizes
cbg_avg_infect_diff_norm = cbg_avg_infect_diff / cbg_sizes
cbg_avg_infect_all_norm = cbg_avg_infect_same_norm + cbg_avg_infect_diff_norm
# alpha_bar
avg_death_rates_scaled = np.matmul(cbg_sizes.T, cbg_death_rates_scaled) / np.sum(cbg_sizes)

# New new method # 20210619
cbg_vulnerability = cbg_avg_infect_all_norm * cbg_death_rates_scaled 
cbg_secondary_damage = cbg_avg_infect_all_norm * (cbg_avg_infect_all_norm*(S_ratio/I_ratio)) * avg_death_rates_scaled
cbg_damage = cbg_vulnerability + cbg_secondary_damage

cbg_age_msa['Vulnerability'] = cbg_vulnerability.copy()
cbg_age_msa['Damage'] = cbg_damage.copy()

###############################################################################
# Load non-vaccination results for comparison

history_D2_no_vaccination = np.fromfile(os.path.join(root,MSA_NAME,'vaccination_results_adaptive_0.1_0.01','20210206_history_D2_no_vaccination_adaptive_0.1_0.01_30seeds_%s'%(MSA_NAME)))
# Average across random seeds
history_D2_no_vaccination = np.reshape(history_D2_no_vaccination,(63,30,M))
avg_final_deaths_no_vaccination = np.mean(history_D2_no_vaccination,axis=1)[-1,:]
#avg_final_deaths_no_vaccination = np.mean(np.array(history_D2_no_vaccination),axis=1)[-1,:]
cbg_age_msa['Final_Deaths_No_Vaccination'] = avg_final_deaths_no_vaccination
cbg_income_msa['Final_Deaths_No_Vaccination'] = avg_final_deaths_no_vaccination
cbg_occupation_msa['Final_Deaths_No_Vaccination'] = avg_final_deaths_no_vaccination
final_deaths_no_vaccination = np.sum(avg_final_deaths_no_vaccination)

###############################################################################
# Collect data together

data = pd.DataFrame()

data['Sum'] = cbg_age_msa['Sum'].copy()
data['Elder_Ratio'] = cbg_age_msa['Elder_Ratio'].copy()
data['Mean_Household_Income'] = cbg_income_msa['Mean_Household_Income'].copy()
data['Essential_Worker_Ratio'] = cbg_occupation_msa['Essential_Worker_Ratio'].copy()
data['Vulnerability'] = cbg_age_msa['Vulnerability'].copy()
data['Damage'] = cbg_age_msa['Damage'].copy()

###############################################################################
# Construct randombags in a semi-random way (分组——采样)
# (1) Grouping
# (2) 检查每组人数，若小于target_num，则与邻组合并。
# (3) 在每个非0的hybrid_group里采样num_groupwise个样本。

# Grouping: 按NUM_GROUPS_FOR_RANDOMBAG分位数，将全体CBG分为NUM_GROUPS_FOR_RANDOMBAG个组，将分割点存储在separators中
separators = functions.get_separators(data, NUM_GROUPS_FOR_RANDOMBAG, 'Elder_Ratio','Sum', normalized=True)
data['Age_Quantile_FOR_RANDOMBAG'] =  data['Elder_Ratio'].apply(lambda x : functions.assign_group(x, separators))

separators = functions.get_separators(data, NUM_GROUPS_FOR_RANDOMBAG, 'Mean_Household_Income','Sum', normalized=False)
data['Income_Quantile_FOR_RANDOMBAG'] =  data['Mean_Household_Income'].apply(lambda x : functions.assign_group(x, separators))

separators = functions.get_separators(data, NUM_GROUPS_FOR_RANDOMBAG, 'Essential_Worker_Ratio','Sum', normalized=True)
data['EW_Quantile_FOR_RANDOMBAG'] =  data['Essential_Worker_Ratio'].apply(lambda x : functions.assign_group(x, separators))

separators = functions.get_separators(data, NUM_GROUPS_FOR_RANDOMBAG, 'Vulnerability','Sum', normalized=False)
data['Vulner_Quantile_FOR_RANDOMBAG'] =  data['Vulnerability'].apply(lambda x : functions.assign_group(x, separators))

separators = functions.get_separators(data, NUM_GROUPS_FOR_RANDOMBAG, 'Damage','Sum', normalized=False)
data['Damage_Quantile_FOR_RANDOMBAG'] =  data['Damage'].apply(lambda x : functions.assign_group(x, separators))


# Hybrid Grouping
def assign_hybrid_group(data):
    #return (data['Age_Quantile']*9 + data['Income_Quantile']*3 + data['EW_Quantile'])
    return (data['Age_Quantile_FOR_RANDOMBAG']*81 + data['Income_Quantile_FOR_RANDOMBAG']*27 + data['EW_Quantile_FOR_RANDOMBAG']*9 + data['Vulner_Quantile_FOR_RANDOMBAG']*3 + data['Damage_Quantile_FOR_RANDOMBAG'])
    
data['Hybrid_Group'] = data.apply(lambda x : assign_hybrid_group(x), axis=1)


# 分层——采样：
# 首先检查每组人数，若小于target_num，则与邻组合并。#(若是第一组，则与后一组合并，否则与前一组合并。)
#(若是最后一组，则与前一组合并，否则与后一组合并。)
target_pop = data['Sum'].sum() * VACCINATION_RATIO 
target_cbg_num = 5 # at least contains 5 CBGs
count = 0
max_group_idx = int(NUM_GROUPS_FOR_RANDOMBAG*NUM_GROUPS_FOR_RANDOMBAG*NUM_GROUPS_FOR_RANDOMBAG*NUM_GROUPS_FOR_RANDOMBAG*NUM_GROUPS_FOR_RANDOMBAG)
for i in range(max_group_idx):
    print(len(data[data['Hybrid_Group']==i]))
    if(len(data[data['Hybrid_Group']==i])>0):
        count += 1
    if((data[data['Hybrid_Group']==i]['Sum'].sum()<target_pop) or (len(data[data['Hybrid_Group']==i])<target_cbg_num)):
        if(i==max_group_idx-1): 
            #data[data['Hybrid_Group']==i]['Hybrid_Group'] = 1
            data['Hybrid_Group'] = data['Hybrid_Group'].apply(lambda x : max_group_idx-2 if x==i else x)
        else:
            #data[data['Hybrid_Group']==i]['Hybrid_Group'] = i-1
            data['Hybrid_Group'] = data['Hybrid_Group'].apply(lambda x : i+1 if x==i else x)
print('Num of groups: ', count)

# Recheck after merging
print('Recheck:')
count = 0
for i in range(max_group_idx):
    print(len(data[data['Hybrid_Group']==i]))
    if(len(data[data['Hybrid_Group']==i])>0):
        count += 1
print('Num of groups: ', count)

        
# 在每个非0的hybrid_group里采样num_groupwise个样本。
print('NUM_GROUPWISE: ',NUM_GROUPWISE)
np.random.seed(RANDOM_SEED)

result_df = pd.DataFrame(columns=['Vaccinated_Idxs',
                                  'Fatality_Rate_Abs','Fatality_Rate_Rel',
                                  'Age_Gini_Abs','Age_Gini_Rel',
                                  'Income_Gini_Abs','Income_Gini_Rel',
                                  'Occupation_Gini_Abs','Occupation_Gini_Rel'])
start = time.time()

for group_idx in range(max_group_idx):
    if(len(data[data['Hybrid_Group']==group_idx])==0): # 跳过0组
        continue
    
    for i in range(NUM_GROUPWISE):
        print('group_idx: ', group_idx, ', sample_idx: ', i)
        start1 = time.time()
        # Construct the vaccination vector
        random_permutation = np.arange(len(data))
        np.random.shuffle(random_permutation)
        not_eligible_idx = len(data)+1
        data['Random_Permutation'] = random_permutation
        data['Random_Permutation'] = data.apply(lambda x : not_eligible_idx if (x['Hybrid_Group']!=group_idx) else x['Random_Permutation'], axis=1)
        print('Check num of eligible CBGs:', len(data)-len(data[data['Random_Permutation']==not_eligible_idx]))
        
        vaccination_vector_randombag = functions.vaccine_distribution_flood(cbg_table=data, 
                                                                            vaccination_ratio=VACCINATION_RATIO, 
                                                                            demo_feat='Random_Permutation', 
                                                                            ascending=True,
                                                                            execution_ratio=1
                                                                            )
         # Retrieve vaccinated CBGs
        data['Vaccination_Vector'] = vaccination_vector_randombag
        data['Vaccinated'] = data['Vaccination_Vector'].apply(lambda x : 1 if x!=0 else 0)
        vaccinated_idxs = data[data['Vaccinated'] != 0].index.tolist()  
        print('Num of vaccinated CBGs: ', len(vaccinated_idxs))
        
        # Run simulations
        _, history_D2_randombag = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                 vaccination_vector=vaccination_vector_randombag,
                                                 protection_rate = PROTECTION_RATE)
                                                
        # Average history records across random seeds
        deaths_cbg_randombag, _ = functions.average_across_random_seeds_only_death(history_D2_randombag, 
                                                                                 M, idxs_msa_nyt, 
                                                                                 print_results=False,draw_results=False)
        # Analyze deaths in each demographic group
        avg_final_deaths_randombag = deaths_cbg_randombag[-1,:]
        # Add simulation results to cbg table
        cbg_age_msa['Final_Deaths_Randombag'] = avg_final_deaths_randombag
        cbg_income_msa['Final_Deaths_Randombag'] = avg_final_deaths_randombag
        cbg_occupation_msa['Final_Deaths_Randombag'] = avg_final_deaths_randombag
        
        gini_df = make_gini_table(policy_list=policy_list, demo_feat_list=demo_feat_list, rel_to=REL_TO, num_groups=NUM_GROUPS_FOR_GINI, save_result=False)
        print(gini_df)

        # Store in df
        result_df = result_df.append({'Vaccinated_Idxs':vaccinated_idxs,
                                      'Fatality_Rate_Abs':gini_df.iloc[0][1],
                                      'Fatality_Rate_Rel':gini_df.iloc[1][1],
                                      'Age_Gini_Abs':gini_df.iloc[2][1],
                                      'Age_Gini_Rel':gini_df.iloc[3][1],
                                      'Income_Gini_Abs':gini_df.iloc[4][1],
                                      'Income_Gini_Rel':gini_df.iloc[5][1],
                                      'Occupation_Gini_Abs':gini_df.iloc[6][1],
                                      'Occupation_Gini_Rel':gini_df.iloc[7][1]}, ignore_index=True)
        print(result_df)   
        print('Time for this sample: ', time.time()-start1);start1=time.time()
        
        result_df.to_csv(os.path.join(root,filename))

end = time.time()
print('Time: ',(end-start))

print('File name: ', filename)

