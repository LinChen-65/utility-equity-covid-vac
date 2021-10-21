# python test_vaccination_adaptive_hybrid_autosearch_conform.py MSA_NAME VACCINATION_TIME VACCINATION_RATIO RECHECK_INTERVAL consider_hesitancy ACCEPTANCE_SCENARIO w1 w2 w3 w4 w5 quick_test 
# python test_vaccination_adaptive_hybrid_autosearch_conform.py Atlanta 15 0.1 0.01 True cf18 1 1 1 1 1 False

from genericpath import exists
import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import sys
import socket
import os
import datetime
import pandas as pd
import numpy as np
import pickle
import time
import pdb

from skcriteria import Data, MIN
from skcriteria.madm import closeness

import constants
import functions
#import disease_model_only_modify_attack_rates
#import disease_model_diff_acceptance
import disease_model_test

###############################################################################
# Constants

# root
hostname = socket.gethostname()
print('hostname: ', hostname)
if(hostname=='fib-dl3'):
    root = '/data/chenlin/COVID-19/Data' #dl3
elif(hostname=='rl4'):
    root = '/home/chenlin/COVID-19/Data' #rl4

timestring='20210206'
MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
NUM_DAYS = 63
NUM_GROUPS = 5

# Vaccination protection rate
PROTECTION_RATE = 1
# Policy execution ratio
EXECUTION_RATIO = 1

###############################################################################
# Main variable settings

MSA_NAME = sys.argv[1]; #MSA_NAME = 'SanFrancisco'
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME] #MSA_NAME_FULL = 'San_Francisco_Oakland_Hayward_CA'
print('MSA_NAME: ',MSA_NAME)

# Policies to compare
policy_to_compare = ['No_Vaccination','Baseline','Age_Flood', 'Income_Flood','JUE_EW_Flood']
policy_to_compare_rel_to_no_vaccination = ['No_Vaccination','Baseline','Age_Flood', 'Income_Flood','JUE_EW_Flood']
policy_to_compare_rel_to_baseline = ['Baseline','No_Vaccination','Age_Flood', 'Income_Flood','JUE_EW_Flood']

# Vaccination time
VACCINATION_TIME = sys.argv[2];print('VACCINATION_TIME:',VACCINATION_TIME)
VACCINATION_TIME_STR = VACCINATION_TIME
VACCINATION_TIME = float(VACCINATION_TIME)
print(VACCINATION_TIME_STR,'\n',VACCINATION_TIME)

policy_savename = 'adaptive_%sd_hybrid'%VACCINATION_TIME_STR
print('policy_savename:',policy_savename)

# Vaccination_Ratio
VACCINATION_RATIO = sys.argv[3]; print('VACCINATION_RATIO:',VACCINATION_RATIO)
VACCINATION_RATIO = float(VACCINATION_RATIO)

# Recheck interval: After distributing some portion of vaccines, recheck the most vulnerable demographic group
RECHECK_INTERVAL = sys.argv[4]; print('RECHECK_INTERVAL:',RECHECK_INTERVAL)
RECHECK_INTERVAL = float(RECHECK_INTERVAL)

# Consider hesitancy or not
consider_hesitancy = sys.argv[5]
print('Consider hesitancy? ', consider_hesitancy)
if(consider_hesitancy not in ['True','False']): 
    print('Invalid value for consider_hesitancy. Please check.')
    pdb.set_trace()

# Acceptance scenario, if considering hesitancy
# if consider_hesitancy=='False', this field does not affect anything
ACCEPTANCE_SCENARIO = sys.argv[6]
print('Vaccine acceptance scenario: ', ACCEPTANCE_SCENARIO)

w1 = float(sys.argv[7])
w2 = float(sys.argv[8])
w3 = float(sys.argv[9])
w4 = float(sys.argv[10])
w5 = float(sys.argv[11])
weights = [w1,w2,w3,w4,w5]
print('Weights:', weights)

# Quick Test: prototyping
quick_test = sys.argv[12]; print('Quick testing?', quick_test)
if(quick_test == 'True'):
    NUM_SEEDS = 2
    NUM_SEEDS_CHECKING = 2
else:
    NUM_SEEDS = 30
    NUM_SEEDS_CHECKING = 30
print('NUM_SEEDS: ', NUM_SEEDS)
print('NUM_SEEDS_CHECKING: ', NUM_SEEDS_CHECKING)
STARTING_SEED = range(NUM_SEEDS)
STARTING_SEED_CHECKING = range(NUM_SEEDS_CHECKING)

distribution_time = VACCINATION_RATIO / RECHECK_INTERVAL # 分几次把疫苗分配完

# Compare all policies with no_vaccination scenario
#REL_TO = 'No_Vaccination'

###############################################################################
# Functions

def run_simulation(starting_seed, num_seeds, vaccination_vector, vaccine_acceptance,protection_rate=1):
    #m = disease_model_only_modify_attack_rates.Model(starting_seed=starting_seed,
    #m = disease_model_diff_acceptance.Model(starting_seed=starting_seed, #20211007
    m = disease_model_test.Model(starting_seed=starting_seed, #20211013
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
                               vaccine_acceptance = vaccine_acceptance,#20211007
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
    return history_C2, history_D2


# Analyze results and produce graphs
def output_result(cbg_table, demo_feat, policy_list, num_groups, rel_to, print_result=True,draw_result=True):
    #print('Observation dimension: ', demo_feat)
    results = {}
    
    for policy in policy_list:
        exec("final_deaths_rate_%s_total = cbg_table['Final_Deaths_%s'].sum()/cbg_table['Sum'].sum()" % (policy.lower(),policy))
        cbg_table['Final_Deaths_' + policy] = eval('avg_final_deaths_' + policy.lower())
        exec("%s = np.zeros(num_groups)" % ('final_deaths_rate_'+ policy.lower()))
        deaths_total_abs = eval('final_deaths_rate_%s_total'%(policy.lower()))
        
        for i in range(num_groups):
            eval('final_deaths_rate_'+ policy.lower())[i] = cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Final_Deaths_' + policy].sum()
            eval('final_deaths_rate_'+ policy.lower())[i] /= cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Sum'].sum()
        deaths_gini_abs = functions.gini(eval('final_deaths_rate_'+ policy.lower()))
        
        if(rel_to=='No_Vaccination'):
            # rel is compared to No_Vaccination
            if(policy=='No_Vaccination'):
                deaths_total_no_vaccination = deaths_total_abs
                deaths_gini_no_vaccination = deaths_gini_abs
                deaths_total_rel = 0; deaths_gini_rel = 0
                results[policy] = {
                                   'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.4f'% deaths_gini_abs, #'%.4f'
                                   'deaths_gini_rel':'%.4f'% deaths_gini_rel}  #'%.4f' 
            else:
                deaths_total_rel = (eval('final_deaths_rate_%s_total'%(policy.lower())) - deaths_total_no_vaccination) / deaths_total_no_vaccination
                deaths_gini_rel = (functions.gini(eval('final_deaths_rate_'+ policy.lower())) - deaths_gini_no_vaccination) / deaths_gini_no_vaccination
                results[policy] = {
                                   'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.4f'% deaths_gini_abs, #'%.4f'
                                   'deaths_gini_rel':'%.4f'% deaths_gini_rel} #'%.4f'   
        
        elif(rel_to=='Baseline'):
            # rel is compared to Baseline
            if(policy=='Baseline'):
                deaths_total_baseline = deaths_total_abs
                deaths_gini_baseline = deaths_gini_abs
                deaths_total_rel = 0
                deaths_gini_rel = 0    
                results[policy] = {
                                   'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.4f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.4f'% deaths_gini_rel}   
            else:
                deaths_total_rel = (eval('final_deaths_rate_%s_total'%(policy.lower())) - deaths_total_baseline) / deaths_total_baseline
                deaths_gini_rel = (functions.gini(eval('final_deaths_rate_'+ policy.lower())) - deaths_gini_baseline) / deaths_gini_baseline
                results[policy] = {                               
                                   'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.4f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.4f'% deaths_gini_rel}                        
        
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


def make_gini_table(policy_list, demo_feat_list, num_groups, rel_to, save_path, save_result=False):
    cbg_table_name_dict=dict()
    cbg_table_name_dict['Age'] = cbg_age_msa
    cbg_table_name_dict['Mean_Household_Income'] = cbg_income_msa
    cbg_table_name_dict['Essential_Worker'] = cbg_occupation_msa
    cbg_table_name_dict['Hybrid'] = cbg_age_msa # randomly choose one. it doesn't matter.

    gini_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('All','deaths_total_abs'),('All','deaths_total_rel')]))
    gini_df['Policy'] = policy_list

    for demo_feat in demo_feat_list:
        results = output_result(cbg_table_name_dict[demo_feat], 
                                demo_feat, policy_list, num_groups=NUM_GROUPS,
                                rel_to=rel_to, print_result=False, draw_result=False)
       
        for i in range(len(policy_list)):
            policy = policy_list[i]
            gini_df.loc[i,('All','deaths_total_abs')] = results[policy]['deaths_total_abs']
            gini_df.loc[i,('All','deaths_total_rel')] = results[policy]['deaths_total_rel'] if abs(float(results[policy]['deaths_total_rel']))>=0.01 else 0
            gini_df.loc[i,(demo_feat,'deaths_gini_abs')] = results[policy]['deaths_gini_abs']
            gini_df.loc[i,(demo_feat,'deaths_gini_rel')] = results[policy]['deaths_gini_rel'] if abs(float(results[policy]['deaths_gini_rel']))>=0.01 else 0

    gini_df.set_index(['Policy'],inplace=True)
    # Transpose
    gini_df_trans = pd.DataFrame(gini_df.values.T, index=gini_df.columns, columns=gini_df.index)#转置
    # Save .csv
    if(save_result==True):
        gini_df_trans.to_csv(save_path)
        
    return gini_df_trans
        

def get_overall_performance(data_column): #20211020
    return -(float(data_column.iloc[1])+float(data_column.iloc[3])+float(data_column.iloc[5])+float(data_column.iloc[7]))

'''
def compare_results(policy_1,policy_2): #20211020
    print('Comparing %s to %s: ' % (policy_1, policy_2))
    print('Death rate: ', eval('%s_death_rate'%policy_1), eval('%s_death_rate'%policy_2), 'Good enough?', (eval('%s_death_rate'%policy_1)<=eval('%s_death_rate'%policy_2)))
    print('Age gini: ', eval('%s_age_gini'%policy_1), eval('%s_age_gini'%policy_2), 'Good enough?', (eval('%s_age_gini'%policy_1)<=eval('%s_age_gini'%policy_2)))
    print('Death rate: ', eval('%s_death_rate'%policy_1), eval('%s_death_rate'%policy_2), 'Good enough?', (eval('%s_death_rate'%policy_1)<=eval('%s_death_rate'%policy_2)))
    print('Death rate: ', eval('%s_death_rate'%policy_1), eval('%s_death_rate'%policy_2), 'Good enough?', (eval('%s_death_rate'%policy_1)<=eval('%s_death_rate'%policy_2)))
    print('Death rate: ', eval('%s_death_rate'%policy_1), eval('%s_death_rate'%policy_2), 'Good enough?', (eval('%s_death_rate'%policy_1)<=eval('%s_death_rate'%policy_2)))
    print('Death rate: ', eval('%s_death_rate'%policy_1), eval('%s_death_rate'%policy_2), 'Good enough?', (eval('%s_death_rate'%policy_1)<=eval('%s_death_rate'%policy_2)))
   
    print('Age gini: ', hybrid_age_gini, best_hybrid_age_gini, 'Good enough?', better_age_gini)
    print('Income gini: ', hybrid_income_gini, best_hybrid_income_gini, 'Good enough?', better_income_gini)
    print('Occupation gini: ', hybrid_occupation_gini, best_hybrid_occupation_gini, 'Good enough?', better_occupation_gini)
    print('Overall performance: ', hybrid_overall_performance, best_hybrid_overall_performance,'Good enough?', better_overall_performance)
'''
###############################################################################
# Load Demographic-Related Data

start = time.time()

# Load POI-CBG visiting matrices
f = open(os.path.join(root, MSA_NAME, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()
# Load precomputed parameters to adjust(clip) POI dwell times
d = pd.read_csv(os.path.join(root,MSA_NAME, 'parameters_%s.csv' % MSA_NAME)) 
# No clipping
new_d = d
all_hours = functions.list_hours_in_range(MIN_DATETIME, MAX_DATETIME)
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
print('Counties included: ', good_list)
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
#print('Number of CBGs in this metro area:', len(idxs_msa_all))
#print('Number of CBGs in to compare with NYT data:', len(idxs_msa_nyt))

# Load other Safegraph demographic data, and perform grouping
#if('Age_Flood' in policy_to_compare):
if(True):
    # Calculate elder ratios
    cbg_age_msa['Elder_Absolute'] = cbg_age_msa.apply(lambda x : x['70 To 74 Years']+x['75 To 79 Years']+x['80 To 84 Years']+x['85 Years And Over'],axis=1)
    cbg_age_msa['Elder_Ratio'] = cbg_age_msa['Elder_Absolute'] / cbg_age_msa['Sum']
    # Grouping
    separators = functions.get_separators(cbg_age_msa, NUM_GROUPS, 'Elder_Ratio','Sum', normalized=True)
    cbg_age_msa['Age_Quantile'] =  cbg_age_msa['Elder_Ratio'].apply(lambda x : functions.assign_group(x, separators))
    
#if('EW_Flood' in policy_to_compare):
if(True):
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
    # Deal with NaN values
    cbg_occupation_msa.fillna(0,inplace=True)
    # Grouping
    separators = functions.get_separators(cbg_occupation_msa, NUM_GROUPS, 'Essential_Worker_Ratio','Sum', normalized=True)
    cbg_occupation_msa['Essential_Worker_Quantile'] =  cbg_occupation_msa['Essential_Worker_Ratio'].apply(lambda x : functions.assign_group(x, separators))


#if('Income_Flood' in policy_to_compare):
if(True):
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
    cbg_income_msa.fillna(0,inplace=True)
    # Grouping
    separators = functions.get_separators(cbg_income_msa, NUM_GROUPS, 'Mean_Household_Income','Sum', normalized=False)
    cbg_income_msa['Mean_Household_Income_Quantile'] =  cbg_income_msa['Mean_Household_Income'].apply(lambda x : functions.assign_group(x, separators))

    if(consider_hesitancy=='True'):
        # Vaccine hesitancy by income #20211007
        if(ACCEPTANCE_SCENARIO in ['real','cf1','cf2','cf3','cf4','cf5','cf6','cf7','cf8']):
            cbg_income_msa['Vaccine_Acceptance'] = cbg_income_msa['Mean_Household_Income'].apply(lambda x:functions.assign_acceptance_absolute(x,ACCEPTANCE_SCENARIO))
        elif(ACCEPTANCE_SCENARIO in ['cf9','cf10','cf11','cf12','cf13','cf14','cf15','cf16','cf17','cf18']):
            cbg_income_msa['Vaccine_Acceptance'] = cbg_income_msa['Mean_Household_Income_Quantile'].apply(lambda x:functions.assign_acceptance_quantile(x,ACCEPTANCE_SCENARIO))
        # Retrieve vaccine acceptance as ndarray
        vaccine_acceptance = np.array(cbg_income_msa['Vaccine_Acceptance'].copy())
    elif(consider_hesitancy=='False'):
        vaccine_acceptance = np.ones(len(cbg_sizes)) # fully accepted scenario


###############################################################################
# Load results of other policies for comparison

RECHECK_INTERVAL_OTHERS = 0.01
subroot = 'vaccination_results_adaptive_%sd_%s_%s' % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL_OTHERS)
print('subroot: ', subroot)

if(consider_hesitancy=='True'):
    notation_string = 'acceptance_%s_'%ACCEPTANCE_SCENARIO
else:
    notation_string = ''   
print('notation_string: ', notation_string)    

for policy in policy_to_compare:
    policy = policy.lower()

    if(policy=='no_vaccination'):
        history_D2_no_vaccination = np.fromfile(os.path.join(root,MSA_NAME,
                                                            'vaccination_results_adaptive_%sd_0.1_0.01'%VACCINATION_TIME_STR,
                                                            '20210206_history_D2_no_vaccination_adaptive_0.1_0.01_30seeds_%s'%MSA_NAME))
        history_D2_no_vaccination = np.reshape(history_D2_no_vaccination,(63,NUM_SEEDS,M))
    else:
        if(consider_hesitancy=='False'):
            exec('history_D2_%s = np.fromfile(os.path.join(root,MSA_NAME,subroot,\'test_history_D2_%s_adaptive_%sd_%s_%s_30seeds_%s\'))' 
                %(policy,policy,VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL_OTHERS,MSA_NAME))
            if(len(eval('history_D2_%s'%policy))==(2*63*NUM_SEEDS*M)):
                print('Get rid of history_C2.')
                exec('history_D2_%s = np.array(np.reshape(history_D2_%s,(2,63,NUM_SEEDS,M)))[1,:,:,:].squeeze()' % (policy,policy))
                # Save back results
                exec('np.array(history_D2_%s).tofile(os.path.join(root,MSA_NAME,subroot,\'test_history_D2_%s_adaptive_%sd_%s_%s_30seeds_%s\'))' 
                %(policy,policy,VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL_OTHERS,MSA_NAME))
            elif(len(eval('history_D2_%s'%policy))==(63*NUM_SEEDS*M)):
                #print('No need to get rid of history_C2.')
                exec('history_D2_%s = np.reshape(history_D2_%s,(63,NUM_SEEDS,M))'%(policy,policy))
            else:
                print('Something wrong with the file for policy: %s. Please check.' %(policy))
        elif(consider_hesitancy=='True'):
            exec('history_D2_%s = np.fromfile(os.path.join(root,MSA_NAME,subroot,\'test_history_D2_%s_adaptive_%sd_%s_%s_30seeds_acceptance_%s_%s\'))' 
                %(policy,policy,VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL_OTHERS,ACCEPTANCE_SCENARIO,MSA_NAME))
            exec('history_D2_%s = np.reshape(history_D2_%s,(63,NUM_SEEDS,M))'%(policy,policy))


# Add simulation results to grouping tables
for policy in policy_to_compare:
    exec("history_D2_%s = np.array(history_D2_%s)" % (policy.lower(),policy.lower()))
    # Average across random seeds
    exec("avg_history_D2_%s = np.mean(history_D2_%s,axis=1)" % (policy.lower(),policy.lower()))
    exec("avg_final_deaths_%s = avg_history_D2_%s[-1,:]" % (policy.lower(),policy.lower()))
    
    exec("cbg_age_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
    exec("cbg_income_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
    exec("cbg_occupation_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))

# Check whether there is NaN in cbg_tables
#print('Any NaN in cbg_age_msa?', cbg_age_msa.isnull().any().any())
#print('Any NaN in cbg_income_msa?', cbg_income_msa.isnull().any().any())
#print('Any NaN in cbg_occupation_msa?', cbg_occupation_msa.isnull().any().any())
if((cbg_age_msa.isnull().any().any()) or (cbg_income_msa.isnull().any().any()) or (cbg_occupation_msa.isnull().any().any())):
    print('There are nan values in cbg_tables. Please check.')
    pdb.set_trace()

# Obtain efficiency and equity of policies
demo_feat_list = ['Age', 'Mean_Household_Income', 'Essential_Worker']
#print('Demographic feature list: ', demo_feat_list)
gini_table_no_vaccination = make_gini_table(policy_list=policy_to_compare_rel_to_no_vaccination, demo_feat_list=demo_feat_list, num_groups=NUM_GROUPS, 
                                            rel_to='No_Vaccination', save_path=None, save_result=False)
print('Gini table of all the other policies: \n', gini_table_no_vaccination)
gini_table_baseline = make_gini_table(policy_list=policy_to_compare_rel_to_baseline, demo_feat_list=demo_feat_list, num_groups=NUM_GROUPS, 
                                      rel_to='Baseline', save_path=None, save_result=False)
print('Gini table of all the other policies: \n', gini_table_baseline)


# Best results from other polices  
lowest_death_rate = float(gini_table_no_vaccination.iloc[0].min())
lowest_age_gini = float(gini_table_no_vaccination.iloc[2].min())
lowest_income_gini = float(gini_table_no_vaccination.iloc[4].min())
lowest_occupation_gini = float(gini_table_no_vaccination.iloc[6].min())

# No_Vaccination results
data_column = gini_table_no_vaccination['No_Vaccination']
no_vaccination_death_rate = float(data_column.iloc[0])
no_vaccination_age_gini = float(data_column.iloc[2])
no_vaccination_income_gini = float(data_column.iloc[4])
no_vaccination_occupation_gini = float(data_column.iloc[6])
#print(no_vaccination_death_rate,no_vaccination_age_gini,no_vaccination_income_gini,no_vaccination_occupation_gini)

# Baseline results
data_column = gini_table_no_vaccination['Baseline']
baseline_death_rate = float(data_column.iloc[0])
baseline_age_gini = float(data_column.iloc[2])
baseline_income_gini = float(data_column.iloc[4])
baseline_occupation_gini = float(data_column.iloc[6])
#print(baseline_death_rate,baseline_age_gini,baseline_income_gini,baseline_occupation_gini)

# target: better of No_Vaccination and Baseline
target_death_rate = min(baseline_death_rate, no_vaccination_death_rate)
target_age_gini = min(baseline_age_gini, no_vaccination_age_gini)
target_income_gini = min(baseline_income_gini, no_vaccination_income_gini)
target_occupation_gini = min(baseline_occupation_gini, no_vaccination_occupation_gini)


# Overall performance, relative to Baseline # 20211020
baseline_overall_performance = 0
no_vaccination_overall_performance = get_overall_performance(gini_table_baseline['No_Vaccination'])
age_flood_overall_performance = get_overall_performance(gini_table_baseline['Age_Flood'])
income_flood_overall_performance = get_overall_performance(gini_table_baseline['Income_Flood'])
jue_ew_flood_overall_performance = get_overall_performance(gini_table_baseline['JUE_EW_Flood'])
# Target overall performance: best of all the others
target_overall_performance = max(baseline_overall_performance,no_vaccination_overall_performance,
                               age_flood_overall_performance, income_flood_overall_performance, jue_ew_flood_overall_performance)

print('baseline_overall_performance: ', baseline_overall_performance)
print('no_vaccination_overall_performance: ', no_vaccination_overall_performance)
print('age_flood_overall_performance: ', age_flood_overall_performance)
print('income_flood_overall_performance: ', income_flood_overall_performance)
print('jue_ew_flood_overall_performance: ', jue_ew_flood_overall_performance)
print('target_overall_performance: ', target_overall_performance)


###############################################################################
# Load and scale age-aware CBG-specific attack/death rates (original)

cbg_death_rates_original = np.loadtxt(os.path.join(root, MSA_NAME, 'cbg_death_rates_original_'+MSA_NAME))
cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)
#print('Age-aware CBG-specific death rates loaded. Attack rates are irrelevant to age.')

# Fix attack_scale
attack_scale = 1
cbg_attack_rates_scaled = cbg_attack_rates_original * attack_scale
# Scale death rates
cbg_death_rates_scaled = cbg_death_rates_original * constants.death_scale_dict[MSA_NAME]
#print('Age-aware CBG-specific death rates scaled.')

cbg_age_msa['Death_Rate'] =  cbg_death_rates_scaled

###############################################################################
# Obtain vulnerability and damage, according to theoretical analysis

nyt_included = np.zeros(len(idxs_msa_all))
for i in range(len(nyt_included)):
    if(i in idxs_msa_nyt):
        nyt_included[i] = 1
cbg_age_msa['NYT_Included'] = nyt_included.copy()

# Retrieve the attack rate for the whole MSA (home_beta, fitted for each MSA)
home_beta = constants.parameters_dict[MSA_NAME][1]
#print('MSA home_beta retrieved.')

# Get cbg_avg_infect_same, cbg_avg_infect_diff
if(os.path.exists(os.path.join(root, '3cbg_avg_infect_same_%s.npy'%MSA_NAME))):
    #print('cbg_avg_infect_same, cbg_avg_infect_diff: Load existing file.')
    cbg_avg_infect_same = np.load(os.path.join(root, '3cbg_avg_infect_same_%s.npy'%MSA_NAME))
    cbg_avg_infect_diff = np.load(os.path.join(root, '3cbg_avg_infect_diff_%s.npy'%MSA_NAME))
else:
    print('cbg_avg_infect_same, cbg_avg_infect_diff: File not found. Please check.')
    pdb.set_trace()
#print('cbg_avg_infect_same.shape:',cbg_avg_infect_same.shape)

SEIR_at_30d = np.load(os.path.join(root, 'SEIR_at_30d.npy'),allow_pickle=True).item()
S_ratio = SEIR_at_30d[MSA_NAME]['S'] / (cbg_sizes.sum())
I_ratio = SEIR_at_30d[MSA_NAME]['I'] / (cbg_sizes.sum())
#print('S_ratio:',S_ratio,'I_ratio:',I_ratio)

# Deal with nan and inf (https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
cbg_avg_infect_same = np.nan_to_num(cbg_avg_infect_same,nan=0,posinf=0,neginf=0)
cbg_avg_infect_diff = np.nan_to_num(cbg_avg_infect_diff,nan=0,posinf=0,neginf=0)
cbg_age_msa['Infect'] = cbg_avg_infect_same + cbg_avg_infect_diff
# Check whether there is NaN in cbg_tables
#print('Any NaN in cbg_age_msa[\'Infect\']?', cbg_age_msa['Infect'].isnull().any().any())
if(cbg_age_msa['Infect'].isnull().any().any()):
    print('There are NaNs in cbg_age_msa[\'Infect\']. Please check.')
    pdb.set_trace()

# Normalize by cbg population
cbg_avg_infect_same_norm = cbg_avg_infect_same / cbg_sizes
cbg_avg_infect_diff_norm = cbg_avg_infect_diff / cbg_sizes
cbg_avg_infect_all_norm = cbg_avg_infect_same_norm + cbg_avg_infect_diff_norm

# Compute the average death rate for the whole MSA: perform another weighted average over all CBGs
avg_death_rates_scaled = np.matmul(cbg_sizes.T, cbg_death_rates_scaled) / np.sum(cbg_sizes)
#print('avg_death_rates_scaled.shape:',avg_death_rates_scaled.shape) # shape: (), because it is a scalar

# Compute vulnerability and damage for each cbg
# New new method # 20210619
cbg_vulnerability = cbg_avg_infect_all_norm * cbg_death_rates_scaled 
cbg_secondary_damage = cbg_avg_infect_all_norm * (cbg_avg_infect_all_norm*(S_ratio/I_ratio)) * avg_death_rates_scaled
cbg_damage = cbg_vulnerability + cbg_secondary_damage

cbg_age_msa['Vulnerability'] = cbg_vulnerability.copy()
cbg_age_msa['Damage'] = cbg_damage.copy()

cbg_age_msa['Vulner_Rank'] = cbg_age_msa['Vulnerability'].rank(ascending=False,method='first') 
cbg_age_msa['Damage_Rank'] = cbg_age_msa['Damage'].rank(ascending=False,method='first')

# Only those belonging to the MSA (according to nyt) is valid for vaccination.
# This is to prevent overlapping of CBGs across MSAs.
cbg_age_msa['Vulner_Rank'] = cbg_age_msa.apply(lambda x :  x['Vulner_Rank'] if x['NYT_Included']==1 else M+1, axis=1)
cbg_age_msa['Vulner_Rank_New'] = cbg_age_msa['Vulner_Rank'].rank(ascending=True,method='first')

cbg_age_msa['Damage_Rank'] = cbg_age_msa.apply(lambda x :  x['Damage_Rank'] if x['NYT_Included']==1 else M+1, axis=1)
cbg_age_msa['Damage_Rank_New'] = cbg_age_msa['Damage_Rank'].rank(ascending=True,method='first')

###############################################################################

columns_of_interest = ['census_block_group','Sum']
cbg_hybrid_msa = cbg_age_msa[columns_of_interest].copy()

cbg_hybrid_msa['Vulner_Rank'] = cbg_age_msa['Vulner_Rank_New'].copy()
cbg_hybrid_msa['Damage_Rank'] = cbg_age_msa['Damage_Rank_New'].copy()
#print('Any NaN in cbg_hybrid_msa?', cbg_hybrid_msa.isnull().any().any())
if(cbg_hybrid_msa.isnull().any().any()):
    print('There are NaNs in cbg_hybrid_msa. Please check.')
    pdb.set_trace()

# Annotate the most vulnerable group. 这是为了配合函数，懒得改函数
# Not grouping, so set all ['Most_Vulnerable']=1.
cbg_hybrid_msa['Most_Vulnerable'] = 1

###############################################################################

cnames=["Age_Flood", "Income_Flood", "EW_Flood", "Vulner", "Damage"]
criteria = [MIN, MIN, MIN, MIN, MIN]
# Initial weights are input by user. [1,1,1,1,1]

num_better_history = 0
refine_mode = False # First-round search
refine_threshold = 6 #0#6 
first_time = True
while(True):
    # if in refine_mode, how to adjust weights
    if(refine_mode==True):
        print('refine_count: ', refine_count)
        if(refine_count<int(0.5*refine_threshold)):
            w1 *= 1.1; w1=round(w1,1)
            w2 *= 1.1; w2=round(w2,1)
            w3 *= 1.1; w3=round(w3,1)
        else:
            if(refine_count==int(0.5*refine_threshold)):
                w1 = refine_w[0]
                w2 = refine_w[1]
                w3 = refine_w[2]
            w1 *= 0.9; w1=round(w1,1)
            w2 *= 0.9; w2=round(w2,1)
            w3 *= 0.9; w3=round(w3,1)
        weights = [w1,w2,w3,w4,w5]    
    print('\nWeights:',weights)

    # path to save comprehensive result
    file_savename = os.path.join(root,MSA_NAME,subroot,
                                     'test_history_D2_adaptive_hybrid_%sd_%s_%s_%s%s%s%s%s_%sseeds_%s%s'
                                     %(VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,w1,w2,w3,w4,w5,NUM_SEEDS,notation_string,MSA_NAME)
                                     )
    # if file for current weights exists, no need to simulate again                                 
    if(os.path.exists(file_savename)):
        print('Result already exists. No need to simulate. Directly load it. Weights: ', weights)  
        history_D2_hybrid_flood = np.fromfile(file_savename)
        history_D2_hybrid_flood = np.reshape(history_D2_hybrid_flood,(63,NUM_SEEDS,M))
    else: # File not exists, start to simulate
        current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
        cbg_hybrid_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
        leftover = 0
        for i in range(int(distribution_time)):
            if i==(int(distribution_time)-1): is_last = True
            else: is_last=False
            cbg_hybrid_msa['Vaccination_Vector'] = current_vector
            
            # Run a simulation to estimate death risks at the moment
            _, history_D2_current = run_simulation(starting_seed=STARTING_SEED_CHECKING, 
                                                num_seeds=NUM_SEEDS_CHECKING, 
                                                vaccination_vector=current_vector,
                                                vaccine_acceptance=vaccine_acceptance,
                                                protection_rate = PROTECTION_RATE)
                                                                    
            # Average history records across random seeds
            deaths_cbg_current, _ = functions.average_across_random_seeds_only_death(history_D2_current, 
                                                                                    M, idxs_msa_nyt, 
                                                                                    print_results=False, draw_results=False)
                                                                                                
            # Analyze deaths in each demographic group
            avg_final_deaths_current = deaths_cbg_current[-1,:]
            
            # Add simulation results to cbg table
            cbg_hybrid_msa['Final_Deaths_Current'] = avg_final_deaths_current
            cbg_age_msa['Final_Deaths_Current'] = avg_final_deaths_current
            cbg_income_msa['Final_Deaths_Current'] = avg_final_deaths_current
            cbg_occupation_msa['Final_Deaths_Current'] = avg_final_deaths_current
            
            # Estimate demographic disparities
            # Age
            demo_feat = 'Age'
            final_deaths_rate_current = np.zeros(NUM_GROUPS)
            for group_id in range(NUM_GROUPS):
                final_deaths_rate_current[group_id] = cbg_age_msa[cbg_age_msa[demo_feat + '_Quantile']==group_id]['Final_Deaths_Current'].sum()
                final_deaths_rate_current[group_id] /= cbg_age_msa[cbg_age_msa[demo_feat + '_Quantile']==group_id]['Sum'].sum()
            #print(demo_feat, ', final_deaths_rate_current: ', final_deaths_rate_current)
            # Sort groups according to vulnerability
            group_vulnerability = np.argsort(-final_deaths_rate_current) # 死亡率从大到小排序
            group_vulner_dict = dict()
            for i in range(NUM_GROUPS):
                for j in range(NUM_GROUPS):
                    if(final_deaths_rate_current[i]==final_deaths_rate_current[group_vulnerability[j]]):
                        group_vulner_dict[i] = j
            #print('group_vulner_dict:', group_vulner_dict)
            # Annotate the CBGs according to the corresponding group vulnerability
            cbg_age_msa['Group_Vulnerability'] = cbg_age_msa.apply(lambda x : group_vulner_dict[x['Age_Quantile']], axis=1)
            
            # Income
            demo_feat = 'Mean_Household_Income'
            final_deaths_rate_current = np.zeros(NUM_GROUPS)
            for group_id in range(NUM_GROUPS):
                final_deaths_rate_current[group_id] = cbg_income_msa[cbg_income_msa['Mean_Household_Income_Quantile']==group_id]['Final_Deaths_Current'].sum()
                final_deaths_rate_current[group_id] /= cbg_income_msa[cbg_income_msa['Mean_Household_Income_Quantile']==group_id]['Sum'].sum()
            #print(demo_feat, ', final_deaths_rate_current: ', final_deaths_rate_current)
            # Sort groups according to vulnerability
            group_vulnerability = np.argsort(-final_deaths_rate_current) # 死亡率从大到小排序
            group_vulner_dict = dict()
            for i in range(NUM_GROUPS):
                for j in range(NUM_GROUPS):
                    if(final_deaths_rate_current[i]==final_deaths_rate_current[group_vulnerability[j]]):
                        group_vulner_dict[i] = j
            #print('group_vulner_dict:', group_vulner_dict)
            # Annotate the CBGs according to the corresponding group vulnerability
            cbg_income_msa['Group_Vulnerability'] = cbg_income_msa.apply(lambda x : group_vulner_dict[x['Mean_Household_Income_Quantile']], axis=1)
            
            # Occupation
            demo_feat = 'EW'
            final_deaths_rate_current = np.zeros(NUM_GROUPS)
            for group_id in range(NUM_GROUPS):
                final_deaths_rate_current[group_id] = cbg_occupation_msa[cbg_occupation_msa['Essential_Worker_Quantile']==group_id]['Final_Deaths_Current'].sum()
                final_deaths_rate_current[group_id] /= cbg_occupation_msa[cbg_occupation_msa['Essential_Worker_Quantile']==group_id]['Sum'].sum()
            #print(demo_feat, ', final_deaths_rate_current: ', final_deaths_rate_current)
            # Sort groups according to vulnerability
            group_vulnerability = np.argsort(-final_deaths_rate_current) # 死亡率从大到小排序
            group_vulner_dict = dict()
            for i in range(NUM_GROUPS):
                for j in range(NUM_GROUPS):
                    if(final_deaths_rate_current[i]==final_deaths_rate_current[group_vulnerability[j]]):
                        group_vulner_dict[i] = j
            #print('group_vulner_dict:', group_vulner_dict)
            # Annotate the CBGs according to the corresponding group vulnerability
            cbg_occupation_msa['Group_Vulnerability'] = cbg_occupation_msa.apply(lambda x : group_vulner_dict[x['Essential_Worker_Quantile']], axis=1)
            
            # Generate scores according to each policy in policy_to_combine
            age_scores = cbg_age_msa['Group_Vulnerability'].copy()  # The smaller the group number, the more vulnerable the group.
            income_scores = cbg_income_msa['Group_Vulnerability'].copy()  # The smaller the group number, the more vulnerable the group.
            occupation_scores = cbg_occupation_msa['Group_Vulnerability'].copy()  # The smaller the group number, the more vulnerable the group.
            # If vaccinated, the ranking must be changed.
            cbg_hybrid_msa['Vulner_Rank'] = cbg_hybrid_msa.apply(lambda x : x['Vulner_Rank'] if x['Vaccination_Vector']==0 else (len(cbg_hybrid_msa)+1), axis=1) # 20210508
            cbg_hybrid_msa['Damage_Rank'] = cbg_hybrid_msa.apply(lambda x : x['Damage_Rank'] if x['Vaccination_Vector']==0 else (len(cbg_hybrid_msa)+1), axis=1) # 20210508
            vulner_scores = cbg_hybrid_msa['Vulner_Rank'].copy()
            damage_scores = cbg_hybrid_msa['Damage_Rank'].copy()
            
            # Normalization
            age_scores += 1; age_scores /= np.max(age_scores)
            income_scores += 1; income_scores /= np.max(income_scores)
            occupation_scores += 1; occupation_scores /= np.max(occupation_scores)
            vulner_scores += 1; vulner_scores /= np.max(vulner_scores)
            damage_scores += 1; damage_scores /= np.max(damage_scores)
            
            # Combine the scores according to policy weights, to get the final ranking of CBGs
            cbg_multi_scores = []
            for i in range(M):
                cbg_multi_scores.append([age_scores[i],income_scores[i],occupation_scores[i],vulner_scores[i], damage_scores[i]])

            data = Data(cbg_multi_scores, criteria, weights=weights, cnames=cnames)
            decider = closeness.TOPSIS() 
            decision = decider.decide(data)
            cbg_hybrid_msa['Hybrid_Sort'] = decision.rank_
            
            # Distribute vaccines in the currently most vulnerable group - flooding
            new_vector = functions.vaccine_distribution_flood_new(cbg_table=cbg_hybrid_msa, 
                                                                #vaccination_ratio=VACCINATION_RATIO, 
                                                                vaccination_ratio=RECHECK_INTERVAL, 
                                                                demo_feat='Hybrid_Sort', ascending=True, 
                                                                execution_ratio=EXECUTION_RATIO,
                                                                leftover=leftover,
                                                                is_last=is_last
                                                                )
            leftover_prev = leftover
            leftover = np.sum(cbg_sizes) * RECHECK_INTERVAL + leftover_prev - np.sum(new_vector) 
            current_vector += new_vector
            assert((current_vector<=cbg_sizes).all())
            #print('Newly distributed vaccines: ', np.sum(new_vector))
            #print('Leftover: ', leftover)
            #print('Total Num of distributed vaccines: ', np.sum(current_vector))
        
        vaccination_vector_hybrid_flood = current_vector

        # Run simulations
        _, history_D2_hybrid_flood = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                    vaccination_vector=vaccination_vector_hybrid_flood,
                                                    vaccine_acceptance=vaccine_acceptance,
                                                    protection_rate = PROTECTION_RATE)


    # Obtain the efficiency and equity of the hybrid policy
    # Check whether the hybrid policy is good enough
    policy_all = policy_to_compare_rel_to_no_vaccination+['Hybrid_Flood']
    policy_all_no_vaccination = policy_to_compare_rel_to_no_vaccination+['Hybrid_Flood']
    policy_all_baseline = policy_to_compare_rel_to_baseline+['Hybrid_Flood']

    # Add simulation results to grouping tables
    for policy in policy_all:
        exec("history_D2_%s = np.array(history_D2_%s)" % (policy.lower(),policy.lower()))
        exec("avg_history_D2_%s = np.mean(history_D2_%s,axis=1)" % (policy.lower(),policy.lower()))
        exec("avg_final_deaths_%s = avg_history_D2_%s[-1,:]" % (policy.lower(),policy.lower()))

        exec("cbg_age_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
        exec("cbg_income_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
        exec("cbg_occupation_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
        
    gini_table_no_vaccination = make_gini_table(policy_list=policy_all_no_vaccination, demo_feat_list=demo_feat_list, num_groups=NUM_GROUPS, 
                                                rel_to='No_Vaccination', save_path=None, save_result=False)
    gini_table_baseline = make_gini_table(policy_list=policy_all_baseline, demo_feat_list=demo_feat_list, num_groups=NUM_GROUPS, 
                                          rel_to='Baseline', save_path=None, save_result=False)                             
    print('Gini table of all policies: \n', gini_table_no_vaccination)

    data_column = gini_table_no_vaccination['Hybrid_Flood']
    hybrid_death_rate = float(data_column.iloc[0])
    hybrid_age_gini = float(data_column.iloc[2])
    hybrid_income_gini = float(data_column.iloc[4])
    hybrid_occupation_gini = float(data_column.iloc[6])
    hybrid_overall_performance = get_overall_performance(gini_table_baseline['Hybrid_Flood'])

    # Compare current results to target results
    better_efficiency = (hybrid_death_rate<=(target_death_rate)) # 20211013
    better_age_gini = (hybrid_age_gini<=(target_age_gini))
    better_income_gini = (hybrid_income_gini<=(target_income_gini))
    better_occupation_gini = (hybrid_occupation_gini<=(target_occupation_gini))
    better_overall_performance = (hybrid_overall_performance>=(target_overall_performance)) #20211020
    
    #########################################
    # Compare to target
    # print('Compare to target:')
    print('Death rate: ', hybrid_death_rate, target_death_rate, 'Good enough?', better_efficiency)
    print('Age gini: ', hybrid_age_gini, target_age_gini, 'Good enough?', better_age_gini)
    print('Income gini: ', hybrid_income_gini, target_income_gini, 'Good enough?', better_income_gini)
    print('Occupation gini: ', hybrid_occupation_gini, target_occupation_gini, 'Good enough?', better_occupation_gini)
    print('Overall performance: ', hybrid_overall_performance, target_overall_performance,'Good enough?', better_overall_performance)
    #print('Weights:',weights)

    # Compared to best in the history, to determine whether to save this one
    # Count number of dimensions that are better than target results
    num_better_now = 0
    if(better_efficiency): num_better_now += 1
    if(better_age_gini): num_better_now += 1
    if(better_income_gini): num_better_now += 1
    if(better_occupation_gini): num_better_now += 1
    print('num_better_now:',num_better_now)
    print('num_better_history:',num_better_history)
    if(num_better_now>=num_better_history):
        num_better_history = num_better_now
        print('Weights:',weights)
        print('Find a better or equal solution. num_better_history updated: ',num_better_history)  

    #########################################
    # Compare to current best
    # If this is the first time we simulate, there are no existing best_xxx, 
    # so directly assign them current hybrid values. 
    if(first_time == True):
        best_weights = weights.copy()
        best_hybrid_death_rate = hybrid_death_rate
        best_hybrid_age_gini = hybrid_age_gini
        best_hybrid_income_gini = hybrid_income_gini
        best_hybrid_occupation_gini = hybrid_occupation_gini
        best_hybrid_overall_performance = hybrid_overall_performance
        history_D2_best_hybrid_flood = history_D2_hybrid_flood.copy()
        first_time = False

    print('Comparing to current best: ')
    print('Death rate: ', hybrid_death_rate, best_hybrid_death_rate, 'Good enough?', (hybrid_death_rate<=best_hybrid_death_rate))
    print('Age gini: ', hybrid_age_gini, best_hybrid_age_gini, 'Good enough?', (hybrid_age_gini<=best_hybrid_age_gini))
    print('Income gini: ', hybrid_income_gini, best_hybrid_income_gini, 'Good enough?', (hybrid_income_gini<=best_hybrid_income_gini))
    print('Occupation gini: ', hybrid_occupation_gini, best_hybrid_occupation_gini, 'Good enough?', (hybrid_occupation_gini<=best_hybrid_occupation_gini))
    print('Overall performance: ', hybrid_overall_performance, best_hybrid_overall_performance,'Good enough?',(hybrid_overall_performance>=best_hybrid_overall_performance))

    # Better than history: save it
    if((num_better_now==4)&(hybrid_overall_performance>=best_hybrid_overall_performance)):
        print('Find a better solution. Weights:',weights)
        if(os.path.exists(file_savename)):
            print('Result already exists. No need to save.')
        else:
            print('Result will be saved. File name: ', file_savename)
            np.array(history_D2_hybrid_flood).tofile(file_savename)

    # Better than history and really good enough
    if((num_better_now==4)&(better_overall_performance)):
        if(refine_mode==False):
            print('Find a better solution. Weights:',weights)
            if(os.path.exists(file_savename)):
                print('Result already exists. No need to save.')
            else:
                print('Result will be saved. File name: ', file_savename)
                np.array(history_D2_hybrid_flood).tofile(file_savename)
            
            refine_mode = True; print('######################Will enter refine_mode next round.######################')
            refine_count = 0
            refine_w = weights.copy()

            best_weights = weights.copy()
            best_hybrid_death_rate = hybrid_death_rate
            best_hybrid_age_gini = hybrid_age_gini
            best_hybrid_income_gini = hybrid_income_gini
            best_hybrid_occupation_gini = hybrid_occupation_gini
            best_hybrid_overall_performance = hybrid_overall_performance
            history_D2_best_hybrid_flood = history_D2_hybrid_flood.copy() 
            #continue
        
        elif(refine_mode==True):
            #if((hybrid_death_rate<=best_hybrid_death_rate)&(hybrid_age_gini<=best_hybrid_age_gini)
            #    &(hybrid_income_gini<best_hybrid_income_gini)&(hybrid_occupation_gini<best_hybrid_occupation_gini)):
            
            if(hybrid_overall_performance>best_hybrid_overall_performance): # 20211020
                print('Find a better solution. Weights:',weights)
                if(os.path.exists(file_savename)):
                    print('Result already exists. No need to save.')
                else:
                    print('Result will be saved. File name: ', file_savename)
                    np.array(history_D2_hybrid_flood).tofile(file_savename)

                best_weights = weights.copy()
                best_hybrid_death_rate = hybrid_death_rate
                best_hybrid_age_gini = hybrid_age_gini
                best_hybrid_income_gini = hybrid_income_gini
                best_hybrid_occupation_gini = hybrid_occupation_gini
                best_hybrid_overall_performance = hybrid_overall_performance
                history_D2_best_hybrid_flood = history_D2_hybrid_flood.copy()
                    
    if(refine_mode==True):
        refine_count += 1
        if(refine_count>=refine_threshold):
            break 

    if(refine_mode==False):
        if(not better_efficiency): 
            dice = np.random.random()
            if(dice>0.5):
                w4 += 0.1; w4=round(w4,1)
            else:
                w5 += 0.1; w5=round(w5,1)
        if(not better_age_gini): 
            w1 += 0.1; w1=round(w1,1)
        if(not better_income_gini): 
            w2 += 0.1; w2=round(w2,1)
        if(not better_occupation_gini): 
            w3 += 0.1; w3=round(w3,1)
        if(not better_overall_performance):
            if(hybrid_overall_performance<age_flood_overall_performance):
                w1 += 0.1; w1=round(w1,1)
            if(hybrid_overall_performance<income_flood_overall_performance):
                w2 += 0.1; w2=round(w2,1)
            if(hybrid_overall_performance<jue_ew_flood_overall_performance):
                w3 += 0.1; w3=round(w3,1)
        weights = [w1,w2,w3,w4,w5]
        print('New weights:',weights)

print('\nFinal weights: ', best_weights) #weights)


# Ablation version
# Ranking
cbg_age_msa['Elder_Ratio_Rank'] = cbg_age_msa['Elder_Ratio'].rank(ascending=False,method='first') 
cbg_occupation_msa['Essential_Worker_Ratio_Rank'] = cbg_occupation_msa['Essential_Worker_Ratio'].rank(ascending=False,method='first') 
cbg_income_msa['Mean_Household_Income_Rank'] = cbg_income_msa['Mean_Household_Income'].rank(ascending=True,method='first') 

ablation_weights = best_weights[:3]
w1 = ablation_weights[0]
w2 = ablation_weights[1]
w3 = ablation_weights[2]
weights = [w1,w2,w3]
print('Ablation version, weights: ',weights)

cnames=["Age_Flood", "Income_Flood", "EW_Flood"]
criteria = [MIN, MIN, MIN]

file_savename = os.path.join(root,MSA_NAME,subroot,
                             'test_history_D2_adaptive_hybrid_ablation_%sd_%s_%s_%s%s%s_%sseeds_%s%s'
                             % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,w1,w2,w3,NUM_SEEDS,notation_string,MSA_NAME))
if(os.path.exists(file_savename)):
    print('Result already exists. No need to simulate. Directly load it. Weights: ', weights)  
    history_D2_hybrid_ablation = np.fromfile(file_savename)
    history_D2_hybrid_ablation = np.reshape(history_D2_hybrid_ablation,(63,NUM_SEEDS,M))
else:
    current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
    cbg_hybrid_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
    leftover = 0
    for i in range(int(distribution_time)):
        if i==(int(distribution_time)-1): is_last = True
        else: is_last=False
        
        cbg_hybrid_msa['Vaccination_Vector'] = current_vector
        
        # Run a simulation to estimate death risks at the moment
        _, history_D2_current = run_simulation(starting_seed=STARTING_SEED_CHECKING, 
                                            num_seeds=NUM_SEEDS_CHECKING, 
                                            vaccination_vector=current_vector,
                                            vaccine_acceptance=vaccine_acceptance,
                                            protection_rate = PROTECTION_RATE)
                                                                
        # Average history records across random seeds
        deaths_cbg_current, _ = functions.average_across_random_seeds_only_death(history_D2_current, 
                                                                                M, idxs_msa_nyt, 
                                                                                print_results=False,draw_results=False)
                                                                                            
        # Analyze deaths in each demographic group
        avg_final_deaths_current = deaths_cbg_current[-1,:]
        
        # Add simulation results to cbg table
        cbg_hybrid_msa['Final_Deaths_Current'] = avg_final_deaths_current
        cbg_age_msa['Final_Deaths_Current'] = avg_final_deaths_current
        cbg_income_msa['Final_Deaths_Current'] = avg_final_deaths_current
        cbg_occupation_msa['Final_Deaths_Current'] = avg_final_deaths_current
        
        # Grouping
        separators = functions.get_separators(cbg_hybrid_msa, NUM_GROUPS, 'Final_Deaths_Current','Sum', normalized=False)
        cbg_hybrid_msa['Final_Deaths_Current_Quantile'] =  cbg_hybrid_msa['Final_Deaths_Current'].apply(lambda x : functions.assign_group(x, separators))
        cbg_age_msa['Final_Deaths_Current_Quantile'] =  cbg_hybrid_msa['Final_Deaths_Current_Quantile'].copy()
        cbg_income_msa['Final_Deaths_Current_Quantile'] =  cbg_hybrid_msa['Final_Deaths_Current_Quantile'].copy()
        cbg_occupation_msa['Final_Deaths_Current_Quantile'] =  cbg_hybrid_msa['Final_Deaths_Current_Quantile'].copy()

        # Generate scores according to each policy in policy_to_combine
        cbg_hybrid_msa['Age_Score'] = cbg_age_msa['Elder_Ratio_Rank'].copy()
        cbg_hybrid_msa['Income_Score'] = cbg_income_msa['Mean_Household_Income_Rank'].copy()
        cbg_hybrid_msa['Occupation_Score'] = cbg_occupation_msa['Essential_Worker_Ratio_Rank'].copy()
        
        # If vaccinated, the ranking must be changed.
        cbg_hybrid_msa['Age_Score'] = cbg_hybrid_msa.apply(lambda x : x['Age_Score'] if x['Vaccination_Vector']==0 else (len(cbg_hybrid_msa)+1), axis=1) # 20210508
        cbg_hybrid_msa['Income_Score'] = cbg_hybrid_msa.apply(lambda x : x['Income_Score'] if x['Vaccination_Vector']==0 else (len(cbg_hybrid_msa)+1), axis=1) # 20210508
        cbg_hybrid_msa['Occupation_Score'] = cbg_hybrid_msa.apply(lambda x : x['Occupation_Score'] if x['Vaccination_Vector']==0 else (len(cbg_hybrid_msa)+1), axis=1) # 20210508
        
        # Only those in the largest 'Final_Deaths_Current_Quantile' is qualified
        cbg_hybrid_msa['Age_Score'] = cbg_hybrid_msa.apply(lambda x : x['Age_Score'] if x['Final_Deaths_Current_Quantile']==NUM_GROUPS-1 else (len(cbg_hybrid_msa)+1), axis=1) # 20210709
        cbg_hybrid_msa['Income_Score'] = cbg_hybrid_msa.apply(lambda x : x['Income_Score'] if x['Final_Deaths_Current_Quantile']==NUM_GROUPS-1 else (len(cbg_hybrid_msa)+1), axis=1) # 20210709
        cbg_hybrid_msa['Occupation_Score'] = cbg_hybrid_msa.apply(lambda x : x['Occupation_Score'] if x['Final_Deaths_Current_Quantile']==NUM_GROUPS-1 else (len(cbg_hybrid_msa)+1), axis=1) # 20210709
        
        age_scores = cbg_hybrid_msa['Age_Score'].copy()
        income_scores = cbg_hybrid_msa['Income_Score'].copy()
        occupation_scores = cbg_hybrid_msa['Occupation_Score'].copy()
        
        # Normalization
        age_scores += 1
        age_scores /= np.max(age_scores)
        income_scores += 1
        income_scores /= np.max(income_scores)
        occupation_scores += 1
        occupation_scores /= np.max(occupation_scores)
        
        # Combine the scores according to policy weights, to get the final ranking of CBGs
        cbg_multi_scores = []
        for i in range(M):
            cbg_multi_scores.append([age_scores[i],income_scores[i],occupation_scores[i]])
        data = Data(cbg_multi_scores, criteria, weights=weights, cnames=cnames)
        
        decider = closeness.TOPSIS() 
        decision = decider.decide(data)
        cbg_hybrid_msa['Hybrid_Sort'] = decision.rank_
        
        # Distribute vaccines in the currently most vulnerable group - flooding
        new_vector = functions.vaccine_distribution_flood_new(cbg_table=cbg_hybrid_msa, 
                                                            #vaccination_ratio=VACCINATION_RATIO, 
                                                            vaccination_ratio=RECHECK_INTERVAL, 
                                                            demo_feat='Hybrid_Sort', ascending=True, 
                                                            execution_ratio=EXECUTION_RATIO,
                                                            leftover=leftover,
                                                            is_last=is_last
                                                            )
                                                            
        leftover_prev = leftover
        leftover = np.sum(cbg_sizes) * RECHECK_INTERVAL + leftover_prev - np.sum(new_vector) 
        current_vector += new_vector
        assert((current_vector<=cbg_sizes).all())
        #print('Newly distributed vaccines: ', np.sum(new_vector))
        #print('Leftover: ', leftover)
        #print('Total Num of distributed vaccines: ', np.sum(current_vector))
        
    vaccination_vector_hybrid_ablation = current_vector

    # Run simulations
    _, history_D2_hybrid_ablation = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                vaccination_vector=vaccination_vector_hybrid_ablation,
                                                vaccine_acceptance=vaccine_acceptance,
                                                protection_rate = PROTECTION_RATE)
                                                                    
policy_all = policy_to_compare_rel_to_no_vaccination+['Best_Hybrid_Flood']+['Hybrid_Ablation']
policy_all_no_vaccination = policy_to_compare_rel_to_no_vaccination+['Best_Hybrid_Flood']+['Hybrid_Ablation']
policy_all_baseline = policy_to_compare_rel_to_baseline+['Best_Hybrid_Flood']+['Hybrid_Ablation']

# Add simulation results to grouping tables
for policy in policy_all:
    exec("history_D2_%s = np.array(history_D2_%s)" % (policy.lower(),policy.lower()))
    exec("avg_history_D2_%s = np.mean(history_D2_%s,axis=1)" % (policy.lower(),policy.lower()))
    exec("avg_final_deaths_%s = avg_history_D2_%s[-1,:]" % (policy.lower(),policy.lower()))
    
    exec("cbg_age_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
    exec("cbg_income_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
    exec("cbg_occupation_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
    
gini_table_no_vaccination = make_gini_table(policy_list=policy_all_no_vaccination, demo_feat_list=demo_feat_list, num_groups=NUM_GROUPS, 
                                            rel_to='No_Vaccination', save_path=None, save_result=False)
gini_table_baseline = make_gini_table(policy_list=policy_all_baseline, demo_feat_list=demo_feat_list, num_groups=NUM_GROUPS, 
                                      rel_to='Baseline', save_path=None, save_result=False)
print('Gini table of all policies: \n', gini_table_no_vaccination)

data_column = gini_table_no_vaccination['Hybrid_Ablation']
hybrid_ablation_death_rate = float(data_column.iloc[0])
hybrid_ablation_age_gini = float(data_column.iloc[2])
hybrid_ablation_income_gini = float(data_column.iloc[4])
hybrid_ablation_occupation_gini = float(data_column.iloc[6])
hybrid_ablation_overall_performance = get_overall_performance(data_column)

print('Best weights: ', best_weights)
print('Ablation weights: ', ablation_weights)

print('###Ablation Compared to Baseline:###')
print('Death rate: ', hybrid_ablation_death_rate, baseline_death_rate, 
      'Good enough?', (hybrid_ablation_death_rate<=(baseline_death_rate)))
print('Age gini: ', hybrid_ablation_age_gini, baseline_age_gini, 
      'Good enough?', (hybrid_ablation_age_gini<=(baseline_age_gini)))
print('Income gini: ', hybrid_ablation_income_gini, baseline_income_gini, 
      'Good enough?', (hybrid_ablation_income_gini<=(baseline_income_gini)))
print('Occupation gini: ', hybrid_ablation_occupation_gini, baseline_occupation_gini, 
      'Good enough?', (hybrid_ablation_occupation_gini<=(baseline_occupation_gini)))
print('Overall performance: ', hybrid_ablation_overall_performance, baseline_overall_performance, 
      'Good enough?', (hybrid_ablation_overall_performance>=(baseline_overall_performance)))

print('###Ablation Compared to Complete_Hybrid:###')
print('Death rate: ', hybrid_ablation_death_rate, best_hybrid_death_rate, 
      'Good enough?', (hybrid_ablation_death_rate<(best_hybrid_death_rate)))
print('Age gini: ', hybrid_ablation_age_gini, best_hybrid_age_gini, 
      'Good enough?', (hybrid_ablation_age_gini<(best_hybrid_age_gini)))
print('Income gini: ', hybrid_ablation_income_gini, best_hybrid_income_gini, 
      'Good enough?', (hybrid_ablation_income_gini<(best_hybrid_income_gini)))
print('Occupation gini: ', hybrid_ablation_occupation_gini, best_hybrid_occupation_gini, 
      'Good enough?', (hybrid_ablation_occupation_gini<(best_hybrid_occupation_gini)))
print('Overall performance: ', hybrid_ablation_overall_performance, best_hybrid_overall_performance, 
      'Good enough?', (hybrid_ablation_overall_performance>=(best_hybrid_overall_performance)))

#file_savename = os.path.join(root,MSA_NAME,subroot,
#                             'test_history_D2_adaptive_hybrid_ablation_%sd_%s_%s_%s%s%s_%sseeds_%s%s'
#                             % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,w1,w2,w3,NUM_SEEDS,notation_string,MSA_NAME))
if(os.path.exists(file_savename)):
    print('Result already exists. No need to save.')
else:
    print('Save hybrid_ablation results. File name: ', file_savename)
    history_D2_hybrid_ablation.tofile(file_savename)

end = time.time()
print('Total time: ', (end-start))
print('Vaccination ratio:',VACCINATION_RATIO,' Vaccination time: ', VACCINATION_TIME)