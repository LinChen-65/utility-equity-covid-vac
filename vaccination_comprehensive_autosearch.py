# python vaccination_comprehensive_autosearch.py --msa_name Atlanta --vaccination_time 31 --vaccination_ratio 0.1 --recheck_interval 0.01 

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import sys
import socket
import os
import datetime
import pandas as pd
import numpy as np
import pickle
import argparse
import time
import pdb
from skcriteria import Data, MIN
from skcriteria.madm import closeness

import constants
import functions
import disease_model_test as disease_model

parser = argparse.ArgumentParser()
parser.add_argument('--msa_name', 
                    help='MSA name. ')
parser.add_argument('--vaccination_time', type=int, default=31,
                    help='Time to distribute vaccines.')
parser.add_argument('--vaccination_ratio' , type=float, default=0.1,
                    help='Vaccination ratio relative to MSA population.')                    
parser.add_argument('--consider_hesitancy', default=False, action='store_true',
                    help='If true, consider vaccine hesitancy.')
parser.add_argument('--acceptance_scenario', 
                    help='Scenario of vaccine hesitancy (fully/real/cf18/cf13/cf17/ALL). Only useful when consider_hesitancy is True.')
parser.add_argument('--consider_accessibility', default=False, action='store_true',
                    help='If true, consider vaccine accessibility.')
parser.add_argument('--quick_test', default=False, action='store_true',
                    help='If true, reduce num_seeds to 2.')
parser.add_argument('--num_groups', type=int, default=5,
                    help='Num of groups to divide CBGs into.') 
parser.add_argument('--execution_ratio', type=float, default=1,
                    help='Policy execution ratio.')                    
parser.add_argument('--recheck_interval', type=float, default = 0.01,
                    help='Recheck interval (After distributing some portion of vaccines, recheck the most vulnerable demographic group).')                             
parser.add_argument('--protection_rate', type=float, default=1, 
                    help='Vaccination protection rate.')      
parser.add_argument('--w1', type=float, default=1, 
                    help='Initial weight 1.')        
parser.add_argument('--w2', type=float, default=1, 
                    help='Initial weight 2.')                                           
parser.add_argument('--w3', type=float, default=1, 
                    help='Initial weight 3.')     
parser.add_argument('--w4', type=float, default=1, 
                    help='Initial weight 4.')     
parser.add_argument('--w5', type=float, default=1, 
                    help='Initial weight 5.')     
parser.add_argument('--w6', type=float, default=1, 
                    help='Initial weight 6.')       
parser.add_argument('--store_history', default=False, action='store_true',
                    help='If true, save history_D2 instead of final_deaths.')                                      
args = parser.parse_args()

print(f'MSA name: {args.msa_name}.')
print(f'vaccination time: {args.vaccination_time}.')
print(f'vaccination ratio: {args.vaccination_ratio}.')
print(f'recheck interval: {args.recheck_interval}.')
if(args.consider_hesitancy):
    print(f'Consider vaccine hesitancy. Vaccine acceptance scenario: {args.acceptance_scenario}.')
print(f'Quick testing? {args.quick_test}')

# root
hostname = socket.gethostname()
print('hostname: ', hostname)
if(hostname=='fib-dl3'):
    root = '/data/chenlin/COVID-19/Data' #dl3
    saveroot = '/data/chenlin/utility-equity-covid-vac/results'
elif(hostname=='rl4'):
    root = '/home/chenlin/COVID-19/Data' #rl4
    saveroot = '/home/chenlin/utility-equity-covid-vac/results'


# Policies to compare
demo_policy_to_compare = ['Age', 'Income', 'Occupation', 'Minority'] #20220305
policy_to_compare_rel_to_no_vaccination = ['No_Vaccination','Baseline'] + demo_policy_to_compare
policy_to_compare_rel_to_baseline = ['Baseline','No_Vaccination'] + demo_policy_to_compare

# Demo feat list
#demo_feat_list = ['Age', 'Mean_Household_Income', 'Essential_Worker']
demo_feat_list = ['Elder_Ratio', 'Mean_Household_Income', 'EW_Ratio', 'Minority_Ratio'] #20220306

# Derived variables
#policy_savename = 'adaptive_%sd_hybrid'%str(args.vaccination_time)
policy_savename = f'{str(args.vaccination_time)}d_hybrid' #20220305
print('policy_savename:',policy_savename)

# Initial weights
weights = [args.w1, args.w2, args.w3, args.w4, args.w5, args.w6]
w1,w2,w3,w4,w5,w6 = weights
print('Weights:', weights)

# Quick Test: prototyping
if(args.quick_test):
    NUM_SEEDS = 2
    NUM_SEEDS_CHECKING = 2
else:
    NUM_SEEDS = 30
    NUM_SEEDS_CHECKING = 30
print('NUM_SEEDS: ', NUM_SEEDS)
print('NUM_SEEDS_CHECKING: ', NUM_SEEDS_CHECKING)
STARTING_SEED = range(NUM_SEEDS)
STARTING_SEED_CHECKING = range(NUM_SEEDS_CHECKING)

# Distribute vaccines in how many rounds
distribution_time = args.vaccination_ratio / args.recheck_interval # 分几次把疫苗分配完

# Recheck interval for other strategies #20220306
recheck_interval_others = 0.01

###############################################################################
# Functions

def run_simulation(starting_seed, num_seeds, vaccination_vector, vaccine_acceptance,protection_rate=1):
    m = disease_model.Model(starting_seed=starting_seed, #20211013
                                 num_seeds=num_seeds,
                                 debug=False,clip_poisson_approximation=True,ipf_final_match='poi',ipf_num_iter=100)

    m.init_exogenous_variables(poi_areas=poi_areas,
                               poi_dwell_time_correction_factors=poi_dwell_time_correction_factors,
                               cbg_sizes=cbg_sizes,
                               poi_cbg_visits_list=poi_cbg_visits_list,
                               all_hours=all_hours,
                               p_sick_at_t0=constants.parameters_dict[args.msa_name][0],
                               vaccination_time=24*args.vaccination_time, # when to apply vaccination (which hour)
                               vaccination_vector = vaccination_vector,
                               vaccine_acceptance = vaccine_acceptance,#20211007
                               protection_rate = protection_rate,
                               home_beta=constants.parameters_dict[args.msa_name][1],
                               cbg_attack_rates_original = cbg_attack_rates_scaled,
                               cbg_death_rates_original = cbg_death_rates_scaled,
                               poi_psi=constants.parameters_dict[args.msa_name][2],
                               just_compute_r0=False,
                               latency_period=96,  # 4 days
                               infectious_period=84,  # 3.5 days
                               confirmation_rate=.1,
                               confirmation_lag=168,  # 7 days
                               death_lag=432
                               )

    m.init_endogenous_variables()

    if(args.store_history): #20220311
        history_C2, history_D2 = m.simulate_disease_spread(no_print=True, store_history=True)    
        return history_D2
    else:
        final_cases, final_deaths = m.simulate_disease_spread(no_print=True, store_history=False) #20220304
        return final_deaths #20220304


# Analyze results and produce graphs
def output_result(cbg_table, demo_feat, policy_list, num_groups, rel_to):
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
            if(policy=='No_Vaccination'):
                deaths_total_no_vaccination = deaths_total_abs
                deaths_gini_no_vaccination = deaths_gini_abs
                deaths_total_rel = 0; deaths_gini_rel = 0
                results[policy] = {'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.4f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.4f'% deaths_gini_rel} 
            else:
                deaths_total_rel = (eval('final_deaths_rate_%s_total'%(policy.lower())) - deaths_total_no_vaccination) / deaths_total_no_vaccination
                deaths_gini_rel = (functions.gini(eval('final_deaths_rate_'+ policy.lower())) - deaths_gini_no_vaccination) / deaths_gini_no_vaccination
                results[policy] = {'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.4f'% deaths_gini_abs, 
                                   'deaths_gini_rel':'%.4f'% deaths_gini_rel} 
        
        elif(rel_to=='Baseline'):
            if(policy=='Baseline'):
                deaths_total_baseline = deaths_total_abs
                deaths_gini_baseline = deaths_gini_abs
                deaths_total_rel = 0
                deaths_gini_rel = 0    
                results[policy] = {'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.4f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.4f'% deaths_gini_rel}   
            else:
                deaths_total_rel = (eval('final_deaths_rate_%s_total'%(policy.lower())) - deaths_total_baseline) / deaths_total_baseline
                deaths_gini_rel = (functions.gini(eval('final_deaths_rate_'+ policy.lower())) - deaths_gini_baseline) / deaths_gini_baseline
                results[policy] = {'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.4f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.4f'% deaths_gini_rel}                        
    return results


def make_gini_table(policy_list, demo_feat_list, num_groups, rel_to, save_path, save_result=False):
    cbg_table_name_dict=dict()
    cbg_table_name_dict['Elder_Ratio'] = cbg_age_msa
    cbg_table_name_dict['Mean_Household_Income'] = cbg_income_msa
    cbg_table_name_dict['EW_Ratio'] = cbg_occupation_msa
    cbg_table_name_dict['Minority_Ratio'] = cbg_minority_msa

    gini_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('All','deaths_total_abs'),('All','deaths_total_rel')]))
    gini_df['Policy'] = policy_list

    for demo_feat in demo_feat_list:
        results = output_result(cbg_table_name_dict[demo_feat], 
                                demo_feat, policy_list, num_groups=args.num_groups, rel_to=rel_to)
       
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
        

def get_overall_performance(data_column): #20211020 #20220306
    #return -(float(data_column.iloc[1])+float(data_column.iloc[3])+float(data_column.iloc[5])+float(data_column.iloc[7]))
    return -(float(data_column.iloc[1])+float(data_column.iloc[3])+float(data_column.iloc[5])+float(data_column.iloc[7])+float(data_column.iloc[9]))


def get_results_from_data_column(data_column): #20220306
    death_rate = float(data_column.iloc[0])
    age_gini = float(data_column.iloc[2])
    income_gini = float(data_column.iloc[4])
    occupation_gini = float(data_column.iloc[6])
    minority_gini = float(data_column.iloc[8])

    return death_rate, age_gini, income_gini, occupation_gini, minority_gini

###############################################################################
# Load Demographic-Related Data

MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[args.msa_name] 

# Load POI-CBG visiting matrices
f = open(os.path.join(root, args.msa_name, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()
# Load precomputed parameters to adjust(clip) POI dwell times
d = pd.read_csv(os.path.join(root,args.msa_name, 'parameters_%s.csv' % args.msa_name)) 
MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
all_hours = functions.list_hours_in_range(MIN_DATETIME, MAX_DATETIME)
poi_areas = d['feet'].values#面积
poi_dwell_times = d['median'].values#平均逗留时间
poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
del d

# Load ACS Data for MSA-county matching
acs_data = pd.read_csv(os.path.join(root,'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
good_list = list(msa_data['FIPS Code'].values) #;print('Counties included: ', good_list)
del acs_data

# Load CBG ids for the MSA
cbg_ids_msa = pd.read_csv(os.path.join(root,args.msa_name,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
M = len(cbg_ids_msa); #print('Number of CBGs in this metro area:', M)

# Mapping from cbg_ids to columns in hourly visiting matrices
cbgs_to_idxs = dict(zip(cbg_ids_msa['census_block_group'].values, range(M)))
x = {}
for i in cbgs_to_idxs: 
    x[str(i)] = cbgs_to_idxs[i]

# Load SafeGraph data to obtain CBG sizes (i.e., populations)
filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_b01.csv")
cbg_agesex = pd.read_csv(filepath)
cbg_age_msa = functions.load_cbg_age_msa(cbg_agesex, cbg_ids_msa) #20220306
del cbg_agesex

# Obtain cbg sizes (populations)
cbg_sizes = cbg_age_msa['Sum'].values
cbg_sizes = np.array(cbg_sizes,dtype='int32'); #print('Total population: ',np.sum(cbg_sizes))

# Select counties belonging to the MSA
y = []
for i in x:
    if((len(i)==12) & (int(i[0:5])in good_list)):
        y.append(x[i])
    if((len(i)==11) & (int(i[0:4])in good_list)):
        y.append(x[i])   
idxs_msa_all = list(x.values()) #; print('Number of CBGs in this metro area:', len(idxs_msa_all))
idxs_msa_nyt = y #; print('Number of CBGs in to compare with NYT data:', len(idxs_msa_nyt))

# Load other Safegraph demographic data, and perform grouping
if('Age' in demo_policy_to_compare):
    # Grouping
    separators = functions.get_separators(cbg_age_msa, args.num_groups, 'Elder_Ratio','Sum', normalized=True)
    cbg_age_msa['Elder_Ratio_Quantile'] =  cbg_age_msa['Elder_Ratio'].apply(lambda x : functions.assign_group(x, separators))
    
if('Occupation' in demo_policy_to_compare):
    filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_c24.csv")
    cbg_occupation = pd.read_csv(filepath)
    cbg_occupation_msa = functions.load_cbg_occupation_msa(cbg_occupation, cbg_ids_msa, cbg_sizes) #20220302
    del cbg_occupation
    cbg_occupation_msa.rename(columns={'Essential_Worker_Ratio':'EW_Ratio'},inplace=True)
    # Grouping
    separators = functions.get_separators(cbg_occupation_msa, args.num_groups, 'EW_Ratio','Sum', normalized=True)
    cbg_occupation_msa['EW_Ratio_Quantile'] =  cbg_occupation_msa['EW_Ratio'].apply(lambda x : functions.assign_group(x, separators))


if('Income' in demo_policy_to_compare):
    filepath = os.path.join(root,"ACS_5years_Income_Filtered_Summary.csv")
    cbg_income = pd.read_csv(filepath)
    cbg_income.drop(['Unnamed: 0'],axis=1, inplace=True)
    cbg_income_msa = functions.load_cbg_income_msa(cbg_income, cbg_ids_msa) #20220302
    del cbg_income
    cbg_income_msa['Sum'] = cbg_age_msa['Sum'].copy()
    # Grouping
    separators = functions.get_separators(cbg_income_msa, args.num_groups, 'Mean_Household_Income','Sum', normalized=False)
    cbg_income_msa['Mean_Household_Income_Quantile'] =  cbg_income_msa['Mean_Household_Income'].apply(lambda x : functions.assign_group(x, separators))

    if(args.consider_hesitancy):
        # Vaccine hesitancy by income #20211007
        if(args.acceptance_scenario in ['real','cf1','cf2','cf3','cf4','cf5','cf6','cf7','cf8']):
            cbg_income_msa['Vaccine_Acceptance'] = cbg_income_msa['Mean_Household_Income'].apply(lambda x:functions.assign_acceptance_absolute(x,args.acceptance_scenario))
        elif(args.acceptance_scenario in ['cf9','cf10','cf11','cf12','cf13','cf14','cf15','cf16','cf17','cf18']):
            cbg_income_msa['Vaccine_Acceptance'] = cbg_income_msa['Mean_Household_Income_Quantile'].apply(lambda x:functions.assign_acceptance_quantile(x,args.acceptance_scenario))
        # Retrieve vaccine acceptance as ndarray
        vaccine_acceptance = np.array(cbg_income_msa['Vaccine_Acceptance'].copy())
    elif(not args.consider_hesitancy):
        vaccine_acceptance = np.ones(len(cbg_sizes)) # fully accepted scenario

    
if('Minority' in demo_policy_to_compare): #20220305
    filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_b03.csv")
    cbg_ethnic = pd.read_csv(filepath)
    cbg_ethnic_msa = functions.load_cbg_ethnic_msa(cbg_ethnic, cbg_ids_msa, cbg_sizes)
    del cbg_ethnic
    # Grouping: 按args.num_groups分位数，将全体CBG分为args.num_groups个组，将分割点存储在separators中
    separators = functions.get_separators(cbg_ethnic_msa, args.num_groups, 'Minority_Ratio','Sum', normalized=False)
    cbg_ethnic_msa['Minority_Ratio_Quantile'] =  cbg_ethnic_msa['Minority_Ratio'].apply(lambda x : functions.assign_group(x, separators))
    cbg_minority_msa = cbg_ethnic_msa

###############################################################################
# Load results of other policies for comparison

subroot = 'vaccination_results_adaptive_%sd_%s_%s' % (str(args.vaccination_time),args.vaccination_ratio,recheck_interval_others)
print('subroot: ', subroot)

###############################################################################
# Preprocessing: from history_D2_xxx to final_deaths_xxx #20220306 # Obtain from make_gini_table.py

file_name_dict = {'baseline': 'baseline',
                  'age': 'age_flood', 
                  'age_reverse': 'age_flood', 
                  'income': 'income_flood',
                  'income_reverse': 'income_flood', 
                  'occupation': 'jue_ew_flood',
                  'occupation_reverse': 'jue_ew_flood'
                  }

for policy in policy_to_compare_rel_to_no_vaccination:
    print(policy)
    policy = policy.lower()

    if((policy in ['minority', 'minority_reverse']) & (args.vaccination_ratio == 0.4)):
        final_deaths_result_path = os.path.join(saveroot, f'vac_results_{args.vaccination_time}d_{args.vaccination_ratio}_0.04', f'final_deaths_{policy}_{args.vaccination_time}d_{args.vaccination_ratio}_0.04_30seeds_{args.msa_name}')
    elif((policy in ['minority', 'minority_reverse']) & (args.vaccination_ratio == 0.56)):
        final_deaths_result_path = os.path.join(saveroot, f'vac_results_{args.vaccination_time}d_{args.vaccination_ratio}_0.056', f'final_deaths_{policy}_{args.vaccination_time}d_{args.vaccination_ratio}_0.056_30seeds_{args.msa_name}')
    else:
        final_deaths_result_path = os.path.join(saveroot, f'vac_results_{args.vaccination_time}d_{args.vaccination_ratio}_{recheck_interval_others}', f'final_deaths_{policy}_{args.vaccination_time}d_{args.vaccination_ratio}_{args.recheck_interval}_30seeds_{args.msa_name}')
    
    if(os.path.exists(final_deaths_result_path)):
        print('Final deaths file already exists.')
        exec(f'final_deaths_{policy} = np.fromfile(\'{final_deaths_result_path}\')')
        exec(f'final_deaths_{policy} = np.reshape(final_deaths_{policy},(NUM_SEEDS,M))')   
    else: # Generate final_deaths_xx files
        if policy in ['minority', 'minority_reverse']:
            if(args.vaccination_ratio in [0.03,0.05,0.08,0.1,0.13,0.15,0.18,0.2]):
                exact_name = f'history_D2_{policy}_{args.vaccination_time}d_{args.vaccination_ratio}_{recheck_interval_others}_30seeds_{args.msa_name}'
                history_D2_result_path = os.path.join(saveroot, f'vac_results_{args.vaccination_time}d_{args.vaccination_ratio}_{recheck_interval_others}', exact_name)
            elif(args.vaccination_ratio in [0.4,0.56]):
                exact_name = f'history_D2_{policy}_{args.vaccination_time}d_{args.vaccination_ratio}_{args.vaccination_ratio/10}_30seeds_{args.msa_name}'
                history_D2_result_path = os.path.join(saveroot, f'vac_results_{args.vaccination_time}d_{args.vaccination_ratio}_{args.vaccination_ratio/10}', exact_name)
        elif(policy=='no_vaccination'):
            history_D2_result_path = os.path.join(root,args.msa_name,'vaccination_results_adaptive_31d_0.1_0.01',f'20210206_history_D2_no_vaccination_adaptive_0.1_0.01_30seeds_{args.msa_name}')
        #elif(policy=='baseline'):
        #    history_D2_result_path = os.path.join(root,args.msa_name,f'vaccination_results_adaptive_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}', f'test_history_D2_{file_name_dict[policy]}_adaptive_{args.vaccination_ratio}_{args.recheck_interval}_30seeds_{args.msa_name}') #20220304
        else:
            #history_D2_result_path = os.path.join(root,args.msa_name,f'vaccination_results_adaptive_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}', f'test_history_D2_{file_name_dict[policy]}_adaptive_{args.vaccination_time}d_{args.vaccination_ratio}_{args.recheck_interval}_30seeds_{args.msa_name}') #20220304
            if('reverse' not in policy):
                if(args.vaccination_time==31):
                    if(args.vaccination_ratio in [0.05,0.1,0.15,0.2]):
                        exact_name = f'20210206_history_D2_{file_name_dict[policy]}_adaptive_{args.vaccination_ratio}_{recheck_interval_others}_30seeds_{args.msa_name}'
                    elif(args.vaccination_ratio in [0.03,0.08,0.13,0.18]):
                        exact_name = f'history_D2_{file_name_dict[policy]}_adaptive_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{recheck_interval_others}_30seeds_{args.msa_name}' #20220304
                    elif(args.vaccination_ratio in [0.4,0.56]):
                        exact_name = f'test_history_D2_{file_name_dict[policy]}_adaptive_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{recheck_interval_others}_30seeds_{args.msa_name}' #20220304
                else:
                    if(args.vaccination_time in [26,36,41]):
                        exact_name = f'20210206_history_D2_{file_name_dict[policy]}_adaptive_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{recheck_interval_others}_30seeds_{args.msa_name}' #20220306
                    elif(args.vaccination_time in [24,29,34,39]):
                        exact_name = f'history_D2_{file_name_dict[policy]}_adaptive_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{recheck_interval_others}_30seeds_{args.msa_name}' #20220306
                history_D2_result_path = os.path.join(root,args.msa_name,f'vaccination_results_adaptive_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{recheck_interval_others}', exact_name) #20220307
            else:
                history_D2_result_path = os.path.join(root,args.msa_name,f'vaccination_results_adaptive_reverse_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{recheck_interval_others}', f'20210206_history_D2_{file_name_dict[policy]}_adaptive_reverse_{args.vaccination_ratio}_{args.recheck_interval}_30seeds_{args.msa_name}') #20220304
            
        exec(f'history_D2_{policy} = np.fromfile(\'{history_D2_result_path}\')')
        exec(f'history_D2_{policy} = np.reshape(history_D2_{policy},(63,NUM_SEEDS,M))')   
        exec(f'final_deaths_{policy} = np.array(history_D2_{policy}[-1,:,:])')
        exec(f'final_deaths_{policy}.tofile(final_deaths_result_path)')
        print(f'Final deaths file saved? {os.path.exists(final_deaths_result_path)}')
    exec(f'avg_final_deaths_{policy} = final_deaths_{policy}.mean(axis=0)')


        
# Add simulation results to grouping tables
for policy in policy_to_compare_rel_to_no_vaccination:
    exec(f"cbg_age_msa['Final_Deaths_{policy}'] = avg_final_deaths_{policy.lower()}")
    exec(f"cbg_income_msa['Final_Deaths_{policy}'] = avg_final_deaths_{policy.lower()}")
    exec(f"cbg_occupation_msa['Final_Deaths_{policy}'] = avg_final_deaths_{policy.lower()}")
    exec(f"cbg_minority_msa['Final_Deaths_{policy}'] = avg_final_deaths_{policy.lower()}")

# Check whether there is NaN in cbg_tables
if((cbg_age_msa.isnull().any().any()) or (cbg_income_msa.isnull().any().any()) or (cbg_occupation_msa.isnull().any().any()) or (cbg_minority_msa.isnull().any().any())):
    print('There are nan values in cbg_tables. Please check.')
    pdb.set_trace()

# Obtain utility and equity of policies
gini_table_no_vac = make_gini_table(policy_list=policy_to_compare_rel_to_no_vaccination, demo_feat_list=demo_feat_list, num_groups=args.num_groups, 
                                    rel_to='No_Vaccination', save_path=None, save_result=False)
print('Gini table of all the other policies (relative to no_vaccination): \n', gini_table_no_vac)
gini_table_baseline = make_gini_table(policy_list=policy_to_compare_rel_to_baseline, demo_feat_list=demo_feat_list, num_groups=args.num_groups, 
                                      rel_to='Baseline', save_path=None, save_result=False)
print('Gini table of all the other policies (relative to baseline): \n', gini_table_baseline)


# Best results from other polices  
'''
lowest_death_rate = float(gini_table_no_vac.iloc[0].min())
lowest_age_gini = float(gini_table_no_vac.iloc[2].min())
lowest_income_gini = float(gini_table_no_vac.iloc[4].min())
lowest_occupation_gini = float(gini_table_no_vac.iloc[6].min())
lowest_minority_gini = float(gini_table_no_vac.iloc[8].min())
'''

# No_Vaccination results
data_column = gini_table_no_vac['No_Vaccination']
no_vaccination_death_rate, no_vaccination_age_gini, no_vaccination_income_gini, no_vaccination_occupation_gini, no_vaccination_minority_gini = get_results_from_data_column(data_column) #20220306
#print(no_vaccination_death_rate,no_vaccination_age_gini,no_vaccination_income_gini,no_vaccination_occupation_gini)

# Baseline results
data_column = gini_table_no_vac['Baseline']
baseline_death_rate, baseline_age_gini, baseline_income_gini, baseline_occupation_gini, baseline_minority_gini = get_results_from_data_column(data_column) #20220306
#print(baseline_death_rate,baseline_age_gini,baseline_income_gini,baseline_occupation_gini)

# target: better of No_Vaccination and Baseline
target_death_rate = min(baseline_death_rate, no_vaccination_death_rate)
target_age_gini = min(baseline_age_gini, no_vaccination_age_gini)
target_income_gini = min(baseline_income_gini, no_vaccination_income_gini)
target_occupation_gini = min(baseline_occupation_gini, no_vaccination_occupation_gini)
target_minority_gini = min(baseline_minority_gini, no_vaccination_minority_gini) #20220306

# Overall performance, relative to Baseline # 20211020
baseline_overall_performance = 0
no_vaccination_overall_performance = get_overall_performance(gini_table_baseline['No_Vaccination'])
age_overall_performance = get_overall_performance(gini_table_baseline['Age'])
income_overall_performance = get_overall_performance(gini_table_baseline['Income'])
occupation_overall_performance = get_overall_performance(gini_table_baseline['Occupation'])
minority_overall_performance = get_overall_performance(gini_table_baseline['Minority'])
# Target overall performance: best of all the others
target_overall_performance = max(baseline_overall_performance,no_vaccination_overall_performance,
                                age_overall_performance, income_overall_performance, occupation_overall_performance, minority_overall_performance)
                                
print('baseline_overall_performance: ', baseline_overall_performance)
print('no_vaccination_overall_performance: ', no_vaccination_overall_performance)
print('age_overall_performance: ', age_overall_performance)
print('income_overall_performance: ', income_overall_performance)
print('occupation_overall_performance: ', occupation_overall_performance)
print('target_overall_performance: ', target_overall_performance)

###############################################################################
# Load and scale age-aware CBG-specific attack/death rates (original)

cbg_death_rates_original = np.loadtxt(os.path.join(root, args.msa_name, 'cbg_death_rates_original_'+args.msa_name))
cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)
attack_scale = 1
cbg_attack_rates_scaled = cbg_attack_rates_original * attack_scale
cbg_death_rates_scaled = cbg_death_rates_original * constants.death_scale_dict[args.msa_name]
cbg_age_msa['Death_Rate'] =  cbg_death_rates_scaled

# Obtain vulnerability and damage, according to theoretical analysis
cbg_age_msa = functions.obtain_vulner_damage(cbg_age_msa, args.msa_name, root, M, idxs_msa_all, idxs_msa_nyt, cbg_death_rates_scaled)

# Construct cbg_hybrid_msa
columns_of_interest = ['census_block_group','Sum']
cbg_hybrid_msa = cbg_age_msa[columns_of_interest].copy()
cbg_hybrid_msa['Vulner_Rank'] = cbg_age_msa['Vulner_Rank_New'].copy()
cbg_hybrid_msa['Damage_Rank'] = cbg_age_msa['Damage_Rank_New'].copy()
if(cbg_hybrid_msa.isnull().any().any()):
    print('There are NaNs in cbg_hybrid_msa. Please check.')
    pdb.set_trace()

# Annotate the most vulnerable group. 这是为了配合函数，懒得改函数
# Not grouping, so set all ['Most_Vulnerable']=1.
cbg_hybrid_msa['Most_Vulnerable'] = 1

###############################################################################
# Start autosearching

start = time.time()

cnames=["Age", "Income", "Occupation", "Vulner", "Damage", "Minority"]
criteria = [MIN, MIN, MIN, MIN, MIN, MIN] # Initial weights are input by user. [1,1,1,1,1,1]

num_better_history = 0
refine_mode = False # First-round search
refine_threshold = 0 #6 #0
first_time = True
while(True):
    # if in refine_mode, how to adjust weights
    if(refine_mode):
        print('refine_count: ', refine_count)
        if(refine_count<int(0.5*refine_threshold)):
            w1 = round((w1 * 1.1), 1)
            w2 = round((w2 * 1.1), 1)
            w3 = round((w3 * 1.1), 1)
            w6 = round((w6 * 1.1), 1)
        else:
            if(refine_count==int(0.5*refine_threshold)):
                w1 = refine_w[0]
                w2 = refine_w[1]
                w3 = refine_w[2]
                w6 = refine_w[5]
            w1 = round((w1 * 0.9), 1)
            w2 = round((w2 * 0.9), 1)
            w3 = round((w3 * 0.9), 1)
            w6 = round((w6 * 0.9), 1)
        weights = [w1,w2,w3,w4,w5,w6]    
    print('\nWeights:',weights)

    # path to save comprehensive result
    subroot = f'comprehensive/vac_results_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}' #20220306
    if not os.path.exists(os.path.join(saveroot, subroot)): # if folder does not exist, create one. #20220302
        os.makedirs(os.path.join(saveroot, subroot))
    file_savename = os.path.join(saveroot, subroot, f'final_deaths_hybrid_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}_{w1}{w2}{w3}{w4}{w5}{w6}_{NUM_SEEDS}seeds_{args.msa_name}')
    vac_vector_savename = os.path.join(saveroot, subroot, f'vac_vector_hybrid_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}_{w1}{w2}{w3}{w4}{w5}{w6}_{NUM_SEEDS}seeds_{args.msa_name}')
    if(args.store_history): #20220311
        history_D2_savename = os.path.join(saveroot, subroot, f'history_D2_hybrid_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}_{w1}{w2}{w3}{w4}{w5}{w6}_{NUM_SEEDS}seeds_{args.msa_name}')

    # if file for current weights exists, no need to simulate again                                 
    if((not args.store_history) & (os.path.exists(file_savename))):
        print('Result already exists. No need to simulate. Directly load it. Weights: ', weights)  
        final_deaths_hybrid = np.fromfile(file_savename) #20220306
        final_deaths_hybrid = np.reshape(final_deaths_hybrid,(NUM_SEEDS,M))  #20220306
        avg_final_deaths_hybrid = np.mean(final_deaths_hybrid, axis=0)
    else: # File not exists, start to simulate
        current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
        cbg_hybrid_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
        leftover = 0
        for i in range(int(distribution_time)):
            if i==(int(distribution_time)-1): is_last = True
            else: is_last=False
            cbg_hybrid_msa['Vaccination_Vector'] = current_vector
            
            # Run a simulation to estimate death risks at the moment
            result = run_simulation(starting_seed=STARTING_SEED_CHECKING, 
                                                  num_seeds=NUM_SEEDS_CHECKING, 
                                                  vaccination_vector=current_vector,
                                                  vaccine_acceptance=vaccine_acceptance,
                                                  protection_rate = args.protection_rate)                                                  
            if(args.store_history): #20220311
                final_deaths_current = np.array(result)[-1,:,:]
            else:
                final_deaths_current = result
            avg_final_deaths_current = final_deaths_current.mean(axis=0)
            
            # Add simulation results to cbg table
            cbg_hybrid_msa['Final_Deaths_Current'] = avg_final_deaths_current
            cbg_age_msa['Final_Deaths_Current'] = avg_final_deaths_current
            cbg_income_msa['Final_Deaths_Current'] = avg_final_deaths_current
            cbg_occupation_msa['Final_Deaths_Current'] = avg_final_deaths_current
            cbg_ethnic_msa['Final_Deaths_Current'] = avg_final_deaths_current
            
            # Generate scores according to each policy in policy_to_combine
            # Estimate demographic disparities: The smaller the group number, the more vulnerable the group.
            # Age
            age_scores = functions.annotate_group_vulnerability(demo_feat='Elder_Ratio', cbg_table=cbg_age_msa, num_groups=args.num_groups)
            # Income
            income_scores = functions.annotate_group_vulnerability(demo_feat='Mean_Household_Income', cbg_table=cbg_income_msa, num_groups=args.num_groups)
            # Occupation
            occupation_scores = functions.annotate_group_vulnerability(demo_feat='EW_Ratio', cbg_table=cbg_occupation_msa, num_groups=args.num_groups)
            # Minority
            minority_scores = functions.annotate_group_vulnerability(demo_feat='Minority_Ratio', cbg_table=cbg_minority_msa, num_groups=args.num_groups)
             
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
            minority_scores += 1; minority_scores /= np.max(minority_scores) #20220306
            
            # Combine the scores according to policy weights, to get the final ranking of CBGs
            cbg_multi_scores = []
            for i in range(M):
                cbg_multi_scores.append([age_scores[i],income_scores[i],occupation_scores[i],vulner_scores[i], damage_scores[i], minority_scores[i]])

            data = Data(cbg_multi_scores, criteria, weights=weights, cnames=cnames)
            decider = closeness.TOPSIS() 
            decision = decider.decide(data)
            cbg_hybrid_msa['Hybrid_Sort'] = decision.rank_
            
            # Distribute vaccines in the currently most vulnerable group - flooding
            new_vector = functions.vaccine_distribution_flood_new(cbg_table=cbg_hybrid_msa, 
                                                                #vaccination_ratio=args.vaccination_ratio, 
                                                                vaccination_ratio=args.recheck_interval, 
                                                                demo_feat='Hybrid_Sort', ascending=True, 
                                                                execution_ratio=args.execution_ratio,
                                                                leftover=leftover,
                                                                is_last=is_last
                                                                )
            leftover_prev = leftover
            leftover = np.sum(cbg_sizes) * args.recheck_interval + leftover_prev - np.sum(new_vector) 
            current_vector += new_vector
            assert((current_vector<=cbg_sizes).all())
        
        vaccination_vector_hybrid = current_vector

        # Run simulations
        result = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                vaccination_vector=vaccination_vector_hybrid,
                                vaccine_acceptance=vaccine_acceptance,protection_rate = args.protection_rate)
        
        if(args.store_history): #20220311
            history_D2_hybrid = result
            history_D2_hybrid.tofile(history_D2_savename)
            vaccination_vector_hybrid.tofile(vac_vector_savename)
            print(f'history_D2 saved at: {history_D2_savename}')
            break
        else:
            final_deaths_hybrid = result
            avg_final_deaths_hybrid = final_deaths_hybrid.mean(axis=0)

    # Obtain the utility and equity of the hybrid policy
    # Check whether the hybrid policy is good enough
    policy_all = policy_to_compare_rel_to_no_vaccination+['Hybrid']
    policy_all_no_vaccination = policy_to_compare_rel_to_no_vaccination+['Hybrid']
    policy_all_baseline = policy_to_compare_rel_to_baseline+['Hybrid']

    # Add simulation results to grouping tables
    for policy in policy_all:
        exec(f"cbg_age_msa['Final_Deaths_{policy}'] = avg_final_deaths_{policy.lower()}")
        exec(f"cbg_income_msa['Final_Deaths_{policy}'] = avg_final_deaths_{policy.lower()}")
        exec(f"cbg_occupation_msa['Final_Deaths_{policy}'] = avg_final_deaths_{policy.lower()}")
        exec(f"cbg_minority_msa['Final_Deaths_{policy}'] = avg_final_deaths_{policy.lower()}")
     
    gini_table_no_vac = make_gini_table(policy_list=policy_all_no_vaccination, demo_feat_list=demo_feat_list, num_groups=args.num_groups, 
                                        rel_to='No_Vaccination', save_path=None, save_result=False)
    gini_table_baseline = make_gini_table(policy_list=policy_all_baseline, demo_feat_list=demo_feat_list, num_groups=args.num_groups, 
                                          rel_to='Baseline', save_path=None, save_result=False)                             
    print('Gini table of all policies: \n', gini_table_no_vac)

    data_column = gini_table_no_vac['Hybrid']
    hybrid_death_rate, hybrid_age_gini, hybrid_income_gini, hybrid_occupation_gini, hybrid_minority_gini = get_results_from_data_column(data_column) #20220306
    hybrid_overall_performance = get_overall_performance(gini_table_baseline['Hybrid'])

    # Compare current results to target results
    better_utility = (hybrid_death_rate<=(target_death_rate)); print('Death rate: ', hybrid_death_rate, target_death_rate, 'Good enough?', better_utility)
    better_age_gini = (hybrid_age_gini<=(target_age_gini)); print('Age gini: ', hybrid_age_gini, target_age_gini, 'Good enough?', better_age_gini)
    better_income_gini = (hybrid_income_gini<=(target_income_gini)); print('Income gini: ', hybrid_income_gini, target_income_gini, 'Good enough?', better_income_gini)
    better_occupation_gini = (hybrid_occupation_gini<=(target_occupation_gini)); print('Occupation gini: ', hybrid_occupation_gini, target_occupation_gini, 'Good enough?', better_occupation_gini)
    better_minority_gini = (hybrid_minority_gini<=(target_minority_gini)); print('Minority gini: ', hybrid_minority_gini, target_minority_gini, 'Good enough?', better_minority_gini) #20220306
    better_overall_performance = (hybrid_overall_performance>=(target_overall_performance)); print('Overall performance: ', hybrid_overall_performance, target_overall_performance,'Good enough?', better_overall_performance)   
    #print('Weights:',weights)

    # Compared to best in the history, to determine whether to save this one
    # Count number of dimensions that are better than target results
    num_better_now = 0
    if(better_utility): num_better_now += 1
    if(better_age_gini): num_better_now += 1
    if(better_income_gini): num_better_now += 1
    if(better_occupation_gini): num_better_now += 1
    if(better_minority_gini): num_better_now += 1 #20220306
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
    if(first_time):
        best_weights = weights.copy()
        best_hybrid_death_rate = hybrid_death_rate
        best_hybrid_age_gini = hybrid_age_gini
        best_hybrid_income_gini = hybrid_income_gini
        best_hybrid_occupation_gini = hybrid_occupation_gini
        best_hybrid_minority_gini = hybrid_minority_gini #20220306
        best_hybrid_overall_performance = hybrid_overall_performance
        final_deaths_best_hybrid = final_deaths_hybrid.copy() #20220306
        first_time = False

    print('Comparing to current best: ')
    print('Death rate: ', hybrid_death_rate, best_hybrid_death_rate, 'Good enough?', (hybrid_death_rate<=best_hybrid_death_rate))
    print('Age gini: ', hybrid_age_gini, best_hybrid_age_gini, 'Good enough?', (hybrid_age_gini<=best_hybrid_age_gini))
    print('Income gini: ', hybrid_income_gini, best_hybrid_income_gini, 'Good enough?', (hybrid_income_gini<=best_hybrid_income_gini))
    print('Occupation gini: ', hybrid_occupation_gini, best_hybrid_occupation_gini, 'Good enough?', (hybrid_occupation_gini<=best_hybrid_occupation_gini))
    print('Minority gini: ', hybrid_minority_gini, best_hybrid_minority_gini, 'Good enough?', (hybrid_minority_gini<=best_hybrid_minority_gini))
    print('Overall performance: ', hybrid_overall_performance, best_hybrid_overall_performance,'Good enough?',(hybrid_overall_performance>=best_hybrid_overall_performance))

    # Better than history: save it
    if((num_better_now==5)&(hybrid_overall_performance>=best_hybrid_overall_performance)):
        print('Find a better solution. Weights:',weights)
        if(os.path.exists(file_savename)):
            print('Result already exists. No need to save.')
        else:
            print('Result will be saved. File name: ', file_savename)
            np.array(final_deaths_hybrid).tofile(file_savename)
            vaccination_vector_hybrid.tofile(vac_vector_savename)

    # Better than history and really good 
    if((num_better_now==5)&(better_overall_performance)):
        if(refine_mode==False):
            print('Find a better solution. Weights:',weights)
            if(os.path.exists(file_savename)):
                print('Result already exists. No need to save.')
            else:
                print('Result will be saved. File name: ', file_savename)
                np.array(final_deaths_hybrid).tofile(file_savename)
                vaccination_vector_hybrid.tofile(vac_vector_savename)
            
            refine_mode = True; print('######################Will enter refine_mode next round.######################')
            refine_count = 0
            refine_w = weights.copy()

            best_weights = weights.copy()
            best_hybrid_death_rate = hybrid_death_rate
            best_hybrid_age_gini = hybrid_age_gini
            best_hybrid_income_gini = hybrid_income_gini
            best_hybrid_occupation_gini = hybrid_occupation_gini
            best_hybrid_minority_gini = hybrid_minority_gini #20220306
            best_hybrid_overall_performance = hybrid_overall_performance
            final_deaths_best_hybrid = final_deaths_hybrid.copy() #20220306
        
        elif(refine_mode):
            #if((hybrid_death_rate<=best_hybrid_death_rate)&(hybrid_age_gini<=best_hybrid_age_gini)
            #    &(hybrid_income_gini<best_hybrid_income_gini)&(hybrid_occupation_gini<best_hybrid_occupation_gini)):   
            if(hybrid_overall_performance>best_hybrid_overall_performance): # 20211020
                print('Find a better solution. Weights:',weights)
                if(os.path.exists(file_savename)):
                    print('Result already exists. No need to save.')
                else:
                    print('Result will be saved. File name: ', file_savename)
                    np.array(final_deaths_hybrid).tofile(file_savename)
                    vaccination_vector_hybrid.tofile(vac_vector_savename)

                best_weights = weights.copy()
                best_hybrid_death_rate = hybrid_death_rate
                best_hybrid_age_gini = hybrid_age_gini
                best_hybrid_income_gini = hybrid_income_gini
                best_hybrid_occupation_gini = hybrid_occupation_gini
                best_hybrid_minority_gini = hybrid_minority_gini #20220306
                best_hybrid_overall_performance = hybrid_overall_performance
                final_deaths_best_hybrid = final_deaths_hybrid.copy() #20220306
                    
    if(refine_mode):
        refine_count += 1
        if(refine_count>=refine_threshold):
            break 

    if(refine_mode==False):
        if(not better_utility): 
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
        if(not better_minority_gini): 
            w6 += 0.1; w6=round(w6,1)
        if(not better_overall_performance):
            if(hybrid_overall_performance<age_overall_performance):
                w1 += 0.1; w1=round(w1,1)
            if(hybrid_overall_performance<income_overall_performance):
                w2 += 0.1; w2=round(w2,1)
            if(hybrid_overall_performance<occupation_overall_performance):
                w3 += 0.1; w3=round(w3,1)
            if(hybrid_overall_performance<minority_overall_performance):
                w6 += 0.1; w6=round(w6,1)    
        weights = [w1,w2,w3,w4,w5,w6]
        print('New weights:',weights)

print('\nFinal weights: ', best_weights) #weights)
avg_final_deaths_best_hybrid = avg_final_deaths_hybrid #20220306

################################################################################################
# Ablation version

# Ranking
cbg_age_msa['Elder_Ratio_Rank'] = cbg_age_msa['Elder_Ratio'].rank(ascending=False,method='first') 
cbg_income_msa['Mean_Household_Income_Rank'] = cbg_income_msa['Mean_Household_Income'].rank(ascending=True,method='first') 
cbg_occupation_msa['EW_Ratio_Rank'] = cbg_occupation_msa['EW_Ratio'].rank(ascending=False,method='first') 
cbg_minority_msa['Minority_Ratio_Rank'] = cbg_minority_msa['Minority_Ratio'].rank(ascending=False,method='first') 

w1 = best_weights[0]
w2 = best_weights[1]
w3 = best_weights[2]
w6 = best_weights[5]
ablation_weights = [w1,w2,w3,w6]
weights = [w1,w2,w3,w6]
print('Ablation version, weights: ', weights)

cnames=["Age", "Income", "Occupation", "Minority"]
criteria = [MIN, MIN, MIN, MIN]

# path to save comprehensive_ablation result
file_savename = os.path.join(saveroot, subroot, f'final_deaths_hybrid_ablation_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}_{w1}{w2}{w3}{w6}_{NUM_SEEDS}seeds_{args.msa_name}')
vac_vector_savename = os.path.join(saveroot, subroot, f'vac_vector_hybrid_ablation_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}_{w1}{w2}{w3}{w6}_{NUM_SEEDS}seeds_{args.msa_name}')
# if file for current weights exists, no need to simulate again                                 
if(os.path.exists(file_savename)):
    print('Result already exists. No need to simulate. Directly load it. Weights: ', weights)  
    final_deaths_hybrid_ablation = np.fromfile(file_savename) #20220306
    final_deaths_hybrid_ablation = np.reshape(final_deaths_hybrid,(NUM_SEEDS,M))  #20220306
    avg_final_deaths_hybrid_ablation = np.mean(final_deaths_hybrid_ablation, axis=0)  
else:
    current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
    cbg_hybrid_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
    leftover = 0
    for i in range(int(distribution_time)):
        if i==(int(distribution_time)-1): is_last = True
        else: is_last=False
        
        cbg_hybrid_msa['Vaccination_Vector'] = current_vector
        
        # Run a simulation to estimate death risks at the moment
        final_deaths_current = run_simulation(starting_seed=STARTING_SEED_CHECKING, 
                                              num_seeds=NUM_SEEDS_CHECKING, 
                                              vaccination_vector=current_vector,
                                              vaccine_acceptance=vaccine_acceptance,
                                              protection_rate = args.protection_rate)
        avg_final_deaths_current = np.mean(final_deaths_current, axis=0)
        
        # Add simulation results to cbg table
        cbg_hybrid_msa['Final_Deaths_Current'] = avg_final_deaths_current
        cbg_age_msa['Final_Deaths_Current'] = avg_final_deaths_current
        cbg_income_msa['Final_Deaths_Current'] = avg_final_deaths_current
        cbg_occupation_msa['Final_Deaths_Current'] = avg_final_deaths_current
        cbg_minority_msa['Final_Deaths_Current'] = avg_final_deaths_current
        
        # Grouping
        separators = functions.get_separators(cbg_hybrid_msa, args.num_groups, 'Final_Deaths_Current','Sum', normalized=False)
        cbg_hybrid_msa['Final_Deaths_Current_Quantile'] =  cbg_hybrid_msa['Final_Deaths_Current'].apply(lambda x : functions.assign_group(x, separators))

        # Generate scores according to each policy in policy_to_combine
        cbg_hybrid_msa['Age_Score'] = cbg_age_msa['Elder_Ratio_Rank'].copy()
        cbg_hybrid_msa['Income_Score'] = cbg_income_msa['Mean_Household_Income_Rank'].copy()
        cbg_hybrid_msa['Occupation_Score'] = cbg_occupation_msa['EW_Ratio_Rank'].copy()
        cbg_hybrid_msa['Minority_Score'] = cbg_minority_msa['Minority_Ratio_Rank'].copy()
        
        # If vaccinated, the ranking must be changed.
        cbg_hybrid_msa['Age_Score'] = cbg_hybrid_msa.apply(lambda x : x['Age_Score'] if x['Vaccination_Vector']==0 else (len(cbg_hybrid_msa)+1), axis=1) # 20210508
        cbg_hybrid_msa['Income_Score'] = cbg_hybrid_msa.apply(lambda x : x['Income_Score'] if x['Vaccination_Vector']==0 else (len(cbg_hybrid_msa)+1), axis=1) # 20210508
        cbg_hybrid_msa['Occupation_Score'] = cbg_hybrid_msa.apply(lambda x : x['Occupation_Score'] if x['Vaccination_Vector']==0 else (len(cbg_hybrid_msa)+1), axis=1) # 20210508
        cbg_hybrid_msa['Minority_Score'] = cbg_hybrid_msa.apply(lambda x : x['Minority_Score'] if x['Vaccination_Vector']==0 else (len(cbg_hybrid_msa)+1), axis=1) # 20220306
        
        # Only those in the largest 'Final_Deaths_Current_Quantile' is qualified
        age_scores = cbg_hybrid_msa.apply(lambda x : x['Age_Score'] if x['Final_Deaths_Current_Quantile']==args.num_groups-1 else (len(cbg_hybrid_msa)+1), axis=1) # 20210709
        income_scores = cbg_hybrid_msa.apply(lambda x : x['Income_Score'] if x['Final_Deaths_Current_Quantile']==args.num_groups-1 else (len(cbg_hybrid_msa)+1), axis=1) # 20210709
        occupation_scores = cbg_hybrid_msa.apply(lambda x : x['Occupation_Score'] if x['Final_Deaths_Current_Quantile']==args.num_groups-1 else (len(cbg_hybrid_msa)+1), axis=1) # 20210709
        minority_scores = cbg_hybrid_msa.apply(lambda x : x['Minority_Score'] if x['Final_Deaths_Current_Quantile']==args.num_groups-1 else (len(cbg_hybrid_msa)+1), axis=1) # 20220306

        # Normalization
        age_scores += 1; age_scores /= np.max(age_scores)
        income_scores += 1; income_scores /= np.max(income_scores)
        occupation_scores += 1; occupation_scores /= np.max(occupation_scores)
        minority_scores += 1; minority_scores /= np.max(minority_scores) #20220306
        
        # Combine the scores according to policy weights, to get the final ranking of CBGs
        cbg_multi_scores = []
        for i in range(M):
            cbg_multi_scores.append([age_scores[i],income_scores[i],occupation_scores[i],minority_scores[i]])
        data = Data(cbg_multi_scores, criteria, weights=weights, cnames=cnames)
        
        decider = closeness.TOPSIS() 
        decision = decider.decide(data)
        cbg_hybrid_msa['Hybrid_Sort'] = decision.rank_
        
        # Distribute vaccines in the currently most vulnerable group - flooding
        new_vector = functions.vaccine_distribution_flood_new(cbg_table=cbg_hybrid_msa, 
                                                            #vaccination_ratio=args.vaccination_ratio, 
                                                            vaccination_ratio=args.recheck_interval, 
                                                            demo_feat='Hybrid_Sort', ascending=True, 
                                                            execution_ratio=args.execution_ratio,
                                                            leftover=leftover,
                                                            is_last=is_last
                                                            )
                                                            
        leftover_prev = leftover
        leftover = np.sum(cbg_sizes) * args.recheck_interval + leftover_prev - np.sum(new_vector) 
        current_vector += new_vector
        assert((current_vector<=cbg_sizes).all())
        
    vaccination_vector_hybrid_ablation = current_vector

    # Run simulations
    final_deaths_hybrid_ablation = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                vaccination_vector=vaccination_vector_hybrid_ablation,
                                                vaccine_acceptance=vaccine_acceptance,
                                                protection_rate = args.protection_rate)
    avg_final_deaths_hybrid_ablation = np.mean(final_deaths_hybrid_ablation, axis=0)   
    print('Save hybrid_ablation results. File name: ', file_savename)
    final_deaths_hybrid_ablation.tofile(file_savename) #20220306        

policy_all = policy_to_compare_rel_to_no_vaccination+['Best_Hybrid']+['Hybrid_Ablation']
policy_all_no_vaccination = policy_to_compare_rel_to_no_vaccination+['Best_Hybrid']+['Hybrid_Ablation']
policy_all_baseline = policy_to_compare_rel_to_baseline+['Best_Hybrid']+['Hybrid_Ablation']

# Add simulation results to grouping tables
for policy in policy_all:
    exec(f"cbg_age_msa['Final_Deaths_{policy}'] = avg_final_deaths_{policy.lower()}")
    exec(f"cbg_income_msa['Final_Deaths_{policy}'] = avg_final_deaths_{policy.lower()}")
    exec(f"cbg_occupation_msa['Final_Deaths_{policy}'] = avg_final_deaths_{policy.lower()}")
    exec(f"cbg_minority_msa['Final_Deaths_{policy}'] = avg_final_deaths_{policy.lower()}")
    
gini_table_no_vac = make_gini_table(policy_list=policy_all_no_vaccination, demo_feat_list=demo_feat_list, num_groups=args.num_groups, 
                                    rel_to='No_Vaccination', save_path=None, save_result=False)
gini_table_baseline = make_gini_table(policy_list=policy_all_baseline, demo_feat_list=demo_feat_list, num_groups=args.num_groups, 
                                      rel_to='Baseline', save_path=None, save_result=False)
print('Gini table of all policies: \n', gini_table_no_vac)

data_column = gini_table_no_vac['Hybrid_Ablation']
hybrid_ablation_death_rate, hybrid_ablation_age_gini, hybrid_ablation_income_gini, hybrid_ablation_occupation_gini, hybrid_ablation_minority_gini = get_results_from_data_column(data_column) #20220306
hybrid_ablation_overall_performance = get_overall_performance(data_column)

print('Best weights: ', best_weights)
print('Ablation weights: ', ablation_weights)

print('###Ablation Compared to Baseline:###')
print('Death rate: ', hybrid_ablation_death_rate, baseline_death_rate, 'Good enough?', (hybrid_ablation_death_rate<=(baseline_death_rate)))
print('Age gini: ', hybrid_ablation_age_gini, baseline_age_gini, 'Good enough?', (hybrid_ablation_age_gini<=(baseline_age_gini)))
print('Income gini: ', hybrid_ablation_income_gini, baseline_income_gini, 'Good enough?', (hybrid_ablation_income_gini<=(baseline_income_gini)))
print('Occupation gini: ', hybrid_ablation_occupation_gini, baseline_occupation_gini, 'Good enough?', (hybrid_ablation_occupation_gini<=(baseline_occupation_gini)))
print('Minority gini: ', hybrid_ablation_minority_gini, baseline_minority_gini, 'Good enough?', (hybrid_ablation_minority_gini<=(baseline_minority_gini))) #20220306
print('Overall performance: ', hybrid_ablation_overall_performance, baseline_overall_performance, 'Good enough?', (hybrid_ablation_overall_performance>=(baseline_overall_performance)))

print('###Ablation Compared to Complete_Hybrid:###')
print('Death rate: ', hybrid_ablation_death_rate, best_hybrid_death_rate, 'Good enough?', (hybrid_ablation_death_rate<(best_hybrid_death_rate)))
print('Age gini: ', hybrid_ablation_age_gini, best_hybrid_age_gini, 'Good enough?', (hybrid_ablation_age_gini<(best_hybrid_age_gini)))
print('Income gini: ', hybrid_ablation_income_gini, best_hybrid_income_gini, 'Good enough?', (hybrid_ablation_income_gini<(best_hybrid_income_gini)))
print('Occupation gini: ', hybrid_ablation_occupation_gini, best_hybrid_occupation_gini, 'Good enough?', (hybrid_ablation_occupation_gini<(best_hybrid_occupation_gini)))
print('Minority gini: ', hybrid_ablation_minority_gini, best_hybrid_minority_gini, 'Good enough?', (hybrid_ablation_minority_gini<(best_hybrid_minority_gini)))
print('Overall performance: ', hybrid_ablation_overall_performance, best_hybrid_overall_performance, 'Good enough?', (hybrid_ablation_overall_performance>=(best_hybrid_overall_performance)))



end = time.time()
print('Total time: ', (end-start))
print('Vaccination ratio:',args.vaccination_ratio,' Vaccination time: ', args.vaccination_time)