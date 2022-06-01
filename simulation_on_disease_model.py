# python simulation_on_disease_model.py --msa_name Atlanta --num_seeds 60

import argparse
import os
import datetime
import pandas as pd
import numpy as np
import pickle

import constants
import functions
import disease_model_only_modify_attack_rates as disease_model

import time

root = os.getcwd()
dataroot = os.path.join(root, 'data')
resultroot = os.path.join(root, 'results')

parser = argparse.ArgumentParser()
parser.add_argument('--msa_name', 
                    help='MSA name.')
parser.add_argument('--safegraph_root', default=dataroot,
                    help='Safegraph data root.')               
parser.add_argument('--num_seeds', type=int, default=30,
                    help='Safegraph data root.') 
args = parser.parse_args()


MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
NUM_DAYS = 63

MSA_NAME = args.msa_name; print('MSA_NAME: ',MSA_NAME)
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME]

policy_list = ['No_Vaccination', 'Age_Agnostic']   
print('Policy list: ', policy_list)


NUM_SEEDS = args.num_seeds #30
print('NUM_SEEDS: ', NUM_SEEDS)
STARTING_SEED = range(NUM_SEEDS)

# Vaccination_Ratio
VACCINATION_RATIO = 0.1
# Vaccination protection rate
PROTECTION_RATE = 1
# Policy execution ratio
EXECUTION_RATIO = 1

###############################################################################
# Functions

def run_simulation(starting_seed, num_seeds, vaccination_vector, protection_rate=1):
    m = disease_model.Model(starting_seed=starting_seed,
                            num_seeds=num_seeds,
                            debug=False,clip_poisson_approximation=True,ipf_final_match='poi',ipf_num_iter=100)

    m.init_exogenous_variables(poi_areas=poi_areas,
                               poi_dwell_time_correction_factors=poi_dwell_time_correction_factors,
                               cbg_sizes=cbg_sizes,
                               poi_cbg_visits_list=poi_cbg_visits_list,
                               all_hours=all_hours,
                               p_sick_at_t0=constants.parameters_dict[MSA_NAME][0],
                               vaccination_time=24*31, # when to apply vaccination (which hour)
                               vaccination_vector = vaccination_vector,
                               protection_rate = protection_rate,
                               home_beta=constants.parameters_dict[MSA_NAME][1],
                               cbg_attack_rates_original = cbg_attack_rates_original_scaled,
                               cbg_death_rates_original = cbg_death_rates_original_scaled,
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

    return history_C2, history_D2


###############################################################################
# Load Data

# Load POI-CBG visiting matrices
f = open(os.path.join(dataroot, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()

# Load precomputed parameters to adjust(clip) POI dwell times
d = pd.read_csv(os.path.join(dataroot, 'parameters_%s.csv' % MSA_NAME)) 
all_hours = functions.list_hours_in_range(MIN_DATETIME, MAX_DATETIME)
poi_areas = d['feet'].values    #Area
poi_dwell_times = d['median'].values    #Average Dwell Time
poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
del d

# Load ACS Data for MSA-county matching
acs_data = pd.read_csv(os.path.join(dataroot,'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
good_list = list(msa_data['FIPS Code'].values)
print('CBG included: ', good_list)
del acs_data

# Load CBG ids for the MSA
cbg_ids_msa = pd.read_csv(os.path.join(dataroot,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
M = len(cbg_ids_msa)

# Mapping from cbg_ids to columns in hourly visiting matrices
cbgs_to_idxs = dict(zip(cbg_ids_msa['census_block_group'].values, range(M)))
x = {}
for i in cbgs_to_idxs:
    x[str(i)] = cbgs_to_idxs[i]
print('Number of CBGs in this metro area:', M)

# Load SafeGraph data to obtain CBG sizes (i.e., populations)
filepath = os.path.join(args.safegraph_root,"safegraph_open_census_data/data/cbg_b01.csv")
cbg_agesex = pd.read_csv(filepath)
# Extract CBGs belonging to the MSA
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

###############################################################################
if ('Age_Agnostic' in policy_list):
    print('Age_Agnostic.')

    cbg_death_rates_original = np.loadtxt(os.path.join(dataroot, 'cbg_death_rates_original_'+MSA_NAME))
    cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)
    cbg_death_rates_original = np.ones(cbg_attack_rates_original.shape)
    attack_scale = 1
    death_scale = 0.0066
    cbg_attack_rates_original_scaled = cbg_attack_rates_original * attack_scale
    cbg_death_rates_original_scaled = death_scale

    # Construct the vaccination vector
    vaccination_vector_age_agnostic = np.zeros(len(cbg_sizes))
    # Run simulations
    _, history_D2_age_agnostic = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                vaccination_vector=vaccination_vector_age_agnostic,
                                                protection_rate = PROTECTION_RATE)


###############################################################################
# Load and scale age-aware CBG-specific death rates (original)

cbg_death_rates_original = np.loadtxt(os.path.join(dataroot, 'cbg_death_rates_original_'+MSA_NAME))
cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)
attack_scale = 1
cbg_attack_rates_original_scaled = cbg_attack_rates_original * attack_scale
cbg_death_rates_original_scaled = cbg_death_rates_original * constants.death_scale_dict[MSA_NAME]
print('Age-aware CBG-specific death rates scaled.')

###############################################################################
# No_Vaccination

if ('No_Vaccination' in policy_list):
    print('Policy: No_Vaccination.')

    # Construct the vaccination vector
    vaccination_vector_no_vaccination = np.zeros(len(cbg_sizes))
    # Run simulations
    _, history_D2_no_vaccination = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                  vaccination_vector=vaccination_vector_no_vaccination,
                                                  protection_rate = PROTECTION_RATE)


#############################################################################
# Save results

print('Policy list: ', policy_list)

policy = 'Age_Agnostic'
policy = policy.lower()
np.array(history_D2_age_agnostic).tofile(os.path.join(resultroot, 'vaccination_results_adaptive_31d_0.1_0.01', r'20210206_history_D2_%s_adaptive_%s_0.01_%sseeds_%s') % (policy,VACCINATION_RATIO,NUM_SEEDS,MSA_NAME))
policy = 'No_Vaccination'
policy = policy.lower()
np.array(history_D2_no_vaccination).tofile(os.path.join(resultroot, 'vaccination_results_adaptive_31d_0.1_0.01', r'20210206_history_D2_%s_adaptive_%s_0.01_%sseeds_%s') % (policy,VACCINATION_RATIO,NUM_SEEDS,MSA_NAME))

print('Results saved.')
