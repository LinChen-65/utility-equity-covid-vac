# python get_s_i_ratio_at_vaccination_moment.py 

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import argparse
import os
import datetime
import pandas as pd
import numpy as np
import pickle

import constants
import functions
import disease_model_returnSEIR as disease_model

import time
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--quick_test', default=False, action='store_true',
                    help='If true, reduce num_seeds to 2.')
parser.add_argument('--safegraph_root', default='/data/chenlin/COVID-19/Data',
                    help='Safegraph data root.') 
parser.add_argument('--save_result', default=False, action='store_true',
                    help='If true, save simulation results.')
args = parser.parse_args()

# root
#root = '/data/chenlin/COVID-19/Data'
root = os.getcwd()
dataroot = os.path.join(root, 'data')
saveroot = os.path.join(root, 'results')

###############################################################################
# Constants

MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 3, 30, 23) #MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
NUM_DAYS = 30 #NUM_DAYS = 63
print('NUM_DAYS:', NUM_DAYS)

# Vaccination protection rate
PROTECTION_RATE = 1
# Policy execution ratio
EXECUTION_RATIO = 1

MSA_NAME_LIST = ['Atlanta','Chicago','Dallas','Houston', 'LosAngeles',
                 'Miami','Philadelphia','SanFrancisco','WashingtonDC']

#############################################################################
# Main variable settings

# Quick Test: prototyping
print('Quick testing?', args.quick_test)
if(args.quick_test): 
    NUM_SEEDS = 2
else: 
    NUM_SEEDS = 30
print('NUM_SEEDS: ', NUM_SEEDS)
STARTING_SEED = range(NUM_SEEDS)

file_path = os.path.join(saveroot, 'SEIR_at_30d.npy')
print('File path: ',file_path)

###############################################################################
# Functions

def run_simulation(starting_seed, num_seeds, vaccination_vector, protection_rate=1):
    m = disease_model.Model(starting_seed=starting_seed,
                            num_seeds=num_seeds,
                            debug=False,clip_poisson_approximation=True,ipf_final_match='poi',ipf_num_iter=100)

    m.init_exogenous_variables(poi_areas=poi_areas,
                               poi_dwell_time_correction_factors=poi_dwell_time_correction_factors,
                               cbg_sizes=cbg_sizes,
                               poi_cbg_visits_list=truncated_list,#poi_cbg_visits_list,
                               all_hours=all_hours,
                               p_sick_at_t0=constants.parameters_dict[MSA_NAME][0],
                               vaccination_time=24*31, # when to apply vaccination (which hour)
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

    S,E,I,R = m.simulate_disease_spread(no_print=True)    
    print('S:',S,'\nE:',E,'\nI:',I,'\nR:',R)
    return S,E,I,R


###############################################################################
result_dict = dict()

for MSA_NAME in MSA_NAME_LIST:
    print('MSA_NAME: ',MSA_NAME)
    MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME] 

    # Load Data
    all_hours = functions.list_hours_in_range(MIN_DATETIME, MAX_DATETIME)
    print('len(all_hours):',len(all_hours))#;pdb.set_trace()

    # Load POI-CBG visiting matrices
    f = open(os.path.join(dataroot, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
    poi_cbg_visits_list = pickle.load(f)
    f.close()
    # truncated list
    truncated_list = []
    for i in range(len(all_hours)):
        truncated_list.append(poi_cbg_visits_list[i])
    print('len(truncated_list):',len(truncated_list))
    del poi_cbg_visits_list

    # Load precomputed parameters to adjust(clip) POI dwell times
    d = pd.read_csv(os.path.join(dataroot, 'parameters_%s.csv' % MSA_NAME)) 
    poi_areas = d['feet'].values#面积
    poi_dwell_times = d['median'].values#平均逗留时间
    poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
    del d
    poi_trans_rate = constants.parameters_dict[MSA_NAME][2] / poi_areas * poi_dwell_time_correction_factors

    # Load ACS Data for MSA-county matching
    acs_data = pd.read_csv(os.path.join(dataroot,'list1.csv'),header=2)
    acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
    msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
    msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
    msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
    good_list = list(msa_data['FIPS Code'].values)
    print('Counties included: ', good_list)
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
    #print('Number of CBGs in this metro area:', len(idxs_msa_all))
    #print('Number of CBGs in to compare with NYT data:', len(idxs_msa_nyt))

    nyt_included = np.zeros(len(idxs_msa_all))
    for i in range(len(nyt_included)):
        if(i in idxs_msa_nyt):
            nyt_included[i] = 1
    cbg_age_msa['NYT_Included'] = nyt_included.copy()

    ###############################################################################
    # Load and scale age-aware CBG-specific death rates (original)

    cbg_death_rates_original = np.loadtxt(os.path.join(dataroot, 'cbg_death_rates_original_'+MSA_NAME))
    cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)

    # The scaling factors are set according to a grid search
    attack_scale = 1
    cbg_attack_rates_scaled = cbg_attack_rates_original * attack_scale
    cbg_death_rates_scaled = cbg_death_rates_original * constants.death_scale_dict[MSA_NAME]
    print('Age-aware CBG-specific death rates scaled.')

    cbg_age_msa['Death_Rate'] =  cbg_death_rates_scaled
    
    # Construct the vaccination vector (no_vaccination)
    vaccination_vector_no_vaccination = np.zeros(len(cbg_sizes))
    # Run a simulation to determine the most vulnerable group
    S,E,I,R = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                             vaccination_vector=vaccination_vector_no_vaccination,
                             protection_rate = PROTECTION_RATE)
                                               

    result_dict[MSA_NAME] = {'S':S,'E':E,'I':I,'R':R}
    print('Current result_dict:', result_dict)

    if(args.save_result):
        np.save(file_path, result_dict)
    