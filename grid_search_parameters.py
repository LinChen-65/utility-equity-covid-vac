# python grid_search_parameters.py --msa_name Atlanta

import argparse
import os
import datetime
import pandas as pd
import numpy as np
import pickle
import time

import constants
import functions
import disease_model_original as disease_model

from math import sqrt
from sklearn.metrics import mean_squared_error

# root
root = os.getcwd()
dataroot = os.path.join(root, 'data')
print(root)

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--msa_name', 
                    help='MSA name.')
parser.add_argument('--safegraph_root', default=dataroot, 
                    help='Safegraph data root.')  
parser.add_argument('--save_result', default=False, action='store_true',
                    help='If true, save simulation results.')  
args = parser.parse_args()

MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
min_datetime=MIN_DATETIME
max_datetime=MAX_DATETIME

############################################################
# Main variable settings

MSA_NAME = args.msa_name; print('MSA_NAME: ',MSA_NAME) #MSA_NAME = 'SanFrancisco'
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME] #MSA_NAME_FULL = 'San_Francisco_Oakland_Hayward_CA'

# how_to_select_best_grid_search_models = ['cases','cases_smooth','deaths','deaths_smooth]
how_to_select_best_grid_search_models = 'cases'

# Parameters to experiment
NUM_SEEDS = 30
p_sick_at_t0_list = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
home_beta_list = np.linspace(constants.BETA_AND_PSI_PLAUSIBLE_RANGE['min_home_beta'],constants.BETA_AND_PSI_PLAUSIBLE_RANGE['max_home_beta'], 10)
poi_psi_list = np.linspace(constants.BETA_AND_PSI_PLAUSIBLE_RANGE['min_poi_psi'], constants.BETA_AND_PSI_PLAUSIBLE_RANGE['max_poi_psi'], 15)

STARTING_SEED = range(NUM_SEEDS)

############################################################
# functions

def match_msa_name_to_msas_in_acs_data(msa_name, acs_msas):
    msa_pieces = msa_name.split('_')
    query_states = set()
    i = len(msa_pieces) - 1
    while True:
        piece = msa_pieces[i]
        if len(piece) == 2 and piece.upper() == piece:
            query_states.add(piece)
            i -= 1
        else:
            break
    query_cities = set(msa_pieces[:i+1])

    for msa in acs_msas:
        if ', ' in msa:
            city_string, state_string = msa.split(', ')
            states = set(state_string.split('-'))
            if states == query_states:
                cities = city_string.split('-')
                overlap = set(cities).intersection(query_cities)
                if len(overlap) > 0:  # same states and at least one city matched
                    return msa
    return None


def get_fips_codes_from_state_and_county_fp(state, county):
    state = str(int(state))
    county = str(int(county))
    if len(state) == 1:
        state = '0' + state
    if len(county) == 1:
        county = '00' + county
    elif len(county) == 2:
        county = '0' + county
    return int(state + county)
    
# Average history records across random seeds
def average_across_random_seeds(history_C2, history_D2, num_cbgs, cbg_idxs, print_results=False):
    num_days = len(history_C2)
    
    # Average history records across random seeds
    avg_history_C2 = np.zeros((num_days,num_cbgs))
    avg_history_D2 = np.zeros((num_days,num_cbgs))
    for i in range(num_days):
        avg_history_C2[i] = np.mean(history_C2[i],axis=0)
        avg_history_D2[i] = np.mean(history_D2[i],axis=0)
    
    # Extract lines corresponding to CBGs in the metro area/county
    cases_msa = np.zeros(num_days)
    deaths_msa = np.zeros(num_days)
    for i in range(num_days):
        for j in cbg_idxs:
            cases_msa[i] += avg_history_C2[i][j]
            deaths_msa[i] += avg_history_D2[i][j]
            
    if(print_results==True):
        print('Cases: ',cases_msa)
        print('Deaths: ',deaths_msa)
    
    return avg_history_C2, avg_history_D2,cases_msa,deaths_msa


def apply_smoothing(x, agg_func=np.mean, before=3, after=3):
    new_x = []
    for i, x_point in enumerate(x):
        before_idx = max(0, i-before)
        after_idx = min(len(x), i+after+1)
        new_x.append(agg_func(x[before_idx:after_idx]))
    return np.array(new_x)

############################################################
# Load Data

# Load POI-CBG visiting matrices
f = open(os.path.join(root, 'data', '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()

# Load precomputed parameters to adjust(clip) POI dwell times
d = pd.read_csv(os.path.join(root,'data', 'parameters_%s.csv' % MSA_NAME)) 
all_hours = functions.list_hours_in_range(min_datetime, max_datetime)
poi_areas = d['feet'].values    #Area
poi_dwell_times = d['median'].values    #Average Dwell Time
poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2

# Load ACS Data for MSA-county matching
acs_data = pd.read_csv(os.path.join(root,'data', 'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
msa_match = match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
msa_data['FIPS Code'] = msa_data.apply(lambda x : get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
good_list = list(msa_data['FIPS Code'].values)
print(good_list)

# Load CBG ids for the MSA
cbg_ids_msa = pd.read_csv(os.path.join(root,'data','%s_cbg_ids.csv'%MSA_NAME_FULL)) 
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
cbg_agesex_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
cbg_age_msa = cbg_agesex_msa.copy()

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

# Deal with CBGs with 0 populations
print(cbg_age_msa[cbg_age_msa['Sum']==0]['census_block_group'])
cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
M = len(cbg_age_msa)

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
        
idxs_msa = list(x.values())
idxs_county = y
print('Number of CBGs in this metro area:', len(idxs_msa))
print('Number of CBGs in to compare with NYT data:', len(idxs_county))

# Load ground truth: NYT Data
nyt_data = pd.read_csv(os.path.join(root, 'data', 'us-counties.csv'))
nyt_data['in_msa'] = nyt_data.apply(lambda x : x['fips'] in good_list , axis=1)
nyt_data_msa = nyt_data[nyt_data['in_msa']==True].copy()
# Extract data according to simulation time range
nyt_data_msa['in_simu_period'] = nyt_data_msa['date'].apply(lambda x : True if (x<'2020-05-10') & (x>'2020-03-07') else False)
nyt_data_msa_in_simu_period = nyt_data_msa[nyt_data_msa['in_simu_period']==True].copy() 
nyt_data_msa_in_simu_period.reset_index(inplace=True)
# Group by date
nyt_data_group = nyt_data_msa_in_simu_period.groupby(nyt_data_msa_in_simu_period["date"])
# Sum up cases/deaths from different counties
# Cumulative
nyt_data_cumulative = nyt_data_group.sum()[['cases','deaths']]

# From cumulative to daily
# Cases
cases_daily = [0]
for i in range(1,len(nyt_data_cumulative)):
    cases_daily.append(nyt_data_cumulative['cases'].values[i]-nyt_data_cumulative['cases'].values[i-1])
# Smoothed ground truth
cases_daily_smooth = apply_smoothing(cases_daily, agg_func=np.mean, before=3, after=3)

# Deaths
deaths_daily = [0]
for i in range(1,len(nyt_data_cumulative)):
    deaths_daily.append(nyt_data_cumulative['deaths'].values[i]-nyt_data_cumulative['deaths'].values[i-1])
# Smoothed ground truth
deaths_daily_smooth = apply_smoothing(deaths_daily, agg_func=np.mean, before=3, after=3)

# Initialization: only need to be performed once    
m = disease_model.Model(starting_seed=STARTING_SEED,
                 num_seeds=NUM_SEEDS,
                 debug=False,
                 clip_poisson_approximation=True,
                 ipf_final_match='poi',
                 ipf_num_iter=100)

rmse_dict_cases_agnostic = dict()
rmse_dict_cases_smooth_agnostic = dict()
rmse_dict_deaths_agnostic = dict()
rmse_dict_deaths_smooth_agnostic = dict()

# Grid search
isfirst = True
start = time.time()

for idx_p_sick_at_t0 in range(len(p_sick_at_t0_list)):
    for idx_home_beta in range(len(home_beta_list)):
        for idx_poi_psi in range(len(poi_psi_list)):
            
            p_sick_at_t0=p_sick_at_t0_list[idx_p_sick_at_t0]
            home_beta=home_beta_list[idx_home_beta]
            poi_psi=poi_psi_list[idx_poi_psi]

            print('\nCurrent parameter set: [%s,%s,%s].'%(p_sick_at_t0, home_beta, poi_psi))
            
            m.init_exogenous_variables(poi_areas=poi_areas,
                                       poi_dwell_time_correction_factors=poi_dwell_time_correction_factors,
                                       cbg_sizes=cbg_sizes,
                                       poi_cbg_visits_list=poi_cbg_visits_list,
                                       all_hours=all_hours,
                                       p_sick_at_t0=p_sick_at_t0,
                                       home_beta=home_beta,
                                       poi_psi=poi_psi,
                                       just_compute_r0=False,
                                       latency_period=96,  # 4 days
                                       infectious_period=84,  # 3.5 days
                                       confirmation_rate=.1,
                                       confirmation_lag=168,  # 7 days
                                       death_rate=.0066,
                                       death_lag=432)
            m.init_endogenous_variables()
            T1,L_1,I_1,R_1,C2,D2, history_C2, history_D2, total_affected_cbg = m.simulate_disease_spread()

            total_affected_cbg_age_agnostic = total_affected_cbg
            history_C2_age_agnostic = history_C2.copy()
            history_D2_age_agnostic = history_D2.copy()

            # Average history records across random seeds
            policy = 'Age_Agnostic'
            _, _, cases_total_age_agnostic, deaths_total_age_agnostic = average_across_random_seeds(history_C2_age_agnostic, history_D2_age_agnostic, M, idxs_county, print_results=False)

            print(cases_total_age_agnostic[-1], deaths_total_age_agnostic[-1])
            cases_total_age_agnostic_final = cases_total_age_agnostic[-1]
            deaths_total_age_agnostic_final = deaths_total_age_agnostic[-1]
            
            # From cumulative to daily
            cases_daily_total_age_agnostic = [0]
            for i in range(1,len(cases_total_age_agnostic)):
                cases_daily_total_age_agnostic.append(cases_total_age_agnostic[i]-cases_total_age_agnostic[i-1])
                
            deaths_daily_total_age_agnostic = [0]
            for i in range(1,len(deaths_total_age_agnostic)):
                deaths_daily_total_age_agnostic.append(deaths_total_age_agnostic[i]-deaths_total_age_agnostic[i-1])
                
            cases = nyt_data_cumulative['cases'].values
            cases_smooth = apply_smoothing(cases, agg_func=np.mean, before=3, after=3)
            
            deaths = nyt_data_cumulative['deaths'].values
            deaths_smooth = apply_smoothing(deaths, agg_func=np.mean, before=3, after=3)
            
            # RMSE across random seeds
            rmse_dict_cases_agnostic['%s,%s,%s'%(p_sick_at_t0, home_beta, poi_psi)] = sqrt(mean_squared_error(cases,cases_total_age_agnostic))
            rmse_dict_cases_smooth_agnostic['%s,%s,%s'%(p_sick_at_t0, home_beta, poi_psi)] = sqrt(mean_squared_error(cases_smooth,cases_total_age_agnostic))
            rmse_dict_deaths_agnostic['%s,%s,%s'%(p_sick_at_t0, home_beta, poi_psi)] = sqrt(mean_squared_error(deaths,deaths_total_age_agnostic))
            rmse_dict_deaths_smooth_agnostic['%s,%s,%s'%(p_sick_at_t0, home_beta, poi_psi)] = sqrt(mean_squared_error(deaths_smooth,deaths_total_age_agnostic))
            
            if(how_to_select_best_grid_search_models == 'cases'):
                if(isfirst==True):
                    best_rmse = sqrt(mean_squared_error(cases,cases_total_age_agnostic))
                    best_parameters = [p_sick_at_t0, home_beta, poi_psi]
                    print('Current best: ', best_rmse, '\nCurrent best parameter set: [%s,%s,%s].'%(p_sick_at_t0, home_beta, poi_psi))
                else:
                    print('Current mse: ', sqrt(mean_squared_error(cases,cases_total_age_agnostic)))
                    print('Previous best: ',best_rmse)
                    if(best_rmse > sqrt(mean_squared_error(cases,cases_total_age_agnostic))):
                        best_rmse = sqrt(mean_squared_error(cases,cases_total_age_agnostic))
                        best_parameters = [p_sick_at_t0, home_beta, poi_psi]
                        print('Current best: ', best_rmse, '\nCurrent best parameter set: [%s,%s,%s].'%(p_sick_at_t0, home_beta, poi_psi))
                    else:
                        print('Current best not changed. \nCurrent best parameter set:',best_parameters)
            
            isfirst = False
            
            # Save rmse dicts
            if(args.save_result):
                np.save(os.path.join(root,'results', '20210127_rmse_cases_%s_%s'%(MSA_NAME,p_sick_at_t0)),rmse_dict_cases_agnostic)
                np.save(os.path.join(root,'results', '20210127_rmse_cases_smooth_%s_%s'%(MSA_NAME,p_sick_at_t0)),rmse_dict_cases_smooth_agnostic)
                np.save(os.path.join(root,'results', '20210127_rmse_deaths_%s_%s'%(MSA_NAME,p_sick_at_t0)),rmse_dict_deaths_agnostic)
                np.save(os.path.join(root,'results', '20210127_rmse_deaths_smooth_%s_%s'%(MSA_NAME,p_sick_at_t0)),rmse_dict_deaths_smooth_agnostic)
                
                # Save best results
                best_results = dict()
                best_results['rmse'] = best_rmse
                best_results['parameters'] = best_parameters
                np.save(os.path.join(root,'results', '20210127_best_results_%s_%s_%s'%(how_to_select_best_grid_search_models,MSA_NAME,p_sick_at_t0)),best_results)

end = time.time()
print('Total Time:',(end-start))

