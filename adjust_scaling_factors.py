# -*- coding: utf-8 -*-
# python adjust_scaling_factors.py --msa_name Atlanta --quick_test

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import os
import datetime
import pandas as pd
import numpy as np
import pickle

import constants
import functions
#import disease_model_original
import disease_model_only_modify_attack_rates as disease_model
#import disease_model_test as disease_model

from math import sqrt
from sklearn.metrics import mean_squared_error

import time

#root = '/data/chenlin/COVID-19/Data'
root = os.getcwd()
dataroot = os.path.join(root, 'data')
print(root)


# Parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--msa_name', 
                    help='MSA name.')
parser.add_argument('--safegraph_root', default=dataroot, #'/data/chenlin/COVID-19/Data',
                    help='Safegraph data root.')
parser.add_argument('--quick_test', default=False, action='store_true',
                    help='If true, reduce number of simulations to test quickly.')
parser.add_argument('--save_result', default=False, action='store_true',
                    help='If true, save simulation results.')  
args = parser.parse_args()




MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
NUM_DAYS = 63

###############################################################################
# Main variable settings
MSA_NAME = args.msa_name; print('MSA_NAME: ',MSA_NAME)
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME]

# Selection criteria
#how_to_select_best_grid_search_models = 'deaths'
how_to_select_best_grid_search_models = 'deaths_daily_smooth'

if(args.quick_test):
    print('Quick testing.')
    NUM_SEEDS = 2
    attack_scale_list = np.arange(15, 16, 0.5)
    death_scale_list = np.arange(0.5, 0.6, 0.05)
else:
    NUM_SEEDS = 60
    #attack_scale_list = np.arange(15, 26, 0.1)
    if(MSA_NAME=='LosAngeles'):
        death_scale_list = [1.52]
    else:
        #death_scale_list = np.arange(0.5, 1.6, 0.01)
        death_scale_list = np.arange(0.6, 3.0, 0.01)
    

STARTING_SEED = range(NUM_SEEDS)
print('NUM_SEEDS: ', NUM_SEEDS)

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
    
    return total_affected, history_C2, history_D2, total_affected_each_cbg


############################################################
# Load Data

# Load POI-CBG visiting matrices
f = open(os.path.join(root, 'data', '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()

# Load precomputed parameters to adjust(clip) POI dwell times
d = pd.read_csv(os.path.join(root,'data', 'parameters_%s.csv' % MSA_NAME)) 
all_hours = functions.list_hours_in_range(MIN_DATETIME, MAX_DATETIME)
poi_areas = d['feet'].values#面积
poi_dwell_times = d['median'].values#平均逗留时间
poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
del d

# Load ACS Data for MSA-county matching
acs_data = pd.read_csv(os.path.join(root, 'data', 'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
good_list = list(msa_data['FIPS Code'].values)
print('CBG included: ',good_list)
del acs_data

# Load CBG ids for the MSA
cbg_ids_msa = pd.read_csv(os.path.join(root, 'data','%s_cbg_ids.csv'%MSA_NAME_FULL)) 
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
        
idxs_msa_all = list(x.values())
idxs_msa_nyt = y
print('Number of CBGs in this metro area:', len(idxs_msa_all))
print('Number of CBGs in to compare with NYT data:', len(idxs_msa_nyt))

# Load ground truth: NYT Data
nyt_data = pd.read_csv(os.path.join(root, 'data', 'us-counties.csv'))
nyt_data['in_msa'] = nyt_data.apply(lambda x : x['fips'] in good_list , axis=1)
nyt_data_msa = nyt_data[nyt_data['in_msa']==True].copy()
del nyt_data
# Extract data according to simulation time range
nyt_data_msa['in_simu_period'] = nyt_data_msa['date'].apply(lambda x : True if (x<'2020-05-10') & (x>'2020-03-07') else False)
nyt_data_msa_in_simu_period = nyt_data_msa[nyt_data_msa['in_simu_period']==True].copy() 
nyt_data_msa_in_simu_period.reset_index(inplace=True)
del nyt_data_msa
# Group by date
nyt_data_group = nyt_data_msa_in_simu_period.groupby(nyt_data_msa_in_simu_period["date"])
# Sum up cases/deaths from different counties
# Cumulative
nyt_data_cumulative = nyt_data_group.sum()[['cases','deaths']]

# NYT data: Accumulated cases and deaths
cases = nyt_data_cumulative['cases'].values
if(len(cases)<NUM_DAYS):
    cases = [0]*(NUM_DAYS-len(cases)) + list(cases)
cases_smooth = functions.apply_smoothing(cases, agg_func=np.mean, before=3, after=3)

deaths = nyt_data_cumulative['deaths'].values
if(len(deaths)<NUM_DAYS):
    deaths = [0]*(NUM_DAYS-len(deaths)) + list(deaths)
deaths_smooth = functions.apply_smoothing(deaths, agg_func=np.mean, before=3, after=3)



# NYT data: From cumulative to daily
# Cases
cases_daily = [0]
for i in range(1,len(nyt_data_cumulative)):
    cases_daily.append(nyt_data_cumulative['cases'].values[i]-nyt_data_cumulative['cases'].values[i-1])
if(len(cases_daily)<NUM_DAYS):
    cases_daily = [0]*(NUM_DAYS-len(cases_daily)) + list(cases_daily)
# Smoothed ground truth
cases_daily_smooth = functions.apply_smoothing(cases_daily, agg_func=np.mean, before=3, after=3)


# Deaths
deaths_daily = [0]
for i in range(1,len(nyt_data_cumulative)):
    deaths_daily.append(nyt_data_cumulative['deaths'].values[i]-nyt_data_cumulative['deaths'].values[i-1])
if(len(deaths_daily)<NUM_DAYS):
    deaths_daily = [0]*(NUM_DAYS-len(deaths_daily)) + list(deaths_daily)
# Smoothed ground truth
deaths_daily_smooth = functions.apply_smoothing(deaths_daily, agg_func=np.mean, before=3, after=3)



###############################################################################
# Load age-aware CBG-specific death rates (original)

#cbg_attack_rates_original = np.loadtxt(os.path.join(root, MSA_NAME, 'cbg_attack_rates_original_'+MSA_NAME)) # Nature Medicine, 3 groups, 0.003, 0.044, 0.084
cbg_death_rates_original = np.loadtxt(os.path.join(root, 'data', 'cbg_death_rates_original_'+MSA_NAME))
cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)

# Construct vaccination vector
vaccination_vector_no_vaccination = np.zeros(len(cbg_sizes))

###############################################################################

start = time.time()

if(how_to_select_best_grid_search_models == 'deaths_daily_smooth'):
    rmse_dict_deaths_daily = dict()
    rmse_dict_deaths_daily_smooth = dict()
if(how_to_select_best_grid_search_models == 'deaths'):
    rmse_dict_deaths = dict()
    rmse_dict_deaths_smooth = dict()

# Fix attack_scale
attack_scale = 1
cbg_attack_rates_original_scaled = cbg_attack_rates_original * attack_scale

# Calibrate the death_scale
isfirst = True
# counter for early stopping
counter = 0

# Start searching
for death_scale in death_scale_list:
    print('Current death_scale:',death_scale)
    
    # Scaling
    cbg_death_rates_original_scaled = cbg_death_rates_original * death_scale
    
    # Run simulation
    _, history_C2_no_vaccination, history_D2_no_vaccination, _ = run_simulation(starting_seed=STARTING_SEED, 
                                                                                num_seeds=NUM_SEEDS, 
                                                                                vaccination_vector=vaccination_vector_no_vaccination)
    # Average history records across random seeds
    policy = 'No_Vaccination'
    _, _, cases_total_no_vaccination, deaths_total_no_vaccination = functions.average_across_random_seeds(
                                                                                                history_C2_no_vaccination, 
                                                                                                history_D2_no_vaccination, 
                                                                                                M, idxs_msa_nyt, 
                                                                                                print_results=False,
                                                                                                )
    print('total:',cases_total_no_vaccination[-1], deaths_total_no_vaccination[-1])
    
    

    if(how_to_select_best_grid_search_models == 'deaths'):
        # RMSE across random seeds
        # Accumulated deaths
        rmse_dict_deaths['%s,%s'%(attack_scale,death_scale)] = sqrt(mean_squared_error(deaths,deaths_total_no_vaccination))
        rmse_dict_deaths_smooth['%s,%s'%(attack_scale,death_scale)] = sqrt(mean_squared_error(deaths_smooth,deaths_total_no_vaccination))
        if(isfirst==True):
            best_rmse_deaths = sqrt(mean_squared_error(deaths,deaths_total_no_vaccination))
            best_rmse_deaths_smooth = sqrt(mean_squared_error(deaths_smooth,deaths_total_no_vaccination))
            best_parameters = [attack_scale,death_scale]
            print('Current best: ', best_rmse_deaths, '\nCurrent best parameters: [%s,%s].'%(attack_scale,death_scale))
            counter = 0
        else:
            print('Current mse: ', sqrt(mean_squared_error(deaths,deaths_total_no_vaccination)))
            print('Previous best: ',best_rmse_deaths)
            if(best_rmse_deaths > sqrt(mean_squared_error(deaths,deaths_total_no_vaccination))):
                best_rmse_deaths = sqrt(mean_squared_error(deaths,deaths_total_no_vaccination))
                best_rmse_deaths_smooth = sqrt(mean_squared_error(deaths_smooth,deaths_total_no_vaccination))
                best_parameters = [attack_scale,death_scale]
                print('Current best: ', best_rmse_deaths, '\nCurrent best parameters: [%s,%s].'%(attack_scale,death_scale))
                counter = 0 # Reset counter
            else:
                print('Current best not changed. \nCurrent best parameters: [%s,%s].'%(best_parameters[0],best_parameters[1]))
                counter += 1
                if counter >= 10:
                    print('Early stopped.')
                    break
        print('Reminder: best rmse on cases: ', sqrt(mean_squared_error(cases,cases_total_no_vaccination)))
    
    if(how_to_select_best_grid_search_models == 'deaths_daily_smooth'):
        # From cumulative to daily
        cases_daily_total_no_vaccination = [0]
        for i in range(1,len(cases_total_no_vaccination)):
            cases_daily_total_no_vaccination.append(cases_total_no_vaccination[i]-cases_total_no_vaccination[i-1])
        # Deaths
        deaths_daily_total_no_vaccination = [0]
        for i in range(1,len(deaths_total_no_vaccination)):
            deaths_daily_total_no_vaccination.append(deaths_total_no_vaccination[i]-deaths_total_no_vaccination[i-1])
        # RMSE across random seeds
        rmse_dict_deaths_daily['%s,%s'%(attack_scale,death_scale)] = sqrt(mean_squared_error(deaths_daily,deaths_daily_total_no_vaccination))
        rmse_dict_deaths_daily_smooth['%s,%s'%(attack_scale,death_scale)] = sqrt(mean_squared_error(deaths_daily_smooth,deaths_daily_total_no_vaccination))
        
        if(isfirst==True):
            #best_rmse_deaths_daily = sqrt(mean_squared_error(deaths_daily,deaths_daily_total_no_vaccination))
            best_rmse_deaths_daily_smooth = sqrt(mean_squared_error(deaths_daily_smooth,deaths_daily_total_no_vaccination))
            best_parameters = [attack_scale,death_scale]
            print('Current best: ', best_rmse_deaths_daily_smooth, '\nCurrent best parameters: [%s,%s].'%(attack_scale,death_scale))
            counter = 0
        else:
            current_mse = sqrt(mean_squared_error(deaths_daily_smooth,deaths_daily_total_no_vaccination))
            print('Current mse: ', current_mse)
            print('Previous best: ',best_rmse_deaths_daily_smooth)
            if(best_rmse_deaths_daily_smooth > current_mse):
                #best_rmse_deaths_daily = sqrt(mean_squared_error(deaths,deaths_total_no_vaccination))
                best_rmse_deaths_daily_smooth = current_mse
                best_parameters = [attack_scale,death_scale]
                print('Current best: ', best_rmse_deaths_daily_smooth, '\nCurrent best parameters: [%s,%s].'%(attack_scale,death_scale))
                counter = 0 # Reset counter
            else:
                print('Current best not changed. \nCurrent best parameters: [%s,%s].'%(best_parameters[0],best_parameters[1]))
                counter += 1
                if counter >= 10:
                    print('Early stopped.')
                    break
        print('Reminder: best rmse on cases : ', 
              sqrt(mean_squared_error(cases,cases_total_no_vaccination)),
              sqrt(mean_squared_error(cases_smooth,cases_total_no_vaccination)),
              sqrt(mean_squared_error(cases_daily_smooth,cases_daily_total_no_vaccination)),
              sqrt(mean_squared_error(cases_daily_smooth,cases_daily_total_no_vaccination))
              )                
    
    isfirst = False


    if(args.save_result):
        # Save rmse dicts
        if(how_to_select_best_grid_search_models == 'deaths_daily_smooth'):
            np.save(os.path.join(root,'result', '20210205_rmse_deaths_%s_age_aware'%MSA_NAME),rmse_dict_deaths_daily)
            np.save(os.path.join(root,'result', '20210205_rmse_deaths_smooth_%s_age_aware'%MSA_NAME),rmse_dict_deaths_daily_smooth)
        if(how_to_select_best_grid_search_models == 'deaths'):
            np.save(os.path.join(root,'result', '20210205_rmse_deaths_%s_age_aware'%MSA_NAME),rmse_dict_deaths)
            np.save(os.path.join(root,'result', '20210205_rmse_deaths_smooth_%s_age_aware'%MSA_NAME),rmse_dict_deaths_smooth)
        
        # Save best results
        best_results = dict()
        best_results['parameters'] = best_parameters
        best_results['case rmse'] = sqrt(mean_squared_error(cases,cases_total_no_vaccination))
        best_results['case smooth rmse'] = sqrt(mean_squared_error(cases_smooth,cases_total_no_vaccination))
        if(how_to_select_best_grid_search_models == 'deaths_daily_smooth'):
            #best_results['daily death rmse'] = best_rmse_deaths_daily
            best_results['daily death smooth rmse'] = best_rmse_deaths_daily_smooth
        if(how_to_select_best_grid_search_models == 'deaths'):
            best_results['death rmse'] = best_rmse_deaths
            best_results['death smooth rmse'] = best_rmse_deaths_smooth

        np.save(os.path.join(root,'result', '20210205_best_results_%s_%s_only_age_aware_death_rates'%(how_to_select_best_grid_search_models,MSA_NAME)),best_results)


end = time.time()
print('Time: ', (end-start))

