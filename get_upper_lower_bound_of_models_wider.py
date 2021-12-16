# python get_upper_lower_bound_of_models_wider.py MSA_NAME quick_test direction tolerance
# python get_upper_lower_bound_of_models_wider.py Atlanta False lower 1.5

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import sys

import os
import datetime
import pandas as pd
import numpy as np
import pickle

import constants
import functions
import disease_model #disease_model_only_modify_attack_rates

from math import sqrt
from sklearn.metrics import mean_squared_error
import time
import pdb


##############################################################################
# Constants

root = '/data/chenlin/COVID-19/Data'

datestring = '20210206'

MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
min_datetime = MIN_DATETIME
max_datetime = MAX_DATETIME

NUM_DAYS = 63

###############################################################################
# Main variable settings

MSA_NAME = sys.argv[1]; print('MSA_NAME: ',MSA_NAME)
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME]

# Quick Test: prototyping
quick_test = sys.argv[2]

# Vaccination_Ratio
VACCINATION_RATIO = 0.1

# Vaccination protection rate
PROTECTION_RATE = 1

# Policy execution ratio
EXECUTION_RATIO = 1

if(quick_test == True):
    print('Quick testing.')
    NUM_SEEDS = 2
else:
    NUM_SEEDS = 30

STARTING_SEED = range(NUM_SEEDS)
print('NUM_SEEDS: ', NUM_SEEDS)

# Upper bound or lower bound
direction = sys.argv[3]
print('direction:',direction)

# RMSE tolerance
tolerance = float(sys.argv[4])
print('tolerance:',tolerance)

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
    
    
    #return total_affected, history_C2, history_D2, total_affected_each_cbg
    return history_C2, history_D2

###############################################################################
# Load Data

print('Start loading data...')

start = time.time()

# Load POI-CBG visiting matrices
f = open(os.path.join(root, MSA_NAME, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()

# Load precomputed parameters to adjust(clip) POI dwell times
d = pd.read_csv(os.path.join(root,MSA_NAME, 'parameters_%s.csv' % MSA_NAME)) 

# No clipping
new_d = d

all_hours = functions.list_hours_in_range(min_datetime, max_datetime)
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

# Rename
cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
# Extract columns of interest
columns_of_interest = ['census_block_group','Sum']
cbg_age_msa = cbg_age_msa[columns_of_interest].copy()
# Deal with CBGs with 0 populations
cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
M = len(cbg_age_msa)

# Obtain cbg sizes (populations)
cbg_sizes = cbg_age_msa['Sum'].values
cbg_sizes = np.array(cbg_sizes,dtype='int32')
print('Total population: ',np.sum(cbg_sizes))
del cbg_age_msa

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

end=time.time()
print('Data loaded, time: ', (end-start))

###############################################################################
# Best model

policy_list = ['Age_Agnostic','No_Vaccination']

for policy in policy_list:
    policy = policy.lower()
    exec('history_D2_%s = np.fromfile(os.path.join(root,MSA_NAME,\'vaccination_results_adaptive_0.1_0.01\',\'%s_history_D2_%s_adaptive_0.1_0.01_%sseeds_%s\'))' % (policy,datestring,policy,NUM_SEEDS,MSA_NAME))
    exec('history_D2_%s = np.reshape(history_D2_%s,(63,NUM_SEEDS,M))'%(policy,policy))
print(history_D2_no_vaccination.shape)

# Average across random seeds
for policy in policy_list:
    policy = policy.lower()
    exec('deaths_cbg_%s, deaths_total_%s = functions.average_across_random_seeds_only_death(history_D2_%s, M, idxs_msa_nyt, print_results=False,draw_results=False)'
         %(policy,policy,policy))
print(deaths_cbg_no_vaccination.shape)

###############################################################################
# Load and scale age-aware CBG-specific attack/death rates (original)

cbg_death_rates_original = np.loadtxt(os.path.join(root, MSA_NAME, 'cbg_death_rates_original_'+MSA_NAME))
cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)
print('Age-aware CBG-specific death rates loaded. Attack rates are irrelevant to age.')

# The scaling factors are set according to a grid search
# Fix attack_scale
attack_scale = 1
cbg_attack_rates_original_scaled = cbg_attack_rates_original * attack_scale

###############################################################################
# Ground truth: NYT Data

nyt_data = pd.read_csv(os.path.join(root, 'us-counties.csv'))

nyt_data['in_msa'] = nyt_data.apply(lambda x : x['fips'] in good_list , axis=1)
nyt_data_msa = nyt_data[nyt_data['in_msa']==True].copy()
# 只取出模拟期间的真实值
# 0310-0509 -> 0308-0509
nyt_data_msa['in_simu_period'] = nyt_data_msa['date'].apply(lambda x : True if (x<'2020-05-10') & (x>'2020-03-07') else False)
nyt_data_msa_in_simu_period = nyt_data_msa[nyt_data_msa['in_simu_period']==True].copy() 
nyt_data_msa_in_simu_period.reset_index(inplace=True)
# Group by date
nyt_data_group = nyt_data_msa_in_simu_period.groupby(nyt_data_msa_in_simu_period["date"])
# Sum up cases/deaths from different counties
nyt_data_cumulative = nyt_data_group.sum()[['cases','deaths']]

# From cumulative to daily
# Deaths
deaths_daily = [0]
for i in range(1,len(nyt_data_cumulative)):
    deaths_daily.append(nyt_data_cumulative['deaths'].values[i]-nyt_data_cumulative['deaths'].values[i-1])
# Smoothed ground truth
deaths_daily_smooth = functions.apply_smoothing(deaths_daily, agg_func=np.mean, before=3, after=3)
if(len(deaths_daily_smooth)<len(deaths_total_no_vaccination)):
    deaths_daily_smooth = [0]*(len(deaths_total_no_vaccination)-len(deaths_daily_smooth)) + list(deaths_daily_smooth)

# Save results
'''
np.save(os.path.join(root,MSA_NAME,'20210206_deaths_total_no_vaccination_%s.npy'%MSA_NAME), deaths_total_no_vaccination)
np.save(os.path.join(root,MSA_NAME,'20210206_deaths_cbg_no_vaccination_%s.npy'%MSA_NAME), deaths_cbg_no_vaccination)
np.save(os.path.join(root,MSA_NAME,'20210206_deaths_total_age_agnostic_%s.npy'%MSA_NAME), deaths_total_age_agnostic)
np.save(os.path.join(root,MSA_NAME,'20210206_deaths_cbg_age_agnostic_%s.npy'%MSA_NAME), deaths_cbg_age_agnostic)

np.save(os.path.join(root,MSA_NAME,'20210206_deaths_daily_nyt_%s.npy'%MSA_NAME),deaths_daily)
np.save(os.path.join(root,MSA_NAME,'20210206_deaths_daily_smooth_nyt_%s.npy'%MSA_NAME),deaths_daily_smooth)
'''

# From cumulative to daily
deaths_daily_no_vaccination = [0]
for i in range(1,len(deaths_total_no_vaccination)):
    deaths_daily_no_vaccination.append(deaths_total_no_vaccination[i]-deaths_total_no_vaccination[i-1])
# Best RMSE
best_rmse = sqrt(mean_squared_error(deaths_daily_smooth,deaths_daily_no_vaccination))
print('best_rmse:',best_rmse)


###############################################################################
# Best model
'''
cbg_death_rates_original_scaled = cbg_death_rates_original * death_scale

# Construct the vaccination vector
vaccination_vector_no_vaccination = np.zeros(len(cbg_sizes))
# Run simulations
history_C2_no_vaccination, history_D2_no_vaccination = run_simulation(starting_seed=STARTING_SEED, 
                                                                      num_seeds=NUM_SEEDS, 
                                                                      vaccination_vector=vaccination_vector_no_vaccination,
                                                                      protection_rate = PROTECTION_RATE)

# Average history records across random seeds
policy = 'No_Vaccination'
_, deaths_cbg_no_vaccination, _, deaths_total_no_vaccination = functions.average_across_random_seeds(
                                                                                                     history_C2_no_vaccination, 
                                                                                                     history_D2_no_vaccination, 
                                                                                                     M, idxs_msa_nyt, 
                                                                                                     print_results=False,
                                                                                                     draw_results=False)

# From cumulative to daily
deaths_daily_no_vaccination = [0]
for i in range(1,len(deaths_total_no_vaccination)):
    deaths_daily_no_vaccination.append(deaths_total_no_vaccination[i]-deaths_total_no_vaccination[i-1])

# Best RMSE
deaths_daily_rmse_no_vaccination = sqrt(mean_squared_error(deaths_daily_smooth,deaths_daily_no_vaccination))

'''

###############################################################################

# RMSE within (1+20%) of the best model (tolerance=1.2)
scales = dict()
scales['max'] = 1.5
scales[1.2] = {
                'Atlanta':[1.01,1.36],
                'Chicago':[1.15,1.45],
                'Dallas':[0.9,1.15],
                'Houston':[0.68,0.97],
                'LosAngeles':[1.34,1.68],
                'Miami':[0.65,0.93],
                'Philadelphia':[1.48,2.63],
                'SanFrancisco':[0.54,0.78],
                'WashingtonDC':[1.23,1.54]
}

scales[1.5] = {
                'Atlanta':[0.9,1.48],
                'Chicago':[1.05,1.55],
                'Dallas':[0.82,1.24],
                'Houston':[0.58,1.07],
                'LosAngeles':[1.23,1.79],
                'Miami':[0.55,1.03],
                'Philadelphia':[1.1,3.02],
                'SanFrancisco':[0.45,0.86],
                'WashingtonDC':[1.14,1.63]


if(tolerance in scales.keys()): # 结果已经跑出来，这次只需要得到30次simulation的具体结果(划掉，还是取平均比较顺滑)
    start = time.time()
    
    # No_Vaccination: Construct the vaccination vector
    vaccination_vector_no_vaccination = np.zeros(len(cbg_sizes))
    
    death_scale = scales[MSA_NAME][0]
    while(death_scale<=scales[MSA_NAME][1]):
        print('Current death scale:',death_scale)
        cbg_death_rates_original_scaled = cbg_death_rates_original * death_scale
        # Run simulations
        _, history_D2_no_vaccination_all = run_simulation(starting_seed=STARTING_SEED, 
                                                      num_seeds=NUM_SEEDS, 
                                                      vaccination_vector=vaccination_vector_no_vaccination,
                                                      protection_rate = PROTECTION_RATE)
        '''
        history_D2_no_vaccination_all = np.array(history_D2_no_vaccination_all)
        # Extract lines corresponding to CBGs in the metro area/county # retrieved from 'make_dict...py'
        history_D2_no_vaccination= np.zeros((NUM_DAYS,NUM_SEEDS))
        for i in range(NUM_SEEDS):
            for j in range(NUM_DAYS):
                for k in idxs_msa_nyt:
                    history_D2_no_vaccination[j][i] += history_D2_no_vaccination_all[j][i][k]
        #print('history_D2_no_vaccination_all.shape:',history_D2_no_vaccination_all.shape)
        #print('history_D2_no_vaccination.shape:',history_D2_no_vaccination.shape)
        '''
        # Average history records across random seeds
        history_D2_no_vaccination, _ = functions.average_across_random_seeds_only_death(history_D2_most_vulner, 
                                                                                        M, idxs_msa_nyt, 
                                                                                        print_results=False,
                                                                                        draw_results=False)
        print('history_D2_no_vaccination.shape:',history_D2_no_vaccination.shape)                                                                                    
        if(death_scale==scales[MSA_NAME][0]):
            age_aware_history_D2_all = history_D2_no_vaccination.copy()
        else:
            age_aware_history_D2_all = np.concatenate((age_aware_history_D2_all, history_D2_no_vaccination),axis=1)
            print('age_aware_history_D2_all.shape:',age_aware_history_D2_all.shape)
            #pdb.set_trace()
        
        #np.save(os.path.join(root,MSA_NAME,'age_aware_history_D2_all_%s.npy'%(MSA_NAME)), age_aware_history_D2_all)
        np.save(os.path.join(root,MSA_NAME,'avg_age_aware_history_D2_all_%s.npy'%(MSA_NAME)), age_aware_history_D2_all)

        death_scale += 0.01
        death_scale = np.around(death_scale,2)
    
    end = time.time()
    print('Time:', end-start)

    
else:
    start = time.time()

    # No_Vaccination: Construct the vaccination vector
    vaccination_vector_no_vaccination = np.zeros(len(cbg_sizes))
    #scale_bound = [constants.death_scale_dict[MSA_NAME][0]*(1-0.1), constants.death_scale_dict[MSA_NAME][0]*(1+0.1)]

    if(tolerance>scales['max']):
        print('Expand from %s'%scales['max'])
        if(direction=='lower'):
            death_scale = scales[scales['max']][MSA_NAME][0]
        elif(direction=='upper'):
            death_scale = scales[scales['max']][MSA_NAME][1]
    else:    
        print('Expand from best result.')
        death_scale = constants.death_scale_dict[MSA_NAME][0]
    print('Starting death_scale: ',death_scale)

    while(True):
        print('Current death scale:',death_scale)
        cbg_death_rates_original_scaled = cbg_death_rates_original * death_scale
        print('Age-aware CBG-specific death rates scaled.')
        # Run simulations
        history_C2_no_vaccination, history_D2_no_vaccination = run_simulation(starting_seed=STARTING_SEED, 
                                                                              num_seeds=NUM_SEEDS, 
                                                                              vaccination_vector=vaccination_vector_no_vaccination,
                                                                              protection_rate = PROTECTION_RATE)

        # Average history records across random seeds
        policy = 'No_Vaccination'
        _, deaths_cbg_no_vaccination, _, deaths_total_no_vaccination = functions.average_across_random_seeds(
                                                                                                             history_C2_no_vaccination, 
                                                                                                             history_D2_no_vaccination, 
                                                                                                             M, idxs_msa_nyt, 
                                                                                                             print_results=False,
                                                                                                             draw_results=False)
        
        # From cumulative to daily
        deaths_daily_no_vaccination = [0]
        for i in range(1,len(deaths_total_no_vaccination)):
            deaths_daily_no_vaccination.append(deaths_total_no_vaccination[i]-deaths_total_no_vaccination[i-1])
        # RMSE
        this_rmse = sqrt(mean_squared_error(deaths_daily_smooth,deaths_daily_no_vaccination))
        
        print('best_scale:',constants.death_scale_dict[MSA_NAME][0],',best_rmse:',best_rmse)
        print('%s * best_rmse:'%tolerance,best_rmse*tolerance)
        print('this_scale:',death_scale,',this_rmse:',this_rmse)
        if(this_rmse<=best_rmse*tolerance):
            if(direction=='lower'):
                death_scale -= 0.01
            elif(direction=='upper'):
                death_scale += 0.01
            death_scale = np.around(death_scale,2)
        else:
            print('death_scale:',death_scale)
            print('If proceed, results will be saved.')
            pdb.set_trace()
            break
        
    end = time.time()
    print('Time:', end-start)

    #np.save(os.path.join(root,MSA_NAME,'age_aware_20percent_%sbound_%s_%s.npy'%(direction, death_scale, MSA_NAME)), deaths_total_no_vaccination)
    np.save(os.path.join(root,MSA_NAME,'age_aware_%s_%sbound_%s_%s.npy'%(tolerance, direction, death_scale, MSA_NAME)), deaths_total_no_vaccination)
