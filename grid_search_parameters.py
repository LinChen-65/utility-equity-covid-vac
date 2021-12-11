# python grid_search_parameters.py MSA_NAME quick_test p_sick_at_t0
# python grid_search_parameters.py Atlanta False 5e-4

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import sys

import os
import datetime
import pandas as pd
import numpy as np
import pickle
import time

import constants
import helper
import disease_model_original

import imp
imp.reload(constants)
imp.reload(helper)
imp.reload(disease_model_original)

from math import sqrt
from sklearn.metrics import mean_squared_error

############################################################
# Constants
root = '/data/chenlin/COVID-19/Data'

MSA_NAME_LIST = ['Atlanta','Chicago','Dallas','Houston', 'LosAngeles','Miami','NewYorkCity','Philadelphia','SanFrancisco','WashingtonDC']
MSA_NAME_FULL_DICT = {
    'Atlanta':'Atlanta_Sandy_Springs_Roswell_GA',
    'Chicago':'Chicago_Naperville_Elgin_IL_IN_WI',
    'Dallas':'Dallas_Fort_Worth_Arlington_TX',
    'Houston':'Houston_The_Woodlands_Sugar_Land_TX',
    'LosAngeles':'Los_Angeles_Long_Beach_Anaheim_CA',
    'Miami':'Miami_Fort_Lauderdale_West_Palm_Beach_FL',
    'NewYorkCity':'New_York_Newark_Jersey_City_NY_NJ_PA',
    'Philadelphia':'Philadelphia_Camden_Wilmington_PA_NJ_DE_MD',
    'SanFrancisco':'San_Francisco_Oakland_Hayward_CA',
    'WashingtonDC':'Washington_Arlington_Alexandria_DC_VA_MD_WV'
}

MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
min_datetime=MIN_DATETIME
max_datetime=MAX_DATETIME

# beta_and_psi_plausible_range is output of make_param_plausibility_plot and should be updated whenever you recalibrate R0. These numbers allow R0_base to range from 0.1 - 2 and R0_PSI to range from 1-3.
BETA_AND_PSI_PLAUSIBLE_RANGE = {"min_home_beta": 0.0011982272027079982,
                                "max_home_beta": 0.023964544054159966,
                                "max_poi_psi": 4886.41659532027,
                                "min_poi_psi": 515.4024854336667}


print('Constants loaded.')

############################################################
# Main variable settings

MSA_NAME = sys.argv[1]; print('MSA_NAME: ',MSA_NAME)
MSA_NAME_FULL = MSA_NAME_FULL_DICT[MSA_NAME]
#MSA_NAME = 'SanFrancisco'
#MSA_NAME_FULL = 'San_Francisco_Oakland_Hayward_CA'

# how_to_select_best_grid_search_models = ['cases','cases_smooth','deaths','deaths_smooth]
how_to_select_best_grid_search_models = 'cases'

# Quick Test: prototyping
quick_test = sys.argv[2]
#quick_test = False

# Which part of parameters to test (ranging from 1 to 10)
p_sick_at_t0 = sys.argv[3]

# Parameters to experiment
if(quick_test == True):
    NUM_SEEDS = 2
    p_sick_at_t0_list = [1e-2, 5e-3]
    home_beta_list = np.linspace(BETA_AND_PSI_PLAUSIBLE_RANGE['min_home_beta'],BETA_AND_PSI_PLAUSIBLE_RANGE['max_home_beta'], 2)
    poi_psi_list = np.linspace(BETA_AND_PSI_PLAUSIBLE_RANGE['min_poi_psi'], BETA_AND_PSI_PLAUSIBLE_RANGE['max_poi_psi'], 2)
else:
    NUM_SEEDS = 30
    #p_sick_at_t0_list = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
    p_sick_at_t0_list = [p_sick_at_t0]
    home_beta_list = np.linspace(BETA_AND_PSI_PLAUSIBLE_RANGE['min_home_beta'],BETA_AND_PSI_PLAUSIBLE_RANGE['max_home_beta'], 10)
    poi_psi_list = np.linspace(BETA_AND_PSI_PLAUSIBLE_RANGE['min_poi_psi'], BETA_AND_PSI_PLAUSIBLE_RANGE['max_poi_psi'], 15)


STARTING_SEED = range(NUM_SEEDS)

############################################################
# functions

def match_msa_name_to_msas_in_acs_data(msa_name, acs_msas):
    '''
    Matches the MSA name from our annotated SafeGraph data to the
    MSA name in the external datasource in MSA_COUNTY_MAPPING
    '''
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
    #state = str(state)
    if len(state) == 1:
        state = '0' + state
    #county = str(county)
    if len(county) == 1:
        county = '00' + county
    elif len(county) == 2:
        county = '0' + county
    #fips_codes.append(np.int64(state + county))
    #return np.int64(state + county)
    return int(state + county)
    
# Average history records across random seeds
def average_across_random_seeds(policy, history_C2, history_D2, num_cbgs, cbg_idxs, print_results=False, draw_results=True):
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
f = open(os.path.join(root, MSA_NAME, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()

# Load precomputed parameters to adjust(clip) POI dwell times
#d = pd.read_csv(os.path.join(root,'data_after_process(gao)\\parameters1.csv')) # Philadelphia MSA
d = pd.read_csv(os.path.join(root,MSA_NAME, 'parameters_%s.csv' % MSA_NAME)) 
#d.rename(columns={"safegraph_computed_area_in_square_feet":"feet"},inplace=True)
#d.rename(columns={"avg_median_dwell":"median"},inplace=True)

# No clipping
new_d = d

all_hours = helper.list_hours_in_range(min_datetime, max_datetime)
#poi_areas = new_d['safegraph_computed_area_in_square_feet'].values#面积
#poi_dwell_times = new_d['avg_median_dwell'].values#平均逗留时间
poi_areas = new_d['feet'].values#面积
poi_dwell_times = new_d['median'].values#平均逗留时间
poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2

# Load ACS Data for MSA-county matching
acs_data = pd.read_csv(os.path.join(root,'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
msa_match = match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
msa_data['FIPS Code'] = msa_data.apply(lambda x : get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
good_list = list(msa_data['FIPS Code'].values)
print(good_list)

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

# Extract CBGs belonging to the MSA
# https://covid-mobility.stanford.edu//datasets/
cbg_agesex_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
cbg_age_msa = cbg_agesex_msa.copy()

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
'''
good_list = {
             'Atlanta':[13013, 13015, 13035, 13045, 13057, 13063, 13067, 13077, 13085, 13089, 13097, 13113, 13117, 13121, 
                        13135, 13143, 13149, 13151, 13159, 13171, 13199, 13211, 13217, 13223, 13227, 13231, 13247, 13255, 13297],
             'Chicago':[17031, 17037, 17043, 17063, 17089, 17093, 17097, 17111, 17197, 18073, 18089, 18111, 18127, 55059],
             'Dallas': [48085, 48113, 48121, 48139, 48221, 48231, 48251, 48257, 48367, 48397, 48425, 48439, 48497],
             'Houston':[48015, 48039, 48071, 48157, 48167, 48201, 48291, 48339, 48473],
             'LosAngeles':[6111, 6071, 6065, 6037, 6059],
             'Miami': [12011, 12086, 12099],
             'NewYorkCity': [34003, 34013, 34017, 34019, 34023, 34025, 34027, 34029, 34031, 34035, 34037, 34039, 36005, 36027, 
                             36047, 36059, 36061, 36071, 36079, 36081, 36085, 36087, 36103, 36119, 42103],
             'Philadelphia':[34005,34007,34015,42017,42029,42091,42045,42101,10003,24015,34033,42011,10001,34001,34009,34011],
             'SanFrancisco': [6001, 6013, 6041, 6075, 6081],
             'WashingtonDC': [24009, 24017, 24021, 24031, 24033, 51013, 51043, 51047, 51059, 51061, 51107, 51153, 51157, 51177, 
                              51179, 51187, 51510, 51600, 51610, 51630, 51683, 51683, 51685]
            }
'''

#good_list = [6111, 6071, 6065, 6037, 6059] # Los Angeles

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
nyt_data = pd.read_csv(os.path.join(root, 'us-counties.csv'))
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
'''
if(len(cases_daily_smooth)<len(cases_total_no_vaccination)):
    cases_daily_smooth = [0]*(len(cases_total_no_vaccination)-len(cases_daily_smooth)) + list(cases_daily_smooth)
'''
# Deaths
deaths_daily = [0]
for i in range(1,len(nyt_data_cumulative)):
    deaths_daily.append(nyt_data_cumulative['deaths'].values[i]-nyt_data_cumulative['deaths'].values[i-1])
# Smoothed ground truth
deaths_daily_smooth = apply_smoothing(deaths_daily, agg_func=np.mean, before=3, after=3)
'''
if(len(deaths_daily_smooth)<len(deaths_total_no_vaccination)):
    deaths_daily_smooth = [0]*(len(deaths_total_no_vaccination)-len(deaths_daily_smooth)) + list(deaths_daily_smooth)
'''


# Initialization: only need to be performed once    
m = disease_model_original.Model(starting_seed=STARTING_SEED,
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
            _, _, cases_total_age_agnostic, deaths_total_age_agnostic = average_across_random_seeds(policy, history_C2_age_agnostic, history_D2_age_agnostic, M, idxs_county, 
                                                                                      print_results=False,draw_results=False)

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
            
            '''
            if(len(cases_smooth)<len(cases_total_no_vaccination)):
                cases_smooth = [0]*(len(cases_total_no_vaccination)-len(cases_smooth)) + list(cases_smooth)
            if(len(cases)<len(cases_total_no_vaccination)):
                cases = [0]*(len(cases_total_no_vaccination)-len(cases)) + list(cases)
            '''

            deaths = nyt_data_cumulative['deaths'].values
            deaths_smooth = apply_smoothing(deaths, agg_func=np.mean, before=3, after=3)
            '''
            if(len(deaths_smooth)<len(deaths_total_no_vaccination)):
                deaths_smooth = [0]*(len(deaths_total_no_vaccination)-len(deaths_smooth)) + list(deaths_smooth)
                print(len(deaths_smooth))
            if(len(deaths)<len(deaths_total_no_vaccination)):
                deaths = [0]*(len(deaths_total_no_vaccination)-len(deaths)) + list(deaths)
            '''
            
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
            np.save(os.path.join(root,MSA_NAME, '20210127_rmse_cases_%s_%s'%(MSA_NAME,p_sick_at_t0)),(rmse_dict_cases_agnostic)
            np.save(os.path.join(root,MSA_NAME, '20210127_rmse_cases_smooth_%s'%(MSA_NAME,p_sick_at_t0)),rmse_dict_cases_smooth_agnostic)
            np.save(os.path.join(root,MSA_NAME, '20210127_rmse_deaths_%s_%s'%(MSA_NAME,p_sick_at_t0)),rmse_dict_deaths_agnostic)
            np.save(os.path.join(root,MSA_NAME, '20210127_rmse_deaths_smooth_%s_%s'%(MSA_NAME,p_sick_at_t0)),rmse_dict_deaths_smooth_agnostic)
            
            # Save best results
            best_results = dict()
            best_results['rmse'] = best_rmse
            best_results['parameters'] = best_parameters
            np.save(os.path.join(root,MSA_NAME, '20210127_best_results_%s_%s_%s'%(how_to_select_best_grid_search_models,MSA_NAME,p_sick_at_t0)),best_results)

end = time.time()
print('Total Time:',(end-start))






