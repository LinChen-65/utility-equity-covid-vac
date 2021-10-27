# python test_vaccination_adaptive_singledemo_svi_accessibility.py MSA_NAME VACCINATION_TIME VACCINATION_RATIO consider_hesitancy ACCEPTANCE_SCENARIO consider_accessibility quick_test
# python test_vaccination_adaptive_singledemo_svi_accessibility.py Atlanta 31 0.1 True real True False

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import sys
import socket
import os
import datetime
import pandas as pd
import numpy as np
import pickle

import constants
import functions
import disease_model_test

import time
import pdb

###############################################################################
# Constants

# root
hostname = socket.gethostname()
print('hostname: ', hostname)
if(hostname=='fib-dl3'):
    root = '/data/chenlin/COVID-19/Data' #dl3
elif(hostname=='rl4'):
    root = '/home/chenlin/COVID-19/Data' #rl4

MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
NUM_DAYS = 63
NUM_GROUPS = 5

# Policy execution ratio
EXECUTION_RATIO = 1
# Recheck interval (After distributing some portion of vaccines, recheck the most vulnerable demographic group)
RECHECK_INTERVAL = 0.01 

###############################################################################
# Main variable settings

MSA_NAME = sys.argv[1]; print('MSA_NAME: ',MSA_NAME) #MSA_NAME = 'SanFrancisco'
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME] #MSA_NAME_FULL = 'San_Francisco_Oakland_Hayward_CA'

#policy_list = ['Age_Agnostic','No_Vaccination', 'Baseline','Age_Flood', 'Income_Flood', 'EW_Flood','SVI']
#policy_list = ['Baseline','Age_Flood', 'Age_Flood_Reverse','Income_Flood','Income_Flood_Reverse', 'JUE_EW_Flood','JUE_EW_Flood_Reverse','SVI']
#policy_list = ['Baseline','Age_Flood', 'Income_Flood', 'JUE_EW_Flood','SVI']
#policy_list = ['Real_Scaled']
#policy_list = ['Real_Scaled_Flood']
policy_list = ['SVI','Age_Flood_Reverse','Income_Flood_Reverse','JUE_EW_Flood_Reverse']
print('Policy list: ', policy_list)


# Vaccination time
VACCINATION_TIME = sys.argv[2];print('VACCINATION_TIME:',VACCINATION_TIME)
VACCINATION_TIME_STR = VACCINATION_TIME
VACCINATION_TIME = float(VACCINATION_TIME)
print(VACCINATION_TIME_STR,'\n',VACCINATION_TIME)

# Vaccination ratio
VACCINATION_RATIO = sys.argv[3]  #0.1
VACCINATION_RATIO = float(VACCINATION_RATIO)
print('Vaccination ratio: ', VACCINATION_RATIO)

# Consider hesitancy or not
consider_hesitancy = sys.argv[4]
print('Consider hesitancy? ', consider_hesitancy)
if(consider_hesitancy not in ['True','False']): 
    print('Invalid value for consider_hesitancy. Please check.')
    pdb.set_trace()

# Vaccine acceptance scenario
if(consider_hesitancy=='True'):
    token = sys.argv[5]
    if(token == 'ALL'):
        ACCEPTANCE_SCENARIO_LIST = ['real','cf18','cf13','cf17']
    else:
        ACCEPTANCE_SCENARIO_LIST = [token]
elif(consider_hesitancy=='False'):
    ACCEPTANCE_SCENARIO_LIST = ['fully']
print('Vaccine acceptance scenario list: ', ACCEPTANCE_SCENARIO_LIST)

# Consider accessibility or not
consider_accessibility = sys.argv[6]; print('Consider accessibility?', consider_accessibility)


# Quick Test: prototyping
quick_test = sys.argv[7]; print('Quick testing?', quick_test)
if(quick_test == 'True'):
    NUM_SEEDS = 2
    NUM_SEEDS_CHECKING = 2
else:
    NUM_SEEDS = 30
    NUM_SEEDS_CHECKING = 30 
print('NUM_SEEDS: ', NUM_SEEDS)
print('NUM_SEEDS_CHECKING: ', NUM_SEEDS_CHECKING)
STARTING_SEED = range(NUM_SEEDS)
STARTING_SEED_CHECKING = range(NUM_SEEDS)

# 分几次把疫苗分配完
distribution_time = VACCINATION_RATIO / RECHECK_INTERVAL 

# Vaccination protection rate (set by willingness, 20211004)
PROTECTION_RATE = 1

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
    del T1
    del L_1
    del I_1
    del C2
    del D2
    #return total_affected, history_C2, history_D2, total_affected_each_cbg
    return history_C2, history_D2


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


# Load other Safegraph demographic data, and perform grouping
if(('Age_Flood' in policy_list) or ('Age_Flood_Reverse' in policy_list)):
    # Calculate elder ratios
    cbg_age_msa['Elder_Absolute'] = cbg_age_msa.apply(lambda x : x['70 To 74 Years']+x['75 To 79 Years']+x['80 To 84 Years']+x['85 Years And Over'],axis=1)
    cbg_age_msa['Elder_Ratio'] = cbg_age_msa['Elder_Absolute'] / cbg_age_msa['Sum']
    # Grouping
    separators = functions.get_separators(cbg_age_msa, NUM_GROUPS, 'Elder_Ratio','Sum', normalized=True)
    cbg_age_msa['Age_Quantile'] =  cbg_age_msa['Elder_Ratio'].apply(lambda x : functions.assign_group(x, separators))


if(('Income_Flood' in policy_list) or ('Income_Flood_Reverse' in policy_list) or (consider_hesitancy=='True') ):
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


if(('JUE_EW_Flood' in policy_list) or ('JUE_EW_Flood_Reverse' in policy_list)):
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


if('SVI' in policy_list):
    cbg_ids_msa['census_tract'] = cbg_ids_msa['census_block_group'].apply(lambda x:int(str(x)[:-1]))
    svidata = pd.read_csv(os.path.join(root, 'SVI2018_US.csv'))
    columns_of_interest = ['FIPS','RPL_THEMES']
    svidata = svidata[columns_of_interest].copy()
    svidata_msa = pd.merge(cbg_ids_msa, svidata, left_on='census_tract', right_on='FIPS', how='left')
    svidata_msa['Sum'] = cbg_age_msa['Sum'].copy()

#if(consider_accessibility=='True'):
if(True):
    # accessibility by race/ethnic
    # cbg_b03.csv: HISPANIC OR LATINO ORIGIN BY RACE
    filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_b03.csv")
    cbg_race = pd.read_csv(filepath)
    # Extract pois corresponding to the metro area, by merging dataframes
    cbg_race_msa = pd.merge(cbg_ids_msa, cbg_race, on='census_block_group', how='left')
    del cbg_race
    cbg_race_msa.rename(columns={'B03002e1':'Sum',
                             'B03002e2':'NH_Total',
                             'B03002e3':'NH_White',
                             'B03002e4':'NH_Black',
                             'B03002e5':'NH_Indian',
                             'B03002e6':'NH_Asian',
                             'B03002e7':'NH_Hawaiian',
                             'B03002e12':'Hispanic' 
                            },inplace=True)
    
    # Extract columns of interest
    columns_of_interest = ['census_block_group','Sum','NH_Total','NH_White','NH_Black','NH_Indian','NH_Asian','NH_Hawaiian','Hispanic']
    cbg_race_msa = cbg_race_msa[columns_of_interest].copy()
    # Deal with NaN values
    cbg_race_msa.fillna(0,inplace=True)
    # Deal with CBGs with 0 populations
    cbg_race_msa['Sum'] = cbg_race_msa['Sum'].apply(lambda x : x if x!=0 else 1)
    # Calculate "Multiple/Other Non_Hispanic"
    cbg_race_msa['NH_Others'] = cbg_race_msa['NH_Total'] - (cbg_race_msa['NH_White']+cbg_race_msa['NH_Black']+cbg_race_msa['NH_Indian']+cbg_race_msa['NH_Asian']+cbg_race_msa['NH_Hawaiian'])

    #20211016, https://covid.cdc.gov/covid-data-tracker/#vaccinations_vacc-total-admin-rate-total
    vac_rate_total = 0.568
    #20211016, https://covid.cdc.gov/covid-data-tracker/#vaccination-demographic
    vac_rate_nh_white = (0.01*61.1) * vac_rate_total / (0.01*61.2)
    vac_rate_nh_black = (0.01*10.2) * vac_rate_total / (0.01*12.4)
    vac_rate_nh_indian = (0.01*1) * vac_rate_total / (0.01*0.8)
    vac_rate_nh_asian = (0.01*6.3) * vac_rate_total / (0.01*5.8)
    vac_rate_nh_hawaiian = (0.01*0.3) * vac_rate_total / (0.01*0.3)
    vac_rate_nh_others = (0.01*4.4) * vac_rate_total / (0.01*2.3)
    vac_rate_hispanic = (0.01*16.7) * vac_rate_total / (0.01*17.2)   
    # Calculate CBG vac_rate by race/ethnic: weighted average
    cbg_race_msa['Vac_Rate_Race'] = (cbg_race_msa['NH_White']*vac_rate_nh_white + cbg_race_msa['NH_Black']*vac_rate_nh_black + cbg_race_msa['NH_Indian']*vac_rate_nh_indian
                                     +cbg_race_msa['NH_Asian']*vac_rate_nh_asian + cbg_race_msa['NH_Hawaiian']*vac_rate_nh_hawaiian
                                     +cbg_race_msa['NH_Others']*vac_rate_nh_others + cbg_race_msa['Hispanic']*vac_rate_hispanic)
    cbg_race_msa['Vac_Rate_Race'] /= cbg_race_msa['Sum']
    #print('cbg_race_msa[\'Vac_Rate_Race\'].max(): ', cbg_race_msa['Vac_Rate_Race'].max(),'\ncbg_race_msa[\'Vac_Rate_Race\'].min(): ', cbg_race_msa['Vac_Rate_Race'].min())

    # accessibility by age
    #20211016, https://covid.cdc.gov/covid-data-tracker/#vaccination-demographic
    vac_rate_0_11 = (0.01*0.1) * vac_rate_total / (0.01*14.4)
    vac_rate_12_15 = (0.01*4) * vac_rate_total / (0.01*5)
    vac_rate_16_17 = (0.01*2.3) * vac_rate_total / (0.01*2.5)
    vac_rate_18_24 = (0.01*8.6) * vac_rate_total / (0.01*9.2)
    vac_rate_25_39 = (0.01*20.7) * vac_rate_total / (0.01*20.5)
    vac_rate_40_49 = (0.01*14.2) * vac_rate_total / (0.01*12.2)
    vac_rate_50_64 = (0.01*25.1) * vac_rate_total / (0.01*19.4)
    vac_rate_65_74 = (0.01*14.8) * vac_rate_total / (0.01*9.8)
    vac_rate_75_up = (0.01*10) * vac_rate_total / (0.01*7)
    # Calculate CBG vac_rate by age: weighted average
    cbg_age_msa['Vac_Rate_Age'] = ((cbg_age_msa['Under 5 Years']+cbg_age_msa['5 To 9 Years']+cbg_age_msa['10 To 14 Years']*2/5)*vac_rate_0_11
                                    +(cbg_age_msa['10 To 14 Years']*3/5+cbg_age_msa['15 To 17 Years']*1/3)*vac_rate_12_15
                                    +(cbg_age_msa['15 To 17 Years']*2/3)*vac_rate_16_17
                                    +(cbg_age_msa['18 To 19 Years']+cbg_age_msa['20 Years']+cbg_age_msa['21 Years']+cbg_age_msa['22 To 24 Years'])*vac_rate_18_24 
                                    +(cbg_age_msa['25 To 29 Years']+cbg_age_msa['30 To 34 Years']+cbg_age_msa['35 To 39 Years'])*vac_rate_25_39
                                    +(cbg_age_msa['40 To 44 Years']+cbg_age_msa['45 To 49 Years'])*vac_rate_40_49
                                    +(cbg_age_msa['50 To 54 Years']+cbg_age_msa['55 To 59 Years']+cbg_age_msa['60 To 61 Years']+cbg_age_msa['62 To 64 Years'])*vac_rate_50_64
                                    +(cbg_age_msa['65 To 66 Years']+cbg_age_msa['67 To 69 Years']+cbg_age_msa['70 To 74 Years'])*vac_rate_65_74
                                    +(cbg_age_msa['75 To 79 Years']+cbg_age_msa['80 To 84 Years']+cbg_age_msa['85 Years And Over'])*vac_rate_75_up
                                    )
    cbg_age_msa['Vac_Rate_Age'] /= cbg_age_msa['Sum']

    # Multiplication of two determinants
    cbg_age_msa['Vac_Rate_Age_Race'] = cbg_race_msa['Vac_Rate_Race'] * cbg_age_msa['Vac_Rate_Age']
    
    print('cbg_race_msa[\'Vac_Rate_Race\'].max(): ', np.round(cbg_race_msa['Vac_Rate_Race'].max(),3),
          '\ncbg_race_msa[\'Vac_Rate_Race\'].min(): ', np.round(cbg_race_msa['Vac_Rate_Race'].min(),3))
    print('cbg_age_msa[\'Vac_Rate_Age\'].max(): ', np.round(cbg_age_msa['Vac_Rate_Age'].max(),3),
          '\ncbg_age_msa[\'Vac_Rate_Age\'].min(): ', np.round(cbg_age_msa['Vac_Rate_Age'].min(),3))
    print('cbg_age_msa[\'Vac_Rate_Age_Race\'].max(): ', np.round(cbg_age_msa['Vac_Rate_Age_Race'].max(),3),
          '\ncbg_age_msa[\'Vac_Rate_Age_Race\'].min(): ', np.round(cbg_age_msa['Vac_Rate_Age_Race'].min(),3))

    # Division by vaccine acceptance to get the final accessibility
    #cbg_age_msa['Accessibility_Age_Race'] = cbg_age_msa['Vac_Rate_Age_Race'] / 


if('Real_Scaled' in policy_list):
    equal_rate_scale_factor = VACCINATION_RATIO/(np.sum(cbg_sizes * cbg_age_msa['Vac_Rate_Age_Race'])/np.sum(cbg_sizes))
    print('equal_rate_scale_factor: ', equal_rate_scale_factor)

##############################################################################
# Load and scale age-aware CBG-specific attack/death rates (original)

cbg_death_rates_original = np.loadtxt(os.path.join(root, MSA_NAME, 'cbg_death_rates_original_'+MSA_NAME))
cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)
#print('Age-aware CBG-specific death rates loaded. Attack rates are irrelevant to age.')

# The scaling factors are set according to a grid search
# Fix attack_scale
attack_scale = 1
cbg_attack_rates_original_scaled = cbg_attack_rates_original * attack_scale
cbg_death_rates_original_scaled = cbg_death_rates_original * constants.death_scale_dict[MSA_NAME]
#print('Age-aware CBG-specific death rates scaled.')

start_all = time.time()

for ACCEPTANCE_SCENARIO in ACCEPTANCE_SCENARIO_LIST:
    print('ACCEPTANCE_SCENARIO: ', ACCEPTANCE_SCENARIO)
    start = time.time()

    # Calculate vaccine acceptance in each CBG
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

    if(consider_accessibility=='True'):
        # Division by vaccine acceptance to get the final accessibility
        cbg_age_msa['Accessibility_Age_Race'] = cbg_age_msa['Vac_Rate_Age_Race'] / vaccine_acceptance
        # Find the minimum non-zero value
        a = np.array(cbg_age_msa['Accessibility_Age_Race'])
        a = a[a>0]
        print('Minimum nonzero accessibility: ', min(a))
        cbg_age_msa['Accessibility_Age_Race'] = cbg_age_msa['Accessibility_Age_Race'].apply(lambda x : min(a) if x==0 else x)
        vaccine_acceptance = np.array(cbg_age_msa['Accessibility_Age_Race']) #以此代替原来的acceptance参数传入函数

        print('cbg_age_msa[\'Accessibility_Age_Race\'].max(): ', np.round(cbg_age_msa['Accessibility_Age_Race'].max(),3),
             '\ncbg_age_msa[\'Accessibility_Age_Race\'].min(): ', np.round(cbg_age_msa['Accessibility_Age_Race'].min(),3))
    #print(np.isnan(np.array(cbg_age_msa['Accessibility_Age_Race'])).any())
   # pdb.set_trace()

    ##############################################################################
    need_to_save_dict = {
        'no_vaccination':False,
        'baseline':False,
        'age_flood':False,
        'age_flood_reverse':False,
        'income_flood':False,
        'income_flood_reverse':False,
        'jue_ew_flood':False,
        'jue_ew_flood_reverse':False,
        'svi':False,
        'real_scaled':False,
        'real_scaled_flood':False
    }
    ###############################################################################
    # No_Vaccination

    if ('No_Vaccination' in policy_list):
        print('\nPolicy: No_Vaccination.')
        need_to_save_dict['no_vaccination'] = True

        # Construct the vaccination vector
        vaccination_vector_no_vaccination = np.zeros(len(cbg_sizes))
        # Run simulations
        _, history_D2_no_vaccination = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                vaccination_vector=vaccination_vector_no_vaccination,
                                                vaccine_acceptance = vaccine_acceptance, #20211007
                                                #protection_rate = protection_rate_most_disadvantaged)
                                                protection_rate = PROTECTION_RATE)

    ###############################################################################
    # Baseline: Flooding on Random Permutation

    if('Baseline' in policy_list):
        print('\nPolicy: Baseline.')
        need_to_save_dict['baseline'] = True
        
        # Construct the vaccination vector
        random_permutation = np.arange(len(cbg_age_msa))
        np.random.seed(42)
        np.random.shuffle(random_permutation)
        cbg_age_msa['Random_Permutation'] = random_permutation
        vaccination_vector_baseline = functions.vaccine_distribution_flood(cbg_table=cbg_age_msa, 
                                                                        vaccination_ratio=VACCINATION_RATIO, 
                                                                        demo_feat='Random_Permutation', 
                                                                        ascending=None,
                                                                        execution_ratio=1
                                                                        )
        pdb.set_trace()
        # Run simulations
        _, history_D2_baseline = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                            vaccination_vector=vaccination_vector_baseline,
                                            vaccine_acceptance = vaccine_acceptance, #20211007
                                            #protection_rate = protection_rate_baseline)
                                            protection_rate = PROTECTION_RATE)
                                        
    ###############################################################################
    print('Experiments for prioritizing the most disadvantaged communities...')         
    subroot = 'vaccination_results_adaptive_%sd_%s_0.01' %(VACCINATION_TIME_STR,VACCINATION_RATIO)    
    if(consider_accessibility=='False'):
        if(consider_hesitancy=='True'):
            notation_string = 'acceptance_%s_'%ACCEPTANCE_SCENARIO
        else:
            notation_string = ''
    else:
        if(consider_hesitancy=='True'):
            notation_string = 'access_acceptance_%s_'%ACCEPTANCE_SCENARIO
        else:
            notation_string = 'access_'
    ###############################################################################
    # Age_Flood, prioritize the most disadvantaged

    if('Age_Flood' in policy_list):
        demo_feat = 'Age'
        print('\nPolicy: Age_Flood.')
        if(os.path.exists(os.path.join(root, MSA_NAME, subroot,
                                'test_history_D2_age_flood_adaptive_%sd_%s_%s_%sseeds_%s%s' % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS,notation_string, MSA_NAME)))):
            print('Results for Age_Flood already exist. No need to simulate again.')      
        else:    
            need_to_save_dict['age_flood'] = True
            # Construct the vaccination vector    
            current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
            cbg_age_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
            leftover = 0
            
            for i in range(int(distribution_time)):
                if i==(int(distribution_time)-1): is_last = True
                else: is_last = False
                    
                cbg_age_msa['Vaccination_Vector'] = current_vector
                
                # Run a simulation to determine the most vulnerable group
                _, history_D2_current = run_simulation(starting_seed=STARTING_SEED_CHECKING, num_seeds=NUM_SEEDS_CHECKING, 
                                                    #vaccination_vector=vaccination_vector_age_flood,
                                                    vaccination_vector=current_vector,
                                                    vaccine_acceptance = vaccine_acceptance, #20211007
                                                    #protection_rate = protection_rate_most_disadvantaged)
                                                    protection_rate = PROTECTION_RATE)

                # Average history records across random seeds
                deaths_cbg_current, _ = functions.average_across_random_seeds_only_death(history_D2_current, 
                                                                                        M, idxs_msa_nyt, 
                                                                                        print_results=False, draw_results=False)
                # Analyze deaths in each demographic group
                avg_final_deaths_current = deaths_cbg_current[-1,:]
                # Add simulation results to cbg table
                cbg_age_msa['Final_Deaths_Current'] = avg_final_deaths_current
                
                final_deaths_rate_current = np.zeros(NUM_GROUPS)
                for group_id in range(NUM_GROUPS):
                    final_deaths_rate_current[group_id] = cbg_age_msa[cbg_age_msa[demo_feat + '_Quantile']==group_id]['Final_Deaths_Current'].sum()
                    final_deaths_rate_current[group_id] /= cbg_age_msa[cbg_age_msa[demo_feat + '_Quantile']==group_id]['Sum'].sum()
                
                # Find the most vulnerable group
                most_vulnerable_group = np.argmax(final_deaths_rate_current)
                # Annotate the most vulnerable group
                cbg_age_msa['Most_Vulnerable'] = cbg_age_msa.apply(lambda x : 1 if x['Age_Quantile']==most_vulnerable_group else 0, axis=1)
                
                # Distribute vaccines in the currently most vulnerable group - flooding
                new_vector = functions.vaccine_distribution_flood_new(cbg_table=cbg_age_msa, 
                                                                    #vaccination_ratio=VACCINATION_RATIO, 
                                                                    vaccination_ratio=RECHECK_INTERVAL, 
                                                                    demo_feat='Elder_Ratio', 
                                                                    ascending=False, 
                                                                    execution_ratio=EXECUTION_RATIO,
                                                                    leftover=leftover,
                                                                    is_last=is_last
                                                                    )
                leftover_prev = leftover
                current_vector_prev = current_vector.copy() # 20210225
                current_vector += new_vector # 20210224
                current_vector = np.clip(current_vector, None, cbg_sizes) # 20210224
                leftover = np.sum(cbg_sizes) * RECHECK_INTERVAL + leftover_prev - (np.sum(current_vector)-np.sum(current_vector_prev))
                assert((current_vector<=cbg_sizes).all())
                #print('Newly distributed vaccines: ', np.sum(new_vector))
                
            vaccination_vector_age_flood = current_vector

            # Run simulations
            _, history_D2_age_flood = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                vaccination_vector=vaccination_vector_age_flood,
                                                vaccine_acceptance = vaccine_acceptance, #20211007
                                                #protection_rate = protection_rate_most_disadvantaged)
                                                protection_rate = PROTECTION_RATE)

    ###############################################################################
    # Income_Flood, prioritize the most disadvantaged

    if('Income_Flood' in policy_list):
        demo_feat = 'Mean_Household_Income'
        print('\nPolicy: Income_Flood.')
        if(os.path.exists(os.path.join(root,MSA_NAME,subroot,
                            'test_history_D2_income_flood_adaptive_%sd_%s_%s_%sseeds_%s%s' % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS,notation_string,MSA_NAME)))):
            print('Results for Income_Flood already exist. No need to simulate again.')   
        else:   
            need_to_save_dict['income_flood'] = True 
            # Construct the vaccination vector
            current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
            cbg_income_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
            leftover = 0
            
            for i in range(int(distribution_time)):
                if i==(int(distribution_time)-1): is_last = True
                else: is_last=False
                    
                cbg_income_msa['Vaccination_Vector'] = current_vector
                
                # Run a simulation to determine the most vulnerable group
                _, history_D2_current = run_simulation(starting_seed=STARTING_SEED_CHECKING, num_seeds=NUM_SEEDS_CHECKING, 
                                                    #vaccination_vector=vaccination_vector_age_flood,
                                                    vaccination_vector=current_vector,
                                                    vaccine_acceptance = vaccine_acceptance, #20211007
                                                    #protection_rate = protection_rate_most_disadvantaged)
                                                    protection_rate = PROTECTION_RATE)
                # Average history records across random seeds
                deaths_cbg_current, _ = functions.average_across_random_seeds_only_death(history_D2_current, 
                                                                                        M, idxs_msa_nyt, 
                                                                                        print_results=False, draw_results=False)
                # Analyze deaths in each demographic group
                avg_final_deaths_current = deaths_cbg_current[-1,:]
                # Add simulation results to cbg table
                cbg_income_msa['Final_Deaths_Current'] = avg_final_deaths_current
                
                final_deaths_rate_current = np.zeros(NUM_GROUPS)
                for group_id in range(NUM_GROUPS):
                    final_deaths_rate_current[group_id] = cbg_income_msa[cbg_income_msa['Mean_Household_Income_Quantile']==group_id]['Final_Deaths_Current'].sum()
                    final_deaths_rate_current[group_id] /= cbg_income_msa[cbg_income_msa['Mean_Household_Income_Quantile']==group_id]['Sum'].sum()
                
                # Find the most vulnerable group
                most_vulnerable_group = np.argmax(final_deaths_rate_current)
                print('Currently, most_vulnerable_group:', most_vulnerable_group)
                # Annotate the most vulnerable group
                cbg_income_msa['Most_Vulnerable'] = cbg_income_msa.apply(lambda x : 1 if x['Mean_Household_Income_Quantile']==most_vulnerable_group else 0, axis=1)
                
                # Distribute vaccines in the currently most vulnerable group - flooding
                new_vector = functions.vaccine_distribution_flood_new(cbg_table=cbg_income_msa, 
                                                                    #vaccination_ratio=VACCINATION_RATIO, 
                                                                    vaccination_ratio=RECHECK_INTERVAL, 
                                                                    demo_feat='Mean_Household_Income', 
                                                                    ascending=True, 
                                                                    execution_ratio=EXECUTION_RATIO,
                                                                    leftover=leftover,
                                                                    is_last=is_last
                                                                    )
                leftover_prev = leftover
                current_vector_prev = current_vector.copy() # 20210225
                current_vector += new_vector # 20210224
                current_vector = np.clip(current_vector, None, cbg_sizes) # 20210224
                leftover = np.sum(cbg_sizes) * RECHECK_INTERVAL + leftover_prev - (np.sum(current_vector)-np.sum(current_vector_prev))
                
                assert((current_vector<=cbg_sizes).all())
                #print('Newly distributed vaccines: ', (np.sum(current_vector)-np.sum(current_vector_prev)))    
            
            vaccination_vector_income_flood = current_vector

            # Run simulations
            _, history_D2_income_flood = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                    vaccination_vector=vaccination_vector_income_flood,
                                                    vaccine_acceptance = vaccine_acceptance, #20211007
                                                    #protection_rate = protection_rate_most_disadvantaged)
                                                    protection_rate = PROTECTION_RATE)
            

    ###############################################################################
    # JUE_EW_Flood, prioritize the most disadvantaged

    if('JUE_EW_Flood' in policy_list):
        demo_feat = 'JUE_EW'
        print('\nPolicy: JUE_EW_Flood.')
        if(os.path.exists(os.path.join(root,MSA_NAME,subroot,
                                'test_history_D2_jue_ew_flood_adaptive_%sd_%s_%s_%sseeds_%s%s' % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS,notation_string,MSA_NAME)))):
            print('Results for JUE_EW_Flood already exist. No need to simulate again.')       
        else:
            need_to_save_dict['jue_ew_flood'] = True
            # Construct the vaccination vector
            current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
            cbg_occupation_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
            leftover = 0
            
            for i in range(int(distribution_time)):
                if i==(int(distribution_time)-1): is_last = True
                else: is_last=False
                    
                cbg_occupation_msa['Vaccination_Vector'] = current_vector
                
                # Run a simulation to determine the most vulnerable group
                _, history_D2_current = run_simulation(starting_seed=STARTING_SEED_CHECKING, num_seeds=NUM_SEEDS_CHECKING, 
                                                    #vaccination_vector=vaccination_vector_mobility_flood,
                                                    vaccination_vector=current_vector,
                                                    vaccine_acceptance = vaccine_acceptance, #20211007
                                                    #protection_rate = protection_rate_most_disadvantaged)
                                                    protection_rate = PROTECTION_RATE)
                # Average history records across random seeds
                deaths_cbg_current, _= functions.average_across_random_seeds_only_death(history_D2_current, 
                                                                                        M, idxs_msa_nyt, 
                                                                                        print_results=False, draw_results=False)
                # Analyze deaths in each demographic group
                avg_final_deaths_current = deaths_cbg_current[-1,:]
                # Add simulation results to cbg table
                cbg_occupation_msa['Final_Deaths_Current'] = avg_final_deaths_current
                
                final_deaths_rate_current = np.zeros(NUM_GROUPS)
                for group_id in range(NUM_GROUPS):
                    final_deaths_rate_current[group_id] = cbg_occupation_msa[cbg_occupation_msa['Essential_Worker_Quantile']==group_id]['Final_Deaths_Current'].sum()
                    final_deaths_rate_current[group_id] /= cbg_occupation_msa[cbg_occupation_msa['Essential_Worker_Quantile']==group_id]['Sum'].sum()
                
                # Find the most vulnerable group
                most_vulnerable_group = np.argmax(final_deaths_rate_current)
                # Annotate the most vulnerable group
                cbg_occupation_msa['Most_Vulnerable'] = cbg_occupation_msa.apply(lambda x : 1 if x['Essential_Worker_Quantile']==most_vulnerable_group else 0, axis=1)
                
                # Distribute vaccines in the currently most vulnerable group - flooding
                new_vector = functions.vaccine_distribution_flood_new(cbg_table=cbg_occupation_msa, 
                                                                    #vaccination_ratio=VACCINATION_RATIO, 
                                                                    vaccination_ratio=RECHECK_INTERVAL, 
                                                                    demo_feat='Essential_Worker_Ratio', 
                                                                    ascending=False, 
                                                                    execution_ratio=EXECUTION_RATIO,
                                                                    leftover=leftover,
                                                                    is_last=is_last
                                                                    )
                leftover_prev = leftover
                current_vector_prev = current_vector.copy() # 20210225
                current_vector += new_vector # 20210224
                current_vector = np.clip(current_vector, None, cbg_sizes) # 20210224
                leftover = np.sum(cbg_sizes) * RECHECK_INTERVAL + leftover_prev - (np.sum(current_vector)-np.sum(current_vector_prev))
                assert((current_vector<=cbg_sizes).all())
                #print('Newly distributed vaccines: ', np.sum(new_vector))
                
            vaccination_vector_jue_ew_flood = current_vector

            # Run simulations
            _, history_D2_jue_ew_flood = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                    vaccination_vector=vaccination_vector_jue_ew_flood,
                                                    vaccine_acceptance = vaccine_acceptance, #20211007
                                                    #protection_rate = protection_rate_most_disadvantaged)
                                                    protection_rate = PROTECTION_RATE)


    ###############################################################################
    # SVI, prioritize the most disadvantaged
    if('SVI' in policy_list):
        print('Policy: SVI.')
        if(os.path.exists(os.path.join(root,MSA_NAME,subroot,
                                'test_history_D2_svi_adaptive_%sd_%s_%s_%sseeds_%s%s' % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS, notation_string,MSA_NAME)))):
            print('Results for SVI already exist. No need to simulate again.')   
        else:
            need_to_save_dict['svi'] = True
            # Construct the vaccination vector    
            vaccination_vector_svi = functions.vaccine_distribution_flood(cbg_table=svidata_msa, 
                                                                        vaccination_ratio=VACCINATION_RATIO, 
                                                                        demo_feat='RPL_THEMES', 
                                                                        ascending=False,
                                                                        execution_ratio=1
                                                                        )
            # Run simulations
            _, history_D2_svi = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                            vaccination_vector=vaccination_vector_svi,
                                            vaccine_acceptance = vaccine_acceptance, #20211007
                                            protection_rate = PROTECTION_RATE)

    ###############################################################################
    if('Real_Scaled' in policy_list):
        #vaccine_acceptance_real_scaled = np.ones(len(cbg_sizes))
        #print('equal_rate_scale_factor: ', equal_rate_scale_factor)
        print('Policy: Real_Scaled.')
        #if(False):
        if(os.path.exists(os.path.join(root,MSA_NAME,subroot,
                                'test_history_D2_real_scaled_adaptive_%sd_%s_%s_%sseeds_%s%s' % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS, notation_string,MSA_NAME)))):
            print('Results for Real_Scaled already exist. No need to simulate again.')   
        else:
            need_to_save_dict['real_scaled'] = True
            # Construct the vaccination vector    
            vaccination_vector_real_scaled = equal_rate_scale_factor * np.array(cbg_age_msa['Vac_Rate_Age_Race']*cbg_sizes)
            #pdb.set_trace()
            
            # Run simulations
            _, history_D2_real_scaled = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                            vaccination_vector=vaccination_vector_real_scaled,
                                            vaccine_acceptance = vaccine_acceptance, #20211018
                                            protection_rate = PROTECTION_RATE)

    ###############################################################################
    if('Real_Scaled_Flood' in policy_list):
        print('Policy: Real_Scaled_Flood.')
        #if(False):
        if(os.path.exists(os.path.join(root,MSA_NAME,subroot,
                                'test_history_D2_real_scaled_flood_adaptive_%sd_%s_%s_%sseeds_%s%s' % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS, notation_string,MSA_NAME)))):
            print('Results for Real_Scaled_Flood already exist. No need to simulate again.')   
        else:
            need_to_save_dict['real_scaled_flood'] = True
        # Construct the vaccination vector    
        vaccination_vector_real_scaled_flood = functions.vaccine_distribution_flood(cbg_table=cbg_age_msa, 
                                                                    vaccination_ratio=VACCINATION_RATIO, 
                                                                    demo_feat='Vac_Rate_Age_Race', 
                                                                    ascending=False,
                                                                    execution_ratio=1
                                                                    )
        #pdb.set_trace()
        # Run simulations
        _, history_D2_real_scaled_flood = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                         vaccination_vector=vaccination_vector_real_scaled_flood,
                                                         vaccine_acceptance = vaccine_acceptance, #20211007
                                                         protection_rate = PROTECTION_RATE)

    ###############################################################################
    # Save results

    print('need_to_save_dict',need_to_save_dict)
    if(quick_test=='True'):
        print('Testing. Not saving results.')
    else:
        print('Saving results...\nPolicy list: ', policy_list)
        #if(consider_accessibility=='False'):
        #    notation_string = 'acceptance'
        #elif(consider_accessibility=='True'):
        #    notation_string = 'access_acceptance'
        #subroot = 'vaccination_results_adaptive_%sd_%s_0.01' %(VACCINATION_TIME_STR,VACCINATION_RATIO)
        for policy in policy_list:
            policy = policy.lower()
            if(need_to_save_dict[policy]==True):
                if(policy=='baseline'):
                    filename = os.path.join(root, MSA_NAME, subroot,       
                                            'test_history_D2_baseline_adaptive_%sd_%s_%s_%sseeds_%s%s' %(VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS,notation_string,MSA_NAME)
                                            )
                    print('Save baseline results at:\n', filename)            
                    np.array(history_D2_baseline).tofile(filename)
                elif(policy =='svi'):
                    filename = os.path.join(root, MSA_NAME, subroot,
                                            'test_history_D2_svi_adaptive_%sd_%s_%s_%sseeds_%s%s'%(VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS,notation_string,MSA_NAME)
                                            )
                    print('Save SVI-informed results at:\n', filename)            
                    np.array(history_D2_svi).tofile(filename)
                else:         
                    exec('np.array(history_D2_%s).tofile(os.path.join(root,MSA_NAME,subroot,\'test_history_D2_%s_adaptive_%sd_%s_%s_%sseeds_%s%s\'))' 
                        % (policy,policy,VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS, notation_string,MSA_NAME))
                need_to_save_dict[policy] = False
        print('Results saved.')


    ###############################################################################
    ###############################################################################
    ###############################################################################
    # Experiments for vaccinating the least disadvantaged communities

    print('Experiments for prioritizing the least disadvantaged communities...')
    subroot = 'vaccination_results_adaptive_reverse_%sd_%s_0.01'% (VACCINATION_TIME_STR,VACCINATION_RATIO) 
    ###############################################################################
    # Age_Flood, prioritize the least disadvantaged

    if('Age_Flood_Reverse' in policy_list):
        demo_feat = 'Age'
        print('Policy: Age_Flood_Reverse.')
        if(os.path.exists(os.path.join(root, MSA_NAME, subroot,
                                'test_history_D2_age_flood_adaptive_reverse_%sd_%s_%s_%sseeds_%s%s' % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS, notation_string,MSA_NAME)))):
            print('Results for Age_Flood_Reverse already exist. No need to simulate again.')                          
        else:
            need_to_save_dict['age_flood_reverse'] = True    
            # Construct the vaccination vector    
            current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
            cbg_age_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
            leftover = 0
            
            for i in range(int(distribution_time)):
                if i==(int(distribution_time)-1): is_last = True
                else: is_last = False
                    
                cbg_age_msa['Vaccination_Vector'] = current_vector
                
                # Run a simulation to determine the most vulnerable group
                _, history_D2_current = run_simulation(starting_seed=STARTING_SEED_CHECKING, num_seeds=NUM_SEEDS_CHECKING, 
                                                    #vaccination_vector=vaccination_vector_age_flood,
                                                    vaccination_vector=current_vector,
                                                    vaccine_acceptance = vaccine_acceptance, #20211007
                                                    #protection_rate = protection_rate_most_disadvantaged)
                                                    protection_rate = PROTECTION_RATE)

                # Average history records across random seeds
                deaths_cbg_current, _ = functions.average_across_random_seeds_only_death(history_D2_current, 
                                                                                        M, idxs_msa_nyt, 
                                                                                        print_results=False, draw_results=False)
                # Analyze deaths in each demographic group
                avg_final_deaths_current = deaths_cbg_current[-1,:]
                # Add simulation results to cbg table
                cbg_age_msa['Final_Deaths_Current'] = avg_final_deaths_current
                
                final_deaths_rate_current = np.zeros(NUM_GROUPS)
                for group_id in range(NUM_GROUPS):
                    final_deaths_rate_current[group_id] = cbg_age_msa[cbg_age_msa[demo_feat + '_Quantile']==group_id]['Final_Deaths_Current'].sum()
                    final_deaths_rate_current[group_id] /= cbg_age_msa[cbg_age_msa[demo_feat + '_Quantile']==group_id]['Sum'].sum()
                
                # Find the most vulnerable group
                #most_vulnerable_group = np.argmax(final_deaths_rate_current)
                most_vulnerable_group = np.argmin(final_deaths_rate_current)
                # Annotate the most vulnerable group
                cbg_age_msa['Most_Vulnerable'] = cbg_age_msa.apply(lambda x : 1 if x['Age_Quantile']==most_vulnerable_group else 0, axis=1)
                
                # Distribute vaccines in the currently most vulnerable group - flooding
                new_vector = functions.vaccine_distribution_flood_new(cbg_table=cbg_age_msa, 
                                                                    #vaccination_ratio=VACCINATION_RATIO, 
                                                                    vaccination_ratio=RECHECK_INTERVAL, 
                                                                    demo_feat='Elder_Ratio', 
                                                                    ascending=False, 
                                                                    execution_ratio=EXECUTION_RATIO,
                                                                    leftover=leftover,
                                                                    is_last=is_last
                                                                    )
                leftover_prev = leftover
                current_vector_prev = current_vector.copy() # 20210225
                current_vector += new_vector # 20210224
                current_vector = np.clip(current_vector, None, cbg_sizes) # 20210224
                leftover = np.sum(cbg_sizes) * RECHECK_INTERVAL + leftover_prev - (np.sum(current_vector)-np.sum(current_vector_prev))
                assert((current_vector<=cbg_sizes).all())
                #print('Newly distributed vaccines: ', np.sum(new_vector))
                
            vaccination_vector_age_flood_reverse = current_vector

            # Run simulations
            _, history_D2_age_flood_reverse = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                vaccination_vector=vaccination_vector_age_flood_reverse,
                                                vaccine_acceptance = vaccine_acceptance, #20211007
                                                #protection_rate = protection_rate_most_disadvantaged)
                                                protection_rate = PROTECTION_RATE)

    ###############################################################################
    # Income_Flood, prioritize the least disadvantaged

    if('Income_Flood_Reverse' in policy_list):
        demo_feat = 'Mean_Household_Income'
        print('Policy: Income_Flood_Reverse.')
        if(os.path.exists(os.path.join(root, MSA_NAME, subroot, 
                                'test_history_D2_income_flood_adaptive_reverse_%sd_%s_%s_%sseeds_%s%s' % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS, notation_string,MSA_NAME)))):
            print('Results for Income_Flood_Reverse already exist. No need to simulate again.')          
        else:
            need_to_save_dict['income_flood_reverse'] = True    
            # Construct the vaccination vector
            current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
            cbg_income_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
            leftover = 0
            
            for i in range(int(distribution_time)):
                if i==(int(distribution_time)-1): is_last = True
                else: is_last=False
                    
                cbg_income_msa['Vaccination_Vector'] = current_vector
                
                # Run a simulation to determine the most vulnerable group
                _, history_D2_current = run_simulation(starting_seed=STARTING_SEED_CHECKING, num_seeds=NUM_SEEDS_CHECKING, 
                                                    #vaccination_vector=vaccination_vector_age_flood,
                                                    vaccination_vector=current_vector,
                                                    vaccine_acceptance = vaccine_acceptance, #20211007
                                                    #protection_rate = protection_rate_most_disadvantaged)
                                                    protection_rate = PROTECTION_RATE)
                # Average history records across random seeds
                deaths_cbg_current, _ = functions.average_across_random_seeds_only_death(history_D2_current, 
                                                                                        M, idxs_msa_nyt, 
                                                                                        print_results=False, draw_results=False)
                # Analyze deaths in each demographic group
                avg_final_deaths_current = deaths_cbg_current[-1,:]
                # Add simulation results to cbg table
                cbg_income_msa['Final_Deaths_Current'] = avg_final_deaths_current
                
                final_deaths_rate_current = np.zeros(NUM_GROUPS)
                for group_id in range(NUM_GROUPS):
                    final_deaths_rate_current[group_id] = cbg_income_msa[cbg_income_msa['Mean_Household_Income_Quantile']==group_id]['Final_Deaths_Current'].sum()
                    final_deaths_rate_current[group_id] /= cbg_income_msa[cbg_income_msa['Mean_Household_Income_Quantile']==group_id]['Sum'].sum()
                
                # Find the most vulnerable group
                #most_vulnerable_group = np.argmax(final_deaths_rate_current)
                most_vulnerable_group = np.argmin(final_deaths_rate_current)
                # Annotate the most vulnerable group
                cbg_income_msa['Most_Vulnerable'] = cbg_income_msa.apply(lambda x : 1 if x['Mean_Household_Income_Quantile']==most_vulnerable_group else 0, axis=1)
                
                # Distribute vaccines in the currently most vulnerable group - flooding
                new_vector = functions.vaccine_distribution_flood_new(cbg_table=cbg_income_msa, 
                                                                    #vaccination_ratio=VACCINATION_RATIO, 
                                                                    vaccination_ratio=RECHECK_INTERVAL, 
                                                                    demo_feat='Mean_Household_Income', 
                                                                    ascending=True, 
                                                                    execution_ratio=EXECUTION_RATIO,
                                                                    leftover=leftover,
                                                                    is_last=is_last
                                                                    )
                leftover_prev = leftover
                current_vector_prev = current_vector.copy() # 20210225
                current_vector += new_vector # 20210224
                current_vector = np.clip(current_vector, None, cbg_sizes) # 20210224
                leftover = np.sum(cbg_sizes) * RECHECK_INTERVAL + leftover_prev - (np.sum(current_vector)-np.sum(current_vector_prev))
                
                assert((current_vector<=cbg_sizes).all())
                #print('Newly distributed vaccines: ', (np.sum(current_vector)-np.sum(current_vector_prev)))    
            
            vaccination_vector_income_flood_reverse = current_vector

            # Run simulations
            _, history_D2_income_flood_reverse = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                    vaccination_vector=vaccination_vector_income_flood_reverse,
                                                    vaccine_acceptance = vaccine_acceptance, #20211007
                                                    #protection_rate = protection_rate_most_disadvantaged)
                                                    protection_rate = PROTECTION_RATE)

    ###############################################################################
    # JUE_EW_Flood, prioritize the least disadvantaged

    if('JUE_EW_Flood_Reverse' in policy_list):
        demo_feat = 'JUE_EW'
        print('Policy: JUE_EW_Flood_Reverse.')
        if(os.path.exists(os.path.join(root, MSA_NAME, subroot,
                                'test_history_D2_jue_ew_flood_adaptive_reverse_%sd_%s_%s_%sseeds_%s%s' % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS, notation_string,MSA_NAME)))):
            print('Results for JUE_EW_Flood_Reverse already exist. No need to simulate again.')          
        else:
            need_to_save_dict['jue_ew_flood_reverse'] = True    
            # Construct the vaccination vector
            current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
            cbg_occupation_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
            leftover = 0
            
            for i in range(int(distribution_time)):
                if i==(int(distribution_time)-1): is_last = True
                else: is_last=False
                    
                cbg_occupation_msa['Vaccination_Vector'] = current_vector
                
                # Run a simulation to determine the most vulnerable group
                _, history_D2_current = run_simulation(starting_seed=STARTING_SEED_CHECKING, num_seeds=NUM_SEEDS_CHECKING, 
                                                    #vaccination_vector=vaccination_vector_mobility_flood,
                                                    vaccination_vector=current_vector,
                                                    vaccine_acceptance = vaccine_acceptance, #20211007
                                                    #protection_rate = protection_rate_most_disadvantaged)
                                                    protection_rate = PROTECTION_RATE)
                # Average history records across random seeds
                deaths_cbg_current, _= functions.average_across_random_seeds_only_death(history_D2_current, 
                                                                                        M, idxs_msa_nyt, 
                                                                                        print_results=False, draw_results=False)
                # Analyze deaths in each demographic group
                avg_final_deaths_current = deaths_cbg_current[-1,:]
                # Add simulation results to cbg table
                cbg_occupation_msa['Final_Deaths_Current'] = avg_final_deaths_current
                
                final_deaths_rate_current = np.zeros(NUM_GROUPS)
                for group_id in range(NUM_GROUPS):
                    final_deaths_rate_current[group_id] = cbg_occupation_msa[cbg_occupation_msa['Essential_Worker_Quantile']==group_id]['Final_Deaths_Current'].sum()
                    final_deaths_rate_current[group_id] /= cbg_occupation_msa[cbg_occupation_msa['Essential_Worker_Quantile']==group_id]['Sum'].sum()
                
                # Find the most vulnerable group
                #most_vulnerable_group = np.argmax(final_deaths_rate_current)
                most_vulnerable_group = np.argmin(final_deaths_rate_current)
                # Annotate the most vulnerable group
                cbg_occupation_msa['Most_Vulnerable'] = cbg_occupation_msa.apply(lambda x : 1 if x['Essential_Worker_Quantile']==most_vulnerable_group else 0, axis=1)
                
                # Distribute vaccines in the currently most vulnerable group - flooding
                new_vector = functions.vaccine_distribution_flood_new(cbg_table=cbg_occupation_msa, 
                                                                    #vaccination_ratio=VACCINATION_RATIO, 
                                                                    vaccination_ratio=RECHECK_INTERVAL, 
                                                                    demo_feat='Essential_Worker_Ratio', 
                                                                    ascending=False, 
                                                                    execution_ratio=EXECUTION_RATIO,
                                                                    leftover=leftover,
                                                                    is_last=is_last
                                                                    )
                leftover_prev = leftover
                current_vector_prev = current_vector.copy() # 20210225
                current_vector += new_vector # 20210224
                current_vector = np.clip(current_vector, None, cbg_sizes) # 20210224
                leftover = np.sum(cbg_sizes) * RECHECK_INTERVAL + leftover_prev - (np.sum(current_vector)-np.sum(current_vector_prev))
                assert((current_vector<=cbg_sizes).all())
                #print('Newly distributed vaccines: ', np.sum(new_vector))
                
            vaccination_vector_jue_ew_flood_reverse = current_vector

            # Run simulations
            _, history_D2_jue_ew_flood_reverse = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                    vaccination_vector=vaccination_vector_jue_ew_flood_reverse,
                                                    vaccine_acceptance = vaccine_acceptance, #20211007
                                                    #protection_rate = protection_rate_most_disadvantaged)
                                                    protection_rate = PROTECTION_RATE)


    ###############################################################################
    # Save results

    print('need_to_save_dict',need_to_save_dict)
    if(quick_test=='True'):
        print('Testing. Not saving results.')
    else:
        print('Saving results...\nPolicy list: ', policy_list)
        #subroot = 'vaccination_results_adaptive_reverse_%sd_%s_0.01'% (VACCINATION_TIME_STR,VACCINATION_RATIO) 
        for policy in policy_list:
            policy = policy.lower()
            if(need_to_save_dict[policy]==True):
                if(policy!='baseline'):
                    policy_savename = policy[:-8]
                    print('policy_savename: ', policy_savename)
                    filename = os.path.join(root,MSA_NAME,subroot,
                                            'test_history_D2_%s_adaptive_reverse_%sd_%s_%s_%sseeds_%s%s'%(policy_savename,VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS, notation_string, MSA_NAME)
                                            )
                    print('Save %s results at:\n'%policy, filename)            
                    #exec('np.array(history_D2_%s).tofile(os.path.join(root,MSA_NAME,subroot,\'test_history_D2_%s_adaptive_reverse_%sd_%s_%s_%sseeds_%s%s\'))' 
                    exec('np.array(history_D2_%s).tofile(filename)' 
                        % (policy))
                    need_to_save_dict[policy] = False
                print('Results saved.')

        end = time.time()
        print('Total time for thie acceptance scenario: ',(end-start))

end_all = time.time()
print('Total time: ',(end_all-start_all))

print('Vac_Rate_Race.max(): ', np.round(cbg_race_msa['Vac_Rate_Race'].max(),3),
      '\nVac_Rate_Race.min(): ', np.round(cbg_race_msa['Vac_Rate_Race'].min(),3))
print('Vac_Rate_Age.max(): ', np.round(cbg_age_msa['Vac_Rate_Age'].max(),3),
      '\nVac_Rate_Age.min(): ', np.round(cbg_age_msa['Vac_Rate_Age'].min(),3))
print('Vac_Rate_Age_Race.max(): ', np.round(cbg_age_msa['Vac_Rate_Age_Race'].max(),3),
      '\nVac_Rate_Age_Race.min(): ', np.round(cbg_age_msa['Vac_Rate_Age_Race'].min(),3))
if(consider_accessibility=='True'):
    print('Accessibility_Age_Race.max(): ', np.round(cbg_age_msa['Accessibility_Age_Race'].max(),3),
          '\nAccessibility_Age_Race.min(): ', np.round(cbg_age_msa['Accessibility_Age_Race'].min(),3))
