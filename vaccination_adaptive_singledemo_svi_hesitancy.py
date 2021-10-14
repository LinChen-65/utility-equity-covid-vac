# python vaccination_adaptive_singledemo_svi_hesitancy.py MSA_NAME VACCINATION_TIME VACCINATION_RATIO consider_hesitancy ACCEPTANCE_SCENARIO quick_test
# python vaccination_adaptive_singledemo_svi_hesitancy.py Atlanta 31 0.1 True real False

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
#import disease_model_only_modify_attack_rates
import disease_model_diff_acceptance

import time
import pdb

###############################################################################
# Constants

root = '/data/chenlin/COVID-19/Data'

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
#policy_list = ['Baseline','Age_Flood', 'Income_Flood', 'JUE_EW_Flood']
policy_list = ['SVI']
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

# Vaccine acceptance scenario: real, cf1, cf2
ACCEPTANCE_SCENARIO = sys.argv[5]
print('Vaccine acceptance scenario: ', ACCEPTANCE_SCENARIO)

# Quick Test: prototyping
quick_test = sys.argv[6]; print('Quick testing?', quick_test)
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
    m = disease_model_diff_acceptance.Model(starting_seed=starting_seed, #20211007
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
if('Age_Flood' in policy_list):
    # Calculate elder ratios
    cbg_age_msa['Elder_Absolute'] = cbg_age_msa.apply(lambda x : x['70 To 74 Years']+x['75 To 79 Years']+x['80 To 84 Years']+x['85 Years And Over'],axis=1)
    cbg_age_msa['Elder_Ratio'] = cbg_age_msa['Elder_Absolute'] / cbg_age_msa['Sum']
    # Grouping
    separators = functions.get_separators(cbg_age_msa, NUM_GROUPS, 'Elder_Ratio','Sum', normalized=True)
    cbg_age_msa['Age_Quantile'] =  cbg_age_msa['Elder_Ratio'].apply(lambda x : functions.assign_group(x, separators))


if(('Income_Flood' in policy_list) or (consider_hesitancy=='True')):
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


if('JUE_EW_Flood' in policy_list):
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

##############################################################################
# Load and scale age-aware CBG-specific attack/death rates (original)

cbg_death_rates_original = np.loadtxt(os.path.join(root, MSA_NAME, 'cbg_death_rates_original_'+MSA_NAME))
cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)
print('Age-aware CBG-specific death rates loaded. Attack rates are irrelevant to age.')

# The scaling factors are set according to a grid search
# Fix attack_scale
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
                                               vaccine_acceptance = vaccine_acceptance, #20211007
                                               #protection_rate = protection_rate_most_disadvantaged)
                                               protection_rate = PROTECTION_RATE)

###############################################################################
# Baseline: Flooding on Random Permutation

if('Baseline' in policy_list):
    print('Policy: Baseline.')
    
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
    # Run simulations
    _, history_D2_baseline = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                           vaccination_vector=vaccination_vector_baseline,
                                           vaccine_acceptance = vaccine_acceptance, #20211007
                                           #protection_rate = protection_rate_baseline)
                                           protection_rate = PROTECTION_RATE)
                                    


if(os.path.exists(os.path.join(root,MSA_NAME,
                               'vaccination_results_adaptive_%sd_%s_0.01'% (VACCINATION_TIME_STR,VACCINATION_RATIO),
                               'history_D2_svi_adaptive_%sd_%s_%s_%sseeds_acceptance_%s_%s' % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS, ACCEPTANCE_SCENARIO,MSA_NAME)))):
                               #'history_D2_jue_ew_flood_adaptive_%sd_%s_%s_%sseeds_will%s_%s' % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS,WILL_1_STR, MSA_NAME)))):
    print('Results for vaccinating the most disadvantaged already exist. No need to simulate again.')                           
else:
    ###############################################################################
    # Age_Flood, prioritize the most disadvantaged

    if('Age_Flood' in policy_list):
        policy = 'Age_Flood'
        demo_feat = 'Age'
        print('Policy: Age_Flood.')
        
        # Construct the vaccination vector    
        current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
        cbg_age_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
        leftover = 0
        
        for i in range(int(distribution_time)):
            if i==(int(distribution_time)-1): 
                is_last = True
            else:
                is_last = False
                
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
        policy = 'Income_Flood'
        demo_feat = 'Mean_Household_Income'
        print('Policy: Income_Flood.')
        
        # Construct the vaccination vector
        current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
        cbg_income_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
        leftover = 0
        
        for i in range(int(distribution_time)):
            if i==(int(distribution_time)-1):
                is_last = True
            else:
                is_last=False
                
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
        policy = 'EW_Flood'
        demo_feat = 'JUE_EW'
        print('Policy: JUE_EW_Flood.')
        
        # Construct the vaccination vector
        current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
        cbg_occupation_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
        leftover = 0
        
        for i in range(int(distribution_time)):
            if i==(int(distribution_time)-1):
                is_last = True
            else:
                is_last=False
                
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
       
        filename = os.path.join(root, MSA_NAME, 
                                'vaccination_results_adaptive_%sd_%s_0.01'%(VACCINATION_TIME_STR,VACCINATION_RATIO),
                                'history_D2_svi_%sd_%s_%sseeds_%s'%(VACCINATION_TIME_STR,VACCINATION_RATIO,NUM_SEEDS,MSA_NAME))
        np.array(history_D2_svi).tofile(filename)

    ###############################################################################
    # Save results

    if(quick_test=='True'):
        print('Testing. Not saving results.')
    else:
        print('Saving results...\nPolicy list: ', policy_list)
        if(consider_hesitancy=='True'):
            for policy in policy_list:
                policy = policy.lower()
                if(policy=='baseline'):
                    filename = os.path.join(root,MSA_NAME,
                                            'vaccination_results_adaptive_%sd_%s_0.01' %(VACCINATION_TIME_STR,VACCINATION_RATIO),
                                            'history_D2_baseline_%sd_%s_%s_%sseeds_acceptance_%s_%s' %(VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS,ACCEPTANCE_SCENARIO,MSA_NAME)
                                            )
                    np.array(history_D2_baseline).tofile(filename)
                    #np.array(history_D2_baseline).tofile(os.path.join(root,MSA_NAME,'vaccination_results_adaptive_%sd_%s_0.01','history_D2_baseline_adaptive_%sd_%s_%s_%sseeds_will%s_%s')
                    #                                                      %(VACCINATION_TIME_STR,VACCINATION_RATIO,VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS,WILL_BASELINE_STR,MSA_NAME))
                elif(policy =='svi'):
                    filename = os.path.join(root, MSA_NAME, 
                                            'vaccination_results_adaptive_%sd_%s_0.01'%(VACCINATION_TIME_STR,VACCINATION_RATIO),
                                            'history_D2_svi_adaptive_%sd_%s_%s_%sseeds_acceptance_%s_%s'%(VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS,ACCEPTANCE_SCENARIO,MSA_NAME)
                                            )
                    np.array(history_D2_svi).tofile(filename)
                else:
                    exec('np.array(history_D2_%s).tofile(os.path.join(root,MSA_NAME,\'vaccination_results_adaptive_%sd_%s_0.01\',\'history_D2_%s_adaptive_%sd_%s_%s_%sseeds_acceptance_%s_%s\'))' 
                        % (policy,VACCINATION_TIME_STR,VACCINATION_RATIO,policy,VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS, ACCEPTANCE_SCENARIO, MSA_NAME))
                    #exec('np.array(history_D2_%s).tofile(os.path.join(root,MSA_NAME,\'vaccination_results_adaptive_%sd_%s_0.01\',\'history_D2_%s_adaptive_%sd_%s_%s_%sseeds_will%s_%s\'))' 
                    #    % (policy,VACCINATION_TIME_STR,VACCINATION_RATIO,policy,VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS,WILL_1_STR, MSA_NAME))
        elif(consider_hesitancy=='False'):
            for policy in policy_list:
                policy = policy.lower()
                if(policy=='baseline'):
                    filename = os.path.join(root,MSA_NAME,
                                            'vaccination_results_adaptive_%sd_%s_0.01' %(VACCINATION_TIME_STR,VACCINATION_RATIO),
                                            'history_D2_baseline_adaptive_%sd_%s_%s_%sseeds_%s' %(VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS,MSA_NAME)
                                            )
                    np.array(history_D2_baseline).tofile(filename)
                elif(policy =='svi'):
                    filename = os.path.join(root, MSA_NAME, 
                                            'vaccination_results_adaptive_%sd_%s_0.01'%(VACCINATION_TIME_STR,VACCINATION_RATIO),
                                            'history_D2_svi_adaptive_%sd_%s_%s_%sseeds_%s'%(VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS,MSA_NAME))
                    np.array(history_D2_svi).tofile(filename)
                else:
                    exec('np.array(history_D2_%s).tofile(os.path.join(root,MSA_NAME,\'vaccination_results_adaptive_%sd_%s_0.01\',\'history_D2_%s_adaptive_%sd_%s_%s_%sseeds_acceptance_%s_%s\'))' 
                        % (policy,VACCINATION_TIME_STR,VACCINATION_RATIO,policy,VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS, ACCEPTANCE_SCENARIO, MSA_NAME))
        
        print('Results saved.')


###############################################################################
###############################################################################
###############################################################################
# Experiments for vaccinating the least disadvantaged communities
if(os.path.exists(os.path.join(root,MSA_NAME,
                               'vaccination_results_adaptive_reverse_%sd_%s_0.01'% (VACCINATION_TIME_STR,VACCINATION_RATIO),
                               'history_D2_jue_ew_flood_adaptive_reverse_%sd_%s_%s_%sseeds_acceptance_%s_%s' % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS, ACCEPTANCE_SCENARIO,MSA_NAME)))):
                               #'history_D2_jue_ew_flood_adaptive_%sd_%s_%s_%sseeds_will%s_%s' % (VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS,WILL_1_STR, MSA_NAME)))):
    print('Results for vaccinating the least disadvantaged already exist. No need to simulate again.')                           
else:
    print('\nExperiments for vaccinating the least disadvantaged communities:\n')

    ###############################################################################
    # Age_Flood, prioritize the least disadvantaged

    if('Age_Flood' in policy_list):
        policy = 'Age_Flood'
        demo_feat = 'Age'
        print('Policy: Age_Flood.')
        
        # Construct the vaccination vector    
        current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
        cbg_age_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
        leftover = 0
        
        for i in range(int(distribution_time)):
            if i==(int(distribution_time)-1): 
                is_last = True
            else:
                is_last = False
                
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
            
        vaccination_vector_age_flood = current_vector

        # Run simulations
        _, history_D2_age_flood = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                              vaccination_vector=vaccination_vector_age_flood,
                                              vaccine_acceptance = vaccine_acceptance, #20211007
                                              #protection_rate = protection_rate_most_disadvantaged)
                                              protection_rate = PROTECTION_RATE)

    ###############################################################################
    # Income_Flood, prioritize the least disadvantaged

    if('Income_Flood' in policy_list):
        policy = 'Income_Flood'
        demo_feat = 'Mean_Household_Income'
        print('Policy: Income_Flood.')
        
        # Construct the vaccination vector
        current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
        cbg_income_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
        leftover = 0
        
        for i in range(int(distribution_time)):
            if i==(int(distribution_time)-1):
                is_last = True
            else:
                is_last=False
                
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
        
        vaccination_vector_income_flood = current_vector

        # Run simulations
        _, history_D2_income_flood = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                 vaccination_vector=vaccination_vector_income_flood,
                                                 vaccine_acceptance = vaccine_acceptance, #20211007
                                                 #protection_rate = protection_rate_most_disadvantaged)
                                                 protection_rate = PROTECTION_RATE)

    ###############################################################################
    # JUE_EW_Flood, prioritize the least disadvantaged

    if('JUE_EW_Flood' in policy_list):
        policy = 'EW_Flood'
        demo_feat = 'JUE_EW'
        print('Policy: JUE_EW_Flood.')
        
        # Construct the vaccination vector
        current_vector = np.zeros(len(cbg_sizes)) # Initially: no vaccines distributed.
        cbg_occupation_msa['Covered'] = 0 # Initially, no CBG is covered by vaccination.
        leftover = 0
        
        for i in range(int(distribution_time)):
            if i==(int(distribution_time)-1):
                is_last = True
            else:
                is_last=False
                
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
            
        vaccination_vector_jue_ew_flood = current_vector

        # Run simulations
        _, history_D2_jue_ew_flood = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                 vaccination_vector=vaccination_vector_jue_ew_flood,
                                                 vaccine_acceptance = vaccine_acceptance, #20211007
                                                 #protection_rate = protection_rate_most_disadvantaged)
                                                 protection_rate = PROTECTION_RATE)


    ###############################################################################
    # Save results

    if(quick_test=='True'):
        print('Testing. Not saving results.')
    else:
        print('Saving results...')
        print('Policy list: ', policy_list)
        for policy in policy_list:
            policy = policy.lower()
            if(policy!='baseline'):
                exec('np.array(history_D2_%s).tofile(os.path.join(root,MSA_NAME,\'vaccination_results_adaptive_reverse_%sd_%s_0.01\',\'history_D2_%s_adaptive_reverse_%sd_%s_%s_%sseeds_acceptance_%s_%s\'))' 
                    % (policy,VACCINATION_TIME_STR,VACCINATION_RATIO,policy,VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS, ACCEPTANCE_SCENARIO, MSA_NAME))
                #exec('np.array(history_D2_%s).tofile(os.path.join(root,MSA_NAME,\'vaccination_results_adaptive_reverse_%sd_%s_0.01\',\'history_D2_%s_adaptive_reverse_%sd_%s_%s_%sseeds_will%s_%s\'))' 
                #    % (policy,VACCINATION_TIME_STR,VACCINATION_RATIO,policy,VACCINATION_TIME_STR,VACCINATION_RATIO,RECHECK_INTERVAL,NUM_SEEDS,WILL_2_STR, MSA_NAME))
        print('Results saved.')


    end = time.time()
    print('Total time: ',(end-start))

