# python test_vaccination_adaptive_singledemo_svi_accessibility.py --msa_name Atlanta 

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import socket
import os
import datetime
import pandas as pd
import numpy as np
import pickle
import argparse

import constants
import functions
import disease_model_test
#import disease_model_test_till20220304 #test #20220304

import time
import pdb

print('20220311')

parser = argparse.ArgumentParser()
parser.add_argument('--msa_name', 
                    help='MSA name.')
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
                    help='Vaccination protection rate')
args = parser.parse_args()

print('Consider hesitancy? ', args.consider_hesitancy)                                   
print('Quick testing?', args.quick_test)
print('Consider accessibility?', args.consider_accessibility)

# root
hostname = socket.gethostname()
print('hostname: ', hostname)
if(hostname in ['fib-dl3','rl3','rl2']):
    root = '/data/chenlin/COVID-19/Data' #dl3
    saveroot = '/data/chenlin/utility-equity-covid-vac/results'
elif(hostname=='rl4'):
    root = '/home/chenlin/COVID-19/Data' #rl4
    saveroot = '/home/chenlin/utility-equity-covid-vac/results'

# Derived variables
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[args.msa_name] 

#policy_list = ['Baseline','Age_Flood', 'Age_Flood_Reverse','Income_Flood','Income_Flood_Reverse', 'Occupation_Flood','Occupation_Flood_Reverse','SVI']
#policy_list = ['Minority', 'Minority_Reverse']
#policy_list = ['Baseline', 'Age', 'Income', 'Occupation', 'Minority']
policy_list = ['SVI_new']
print('Policy list: ', policy_list)

# Vaccine acceptance scenario
if(args.consider_hesitancy):
    if(args.acceptance_scenario == 'ALL'): ACCEPTANCE_SCENARIO_LIST = ['real','cf18','cf13','cf17']
    else: ACCEPTANCE_SCENARIO_LIST = [args.acceptance_scenario]
else:
    ACCEPTANCE_SCENARIO_LIST = ['fully']
print('Vaccine acceptance scenario list: ', ACCEPTANCE_SCENARIO_LIST)

# Setting num_seeds
if(args.quick_test):
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
distribution_time = args.vaccination_ratio / args.recheck_interval 

###############################################################################
# Functions

def run_simulation(starting_seed, num_seeds, vaccination_vector, vaccine_acceptance,protection_rate=1):
    m = disease_model_test.Model(starting_seed=starting_seed, #20211013
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
                               cbg_attack_rates_original = cbg_attack_rates_original_scaled,
                               cbg_death_rates_original = cbg_death_rates_original_scaled,
                               poi_psi=constants.parameters_dict[args.msa_name][2],
                               just_compute_r0=False,
                               latency_period=96,  # 4 days
                               infectious_period=84,  # 3.5 days
                               confirmation_rate=.1,
                               confirmation_lag=168,  # 7 days
                               death_lag=432
                               )

    m.init_endogenous_variables()

    #T1,L_1,I_1,R_1,C2,D2,total_affected, history_C2, history_D2, total_affected_each_cbg = m.simulate_disease_spread(no_print=True)    
    #return history_C2, history_D2
    final_cases, final_deaths = m.simulate_disease_spread(no_print=True, store_history=False) #20220304
    return final_deaths #20220304
    

def distribute_and_check(cbg_table, demo_feat, vaccine_acceptance, reverse=False): #20220302
    # Construct the vaccination vector
    current_vector = np.zeros(len(cbg_table)) # Initially: no vaccines distributed.
    leftover = 0
    
    for i in range(int(distribution_time)):
        if i==(int(distribution_time)-1): is_last = True
        else: is_last=False
        cbg_table['Vaccination_Vector'] = current_vector
        # Run a simulation to determine the most vulnerable group
                                            
        final_deaths_current = run_simulation(starting_seed=STARTING_SEED_CHECKING, num_seeds=NUM_SEEDS_CHECKING, #20220304
                                              vaccination_vector=current_vector,
                                              vaccine_acceptance=vaccine_acceptance, #20211007
                                              protection_rate=args.protection_rate)
        # Average history records across random seeds
        avg_final_deaths_current = np.mean(final_deaths_current, axis=0); #print(avg_final_deaths_current.shape)
        cbg_table['Final_Deaths_Current'] = avg_final_deaths_current
        
        final_deaths_rate_current = np.zeros(args.num_groups)
        for group_id in range(args.num_groups):
            final_deaths_rate_current[group_id] = cbg_table[cbg_table[demo_feat+'_Quantile']==group_id]['Final_Deaths_Current'].sum()
            final_deaths_rate_current[group_id] /= cbg_table[cbg_table[demo_feat+'_Quantile']==group_id]['Sum'].sum()
        
        # Find the most/least vulnerable group (althouth they're both named 'most_vulnerable_group')
        if(reverse):
            most_vulnerable_group = np.argmin(final_deaths_rate_current)
        else:    
            most_vulnerable_group = np.argmax(final_deaths_rate_current)
        # Annotate the most vulnerable group
        cbg_table['Most_Vulnerable'] = cbg_table.apply(lambda x : 1 if x[demo_feat+'_Quantile']==most_vulnerable_group else 0, axis=1)
        
        # Distribute vaccines in the currently most vulnerable group - flooding
        new_vector = functions.vaccine_distribution_flood_new(cbg_table=cbg_table, 
                                                            vaccination_ratio=args.recheck_interval, 
                                                            demo_feat=demo_feat, 
                                                            ascending=False, 
                                                            execution_ratio=args.execution_ratio,
                                                            leftover=leftover,
                                                            is_last=is_last
                                                            )
        leftover_prev = leftover
        current_vector_prev = current_vector.copy() # 20210225
        current_vector += new_vector # 20210224
        #print((cbg_sizes-current_vector)[current_vector.nonzero()][np.where((cbg_sizes-current_vector)[current_vector.nonzero()]!=0)])
        current_vector = np.clip(current_vector, None, cbg_sizes) # 20210224
        leftover = np.sum(cbg_sizes) * args.recheck_interval + leftover_prev - (np.sum(current_vector)-np.sum(current_vector_prev))
        assert((current_vector<=cbg_sizes).all())

    return current_vector

###############################################################################
# Load Data

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
good_list = list(msa_data['FIPS Code'].values)
print('County included: ', good_list)
del acs_data

# Load CBG ids for the MSA
cbg_ids_msa = pd.read_csv(os.path.join(root,args.msa_name,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
M = len(cbg_ids_msa)

# Mapping from cbg_ids to columns in hourly visiting matrices
cbgs_to_idxs = dict(zip(cbg_ids_msa['census_block_group'].values, range(M)))
x = {}
for i in cbgs_to_idxs:
    x[str(i)] = cbgs_to_idxs[i]

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
        
idxs_msa_all = list(x.values()) #; print('Number of CBGs in this metro area:', len(idxs_msa_all))
idxs_msa_nyt = y #; print('Number of CBGs in to compare with NYT data:', len(idxs_msa_nyt))


# Load other Safegraph demographic data, and perform grouping
#if(('Age' in policy_list) or ('Age_Reverse' in policy_list)):
if(True):
    # Calculate elder ratios
    cbg_age_msa['Elder_Absolute'] = cbg_age_msa.apply(lambda x : x['70 To 74 Years']+x['75 To 79 Years']+x['80 To 84 Years']+x['85 Years And Over'],axis=1)
    cbg_age_msa['Elder_Ratio'] = cbg_age_msa['Elder_Absolute'] / cbg_age_msa['Sum']
    # Grouping
    separators = functions.get_separators(cbg_age_msa, args.num_groups, 'Elder_Ratio','Sum', normalized=True)
    cbg_age_msa['Elder_Ratio_Quantile'] =  cbg_age_msa['Elder_Ratio'].apply(lambda x : functions.assign_group(x, separators))


if(('Income' in policy_list) or ('Income_Reverse' in policy_list) or args.consider_hesitancy):
    filepath = os.path.join(root,"ACS_5years_Income_Filtered_Summary.csv")
    cbg_income = pd.read_csv(filepath)
    cbg_income.drop(['Unnamed: 0'],axis=1, inplace=True)
    cbg_income_msa = functions.load_cbg_income_msa(cbg_income, cbg_ids_msa) #20220302
    del cbg_income
    cbg_income_msa['Sum'] = cbg_age_msa['Sum'].copy()
    # Grouping
    separators = functions.get_separators(cbg_income_msa, args.num_groups, 'Mean_Household_Income','Sum', normalized=False)
    cbg_income_msa['Mean_Household_Income_Quantile'] =  cbg_income_msa['Mean_Household_Income'].apply(lambda x : functions.assign_group(x, separators))


#if(('Occupation' in policy_list) or ('Occupation_Reverse' in policy_list)):
if(True):
    filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_c24.csv")
    cbg_occupation = pd.read_csv(filepath)
    cbg_occupation_msa = functions.load_cbg_occupation_msa(cbg_occupation, cbg_ids_msa, cbg_sizes) #20220302
    del cbg_occupation
    cbg_occupation_msa.rename(columns={'Essential_Worker_Ratio':'EW_Ratio'},inplace=True)
    # Grouping
    separators = functions.get_separators(cbg_occupation_msa, args.num_groups, 'EW_Ratio','Sum', normalized=True)
    cbg_occupation_msa['EW_Ratio_Quantile'] = cbg_occupation_msa['EW_Ratio'].apply(lambda x : functions.assign_group(x, separators))


if(('SVI' in policy_list) or ('SVI_new' in policy_list)):
    cbg_ids_msa['census_tract'] = cbg_ids_msa['census_block_group'].apply(lambda x:int(str(x)[:-1]))
    svidata = pd.read_csv(os.path.join(root, 'SVI2018_US.csv'))
    columns_of_interest = ['FIPS','RPL_THEMES']
    svidata = svidata[columns_of_interest].copy()
    svidata_msa = pd.merge(cbg_ids_msa, svidata, left_on='census_tract', right_on='FIPS', how='left')
    svidata_msa['Sum'] = cbg_age_msa['Sum'].copy()
    # Grouping #20220311
    separators = functions.get_separators(svidata_msa, args.num_groups, 'RPL_THEMES','Sum', normalized=True)
    svidata_msa['RPL_THEMES_Quantile'] = svidata_msa['RPL_THEMES'].apply(lambda x: functions.assign_group(x, separators))

#if(args.consider_accessibility=='True'):
if(True):
    # accessibility by race/ethnic
    # cbg_b03.csv: HISPANIC OR LATINO ORIGIN BY RACE
    filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_b03.csv")
    cbg_ethnic = pd.read_csv(filepath)
    # Extract pois corresponding to the metro area, by merging dataframes
    cbg_ethnic_msa = pd.merge(cbg_ids_msa, cbg_ethnic, on='census_block_group', how='left')
    del cbg_ethnic
    cbg_ethnic_msa.rename(columns={'B03002e1':'Sum',
                                   'B03002e2':'NH_Total',
                                   'B03002e3':'NH_White',
                                   'B03002e4':'NH_Black',
                                   'B03002e5':'NH_Indian',
                                   'B03002e6':'NH_Asian',
                                   'B03002e7':'NH_Hawaiian',
                                   'B03002e12':'Hispanic'}, inplace=True)
    
    # Extract columns of interest
    columns_of_interest = ['census_block_group','Sum','NH_Total','NH_White','NH_Black','NH_Indian','NH_Asian','NH_Hawaiian','Hispanic']
    cbg_ethnic_msa = cbg_ethnic_msa[columns_of_interest].copy()
    # Deal with NaN values
    cbg_ethnic_msa.fillna(0,inplace=True)
    # Deal with CBGs with 0 populations
    cbg_ethnic_msa['Sum'] = cbg_ethnic_msa['Sum'].apply(lambda x : x if x!=0 else 1)
    # Calculate "Multiple/Other Non_Hispanic"
    cbg_ethnic_msa['NH_Others'] = cbg_ethnic_msa['NH_Total'] - (cbg_ethnic_msa['NH_White']+cbg_ethnic_msa['NH_Black']+cbg_ethnic_msa['NH_Indian']+cbg_ethnic_msa['NH_Asian']+cbg_ethnic_msa['NH_Hawaiian'])

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
    cbg_ethnic_msa['Vac_Rate_Race'] = (cbg_ethnic_msa['NH_White']*vac_rate_nh_white + cbg_ethnic_msa['NH_Black']*vac_rate_nh_black + cbg_ethnic_msa['NH_Indian']*vac_rate_nh_indian
                                     +cbg_ethnic_msa['NH_Asian']*vac_rate_nh_asian + cbg_ethnic_msa['NH_Hawaiian']*vac_rate_nh_hawaiian
                                     +cbg_ethnic_msa['NH_Others']*vac_rate_nh_others + cbg_ethnic_msa['Hispanic']*vac_rate_hispanic)
    cbg_ethnic_msa['Vac_Rate_Race'] /= cbg_ethnic_msa['Sum']
    #print('cbg_ethnic_msa[\'Vac_Rate_Race\'].max(): ', cbg_ethnic_msa['Vac_Rate_Race'].max(),'\ncbg_ethnic_msa[\'Vac_Rate_Race\'].min(): ', cbg_ethnic_msa['Vac_Rate_Race'].min())

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
    cbg_age_msa['Vac_Rate_Age_Race'] = cbg_ethnic_msa['Vac_Rate_Race'] * cbg_age_msa['Vac_Rate_Age']    
    '''print('cbg_ethnic_msa[\'Vac_Rate_Race\'].max(): ', np.round(cbg_ethnic_msa['Vac_Rate_Race'].max(),3),
          '\ncbg_ethnic_msa[\'Vac_Rate_Race\'].min(): ', np.round(cbg_ethnic_msa['Vac_Rate_Race'].min(),3))
    print('cbg_age_msa[\'Vac_Rate_Age\'].max(): ', np.round(cbg_age_msa['Vac_Rate_Age'].max(),3),
          '\ncbg_age_msa[\'Vac_Rate_Age\'].min(): ', np.round(cbg_age_msa['Vac_Rate_Age'].min(),3))
    print('cbg_age_msa[\'Vac_Rate_Age_Race\'].max(): ', np.round(cbg_age_msa['Vac_Rate_Age_Race'].max(),3),
          '\ncbg_age_msa[\'Vac_Rate_Age_Race\'].min(): ', np.round(cbg_age_msa['Vac_Rate_Age_Race'].min(),3))
    '''
    # Division by vaccine acceptance to get the final accessibility
    #cbg_age_msa['Accessibility_Age_Race'] = cbg_age_msa['Vac_Rate_Age_Race'] / 



if(('Minority' in policy_list) or ('Minority_Reverse' in policy_list)): #20220302
    #cbg_ethnic_msa['Minority_Absolute'] = cbg_ethnic_msa['NH_White'].copy()
    cbg_ethnic_msa['Minority_Absolute'] = cbg_ethnic_msa['Sum'] - cbg_ethnic_msa['NH_White'] 
    cbg_ethnic_msa['Minority_Ratio'] = cbg_ethnic_msa['Minority_Absolute'] / cbg_ethnic_msa['Sum']
    # Deal with NaN values
    cbg_ethnic_msa.fillna(0,inplace=True)
    # Check whether there is NaN in cbg_tables
    if(cbg_ethnic_msa.isnull().any().any()):
        print('NaN exists in cbg_ethnic_msa. Please check.')
        pdb.set_trace()
    # Grouping: 按args.num_groups分位数，将全体CBG分为args.num_groups个组，将分割点存储在separators中
    separators = functions.get_separators(cbg_ethnic_msa, args.num_groups, 'Minority_Ratio','Sum', normalized=False)
    cbg_ethnic_msa['Minority_Ratio_Quantile'] =  cbg_ethnic_msa['Minority_Ratio'].apply(lambda x : functions.assign_group(x, separators))

    if((args.consider_hesitancy) & (args.acceptance_scenario=='new1')): #20220309
        cbg_ethnic_msa['Vac_Accept_By_Ethnicity'] = (cbg_ethnic_msa['Hispanic']*(255/357) + cbg_ethnic_msa['NH_Total']*(1212/1521)) / cbg_ethnic_msa['Sum']

if((args.consider_hesitancy) & (args.acceptance_scenario=='new1')): #20220309
    filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_b02.csv")
    cbg_race = pd.read_csv(filepath)
    cbg_race_msa = pd.merge(cbg_ids_msa, cbg_race, on='census_block_group', how='left')
    del cbg_race
    cbg_race_msa['Sum'] = cbg_age_msa['Sum']
    # Rename
    cbg_race_msa.rename(columns={'B02001e2':'White_Absolute',
                                 'B02001e3':'Black_Absolute',
                                 'B02001e5':'Asian_Absolute',
                                 'B02001e8':'Multiracial_Absolute'}, inplace=True)
    # Extract columns of interest
    columns_of_interest = ['census_block_group', 'Sum', 'White_Absolute','Black_Absolute','Asian_Absolute','Multiracial_Absolute']
    cbg_race_msa = cbg_race_msa[columns_of_interest].copy()
    cbg_race_msa['White_Ratio'] = cbg_race_msa['White_Absolute'] / cbg_race_msa['Sum']
    cbg_race_msa['Black_Ratio'] = cbg_race_msa['Black_Absolute'] / cbg_race_msa['Sum']
    cbg_race_msa['Other_Absolute'] = cbg_race_msa['Sum'] - (cbg_race_msa['White_Absolute']+cbg_race_msa['Black_Absolute']+cbg_race_msa['Asian_Absolute']+cbg_race_msa['Multiracial_Absolute'])
    cbg_race_msa['Vac_Accept_By_Race'] = (cbg_race_msa['White_Absolute']*(1083/1384) + cbg_race_msa['Black_Absolute']*(142/214) + cbg_race_msa['Asian_Absolute']*(159/173) + cbg_race_msa['Multiracial_Absolute']*(36/43) + cbg_race_msa['Other_Absolute']*(47/58))/cbg_race_msa['Sum']

##############################################################################
# Load and scale age-aware CBG-specific attack/death rates (original)

cbg_death_rates_original = np.loadtxt(os.path.join(root, args.msa_name, 'cbg_death_rates_original_'+args.msa_name))
cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)

# The scaling factors are set according to a grid search
attack_scale = 1
cbg_attack_rates_original_scaled = cbg_attack_rates_original * attack_scale
cbg_death_rates_original_scaled = cbg_death_rates_original * constants.death_scale_dict[args.msa_name]

start_all = time.time()
for ACCEPTANCE_SCENARIO in ACCEPTANCE_SCENARIO_LIST:
    print('ACCEPTANCE_SCENARIO: ', ACCEPTANCE_SCENARIO)
    start = time.time()

    # Calculate vaccine acceptance in each CBG
    if(args.consider_hesitancy):
        # Vaccine hesitancy by income #20211007
        if(ACCEPTANCE_SCENARIO in ['real','cf1','cf2','cf3','cf4','cf5','cf6','cf7','cf8']):
            cbg_income_msa['Vaccine_Acceptance'] = cbg_income_msa['Mean_Household_Income'].apply(lambda x:functions.assign_acceptance_absolute(x,ACCEPTANCE_SCENARIO))
            # Retrieve vaccine acceptance as ndarray
            vaccine_acceptance = np.array(cbg_income_msa['Vaccine_Acceptance'].copy())
        elif(ACCEPTANCE_SCENARIO in ['cf9','cf10','cf11','cf12','cf13','cf14','cf15','cf16','cf17','cf18']):
            cbg_income_msa['Vaccine_Acceptance'] = cbg_income_msa['Mean_Household_Income_Quantile'].apply(lambda x:functions.assign_acceptance_quantile(x,ACCEPTANCE_SCENARIO))
            # Retrieve vaccine acceptance as ndarray
            vaccine_acceptance = np.array(cbg_income_msa['Vaccine_Acceptance'].copy())
        elif(ACCEPTANCE_SCENARIO in ['new1']): #20220309
            cbg_income_msa['Vac_Accept_By_Income'] = cbg_income_msa['Mean_Household_Income'].apply(lambda x:functions.assign_acceptance_absolute(x,'real')) 
            # Retrieve vaccine acceptance as ndarray
            vaccine_acceptance = np.array(cbg_income_msa['Vac_Accept_By_Income'] * cbg_ethnic_msa['Vac_Accept_By_Ethnicity'] * cbg_race_msa['Vac_Accept_By_Race'])
            print('Mean: ', np.mean(vaccine_acceptance), '\nStd: ', np.std(vaccine_acceptance), '\nMax: ', np.max(vaccine_acceptance), '\nMin: ', np.min(vaccine_acceptance))
            print('Check correlation with demo feats:', 
                  '\nwith Elder_Ratio: ', np.round(np.corrcoef(vaccine_acceptance, cbg_age_msa['Elder_Ratio'])[0][1], 3),
                  '\nwith Mean_Household_Income: ', np.round(np.corrcoef(vaccine_acceptance, cbg_income_msa['Mean_Household_Income'])[0][1], 3),
                  '\nwith EW_Ratio: ', np.round(np.corrcoef(vaccine_acceptance, cbg_occupation_msa['EW_Ratio'])[0][1], 3),
                  '\nwith White_Ratio: ', np.round(np.corrcoef(vaccine_acceptance, cbg_race_msa['White_Ratio'])[0][1], 3),
                  '\nwith Black_Ratio: ', np.round(np.corrcoef(vaccine_acceptance, cbg_race_msa['Black_Ratio'])[0][1], 3),
                  '\nwith Minority_Ratio: ', np.round(np.corrcoef(vaccine_acceptance, cbg_ethnic_msa['Minority_Ratio'])[0][1], 3)
                  )
            #pdb.set_trace()
    else:
        vaccine_acceptance = np.ones(len(cbg_sizes)) # fully accepted scenario

    if(args.consider_accessibility):
        # Division by vaccine acceptance to get the final accessibility
        cbg_age_msa['Accessibility_Age_Race'] = cbg_age_msa['Vac_Rate_Age_Race'] / vaccine_acceptance
        # Find the minimum non-zero value
        a = np.array(cbg_age_msa['Accessibility_Age_Race'])
        a = a[a>0]
        print('Minimum nonzero accessibility: ', min(a))
        cbg_age_msa['Accessibility_Age_Race'] = cbg_age_msa['Accessibility_Age_Race'].apply(lambda x : min(a) if ((x==0) or (np.isnan(x))) else x) #20220311
        vaccine_accessibility = np.array(cbg_age_msa['Accessibility_Age_Race']) #20220225
        print('vaccine_accessibility: ', vaccine_accessibility)
        print('vaccine_acceptance: ', vaccine_acceptance)
        #print('accessibility always smaller than acceptance?', (vaccine_accessibility<=vaccine_acceptance).all())
        if(cbg_age_msa.isnull().any().any()):
            print('NaN exists in cbg_ethnic_msa. Please check.')
            pdb.set_trace()
        vaccine_acceptance = np.array(cbg_age_msa['Accessibility_Age_Race']) #以此代替原来的acceptance参数传入函数

        print('cbg_age_msa[\'Accessibility_Age_Race\'].max(): ', np.round(cbg_age_msa['Accessibility_Age_Race'].max(),3),
             '\ncbg_age_msa[\'Accessibility_Age_Race\'].min(): ', np.round(cbg_age_msa['Accessibility_Age_Race'].min(),3))


    for policy in policy_list: 
        # Subroot to store result files
        subroot = f'vac_results_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}' #20220302
        if not os.path.exists(os.path.join(saveroot, subroot)): # if folder does not exist, create one. #2022032
            os.makedirs(os.path.join(saveroot, subroot))

        # Notation string to distinguish different vaccine acceptance/accessibility scenarios
        if(not args.consider_accessibility):
            if(args.consider_hesitancy):
                notation_string = 'acceptance_%s_'%ACCEPTANCE_SCENARIO
            else:
                notation_string = ''
        else:
            if(args.consider_hesitancy):
                notation_string = 'access_acceptance_%s_'%ACCEPTANCE_SCENARIO
            else:
                notation_string = 'access_'

        ###############################################################################
        # No_Vaccination

        if (policy == 'No_Vaccination'):
            print('\nPolicy: No_Vaccination.')
            # Construct the vaccination vector
            vaccination_vector_no_vaccination = np.zeros(len(cbg_sizes))
            # Run simulations
            this_start = time.time()
            final_deaths = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                          vaccination_vector=vaccination_vector_no_vaccination,
                                          vaccine_acceptance = vaccine_acceptance, #20211007
                                          protection_rate = args.protection_rate)                                   
            print('Time: ', time.time() - this_start)
            pdb.set_trace()

        ###############################################################################
        # Baseline: Flooding on Random Permutation

        if(policy == 'Baseline'):
            print('\nPolicy: Baseline.')
            filename = os.path.join(saveroot, subroot, f'final_deaths_{policy.lower()}_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}_{NUM_SEEDS}seeds_{notation_string}{args.msa_name}') #20220304
            if(os.path.exists(filename)):
                print(f'Results for {policy} already exist. No need to simulate again.')     
            else:
                # Construct the vaccination vector
                random_permutation = np.arange(len(cbg_age_msa))
                np.random.seed(42)
                np.random.shuffle(random_permutation)
                cbg_age_msa['Random_Permutation'] = random_permutation
                vaccination_vector_baseline = functions.vaccine_distribution_flood(cbg_table=cbg_age_msa, 
                                                                                vaccination_ratio=args.vaccination_ratio, 
                                                                                demo_feat='Random_Permutation', 
                                                                                ascending=None,
                                                                                execution_ratio=1
                                                                                )
                # Run simulations
                final_deaths = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                        vaccination_vector=vaccination_vector_baseline,
                                                        vaccine_acceptance = vaccine_acceptance, #20211007
                                                        protection_rate = args.protection_rate)
                if(args.quick_test): print('Testing. Not saving results.')
                else:
                    print(f'Save {policy} results at:\n{filename}.')
                    final_deaths.tofile(filename) 

        ###############################################################################
        # Experiments for vaccinating the least disadvantaged communities
        ###############################################################################
   
        print('\nExperiments for prioritizing the most disadvantaged communities...')          

        ###############################################################################
        # Age, prioritize the most disadvantaged

        if(policy == 'Age'):
            print('\nPolicy: Age.')
            filename = os.path.join(saveroot, subroot, f'final_deaths_{policy.lower()}_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}_{NUM_SEEDS}seeds_{notation_string}{args.msa_name}') #20220304
            if(os.path.exists(filename)):
                print(f'Results for {policy} already exist. No need to simulate again.')     
            else:
                cbg_table = cbg_age_msa
                demo_feat = 'Elder_Ratio'
                vaccination_vector_age = distribute_and_check(cbg_table, demo_feat, vaccine_acceptance, reverse=False) #20220302

                # Run simulations
                final_deaths = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                   vaccination_vector=vaccination_vector_age,
                                                   vaccine_acceptance = vaccine_acceptance, #20211007
                                                   protection_rate = args.protection_rate)
                if(args.quick_test): print('Testing. Not saving results.')
                else:
                    print(f'Save {policy} results at:\n{filename}.')
                    final_deaths.tofile(filename) 

        ###############################################################################
        # Income, prioritize the most disadvantaged

        if(policy == 'Income'):
            print('\nPolicy: Income.')
            filename = os.path.join(saveroot, subroot, f'final_deaths_{policy.lower()}_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}_{NUM_SEEDS}seeds_{notation_string}{args.msa_name}') #20220304
            if(os.path.exists(filename)):
                print(f'Results for {policy} already exist. No need to simulate again.')     
            else:
                cbg_table = cbg_income_msa
                demo_feat = 'Mean_Household_Income'
                vaccination_vector_income = distribute_and_check(cbg_table, demo_feat, vaccine_acceptance, reverse=False) #20220302

                # Run simulations
                final_deaths = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                            vaccination_vector=vaccination_vector_income,
                                                            vaccine_acceptance = vaccine_acceptance, #20211007
                                                            protection_rate = args.protection_rate)
                if(args.quick_test): print('Testing. Not saving results.')
                else:
                    print(f'Save {policy} results at:\n{filename}.')
                    final_deaths.tofile(filename) 

        ###############################################################################
        # Occupation, prioritize the most disadvantaged

        if(policy == 'Occupation'):
            print('\nPolicy: Occupation.')
            filename = os.path.join(saveroot, subroot, f'final_deaths_{policy.lower()}_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}_{NUM_SEEDS}seeds_{notation_string}{args.msa_name}') #20220304
            if(os.path.exists(filename)):
                print(f'Results for {policy} already exist. No need to simulate again.')     
            else:
                cbg_table = cbg_occupation_msa
                demo_feat = 'EW_Ratio'
                vaccination_vector_occupation = distribute_and_check(cbg_table, demo_feat, vaccine_acceptance, reverse=False) #20220302

                # Run simulations
                final_deaths = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                        vaccination_vector=vaccination_vector_occupation,
                                                        vaccine_acceptance = vaccine_acceptance, #20211007
                                                        protection_rate = args.protection_rate)
                if(args.quick_test): print('Testing. Not saving results.')
                else:
                    print(f'Save {policy} results at:\n{filename}.')
                    final_deaths.tofile(filename) 

        ###############################################################################
        # SVI, prioritize the most disadvantaged

        if(policy == 'SVI'):
            print('Policy: SVI.')
            if(os.path.exists(os.path.join(root,args.msa_name,subroot,
                                    'test_history_D2_svi_adaptive_%sd_%s_%s_%sseeds_%s%s' % (str(args.vaccination_time),args.vaccination_ratio,args.recheck_interval,NUM_SEEDS, notation_string,args.msa_name)))):
                print('Results for SVI already exist. No need to simulate again.')   
            else:
                # Construct the vaccination vector    
                vaccination_vector_svi = functions.vaccine_distribution_flood(cbg_table=svidata_msa, 
                                                                            vaccination_ratio=args.vaccination_ratio, 
                                                                            demo_feat='RPL_THEMES', 
                                                                            ascending=False,
                                                                            execution_ratio=1
                                                                            )
                # Run simulations
                _, history_D2_svi = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                vaccination_vector=vaccination_vector_svi,
                                                vaccine_acceptance = vaccine_acceptance, #20211007
                                                protection_rate = args.protection_rate)

        # New version #20220311
        if(policy == 'SVI_new'):
            print(f'Policy: {policy}.')
            filename = os.path.join(saveroot, subroot, f'final_deaths_{policy.lower()}_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}_{NUM_SEEDS}seeds_{notation_string}{args.msa_name}') #20220304
            if(os.path.exists(filename)):
                print(f'Results for {policy} already exist. No need to simulate again.')   
            else:   
                cbg_table = svidata_msa
                demo_feat = 'RPL_THEMES'
                this_start = time.time()
                vaccination_vector_svi_new = distribute_and_check(cbg_table, demo_feat, vaccine_acceptance, reverse=False)
                print('Policy constructed. Time: ', time.time()-this_start)
                final_deaths = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, #20220304
                                              vaccination_vector=vaccination_vector_svi_new,
                                              vaccine_acceptance = vaccine_acceptance, #20211007
                                              protection_rate = args.protection_rate)
                if(args.quick_test): print('Testing. Not saving results.')
                else:
                    print(f'Save {policy} results at:\n{filename}.')
                    final_deaths.tofile(filename) 

        ###############################################################################
        # Minority, prioritized the most disadvantaged

        if(policy == 'Minority'):
            print('\nPolicy: Minority.')  
            #filename = os.path.join(saveroot, subroot, f'history_D2_{policy.lower()}_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}_{NUM_SEEDS}seeds_{notation_string}{args.msa_name}')      
            filename = os.path.join(saveroot, subroot, f'final_deaths_{policy.lower()}_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}_{NUM_SEEDS}seeds_{notation_string}{args.msa_name}') #20220304
            if(os.path.exists(filename)):
                print('Results for Minority already exist. No need to simulate again.')       
            else:
                cbg_table = cbg_ethnic_msa
                demo_feat = 'Minority_Ratio'
                this_start = time.time()
                vaccination_vector_minority = distribute_and_check(cbg_table, demo_feat, vaccine_acceptance, reverse=False) #20220302
                print('Policy constructed. Time: ', time.time()-this_start)
                # Run simulations
                #_, history_D2_minority = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                final_deaths = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, #20220304
                                              vaccination_vector=vaccination_vector_minority,
                                              vaccine_acceptance = vaccine_acceptance, #20211007
                                              protection_rate = args.protection_rate)
                if(args.quick_test): print('Testing. Not saving results.')
                else:
                    print(f'Save {policy} results at:\n{filename}.')            
                    #np.array(history_D2_minority).tofile(filename)
                    final_deaths.tofile(filename) #20220304
                
    ###############################################################################
    # Experiments for vaccinating the least disadvantaged communities
    ###############################################################################

    print('\nExperiments for prioritizing the least disadvantaged communities...')

    ###############################################################################
    # Age, prioritize the least disadvantaged

    if(policy == 'Age_Reverse'):
        print('Policy: Age_Reverse.')
        if(os.path.exists(os.path.join(root, args.msa_name, subroot,
                                'test_history_D2_age_flood_adaptive_reverse_%sd_%s_%s_%sseeds_%s%s' % (str(args.vaccination_time),args.vaccination_ratio,args.recheck_interval,NUM_SEEDS, notation_string,args.msa_name)))):
            print('Results for Age_Reverse already exist. No need to simulate again.')                          
        else: 
            cbg_table = cbg_age_msa
            demo_feat = 'Elder_Ratio'                  
            vaccination_vector_age_reverse = distribute_and_check(cbg_table, demo_feat, reverse=True) #20220302

            # Run simulations
            _, history_D2_age_reverse = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                vaccination_vector=vaccination_vector_age_reverse,
                                                vaccine_acceptance = vaccine_acceptance, #20211007
                                                protection_rate = args.protection_rate)

    ###############################################################################
    # Income, prioritize the least disadvantaged

    if(policy == 'Income_Reverse'):
        print('Policy: Income_Reverse.')
        if(os.path.exists(os.path.join(root, args.msa_name, subroot, 
                                'test_history_D2_income_flood_adaptive_reverse_%sd_%s_%s_%sseeds_%s%s' % (str(args.vaccination_time),args.vaccination_ratio,args.recheck_interval,NUM_SEEDS, notation_string,args.msa_name)))):
            print('Results for Income_Reverse already exist. No need to simulate again.')          
        else:
            cbg_table = cbg_income_msa
            demo_feat = 'Mean_Household_Income'            
            vaccination_vector_income_reverse = distribute_and_check(cbg_table, demo_feat, reverse=True) #20220302

            # Run simulations
            _, history_D2_income_reverse = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                    vaccination_vector=vaccination_vector_income_reverse,
                                                    vaccine_acceptance = vaccine_acceptance, #20211007
                                                    protection_rate = args.protection_rate)

    ###############################################################################
    # Occupation, prioritize the least disadvantaged

    if(policy == 'Occupation_Reverse'):     
        print('Policy: Occupation_Reverse.')
        if(os.path.exists(os.path.join(root, args.msa_name, subroot,
                                'test_history_D2_jue_ew_flood_adaptive_reverse_%sd_%s_%s_%sseeds_%s%s' % (str(args.vaccination_time),args.vaccination_ratio,args.recheck_interval,NUM_SEEDS, notation_string,args.msa_name)))):
            print('Results for Occupation_Reverse already exist. No need to simulate again.')          
        else: 
            cbg_table = cbg_occupation_msa
            demo_feat = 'EW_Ratio'            
            vaccination_vector_occupation_reverse = distribute_and_check(cbg_table, demo_feat, reverse=True) #20220302

            # Run simulations
            _, history_D2_occupation_reverse = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
                                                    vaccination_vector=vaccination_vector_occupation_reverse,
                                                    vaccine_acceptance = vaccine_acceptance, #20211007
                                                    protection_rate = args.protection_rate)

    ###############################################################################
    # Minority, prioritize the least disadvantaged

    if(policy == 'Minority_Reverse'):
        print('\nPolicy: Minority_Reverse.')    
        #filename = os.path.join(saveroot, subroot, f'history_D2_{policy.lower()}_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}_{NUM_SEEDS}seeds_{notation_string}{args.msa_name}')          
        filename = os.path.join(saveroot, subroot, f'final_deaths_{policy.lower()}_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}_{NUM_SEEDS}seeds_{notation_string}{args.msa_name}') #20220304         
        if(os.path.exists(filename)):
            print('Results for Minority_Reverse already exist. No need to simulate again.')         
        else:
            cbg_table = cbg_ethnic_msa
            demo_feat = 'Minority_Ratio'
            vaccination_vector_minority_reverse = distribute_and_check(cbg_table, demo_feat, vaccine_acceptance, reverse=True) #20220302
            # Run simulations
            #_, history_D2_minority_reverse = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, 
            final_deaths = run_simulation(starting_seed=STARTING_SEED, num_seeds=NUM_SEEDS, #20220304
                                          vaccination_vector=vaccination_vector_minority_reverse,
                                          vaccine_acceptance = vaccine_acceptance, #20211007
                                          protection_rate = args.protection_rate)
            # Save results
            if(args.quick_test): print('Testing. Not saving results.')
            else:
                policy = policy.lower()
                print(f'Save {policy} results at:\n{filename}.')            
                #np.array(history_D2_minority_reverse).tofile(filename)
                final_deaths.tofile(filename) #20220304

     
    end = time.time()
    print('Total time for thie acceptance scenario: ',(end-start))

end_all = time.time()
print('Total time: ',(end_all-start_all))