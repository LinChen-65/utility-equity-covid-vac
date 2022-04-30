# python generate_infect_same_diff.py --msa_name Atlanta 

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

import time
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--msa_name', 
                    help='MSA name.')
parser.add_argument('--safegraph_root', default='/data/chenlin/COVID-19/Data',
                    help='Safegraph data root.') 
args = parser.parse_args()                    

###############################################################################
# Constants

#root = '/data/chenlin/COVID-19/Data'
root = os.getcwd()
dataroot = os.path.join(root, 'data')
saveroot = os.path.join(root, 'results')

MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)

MSA_NAME = args.msa_name; print('MSA_NAME: ',MSA_NAME)
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME] 

###############################################################################
# Load Data

# Load POI-CBG visiting matrices
f = open(os.path.join(dataroot, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()

# Load precomputed parameters to adjust(clip) POI dwell times
d = pd.read_csv(os.path.join(dataroot, 'parameters_%s.csv' % MSA_NAME)) 
all_hours = functions.list_hours_in_range(MIN_DATETIME, MAX_DATETIME)
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

# Income Data Resource 1: ACS 5-year (2013-2017) Data
filepath = os.path.join(args.safegraph_root,"ACS_5years_Income_Filtered_Summary.csv")
cbg_income = pd.read_csv(filepath)
# Drop duplicate column 'Unnamed:0'
cbg_income.drop(['Unnamed: 0'],axis=1, inplace=True)
# Extract pois corresponding to the metro area (Philadelphia), by merging dataframes
cbg_income_msa = pd.merge(cbg_ids_msa, cbg_income, on='census_block_group', how='left')
# Deal with NaN values
cbg_income_msa.fillna(0,inplace=True)
# Rename
cbg_income_msa.rename(columns = {'total_households':'Total_Households',},inplace=True)
cbg_age_msa['Total_Households'] = cbg_income_msa['Total_Households'].copy()
# Average people in a household
avg_household_size = cbg_age_msa['Sum'] / cbg_age_msa['Total_Households']

# Check whether there is NaN or inf in avg_household_size
print('Any NaN in avg_household_size?', np.isnan(avg_household_size).any())
where_are_inf = np.isinf(avg_household_size)
avg_household_size[where_are_inf] = np.nan
mean_value = np.nanmean(avg_household_size)
# Deal with nan and inf (https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
avg_household_size = np.nan_to_num(avg_household_size,nan=mean_value,posinf=mean_value,neginf=mean_value)
avg_household_size=np.clip(avg_household_size,1,None)
avg_household_size=np.clip(avg_household_size,None,10)
# Check whether there is NaN or inf in avg_household_size
print('Any NaN in avg_household_size?', np.isnan(avg_household_size).any())
print('Average people in a household:', avg_household_size)


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

nyt_included = np.zeros(len(idxs_msa_all))
for i in range(len(nyt_included)):
    if(i in idxs_msa_nyt):
        nyt_included[i] = 1
cbg_age_msa['NYT_Included'] = nyt_included.copy()

# Retrieve the attack rate for the whole MSA (home_beta, fitted for each MSA)
home_beta = constants.parameters_dict[MSA_NAME][1]
print('MSA home_beta retrieved.')

# Compute cbg_avg_N
start = time.time()

if(os.path.exists(os.path.join(saveroot, '3cbg_avg_infect_same_%s.npy'%MSA_NAME))):
    print('avg_N: Load existing file.')
    cbg_avg_infect_same = np.load(os.path.join(saveroot, '3cbg_avg_infect_same_%s.npy'%MSA_NAME))
    cbg_avg_infect_diff = np.load(os.path.join(saveroot, '3cbg_avg_infect_diff_%s.npy'%MSA_NAME))
else:
    print('avg_N: Compute on the fly.')
    hourly_N_same_list = []
    hourly_N_diff_list = []
    for hour_idx in range(len(poi_cbg_visits_list)):
        if(hour_idx%100==0): 
            print(hour_idx)
        poi_cbg_visits_array = poi_cbg_visits_list[hour_idx].toarray() # Extract the visit matrix for this hour
        # poi_cbg_visits_array.shape: (num_poi,num_cbg) e.g.(28713, 2943)
        cbg_out_pop = np.sum(poi_cbg_visits_array, axis=0)
        cbg_out_rate = cbg_out_pop / cbg_sizes # 每个CBG当前外出人数(去往任何POI)占总人数比例
        cbg_in_pop = cbg_sizes - cbg_out_pop
        cbg_in_rate = cbg_in_pop / cbg_sizes # 每个CBG当前留守人数占总人数比例
        poi_pop = np.sum(poi_cbg_visits_array, axis=1) # 每个POI当前人数(来自所有CBG) 

        hourly_N_same = cbg_in_pop * avg_household_size * home_beta
        hourly_N_diff = np.matmul(poi_cbg_visits_array.T, poi_pop * poi_trans_rate)
        
        hourly_N_same_list.append(hourly_N_same)
        hourly_N_diff_list.append(hourly_N_diff)
        
        if(hour_idx==10):
            print('cbg_out_pop.shape:',cbg_out_pop.shape)
            print('poi_pop.shape:',poi_pop.shape)
            print('len(hourly_N_same_list):',len(hourly_N_same_list))

    cbg_avg_infect_same = np.mean(np.array(hourly_N_same_list),axis=0)
    cbg_avg_infect_diff = np.mean(np.array(hourly_N_diff_list),axis=0)

    np.save(os.path.join(root, '3cbg_avg_infect_same_%s'%MSA_NAME), cbg_avg_infect_same) 
    np.save(os.path.join(root, '3cbg_avg_infect_diff_%s'%MSA_NAME), cbg_avg_infect_diff) 
    
print('cbg_avg_infect_same.shape:',cbg_avg_infect_same.shape)

end = time.time()
print('Time: ',(end-start)) # (SanFrancisco used around 10min.)