# python regression_randombag_sample_average.py MSA_NAME LEN_SEEDS NUM_SAMPLE SAMPLE_FRAC
# python regression_randombag_sample_average.py Atlanta 3 20 0.2

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import sys
import os

import constants
import functions

import numpy as np
import pandas as pd
import pickle

from sklearn import preprocessing
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from scipy import stats
import math

import pdb

###############################################################################
# Main variables

root = '/data/chenlin/COVID-19/Data'

timestring = '20210206'
MSA_NAME = sys.argv[1]
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME]
print('\nMSA_NAME: ',MSA_NAME)

RANDOM_SEED_LIST = [66,42,5]
LEN_SEEDS = int(sys.argv[2])
print('Num of random seeds: ', LEN_SEEDS)
print('Random seeds: ',RANDOM_SEED_LIST[:LEN_SEEDS])

NUM_SAMPLE = int(sys.argv[3]); print('Num of samples: ', NUM_SAMPLE)
SAMPLE_FRAC = float(sys.argv[4]); print('Sample fraction: ', SAMPLE_FRAC)

###############################################################################
# Functions

# For each randombag, compute its average features.
def get_avg_feat(cbg_list, data_df, feat_str):
    values = []
    weights = []
    for cbg in cbg_list:
        values.append(data_df.iloc[cbg][feat_str])
        weights.append(data_df.iloc[cbg]['Sum'])
    return np.average(np.array(values),weights=weights)

# For each randombag, compute weighted std of its features.
def get_std_feat(cbg_list, data_df, feat_str):
    # Ref: https://www.codenong.com/2413522/
    values = []
    weights = []
    for cbg in cbg_list:
        values.append(data_df.iloc[cbg][feat_str])
        weights.append(data_df.iloc[cbg]['Sum'])
    average = np.average(np.array(values),weights=weights)
    return math.sqrt(np.average((values-average)**2, weights=weights))

###############################################################################
# Load Common Data: No need for reloading when switching among differet MSAs.

# Load POI-CBG visiting matrices
f = open(os.path.join(root, MSA_NAME, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
poi_cbg_visits_list = pickle.load(f)
f.close()

# Load ACS Data for matching with NYT Data
acs_data = pd.read_csv(os.path.join(root,'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]

# Load NYT Data
nyt_data = pd.read_csv(os.path.join(root, 'us-counties.csv'))

# Load Demographic Data
filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_b01.csv")
cbg_agesex = pd.read_csv(filepath)

filepath = os.path.join(root,"ACS_5years_Income_Filtered_Summary.csv")
cbg_income = pd.read_csv(filepath)
# Drop duplicate column 'Unnamed:0'
cbg_income.drop(['Unnamed: 0'],axis=1, inplace=True)

filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_c24.csv")
cbg_occupation = pd.read_csv(filepath)

# cbg_b03.csv: Ethnic #20220308
filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_b03.csv")
cbg_ethnic = pd.read_csv(filepath)

###############################################################################
data = pd.DataFrame()

cbg_death_rates_original = np.loadtxt(os.path.join(root, MSA_NAME, 'cbg_death_rates_original_'+MSA_NAME))
# The scaling factors are set according to a grid search
cbg_death_rates_scaled = cbg_death_rates_original * constants.death_scale_dict[MSA_NAME]

# Extract data specific to one msa, according to ACS data
# MSA list
msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
good_list = list(msa_data['FIPS Code'].values)

# Load CBG ids belonging to a specific metro area
# cbg_ids_msa
cbg_ids_msa = pd.read_csv(os.path.join(root,MSA_NAME,'%s_cbg_ids.csv' % MSA_NAME_FULL))
cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
M = len(cbg_ids_msa)
# Mapping from cbg_ids to columns in hourly visiting matrices
cbgs_to_idxs = dict(zip(cbg_ids_msa['census_block_group'].values, range(M)))
x = {}
for i in cbgs_to_idxs:
    x[str(i)] = cbgs_to_idxs[i]
idxs_msa_all = list(x.values())

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

# Extract CBGs belonging to the MSA - https://covid-mobility.stanford.edu//datasets/
cbg_agesex_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
cbg_age_msa = cbg_agesex_msa.copy()
del cbg_agesex_msa
# Add up males and females of the same age, according to the detailed age list (DETAILED_AGE_LIST)
# which is defined in Constants.py
for i in range(3,25+1): # 'B01001e3'~'B01001e25'
    male_column = 'B01001e'+str(i)
    female_column = 'B01001e'+str(i+24)
    cbg_age_msa[constants.DETAILED_AGE_LIST[i-3]] = cbg_age_msa.apply(lambda x : x[male_column]+x[female_column],axis=1)
# Rename
cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
columns_of_interest = ['census_block_group','Sum'] + constants.DETAILED_AGE_LIST
cbg_age_msa = cbg_age_msa[columns_of_interest].copy()
# Deal with NaN values
cbg_age_msa.fillna(0,inplace=True) # print('Any NaN?', cbg_age_msa.isnull().any().any())
cbg_age_msa['Elder_Absolute'] = cbg_age_msa.apply(lambda x : x['70 To 74 Years']+x['75 To 79 Years']+x['80 To 84 Years']+x['85 Years And Over'],axis=1)
# Deal with CBGs with 0 populations
cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
cbg_age_msa['Elder_Ratio'] = cbg_age_msa['Elder_Absolute'] / cbg_age_msa['Sum']

cbg_sizes = cbg_age_msa['Sum'].values
cbg_sizes = np.array(cbg_sizes,dtype='int32')
    
# Income Data Resource 1: ACS 5-year (2013-2017) Data
# Extract pois corresponding to the metro area (Philadelphia), by merging dataframes
cbg_income_msa = pd.merge(cbg_ids_msa, cbg_income, on='census_block_group', how='left')
# Deal with NaN values
cbg_income_msa.fillna(0,inplace=True)
# Add information of cbg populations, from cbg_age_Phi(cbg_b01.csv)
cbg_income_msa['Sum'] = cbg_age_msa['Sum'].copy()
# Rename
cbg_income_msa.rename(columns = {'total_household_income':'Total_Household_Income', 
                                 'total_households':'Total_Households',
                                 'mean_household_income':'Mean_Household_Income'},inplace=True)

# Extract pois corresponding to the metro area, by merging dataframes
cbg_occupation_msa = pd.merge(cbg_ids_msa, cbg_occupation, on='census_block_group', how='left')
columns_of_essential_workers = list(constants.ew_rate_dict.keys())
for column in columns_of_essential_workers:
    cbg_occupation_msa[column] = cbg_occupation_msa[column].apply(lambda x : x*constants.ew_rate_dict[column])
cbg_occupation_msa['Essential_Worker_Absolute'] = cbg_occupation_msa.apply(lambda x : x[columns_of_essential_workers].sum(), axis=1)
cbg_occupation_msa['Sum'] = cbg_age_msa['Sum']
cbg_occupation_msa['EW_Ratio'] = cbg_occupation_msa['Essential_Worker_Absolute'] / cbg_occupation_msa['Sum']
columns_of_interest = ['census_block_group','Sum','Essential_Worker_Absolute','EW_Ratio']
cbg_occupation_msa = cbg_occupation_msa[columns_of_interest].copy()
# Deal with NaN values
cbg_occupation_msa.fillna(0,inplace=True)

# Ethnicity #20220308
cbg_ethnic_msa = functions.load_cbg_ethnic_msa(cbg_ethnic, cbg_ids_msa, cbg_sizes)
del cbg_ethnic
cbg_minority_msa = cbg_ethnic_msa

###############################################################################
# Obtain vulnerability and damage, according to theoretical analysis

nyt_included = np.zeros(len(idxs_msa_all))
for i in range(len(nyt_included)):
    if(i in idxs_msa_nyt):
        nyt_included[i] = 1
cbg_age_msa['NYT_Included'] = nyt_included.copy()

# Retrieve the attack rate for the whole MSA (home_beta, fitted for each MSA)
home_beta = constants.parameters_dict[MSA_NAME][1]
print('MSA home_beta retrieved.')

# Retrieve cbg_avg_infect_same, cbg_avg_infect_diff
# As planned, they have been computed in 'tradeoff_md_mv_theory.py'.
# Use them to get data['Vulnerability'] and data['Damage']
if(os.path.exists(os.path.join(root, '3cbg_avg_infect_same_%s.npy'%MSA_NAME))):
    print('cbg_avg_infect_same, cbg_avg_infect_diff: Load existing file.')
    cbg_avg_infect_same = np.load(os.path.join(root, '3cbg_avg_infect_same_%s.npy'%MSA_NAME))
    cbg_avg_infect_diff = np.load(os.path.join(root, '3cbg_avg_infect_diff_%s.npy'%MSA_NAME))
else:
    print('cbg_avg_infect_same, cbg_avg_infect_diff: Compute on the fly.')
    pdb.set_trace()
    hourly_N_same_list = []
    hourly_N_diff_list = []
    for hour_idx in range(len(poi_cbg_visits_list)):
        if(hour_idx%100==0): print(hour_idx)
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

cbg_age_msa['Death_Rate'] =  cbg_death_rates_scaled

# Deal with nan and inf (https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
cbg_avg_infect_same = np.nan_to_num(cbg_avg_infect_same,nan=0,posinf=0,neginf=0)
cbg_avg_infect_diff = np.nan_to_num(cbg_avg_infect_diff,nan=0,posinf=0,neginf=0)
cbg_age_msa['Infect'] = cbg_avg_infect_same + cbg_avg_infect_diff
# Check whether there is NaN in cbg_tables
print('Any NaN in cbg_age_msa[\'Infect\']?', cbg_age_msa['Infect'].isnull().any().any())

SEIR_at_30d = np.load(os.path.join(root, 'SEIR_at_30d.npy'),allow_pickle=True).item()
S_ratio = SEIR_at_30d[MSA_NAME]['S'] / (cbg_sizes.sum())
I_ratio = SEIR_at_30d[MSA_NAME]['I'] / (cbg_sizes.sum())
print('S_ratio:',S_ratio,'I_ratio:',I_ratio)

# Compute the average death rate for the whole MSA: perform another weighted average over all CBGs
avg_death_rates_scaled = np.matmul(cbg_sizes.T, cbg_death_rates_scaled) / np.sum(cbg_sizes)
print('avg_death_rates_scaled.shape:',avg_death_rates_scaled.shape) # shape: (), because it is a scalar


# Normalize by cbg population
cbg_avg_infect_same_norm = cbg_avg_infect_same / cbg_sizes
cbg_avg_infect_diff_norm = cbg_avg_infect_diff / cbg_sizes
cbg_avg_infect_all_norm = cbg_avg_infect_same_norm + cbg_avg_infect_diff_norm
# alpha_bar
avg_death_rates_scaled = np.matmul(cbg_sizes.T, cbg_death_rates_scaled) / np.sum(cbg_sizes)

# New new method # 20210619
cbg_vulnerability = cbg_avg_infect_all_norm * cbg_death_rates_scaled 
cbg_secondary_damage = cbg_avg_infect_all_norm * (cbg_avg_infect_all_norm*(S_ratio/I_ratio)) * avg_death_rates_scaled
cbg_damage = cbg_vulnerability + cbg_secondary_damage

cbg_age_msa['Vulnerability'] = cbg_vulnerability.copy()
cbg_age_msa['Damage'] = cbg_damage.copy()

cbg_age_msa['Vulner_Rank'] = cbg_age_msa['Vulnerability'].rank(ascending=False,method='first') 
cbg_age_msa['Damage_Rank'] = cbg_age_msa['Damage'].rank(ascending=False,method='first')

###############################################################################
# Collect data together

data = pd.DataFrame()

data['Sum'] = cbg_age_msa['Sum'].copy()
data['Elder_Ratio'] = cbg_age_msa['Elder_Ratio'].copy()
data['Mean_Household_Income'] = cbg_income_msa['Mean_Household_Income'].copy()
data['EW_Ratio'] = cbg_occupation_msa['EW_Ratio'].copy()
data['Minority_Ratio'] = cbg_minority_msa['Minority_Ratio'].copy() #20220308

data['Vulnerability'] = cbg_age_msa['Vulnerability'].copy()
data['Damage'] = cbg_age_msa['Damage'].copy()

data['Vulner_Rank'] = cbg_age_msa['Vulner_Rank'].copy()
data['Damage_Rank'] = cbg_age_msa['Damage_Rank'].copy()

###############################################################################
# Load detailed results and average across random seeds

# Simulation times and random seeds
NUM_SEEDS = 60 # 30
STARTING_SEED = range(NUM_SEEDS)
# The following parameters are just to make sure the name is correct. 'No_Vaccination' actually does not need these parameters.
# Vaccination_Ratio
VACCINATION_RATIO = 0.1
# Recheck Interval
RECHECK_INTERVAL = 0.01
# Vaccination protection rate
PROTECTION_RATE = 1
# Policy execution ratio
EXECUTION_RATIO = 1

history_D2_no_vaccination = np.fromfile(os.path.join(root,MSA_NAME,'vaccination_results_adaptive_31d_%s_0.01'% VACCINATION_RATIO,
                                                    '20210206_history_D2_no_vaccination_adaptive_%s_0.01_%sseeds_%s'% (VACCINATION_RATIO,NUM_SEEDS,MSA_NAME))) 
history_D2_no_vaccination = np.array(np.reshape(history_D2_no_vaccination,(63,NUM_SEEDS,M)))

avg_history_D2_no_vaccination = np.mean(history_D2_no_vaccination,axis=1)
avg_final_deaths_no_vaccination = avg_history_D2_no_vaccination[-1,:]

final_deaths_no_vaccination = np.sum(avg_final_deaths_no_vaccination)

###############################################################################
# Load data for randombag vaccination results, and compute the average features

#VACCINATION_RATIO = 0.02
#RANDOM_SEED_1 = 66
#RANDOM_SEED_2 = 42 
#RANDOM_SEED_3 = 5

# Group random
NUM_GROUPWISE = 5 
for i in range(LEN_SEEDS):
    current_seed = RANDOM_SEED_LIST[i]
    current_results = pd.read_csv(os.path.join(root,'newdamage_group_randombag_vaccination_results_withgini_0.02_%s_%s_%s_%sseeds.csv'
                                                  %(MSA_NAME,current_seed, NUM_GROUPWISE, NUM_SEEDS)))  
    if(i==0):
        randombag_results = pd.DataFrame(current_results)
        print('Check:',len(current_results),len(randombag_results))
    else:
        len_results_old = len(randombag_results)
        randombag_results = pd.concat([randombag_results,current_results],axis=0)
        len_results_new = len(randombag_results)
        print('Check:',len_results_old,len(current_results),len_results_new)
    
randombag_results = randombag_results.drop_duplicates()
print('After dropping duplicates: ', len(randombag_results))           
randombag_results.drop(labels=['Unnamed: 0'],axis=1,inplace=True)
# 把str转为list，split flag是', '，然后再把其中每个元素由str转为int(用map函数)
randombag_results['Vaccinated_Idxs'] = randombag_results['Vaccinated_Idxs'].apply(lambda x : list(map(int, (x.strip('[').strip(']').split(', ')))))

    
randombag_results['Avg_Elder_Ratio'] = randombag_results.apply(lambda x: get_avg_feat(x['Vaccinated_Idxs'],data,'Elder_Ratio'), axis=1)
randombag_results['Avg_Mean_Household_Income'] = randombag_results.apply(lambda x: get_avg_feat(x['Vaccinated_Idxs'],data,'Mean_Household_Income'), axis=1)
randombag_results['Avg_EW_Ratio'] = randombag_results.apply(lambda x: get_avg_feat(x['Vaccinated_Idxs'],data,'EW_Ratio'), axis=1)
randombag_results['Avg_Minority_Ratio'] = randombag_results.apply(lambda x: get_avg_feat(x['Vaccinated_Idxs'],data,'Minority_Ratio'), axis=1) #20220308
randombag_results['Avg_Vulnerability'] = randombag_results.apply(lambda x: get_avg_feat(x['Vaccinated_Idxs'],data,'Vulnerability'), axis=1)
randombag_results['Avg_Damage'] = randombag_results.apply(lambda x: get_avg_feat(x['Vaccinated_Idxs'],data,'Damage'), axis=1)
#randombag_results['Avg_Vulnerability'] = randombag_results.apply(lambda x: get_avg_feat(x['Vaccinated_Idxs'],data,'Vulner_Rank'), axis=1)
#randombag_results['Avg_Damage'] = randombag_results.apply(lambda x: get_avg_feat(x['Vaccinated_Idxs'],data,'Damage_Rank'), axis=1)

randombag_results['Std_Elder_Ratio'] = randombag_results.apply(lambda x: get_std_feat(x['Vaccinated_Idxs'],data,'Elder_Ratio'), axis=1)
randombag_results['Std_Mean_Household_Income'] = randombag_results.apply(lambda x: get_std_feat(x['Vaccinated_Idxs'],data,'Mean_Household_Income'), axis=1)
randombag_results['Std_EW_Ratio'] = randombag_results.apply(lambda x: get_std_feat(x['Vaccinated_Idxs'],data,'EW_Ratio'), axis=1)
randombag_results['Std_Minority_Ratio'] = randombag_results.apply(lambda x: get_std_feat(x['Vaccinated_Idxs'],data,'Minority_Ratio'), axis=1) #20220308
randombag_results['Std_Vulnerability'] = randombag_results.apply(lambda x: get_std_feat(x['Vaccinated_Idxs'],data,'Vulnerability'), axis=1)
randombag_results['Std_Damage'] = randombag_results.apply(lambda x: get_std_feat(x['Vaccinated_Idxs'],data,'Damage'), axis=1)

del cbg_age_msa
del cbg_income_msa
del cbg_occupation_msa
del cbg_ethnic_msa

# Check range
print('Avg_Elder_Ratio',randombag_results['Avg_Elder_Ratio'].max(),randombag_results['Avg_Elder_Ratio'].min())
print('Avg_Mean_Household_Income',randombag_results['Avg_Mean_Household_Income'].max(),randombag_results['Avg_Mean_Household_Income'].min())
print('Avg_EW_Ratio',randombag_results['Avg_EW_Ratio'].max(),randombag_results['Avg_EW_Ratio'].min())
print('Avg_EW_Ratio',randombag_results['Avg_Minority_Ratio'].max(),randombag_results['Avg_Minority_Ratio'].min()) #20220308
print('Avg_Vulnerability',randombag_results['Avg_Vulnerability'].max(),randombag_results['Avg_Vulnerability'].min())
print('Avg_Damage',randombag_results['Avg_Damage'].max(),randombag_results['Avg_Damage'].min())

print('Fatality_Rate_Rel',randombag_results['Fatality_Rate_Rel'].max(),randombag_results['Fatality_Rate_Rel'].min())
print('Age_Gini_Rel',randombag_results['Age_Gini_Rel'].max(),randombag_results['Age_Gini_Rel'].min())
print('Income_Gini_Rel',randombag_results['Income_Gini_Rel'].max(),randombag_results['Income_Gini_Rel'].min())
print('Occupation_Gini_Rel',randombag_results['Occupation_Gini_Rel'].max(),randombag_results['Occupation_Gini_Rel'].min())
#print('Minority_Gini_Rel',randombag_results['Minority_Gini_Rel'].max(),randombag_results['Minority_Gini_Rel'].min()) #20220308

###############################################################################
# Preprocessing: Standardization

scaler = preprocessing.StandardScaler() # standard scaler (z-score)

for column in randombag_results.columns:
    if(column=='Vaccinated_Idxs'):continue
    randombag_results[column] = scaler.fit_transform(np.array(randombag_results[column]).reshape(-1,1))
print('Standardized.')

###############################################################################
# Sample and regress

# Regression target
target_list = ['Fatality_Rate_Rel','Age_Gini_Rel','Income_Gini_Rel','Occupation_Gini_Rel','Minority_Gini_Rel']
demo_feat_list = ['Avg_Elder_Ratio','Avg_Mean_Household_Income','Avg_EW_Ratio','Avg_Minority_Ratio',
                  'Std_Elder_Ratio','Std_Mean_Household_Income','Std_EW_Ratio','Avg_EW_Ratio']
                  
# Get averaged params and stds (Ref: https://blog.csdn.net/chongminglun/article/details/104242342)
'''
for target in target_list:
    print('\nRegression target: ', target)
    avg_elder_ratio = np.zeros((4,NUM_SAMPLE))
    avg_income = np.zeros((4,NUM_SAMPLE))
    avg_ew_ratio = np.zeros((4,NUM_SAMPLE))
    std_elder_ratio = np.zeros((4,NUM_SAMPLE))
    std_income = np.zeros((4,NUM_SAMPLE))
    std_ew_ratio = np.zeros((4,NUM_SAMPLE))
    rsquared_adj = np.zeros((2,NUM_SAMPLE))
    
    if(target=='Fatality_Rate_Rel'):
        avg_damage = np.zeros((4,NUM_SAMPLE))
        std_damage = np.zeros((4,NUM_SAMPLE))
    else:
        avg_vulner = np.zeros((4,NUM_SAMPLE))
        std_vulner = np.zeros((4,NUM_SAMPLE))
        
    for sample_idx in range(NUM_SAMPLE):
        sample = randombag_results.sample(frac=SAMPLE_FRAC,random_state=sample_idx)
        if(sample_idx==0): print('len(sample):',len(sample))
        
        Y = sample[target]

        # Regression only with demo_feats
        X = sample[demo_feat_list]
        X = sm.add_constant(X) # adding a constant
        model = sm.OLS(Y, X).fit()
        predictions = model.predict(X) 
        #print(model.summary())
        #print('model.params:',model.params)
        #print('model.bse:',model.bse)
        avg_elder_ratio[0][sample_idx] = model.params['Avg_Elder_Ratio']
        avg_income[0][sample_idx] = model.params['Avg_Mean_Household_Income']
        avg_ew_ratio[0][sample_idx] = model.params['Avg_EW_Ratio']
        std_elder_ratio[0][sample_idx] = model.params['Std_Elder_Ratio']
        std_elder_ratio[0][sample_idx] = model.params['Std_Mean_Household_Income']
        std_income[0][sample_idx] = model.params['Std_EW_Ratio']
        
        avg_elder_ratio[1][sample_idx] = model.bse['Avg_Elder_Ratio']
        avg_income[1][sample_idx] = model.bse['Avg_Mean_Household_Income']
        avg_ew_ratio[1][sample_idx] = model.bse['Avg_EW_Ratio']
        std_elder_ratio[1][sample_idx] = model.bse['Std_Elder_Ratio']
        std_income[1][sample_idx] = model.bse['Std_Mean_Household_Income']
        std_ew_ratio[1][sample_idx] = model.bse['Std_EW_Ratio']
        
        rsquared_adj[0][sample_idx] = model.rsquared_adj

        if(target=='Fatality_Rate_Rel'):
            # Regression with demo_feats and inner mechanisms: Damage
            mediator_list = ['Avg_Damage','Std_Damage']
            X = sample[demo_feat_list+mediator_list]
            X = sm.add_constant(X) # adding a constant
            model = sm.OLS(Y, X).fit()
            predictions = model.predict(X) 
            #print(model.summary())
            avg_elder_ratio[2][sample_idx] = model.params['Avg_Elder_Ratio']
            avg_income[2][sample_idx] = model.params['Avg_Mean_Household_Income']
            avg_ew_ratio[2][sample_idx] = model.params['Avg_EW_Ratio']
            avg_damage[2][sample_idx] = model.params['Avg_Damage']
            std_elder_ratio[2][sample_idx] = model.params['Std_Elder_Ratio']
            std_income[2][sample_idx] = model.params['Std_Mean_Household_Income']
            std_ew_ratio[2][sample_idx] = model.params['Std_EW_Ratio']
            std_damage[2][sample_idx] = model.params['Std_Damage']
            
            avg_elder_ratio[3][sample_idx] = model.bse['Avg_Elder_Ratio']
            avg_income[3][sample_idx] = model.bse['Avg_Mean_Household_Income']
            avg_ew_ratio[3][sample_idx] = model.bse['Avg_EW_Ratio']
            avg_damage[3][sample_idx] = model.bse['Avg_Damage']
            std_elder_ratio[3][sample_idx] = model.bse['Std_Elder_Ratio']
            std_income[3][sample_idx] = model.bse['Std_Mean_Household_Income']
            std_ew_ratio[3][sample_idx] = model.bse['Std_EW_Ratio']
            std_damage[3][sample_idx] = model.bse['Std_Damage']
            
            rsquared_adj[1][sample_idx] = model.rsquared_adj
        
        else:
            # Regression with demo_feats and inner mechanisms: Vulnerability
            mediator_list = ['Avg_Vulnerability','Std_Vulnerability']
            X = sample[demo_feat_list+mediator_list]
            X = sm.add_constant(X) # adding a constant
            model = sm.OLS(Y, X).fit()
            predictions = model.predict(X) 
            #print(model.summary())
            avg_elder_ratio[2][sample_idx] = model.params['Avg_Elder_Ratio']
            avg_income[2][sample_idx] = model.params['Avg_Mean_Household_Income']
            avg_ew_ratio[2][sample_idx] = model.params['Avg_EW_Ratio']
            avg_vulner[2][sample_idx] = model.params['Avg_Vulnerability']
            std_elder_ratio[2][sample_idx] = model.params['Std_Elder_Ratio']
            std_income[2][sample_idx] = model.params['Std_Mean_Household_Income']
            std_ew_ratio[2][sample_idx] = model.params['Std_EW_Ratio']
            std_vulner[2][sample_idx] = model.params['Std_Vulnerability']
            
            avg_elder_ratio[3][sample_idx] = model.bse['Avg_Elder_Ratio']
            avg_income[3][sample_idx] = model.bse['Avg_Mean_Household_Income']
            avg_ew_ratio[3][sample_idx] = model.bse['Avg_EW_Ratio']
            avg_vulner[3][sample_idx] = model.bse['Avg_Vulnerability']
            std_elder_ratio[3][sample_idx] = model.bse['Std_Elder_Ratio']
            std_income[3][sample_idx] = model.bse['Std_Mean_Household_Income']
            std_ew_ratio[3][sample_idx] = model.bse['Std_EW_Ratio']
            std_vulner[3][sample_idx] = model.bse['Std_Vulnerability']
            
            rsquared_adj[1][sample_idx] = model.rsquared_adj
            
            #X = sample[demo_feat_list+mediator_list]
            #reg = linear_model.LinearRegression()
            #reg.fit(X,Y)
            #r2 = reg.score(X,Y)
            #rsquared_adj[1][sample_idx] = (1-(1-r2)*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1))
            
    
    print('Elder_Ratio:', np.round(np.mean(avg_elder_ratio[0]),3), np.round(np.mean(avg_elder_ratio[1]),3), np.round(np.mean(avg_elder_ratio[2]),3), np.round(np.mean(avg_elder_ratio[3]),3))
    print('Income:', np.round(np.mean(avg_income[0]),3), np.round(np.mean(avg_income[1]),3),np.round(np.mean(avg_income[2]),3),np.round(np.mean(avg_income[3]),3))
    print('EW_Ratio:', np.round(np.mean(avg_ew_ratio[0]),3), np.round(np.mean(avg_ew_ratio[1]),3),np.round(np.mean(avg_ew_ratio[2]),3),np.round(np.mean(avg_ew_ratio[3]),3))
    if(target=='Fatality_Rate_Rel'):
        print('Damage:', np.round(np.mean(avg_damage[0]),3),np.round(np.mean(avg_damage[1]),3),np.round(np.mean(avg_damage[2]),3),np.round(np.mean(avg_damage[3]),3) )
    else:
        print('Vulner:', np.round(np.mean(avg_vulner[0]),3),np.round(np.mean(avg_vulner[1]),3),np.round(np.mean(avg_vulner[2]),3),np.round(np.mean(avg_vulner[3]),3) )
    print('Adjusted R-squared: ', np.round(np.mean(rsquared_adj[0]),3), np.round(np.mean(rsquared_adj[1]),3))    
    
pdb.set_trace()
         
'''
###############################################################################
# Just retrieve adj_r2

fatality_adj_r2_model1 = []
fatality_adj_r2_model2 = []
age_adj_r2_model1 = []
age_adj_r2_model2 = []
income_adj_r2_model1 = []
income_adj_r2_model2 = []
occupation_adj_r2_model1 = []
occupation_adj_r2_model2 = []

for sample_idx in range(NUM_SAMPLE):
    sample = randombag_results.sample(frac=SAMPLE_FRAC,random_state=sample_idx)
    if(sample_idx==0): print('len(sample):',len(sample))
    
    '''
    # Check range
    print('Avg_Elder_Ratio',randombag_results['Avg_Elder_Ratio'].max(),randombag_results['Avg_Elder_Ratio'].min())
    print('Avg_Mean_Household_Income',randombag_results['Avg_Mean_Household_Income'].max(),randombag_results['Avg_Mean_Household_Income'].min())
    print('Avg_EW_Ratio',randombag_results['Avg_EW_Ratio'].max(),randombag_results['Avg_EW_Ratio'].min())
    print('Avg_Vulnerability',randombag_results['Avg_Vulnerability'].max(),randombag_results['Avg_Vulnerability'].min())
    print('Avg_Damage',randombag_results['Avg_Damage'].max(),randombag_results['Avg_Damage'].min())
    #print(randombag_results['Saved_Lives'].max(),randombag_results['Saved_Lives'].min())
    print('Fatality_Rate_Rel',randombag_results['Fatality_Rate_Rel'].max(),randombag_results['Fatality_Rate_Rel'].min())
    print('Age_Gini_Rel',randombag_results['Age_Gini_Rel'].max(),randombag_results['Age_Gini_Rel'].min())
    print('Income_Gini_Rel',randombag_results['Income_Gini_Rel'].max(),randombag_results['Income_Gini_Rel'].min())
    print('Occupation_Gini_Rel',randombag_results['Occupation_Gini_Rel'].max(),randombag_results['Occupation_Gini_Rel'].min())
    '''
    ###############################################################################
    # Linear Regression (statsmodels)

    # Target: 
    #demo_feat_list = ['Avg_Elder_Ratio','Avg_Mean_Household_Income','Avg_EW_Ratio']
    demo_feat_list = ['Avg_Elder_Ratio','Avg_Mean_Household_Income','Avg_EW_Ratio',#'Avg_Minority_Ratio',
                      'Std_Elder_Ratio','Std_Mean_Household_Income','Std_EW_Ratio',#'Avg_Minority_Ratio'
                      ]

    # Regression target
    target_list = ['Fatality_Rate_Rel','Age_Gini_Rel','Income_Gini_Rel','Occupation_Gini_Rel']#,'Minority_Gini_Rel']

    for target in target_list:
        #print('Regression target: ', target)
        Y = sample[target]

        # Regression only with demo_feats
        X = sample[demo_feat_list]
        #print('Independent Variables: ',demo_feat_list)
        #X = sm.add_constant(X) # adding a constant
        '''
        model = sm.OLS(Y, X).fit()
        predictions = model.predict(X) 
        print(model.summary())
        print('model.params:',model.params)
        print('model.bse:',model.bse)
        '''
        reg = linear_model.LinearRegression()
        reg.fit(X,Y)
        r2 = reg.score(X,Y)
        adjusted_r2_model1 = (1-(1-r2)*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1))
        #print('adjusted_r2_model1:',adjusted_r2_model1)
        

        
        # Regression with demo_feats and inner mechanisms: Vulnerability
        #mediator_list = ['Avg_Vulnerability']
        mediator_list = ['Avg_Vulnerability','Std_Vulnerability']
        X = sample[demo_feat_list+mediator_list]
        #print('Independent Variables: ',demo_feat_list+mediator_list)
        #X = sm.add_constant(X) # adding a constant
        '''
        model = sm.OLS(Y, X).fit()
        predictions = model.predict(X) 
        print(model.summary())
        '''
        reg = linear_model.LinearRegression()
        reg.fit(X,Y)
        r2 = reg.score(X,Y)
        adjusted_r2_model2 = (1-(1-r2)*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1))
        
        
        # Regression with demo_feats and inner mechanisms: Damage
        #mediator_list = ['Avg_Damage']
        mediator_list = ['Avg_Damage','Std_Damage']
        X = sample[demo_feat_list+mediator_list]
        #print('Independent Variables: ',demo_feat_list+mediator_list)
        #X = sm.add_constant(X) # adding a constant
        '''
        model = sm.OLS(Y, X).fit()
        predictions = model.predict(X) 
        print(model.summary())
        '''
        reg = linear_model.LinearRegression()
        reg.fit(X,Y)
        r2 = reg.score(X,Y)
        adjusted_r2_model3 = (1-(1-r2)*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1))
        
        # Regression with demo_feats and inner mechanisms: Vulnerability, Damage
        '''
        #mediator_list = ['Avg_Vulnerability','Avg_Damage']
        mediator_list = ['Avg_Vulnerability','Avg_Damage','Std_Vulnerability','Std_Damage']
        X = randombag_results[demo_feat_list+mediator_list]
        print('Independent Variables: ',demo_feat_list+mediator_list)
        X = sm.add_constant(X) # adding a constant
        model = sm.OLS(Y, X).fit()
        predictions = model.predict(X) 
        print(model.summary())
        '''
        
        if(target=='Fatality_Rate_Rel'):
            fatality_adj_r2_model1.append(adjusted_r2_model1)
            fatality_adj_r2_model2.append(adjusted_r2_model3)
        elif(target=='Age_Gini_Rel'):
            age_adj_r2_model1.append(adjusted_r2_model1)
            age_adj_r2_model2.append(adjusted_r2_model2)
        elif(target=='Income_Gini_Rel'):
            income_adj_r2_model1.append(adjusted_r2_model1)
            income_adj_r2_model2.append(adjusted_r2_model2)
        elif(target=='Occupation_Gini_Rel'):
            occupation_adj_r2_model1.append(adjusted_r2_model1)
            occupation_adj_r2_model2.append(adjusted_r2_model2)    
        elif(target=='Minority_Gini_Rel'): #20220308
            minority_adj_r2_model1.append(adjusted_r2_model1)
            minority_adj_r2_model2.append(adjusted_r2_model2)       

print('Mean and Std: ')
print('fatality_adj_r2_model1:',np.mean(np.array(fatality_adj_r2_model1)),np.std(np.array(fatality_adj_r2_model1)))
print('fatality_adj_r2_model2:',np.mean(np.array(fatality_adj_r2_model2)),np.std(np.array(fatality_adj_r2_model2)))         
print('age_adj_r2_model1:',np.mean(np.array(age_adj_r2_model1)),np.std(np.array(age_adj_r2_model1)))
print('age_adj_r2_model2:',np.mean(np.array(age_adj_r2_model2)),np.std(np.array(age_adj_r2_model2)))  
print('income_adj_r2_model1:',np.mean(np.array(income_adj_r2_model1)),np.std(np.array(income_adj_r2_model1)))
print('income_adj_r2_model2:',np.mean(np.array(income_adj_r2_model2)),np.std(np.array(income_adj_r2_model2))) 
print('occupation_adj_r2_model1:',np.mean(np.array(occupation_adj_r2_model1)),np.std(np.array(occupation_adj_r2_model1)))
print('occupation_adj_r2_model2:',np.mean(np.array(occupation_adj_r2_model2)),np.std(np.array(occupation_adj_r2_model2)))
print('minority_adj_r2_model1:',np.mean(np.array(minority_adj_r2_model1)),np.std(np.array(minority_adj_r2_model1))) #20220308
print('minority_adj_r2_model2:',np.mean(np.array(minority_adj_r2_model2)),np.std(np.array(minority_adj_r2_model2))) #20220308
'''        
print('Details: ')
print('fatality_adj_r2_model1:',fatality_adj_r2_model1)
print('fatality_adj_r2_model2:',fatality_adj_r2_model2)            
print('age_adj_r2_model1:',age_adj_r2_model1)
print('age_adj_r2_model2:',age_adj_r2_model2)  
print('income_adj_r2_model1:',income_adj_r2_model1)
print('income_adj_r2_model2:',income_adj_r2_model2)  
print('occupation_adj_r2_model1:',occupation_adj_r2_model1)
print('occupation_adj_r2_model2:',occupation_adj_r2_model2)  
''' 


###############################################################################
# Linear Regression (sklearn)
'''
print('Linear Regression (sklearn), not forced positive:')
Forced_Positive = False

# Start with demo_feats
demo_feat_list = ['Avg_Elder_Ratio','Neg_Avg_Mean_Household_Income','Avg_EW_Ratio']
# Regression target: fatality rate in each CBG
Y = randombag_results['Saved_Lives']

# Regression only with demo_feats
X = randombag_results[demo_feat_list]
print('Independent Variables: ',demo_feat_list) # Unlike in statsmodels, here no need to add a constant.
reg = linear_model.LinearRegression(positive=Forced_Positive)
reg.fit(X,Y)
r2 = reg.score(X,Y)
adjusted_r2 = (1-(1-r2)*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1))
print('reg.score (R-squared): %.4f' % r2)
print('Adj. R-squared: %.4f' % adjusted_r2)
print('reg.coef_: ',reg.coef_)

# Regression with demo_feats and inner mechanisms: Damage
mediator_list = ['Avg_Damage']
X = randombag_results[demo_feat_list+mediator_list]
print('Independent Variables: ',demo_feat_list+mediator_list)
reg = linear_model.LinearRegression(positive=Forced_Positive)
reg.fit(X,Y)
r2 = reg.score(X,Y)
adjusted_r2 = (1-(1-r2)*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1))
print('reg.score (R-squared): %.4f' % r2)
print('Adj. R-squared: %.4f' % adjusted_r2)
print('reg.coef_: ',reg.coef_)

# Regression with demo_feats and inner mechanisms: Vulnerability
mediator_list = ['Avg_Vulnerability']
X = randombag_results[demo_feat_list+mediator_list]
print('Independent Variables: ',demo_feat_list+mediator_list)
reg = linear_model.LinearRegression(positive=Forced_Positive)
reg.fit(X,Y)
r2 = reg.score(X,Y)
adjusted_r2 = (1-(1-r2)*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1))
print('reg.score (R-squared): %.4f' % r2)
print('Adj. R-squared: %.4f' % adjusted_r2)
print('reg.coef_: ',reg.coef_)

# Regression with demo_feats and inner mechanisms: Vulnerability, Damage
mediator_list = ['Avg_Vulnerability','Avg_Damage']
X = randombag_results[demo_feat_list+mediator_list]
print('Independent Variables: ',demo_feat_list+mediator_list)
reg = linear_model.LinearRegression(positive=Forced_Positive)
reg.fit(X,Y)
r2 = reg.score(X,Y)
adjusted_r2 = (1-(1-r2)*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1))
print('reg.score (R-squared): %.4f' % r2)
print('Adj. R-squared: %.4f' % adjusted_r2)
print('reg.coef_: ',reg.coef_)
'''