# python plot_regression_randombag_sample_average.py --msa_name all 

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import socket
import os
import argparse
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import constants
import functions

import pdb

# root
'''
hostname = socket.gethostname()
print('hostname: ', hostname)
if(hostname in ['fib-dl3','rl3','rl2']):
    root = '/data/chenlin/COVID-19/Data' #dl3
    saveroot = '/data/chenlin/utility-equity-covid-vac/results/'
elif(hostname=='rl4'):
    root = '/home/chenlin/COVID-19/Data' #rl4
    saveroot = '/home/chenlin/utility-equity-covid-vac/results/'
'''
root = os.getcwd()
dataroot = os.path.join(root, 'data')
resultroot = os.path.join(root, 'results')
fig_save_root = os.path.join(root, 'figures')

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--msa_name', 
                    help='MSA name. If \'all\', then iterate over all MSAs.')
parser.add_argument('--len_seeds',  type=int, default=3,
                    help='Number of random seeds.')         
parser.add_argument('--num_samples',  type=int, default=20,
                    help='Number of samples.')
parser.add_argument('--sample_frac',  type=float, default=0.2,
                    help='Fraction each sample contains.')
parser.add_argument('--safegraph_root', default=dataroot, '/data/chenlin/COVID-19/Data',
                    help='Safegraph data root.') 
parser.add_argument('--stop_to_observe', default=False, action='store_true',
                    help='If true, stop after regression.')
args = parser.parse_args()
print('\nargs.msa_name: ',args.msa_name)




# Derived variables
if(args.msa_name=='all'):
    print('all msa.')
    msa_name_list = ['Atlanta', 'Chicago', 'Dallas', 'Houston', 'LosAngeles', 'Miami', 'Philadelphia', 'SanFrancisco', 'WashingtonDC']
else:
    print('msa name:',args.msa_name)
    msa_name_list = [args.msa_name]

RANDOM_SEED_LIST = [38,39,40] #RANDOM_SEED_LIST = [66,42,5]
print('Num of random seeds: ', args.len_seeds)
print('Random seeds: ',RANDOM_SEED_LIST[:args.len_seeds])

print('Num of samples: ', args.num_samples)
print('Sample fraction: ', args.sample_frac)

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

#if(os.path.exists(os.path.join(saveroot, 'minority_adj_r2_model2_std_array'))):
if(False):
    fatality_adj_r2_model1_mean_array = np.fromfile(os.path.join(saveroot, 'fatality_adj_r2_model1_mean_array'))
    fatality_adj_r2_model2_mean_array = np.fromfile(os.path.join(saveroot, 'fatality_adj_r2_model2_mean_array'))
    age_adj_r2_model1_mean_array = np.fromfile(os.path.join(saveroot, 'age_adj_r2_model1_mean_array'))
    age_adj_r2_model2_mean_array = np.fromfile(os.path.join(saveroot, 'age_adj_r2_model2_mean_array'))
    income_adj_r2_model1_mean_array = np.fromfile(os.path.join(saveroot, 'income_adj_r2_model1_mean_array'))
    income_adj_r2_model2_mean_array = np.fromfile(os.path.join(saveroot, 'income_adj_r2_model2_mean_array'))
    occupation_adj_r2_model1_mean_array = np.fromfile(os.path.join(saveroot, 'occupation_adj_r2_model1_mean_array'))
    occupation_adj_r2_model2_mean_array = np.fromfile(os.path.join(saveroot, 'occupation_adj_r2_model2_mean_array'))
    minority_adj_r2_model1_mean_array = np.fromfile(os.path.join(saveroot, 'minority_adj_r2_model1_mean_array'))
    minority_adj_r2_model2_mean_array = np.fromfile(os.path.join(saveroot, 'minority_adj_r2_model2_mean_array'))

    fatality_adj_r2_model1_std_array = np.fromfile(os.path.join(saveroot, 'fatality_adj_r2_model1_std_array'))
    fatality_adj_r2_model2_std_array = np.fromfile(os.path.join(saveroot, 'fatality_adj_r2_model2_std_array'))
    age_adj_r2_model1_std_array = np.fromfile(os.path.join(saveroot, 'age_adj_r2_model1_std_array'))
    age_adj_r2_model2_std_array = np.fromfile(os.path.join(saveroot, 'age_adj_r2_model2_std_array'))
    income_adj_r2_model1_std_array = np.fromfile(os.path.join(saveroot, 'income_adj_r2_model1_std_array'))
    income_adj_r2_model2_std_array = np.fromfile(os.path.join(saveroot, 'income_adj_r2_model2_std_array'))
    occupation_adj_r2_model1_std_array = np.fromfile(os.path.join(saveroot, 'occupation_adj_r2_model1_std_array'))
    occupation_adj_r2_model2_std_array = np.fromfile(os.path.join(saveroot, 'occupation_adj_r2_model2_std_array'))
    minority_adj_r2_model1_std_array = np.fromfile(os.path.join(saveroot, 'minority_adj_r2_model1_std_array'))
    minority_adj_r2_model2_std_array = np.fromfile(os.path.join(saveroot, 'minority_adj_r2_model2_std_array'))
    
else:
    ###############################################################################
    # Load Common Data: No need for reloading when switching among differet MSAs.


    # Load ACS Data for matching with NYT Data
    acs_data = pd.read_csv(os.path.join(dataroot,'list1.csv'),header=2)
    acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]

    # Load NYT Data
    nyt_data = pd.read_csv(os.path.join(dataroot, 'us-counties.csv'))

    # Load Demographic Data
    filepath = os.path.join(args.safegraph_root,"safegraph_open_census_data/data/cbg_b01.csv")
    cbg_agesex = pd.read_csv(filepath)

    filepath = os.path.join(dataroot,"ACS_5years_Income_Filtered_Summary.csv")
    cbg_income = pd.read_csv(filepath)
    # Drop duplicate column 'Unnamed:0'
    cbg_income.drop(['Unnamed: 0'],axis=1, inplace=True)

    filepath = os.path.join(args.safegraph_root,"safegraph_open_census_data/data/cbg_c24.csv")
    cbg_occupation = pd.read_csv(filepath)

    # cbg_b03.csv: Ethnic #20220308
    filepath = os.path.join(args.safegraph_root,"safegraph_open_census_data/data/cbg_b03.csv")
    cbg_ethnic = pd.read_csv(filepath)

    ###############################################################################
    fatality_adj_r2_model1_mean_array = np.zeros(len(msa_name_list))
    fatality_adj_r2_model2_mean_array = np.zeros(len(msa_name_list))
    age_adj_r2_model1_mean_array = np.zeros(len(msa_name_list))
    age_adj_r2_model2_mean_array = np.zeros(len(msa_name_list))
    income_adj_r2_model1_mean_array = np.zeros(len(msa_name_list))
    income_adj_r2_model2_mean_array = np.zeros(len(msa_name_list))
    occupation_adj_r2_model1_mean_array = np.zeros(len(msa_name_list))
    occupation_adj_r2_model2_mean_array = np.zeros(len(msa_name_list))
    minority_adj_r2_model1_mean_array = np.zeros(len(msa_name_list))
    minority_adj_r2_model2_mean_array = np.zeros(len(msa_name_list))
    fatality_adj_r2_model1_std_array = np.zeros(len(msa_name_list))
    fatality_adj_r2_model2_std_array = np.zeros(len(msa_name_list))
    age_adj_r2_model1_std_array = np.zeros(len(msa_name_list))
    age_adj_r2_model2_std_array = np.zeros(len(msa_name_list))
    income_adj_r2_model1_std_array = np.zeros(len(msa_name_list))
    income_adj_r2_model2_std_array = np.zeros(len(msa_name_list))
    occupation_adj_r2_model1_std_array = np.zeros(len(msa_name_list))
    occupation_adj_r2_model2_std_array = np.zeros(len(msa_name_list))
    minority_adj_r2_model1_std_array = np.zeros(len(msa_name_list))
    minority_adj_r2_model2_std_array = np.zeros(len(msa_name_list))

    for msa_idx in range(len(msa_name_list)):
        this_msa = msa_name_list[msa_idx]
        print(this_msa)
        MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[this_msa]
        # Load POI-CBG visiting matrices
        f = open(os.path.join(dataroot, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
        poi_cbg_visits_list = pickle.load(f)
        f.close()

        data = pd.DataFrame()

        cbg_death_rates_original = np.loadtxt(os.path.join(dataroot, 'cbg_death_rates_original_'+this_msa))
        cbg_death_rates_scaled = cbg_death_rates_original * constants.death_scale_dict[this_msa]

        # Extract data specific to one msa, according to ACS data
        # MSA list
        msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
        msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
        msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
        good_list = list(msa_data['FIPS Code'].values)

        # Load CBG ids belonging to a specific metro area
        # cbg_ids_msa
        cbg_ids_msa = pd.read_csv(os.path.join(dataroot,'%s_cbg_ids.csv' % MSA_NAME_FULL))
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
        idxs_msa_all = list(x.values()) #;print('Number of CBGs in this metro area:', len(idxs_msa_all))
        idxs_msa_nyt = y #;print('Number of CBGs in to compare with NYT data:', len(idxs_msa_nyt))
        
        # Extract CBGs belonging to the MSA 
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
            
        # Income Data
        # Extract pois corresponding to the metro area, by merging dataframes
        cbg_income_msa = pd.merge(cbg_ids_msa, cbg_income, on='census_block_group', how='left')
        # Deal with NaN values
        cbg_income_msa.fillna(0,inplace=True)
        # Add information of cbg populations
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
        cbg_minority_msa = cbg_ethnic_msa

        ###############################################################################
        # Obtain vulnerability and damage, according to theoretical analysis

        nyt_included = np.zeros(len(idxs_msa_all))
        for i in range(len(nyt_included)):
            if(i in idxs_msa_nyt):
                nyt_included[i] = 1
        cbg_age_msa['NYT_Included'] = nyt_included.copy()

        # Retrieve the attack rate for the whole MSA (home_beta, fitted for each MSA)
        home_beta = constants.parameters_dict[this_msa][1]
        print('MSA home_beta retrieved.')

        # Retrieve cbg_avg_infect_same, cbg_avg_infect_diff
        # As planned, they have been computed in 'tradeoff_md_mv_theory.py'.
        # Use them to get data['Vulnerability'] and data['Damage']
        if(os.path.exists(os.path.join(resultroot, '3cbg_avg_infect_same_%s.npy'%this_msa))):
            print('cbg_avg_infect_same, cbg_avg_infect_diff: Load existing file.')
            cbg_avg_infect_same = np.load(os.path.join(resultroot, '3cbg_avg_infect_same_%s.npy'%this_msa))
            cbg_avg_infect_diff = np.load(os.path.join(resultroot, '3cbg_avg_infect_diff_%s.npy'%this_msa))
        else:
            print('cbg_avg_infect_same, cbg_avg_infect_diff: not found, please check.')
            pdb.set_trace()
            
        #print('cbg_avg_infect_same.shape:',cbg_avg_infect_same.shape)

        cbg_age_msa['Death_Rate'] =  cbg_death_rates_scaled

        # Deal with nan and inf (https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
        cbg_avg_infect_same = np.nan_to_num(cbg_avg_infect_same,nan=0,posinf=0,neginf=0)
        cbg_avg_infect_diff = np.nan_to_num(cbg_avg_infect_diff,nan=0,posinf=0,neginf=0)
        cbg_age_msa['Infect'] = cbg_avg_infect_same + cbg_avg_infect_diff
        # Check whether there is NaN in cbg_tables
        print('Any NaN in cbg_age_msa[\'Infect\']?', cbg_age_msa['Infect'].isnull().any().any())

        SEIR_at_30d = np.load(os.path.join(resultroot, 'SEIR_at_30d.npy'),allow_pickle=True).item()
        S_ratio = SEIR_at_30d[this_msa]['S'] / (cbg_sizes.sum())
        I_ratio = SEIR_at_30d[this_msa]['I'] / (cbg_sizes.sum())
        print('S_ratio:',S_ratio,'I_ratio:',I_ratio)

        # Compute the average death rate for the whole MSA: perform another weighted average over all CBGs
        avg_death_rates_scaled = np.matmul(cbg_sizes.T, cbg_death_rates_scaled) / np.sum(cbg_sizes)
        #print('avg_death_rates_scaled.shape:',avg_death_rates_scaled.shape) # shape: (), because it is a scalar

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

        history_D2_no_vaccination = np.fromfile(os.path.join(resultroot,'vaccination_results_adaptive_31d_%s_0.01'% VACCINATION_RATIO,
                                                            r'20210206_history_D2_no_vaccination_adaptive_%s_0.01_%sseeds_%s'% (VACCINATION_RATIO,NUM_SEEDS,this_msa))) 
        history_D2_no_vaccination = np.array(np.reshape(history_D2_no_vaccination,(63,NUM_SEEDS,M)))

        avg_history_D2_no_vaccination = np.mean(history_D2_no_vaccination,axis=1)
        avg_final_deaths_no_vaccination = avg_history_D2_no_vaccination[-1,:]

        final_deaths_no_vaccination = np.sum(avg_final_deaths_no_vaccination)

        ###############################################################################
        # Load data for randombag vaccination results, and compute the average features

        # Group random
        NUM_GROUPWISE = 5 
        for i in range(args.len_seeds):
            current_seed = RANDOM_SEED_LIST[i]
            current_results = pd.read_csv(os.path.join(resultroot, 'vac_randombag_results',f'group_randombag_vac_results_0.02_{this_msa}_{current_seed}_{NUM_GROUPWISE}_60seeds.csv'))
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
        print('Minority_Gini_Rel',randombag_results['Minority_Gini_Rel'].max(),randombag_results['Minority_Gini_Rel'].min()) #20220308

        ###############################################################################
        # Preprocessing: Standardization

        scaler = preprocessing.StandardScaler() # standard scaler (z-score)

        for column in randombag_results.columns:
            if(column=='Vaccinated_Idxs'):continue
            randombag_results[column] = scaler.fit_transform(np.array(randombag_results[column]).reshape(-1,1))
        print('Standardized.')

        ###############################################################################
        # Sample and regress
    
        # Just retrieve adj_r2
        fatality_adj_r2_model1 = []
        fatality_adj_r2_model2 = []
        age_adj_r2_model1 = []
        age_adj_r2_model2 = []
        income_adj_r2_model1 = []
        income_adj_r2_model2 = []
        occupation_adj_r2_model1 = []
        occupation_adj_r2_model2 = []
        minority_adj_r2_model1 = []
        minority_adj_r2_model2 = []

        for sample_idx in range(args.num_samples):
            sample = randombag_results.sample(frac=args.sample_frac,random_state=sample_idx)
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
            demo_feat_list = ['Avg_Elder_Ratio','Avg_Mean_Household_Income','Avg_EW_Ratio','Avg_Minority_Ratio',
                            'Std_Elder_Ratio','Std_Mean_Household_Income','Std_EW_Ratio','Std_Minority_Ratio'
                            ]

            # Regression target
            target_list = ['Fatality_Rate_Rel','Age_Gini_Rel','Income_Gini_Rel','Occupation_Gini_Rel','Minority_Gini_Rel']

            for target in target_list:
                if(args.stop_to_observe):
                    print(f'###################################### target: {target} ########################################')
                #Y = sample[target]
                Y = -sample[target] #20220314,social utility, equity

                X = sample[demo_feat_list]
                
                if(args.stop_to_observe):
                    model = sm.OLS(Y, X).fit()
                    predictions = model.predict(X) 
                    #print(model.summary())
                    #print('model.params:',np.round(model.params.values, 3)) 
                    #print('model.bse:',np.round(model.bse.values, 2)) #standard errors of the parameter estimates
                    print(f'Adj. r2: {np.round(model.rsquared_adj,3)}')
                    params = list(np.round(model.params.values, 3))
                    bse = list(np.round(model.bse.values, 2))
                    for idx in range(len(list(bse))):
                        print(f'{params[idx]} ({bse[idx]})')
                    pdb.set_trace()
                else:
                    reg = linear_model.LinearRegression()
                    reg.fit(X,Y)
                    r2 = reg.score(X,Y)
                    adjusted_r2_model1 = (1-(1-r2)*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1))
                    #print('adjusted_r2_model1:',adjusted_r2_model1)


                # Regression with demo_feats and inner mechanisms: Vulnerability
                mediator_list = ['Avg_Vulnerability','Std_Vulnerability']
                X = sample[demo_feat_list+mediator_list]
                if(args.stop_to_observe):
                    model = sm.OLS(Y, X).fit()
                    predictions = model.predict(X) 
                    #print(model.summary())
                    #print('model.params:',np.round(model.params.values, 3)) 
                    #print('model.bse:',np.round(model.bse.values, 2)) #standard errors of the parameter estimates
                    print(f'Adj. r2: {np.round(model.rsquared_adj,3)}')
                    params = list(np.round(model.params.values, 3))
                    bse = list(np.round(model.bse.values, 2))
                    for idx in range(len(list(bse))):
                        print(f'{params[idx]} ({bse[idx]})')
                    pdb.set_trace()
                else:
                    reg = linear_model.LinearRegression()
                    reg.fit(X,Y)
                    r2 = reg.score(X,Y)
                    adjusted_r2_model2 = (1-(1-r2)*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1))

                 
                # Regression with demo_feats and inner mechanisms: Damage
                mediator_list = ['Avg_Damage','Std_Damage']
                X = sample[demo_feat_list+mediator_list]
                if(args.stop_to_observe):
                    model = sm.OLS(Y, X).fit()
                    predictions = model.predict(X) 
                    #print(model.summary())
                    #print('model.params:',np.round(model.params.values, 3)) 
                    #print('model.bse:',np.round(model.bse.values, 2)) #standard errors of the parameter estimates
                    print(f'Adj. r2: {np.round(model.rsquared_adj,3)}')
                    params = list(np.round(model.params.values, 3))
                    bse = list(np.round(model.bse.values, 2))
                    for idx in range(len(list(bse))):
                        print(f'{params[idx]} ({bse[idx]})')
                    pdb.set_trace()
                else:
                    reg = linear_model.LinearRegression()
                    reg.fit(X,Y)
                    r2 = reg.score(X,Y)
                    adjusted_r2_model3 = (1-(1-r2)*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1))


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

        '''
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

        fatality_adj_r2_model1_mean_array[msa_idx] = np.mean(np.array(fatality_adj_r2_model1))
        fatality_adj_r2_model2_mean_array[msa_idx] = np.mean(np.array(fatality_adj_r2_model2))
        age_adj_r2_model1_mean_array[msa_idx] = np.mean(np.array(age_adj_r2_model1))
        age_adj_r2_model2_mean_array[msa_idx] = np.mean(np.array(age_adj_r2_model2))
        income_adj_r2_model1_mean_array[msa_idx] = np.mean(np.array(income_adj_r2_model1))
        income_adj_r2_model2_mean_array[msa_idx] = np.mean(np.array(income_adj_r2_model2))
        occupation_adj_r2_model1_mean_array[msa_idx] = np.mean(np.array(occupation_adj_r2_model1))
        occupation_adj_r2_model2_mean_array[msa_idx] = np.mean(np.array(occupation_adj_r2_model2))
        minority_adj_r2_model1_mean_array[msa_idx] = np.mean(np.array(minority_adj_r2_model1))
        minority_adj_r2_model2_mean_array[msa_idx] = np.mean(np.array(minority_adj_r2_model2))

        fatality_adj_r2_model1_std_array[msa_idx] = np.std(np.array(fatality_adj_r2_model1))
        fatality_adj_r2_model2_std_array[msa_idx] = np.std(np.array(fatality_adj_r2_model2))
        age_adj_r2_model1_std_array[msa_idx] = np.std(np.array(age_adj_r2_model1))
        age_adj_r2_model2_std_array[msa_idx] = np.std(np.array(age_adj_r2_model2))
        income_adj_r2_model1_std_array[msa_idx] = np.std(np.array(income_adj_r2_model1))
        income_adj_r2_model2_std_array[msa_idx] = np.std(np.array(income_adj_r2_model2))
        occupation_adj_r2_model1_std_array[msa_idx] = np.std(np.array(occupation_adj_r2_model1))
        occupation_adj_r2_model2_std_array[msa_idx] = np.std(np.array(occupation_adj_r2_model2))
        minority_adj_r2_model1_std_array[msa_idx] = np.std(np.array(minority_adj_r2_model1))
        minority_adj_r2_model2_std_array[msa_idx] = np.std(np.array(minority_adj_r2_model2))

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


    # Save results
    fatality_adj_r2_model1_mean_array.tofile(os.path.join(resultroot, 'fatality_adj_r2_model1_mean_array'))
    fatality_adj_r2_model2_mean_array.tofile(os.path.join(resultroot, 'fatality_adj_r2_model2_mean_array'))
    age_adj_r2_model1_mean_array.tofile(os.path.join(resultroot, 'age_adj_r2_model1_mean_array'))
    age_adj_r2_model2_mean_array.tofile(os.path.join(resultroot, 'age_adj_r2_model2_mean_array'))
    income_adj_r2_model1_mean_array.tofile(os.path.join(resultroot, 'income_adj_r2_model1_mean_array'))
    income_adj_r2_model2_mean_array.tofile(os.path.join(resultroot, 'income_adj_r2_model2_mean_array'))
    occupation_adj_r2_model1_mean_array.tofile(os.path.join(resultroot, 'occupation_adj_r2_model1_mean_array'))
    occupation_adj_r2_model2_mean_array.tofile(os.path.join(resultroot, 'occupation_adj_r2_model2_mean_array'))
    minority_adj_r2_model1_mean_array.tofile(os.path.join(resultroot, 'minority_adj_r2_model1_mean_array'))
    minority_adj_r2_model2_mean_array.tofile(os.path.join(resultroot, 'minority_adj_r2_model2_mean_array'))

    fatality_adj_r2_model1_std_array.tofile(os.path.join(resultroot, 'fatality_adj_r2_model1_std_array'))
    fatality_adj_r2_model2_std_array.tofile(os.path.join(resultroot, 'fatality_adj_r2_model2_std_array'))
    age_adj_r2_model1_std_array.tofile(os.path.join(resultroot, 'age_adj_r2_model1_std_array'))
    age_adj_r2_model2_std_array.tofile(os.path.join(resultroot, 'age_adj_r2_model2_std_array'))
    income_adj_r2_model1_std_array.tofile(os.path.join(resultroot, 'income_adj_r2_model1_std_array'))
    income_adj_r2_model2_std_array.tofile(os.path.join(resultroot, 'income_adj_r2_model2_std_array'))
    occupation_adj_r2_model1_std_array.tofile(os.path.join(resultroot, 'occupation_adj_r2_model1_std_array'))
    occupation_adj_r2_model2_std_array.tofile(os.path.join(resultroot, 'occupation_adj_r2_model2_std_array'))
    minority_adj_r2_model1_std_array.tofile(os.path.join(resultroot, 'minority_adj_r2_model1_std_array'))
    minority_adj_r2_model2_std_array.tofile(os.path.join(resultroot, 'minority_adj_r2_model2_std_array'))

    print('File saved. Example path:', os.path.join(resultroot, 'minority_adj_r2_model2_std_array'))

##########################################################################################
# Draw figures

anno_list=['Atlanta','Chicago','Dallas','Houston','L.A.','Miami','Phila.','S.F.','D.C.']

##########################################
# Fig3b
plt.figure(figsize=(14,7))
plt.bar(np.arange(9)*3,fatality_adj_r2_model1_mean_array,width=1,label='Only demo-feats',color='k',alpha=0.8,yerr=fatality_adj_r2_model1_std_array) #color='dimgrey',
plt.bar(np.arange(9)*3+1.05,fatality_adj_r2_model2_mean_array,width=1,label='With $\it{societal~risk}$',color='r',alpha=0.65,yerr=fatality_adj_r2_model2_std_array) #color='cornflowerblue'
#plt.legend(fontsize=18,ncol=2,loc='upper center')
plt.xticks(np.arange(9)*3+0.5,anno_list, fontsize=25,rotation=20)  
plt.ylim(0,1)
plt.yticks(fontsize=16) #fontsize=14
plt.ylabel('Explained variance\nof social uility',fontsize=28)
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8]) #https://blog.csdn.net/yywan1314520/article/details/53740001
#plt.legend(ncol=3,fontsize=18,bbox_to_anchor=(0.8,-0.1)) #,title='Policy',title_fontsize='x-large')
plt.legend(fontsize=24,ncol=2,loc='upper center',bbox_to_anchor=(0.5,1.2))
# Save figure
savepath = os.path.join(fig_save_root, 'fig3b.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Fig3b, saved at {savepath}')

##########################################
# Fig3c

def plot_equity_explained_variance(demo_feat, show_yticks=True):
    plt.figure(figsize=(3,5))
    step_1 = 3
    step_2 = 1.1
    plt.barh(np.arange(9)[::-1]*step_1,eval(f'{demo_feat}_adj_r2_model1_mean_array'),xerr=eval(f'{demo_feat}_adj_r2_model1_std_array'),
         height=1,label='Only Demo-Feat',color='k',alpha=0.8) #color='orange',alpha=1
    plt.barh(np.arange(9)[::-1]*step_1-1*step_2,eval(f'{demo_feat}_adj_r2_model2_mean_array'),xerr=eval(f'{demo_feat}_adj_r2_model2_std_array'),
            height=1,label='With Vulnerability',color='green',alpha=0.7)#color='orange'
    #plt.legend(fontsize=12)
    if(show_yticks):
        plt.yticks(np.arange(9)*step_1-0.5*step_2,anno_list[::-1], fontsize=20)  
    else:
        empty_list = ['','','','','','','','','']
        plt.yticks(np.arange(9)*step_1-0.5*step_2,empty_list, fontsize=20)  
    #plt.xlim(0,1)
    plt.xticks(fontsize=14) 
    plt.xlabel(f'Explained variance\n',fontsize=18) #20220313
    '''
    if(demo_feat=='minority'):
        plt.xlabel(f'Explained variance\nof equity-by-race-ethnicity',fontsize=18)
    else:
        plt.xlabel(f'Explained variance\nof equity-by-{demo_feat}',fontsize=18)
    '''
    # Save figure
    savepath = os.path.join(fig_save_root, f'fig3c_{demo_feat}.pdf')
    plt.savefig(savepath,bbox_inches = 'tight')
    print(f'Fig3c_{demo_feat}, saved at {savepath}')
    

# Fig3c, age
plot_equity_explained_variance(demo_feat='age', show_yticks=True)
# Fig3c, income
plot_equity_explained_variance(demo_feat='income', show_yticks=False)
# Fig3c, occupation
plot_equity_explained_variance(demo_feat='occupation', show_yticks=False)
# Fig3c, minority
plot_equity_explained_variance(demo_feat='minority', show_yticks=False)

# Fig3c, legend
plt.figure()
label_list = ['Only demo-feats','With $\it{community~risk}$']
color_list = ['k','green']
alpha_list = [0.8, 0.7]
patches = [mpatches.Patch(color=color_list[i], alpha=alpha_list[i], label="{:s}".format(label_list[i])) for i in range(2) ]
plt.legend(handles=patches,ncol=2,fontsize=20,bbox_to_anchor=(0.8,-0.1)) 
# Save figure
savepath = os.path.join(fig_save_root, f'fig3c_legend.pdf')
plt.savefig(savepath,bbox_inches = 'tight')
print(f'Fig3c_legend, saved at {savepath}')
