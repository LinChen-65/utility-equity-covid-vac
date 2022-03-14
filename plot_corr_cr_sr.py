# python correlation_cr_sr.py 

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import socket
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse

import constants
import functions

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--num_groups', type=int, default=50,
                    help='Number of groups for quantization.')
parser.add_argument('--color_map', default='hot',
                    help='Color map for scatter plot.')
#parser.add_argument('--time_string', default='20210206',
#                    help='Time string to specify the model.')                    
args = parser.parse_args()
print('args.num_groups: ',args.num_groups)
print('Color map:', args.color_map)

# root
hostname = socket.gethostname()
print('hostname: ', hostname)
if(hostname in ['fib-dl3','rl3','rl2']):
    root = '/data/chenlin/COVID-19/Data' #dl3
    saveroot = '/data/chenlin/utility-equity-covid-vac/results/'
elif(hostname=='rl4'):
    root = '/home/chenlin/COVID-19/Data' #rl4
    saveroot = '/home/chenlin/utility-equity-covid-vac/results/'

############################################################
# Functions

# Compute the average features of a group of CBGs.
def get_avg_feat(cbg_list, data_df, feat_str):
    values = []
    weights = []
    for cbg in cbg_list:
        values.append(data_df.iloc[cbg][feat_str])
        weights.append(data_df.iloc[cbg]['Sum'])
    return np.average(np.array(values),weights=weights) 
    

# Scatter plot with density
def scatter_kde(df, col_x, col_y, savepath, color_map='Spectral_r'): 
    plt.figure()
    # Calculate the point density
    xystack = np.vstack([df[col_x],df[col_y]])
    z = gaussian_kde(xystack)(xystack)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = df[col_x][idx], df[col_y][idx], z[idx]
    plt.scatter(x, y, c=z, s=20,cmap=args.color_map)
    plt.colorbar()
    if(col_x=='Vulnerability'):
        label_x = 'Community Risk' 
    else:
        label_x = col_x
    if(col_y=='Damage'):
        label_y = 'Societal Risk'  
    else:
        label_y = col_y
    plt.xlabel(label_x.replace("_", " "),fontsize=20)
    plt.ylabel(label_y.replace("_", " "),fontsize=20)
    plt.xticks(fontsize=15) #fontsize=12
    plt.yticks(fontsize=15) #fontsize=12
    #plt.text(0.4, 0.9, 'Corr: %s'%(np.round(df[col_x].corr(df[col_y]), 2)), fontsize=18)
    plt.savefig(savepath,bbox_inches = 'tight')
    print('Figure saved. Path: ', savepath)
    
############################################################
# Load Data

# Load ACS Data for matching with NYT Data
acs_data = pd.read_csv(os.path.join(root,'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
# Load SafeGraph data to obtain CBG sizes (i.e., populations)
filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_b01.csv")
cbg_agesex = pd.read_csv(filepath)
# cbg_c24.csv: Occupation
filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_c24.csv")
cbg_occupation = pd.read_csv(filepath)
# Load ACS 5-year (2013-2017) Data: Mean Household Income
filepath = os.path.join(root,"ACS_5years_Income_Filtered_Summary.csv")
cbg_income = pd.read_csv(filepath)
# Drop duplicate column 'Unnamed:0'
cbg_income.drop(['Unnamed: 0'],axis=1, inplace=True)

data = pd.DataFrame()
msa_count = 0
vd_corr_list = []
for msa_idx in range(len(constants.MSA_NAME_LIST)):
    MSA_NAME = constants.MSA_NAME_LIST[msa_idx]
    MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME]
    if(MSA_NAME=='NewYorkCity'):continue
    print('\nMSA_NAME: ',MSA_NAME)
    
    # Extract data specific to one msa, according to ACS data
    msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
    msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
    msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
    good_list = list(msa_data['FIPS Code'].values)

    # Load CBG ids belonging to a specific metro area
    cbg_ids_msa = pd.read_csv(os.path.join(root,MSA_NAME,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
    cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
    M = len(cbg_ids_msa)
    # Mapping from cbg_ids to columns in hourly visiting matrices
    cbgs_to_idxs = dict(zip(cbg_ids_msa['census_block_group'].values, range(M)))
    x = {}
    for i in cbgs_to_idxs:
        x[str(i)] = cbgs_to_idxs[i]
    #print('Number of CBGs in this metro area:', M)

    # Select counties belonging to the MSA
    y = []
    for i in x:
        if((len(i)==12) & (int(i[0:5])in good_list)):
            y.append(x[i])
        if((len(i)==11) & (int(i[0:4])in good_list)):
            y.append(x[i])
            
    idxs_msa_all = list(x.values()) #;print('Number of CBGs in this metro area:', len(idxs_msa_all))
    idxs_msa_nyt = y #;print('Number of CBGs in to compare with NYT data:', len(idxs_msa_nyt))

    # Load ACS Data for MSA-county matching
    acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
    msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
    msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
    msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
    good_list = list(msa_data['FIPS Code'].values)

    # Extract CBGs belonging to the MSA - https://covid-mobility.stanford.edu//datasets/
    cbg_age_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')

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
    cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
    # Calculate elder ratios
    cbg_age_msa['Elder_Absolute'] = cbg_age_msa.apply(lambda x : x['70 To 74 Years']+x['75 To 79 Years']+x['80 To 84 Years']+x['85 Years And Over'],axis=1)
    cbg_age_msa['Elder_Ratio'] = cbg_age_msa['Elder_Absolute'] / cbg_age_msa['Sum']

    # Obtain cbg sizes (populations)
    cbg_sizes = cbg_age_msa['Sum'].values
    cbg_sizes = np.array(cbg_sizes,dtype='int32')
    '''
    # Load other Safegraph demographic data
    # Extract pois corresponding to the metro area, by merging dataframes
    cbg_occupation_msa = pd.merge(cbg_ids_msa, cbg_occupation, on='census_block_group', how='left')

    columns_of_essential_workers = list(constants.ew_rate_dict.keys())
    for column in columns_of_essential_workers:
        cbg_occupation_msa[column] = cbg_occupation_msa[column].apply(lambda x : x*constants.ew_rate_dict[column])
    cbg_occupation_msa['Essential_Worker_Absolute'] = cbg_occupation_msa.apply(lambda x : x[columns_of_essential_workers].sum(), axis=1)
    cbg_occupation_msa['Sum'] = cbg_age_msa['Sum']
    cbg_occupation_msa['Essential_Worker_Ratio'] = cbg_occupation_msa['Essential_Worker_Absolute'] / cbg_occupation_msa['Sum']
    columns_of_interest = ['census_block_group','Sum','Essential_Worker_Absolute','Essential_Worker_Ratio']
    cbg_occupation_msa = cbg_occupation_msa[columns_of_interest].copy()

    # Extract pois corresponding to the metro area (Philadelphia), by merging dataframes
    cbg_income_msa = pd.merge(cbg_ids_msa, cbg_income, on='census_block_group', how='left')
    # Add information of cbg populations, from cbg_age_Phi(cbg_b01.csv)
    cbg_income_msa['Sum'] = cbg_age_msa['Sum'].copy()
    # Rename
    cbg_income_msa.rename(columns = {'total_households':'Total_Households',
                                     'mean_household_income':'Mean_Household_Income'},inplace=True)

    # Deal with NaN values
    cbg_age_msa.fillna(0,inplace=True)
    cbg_income_msa.fillna(0,inplace=True)
    cbg_occupation_msa.fillna(0,inplace=True)
    '''

    ###############################################################################
    # Load and scale age-aware CBG-specific attack/death rates (original)
    
    cbg_death_rates_original = np.loadtxt(os.path.join(root, MSA_NAME, 'cbg_death_rates_original_'+MSA_NAME))
    cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape) 
    attack_scale = 1
    cbg_attack_rates_scaled = cbg_attack_rates_original * attack_scale
    cbg_death_rates_scaled = cbg_death_rates_original * constants.death_scale_dict[MSA_NAME]
    cbg_age_msa['Death_Rate'] =  cbg_death_rates_scaled
    
    ###############################################################################
    # Obtain vulnerability and damage, according to theoretical analysis

    nyt_included = np.zeros(len(idxs_msa_all))
    for i in range(len(nyt_included)):
        if(i in idxs_msa_nyt):
            nyt_included[i] = 1
    cbg_age_msa['NYT_Included'] = nyt_included.copy()

    # Retrieve the attack rate for the whole MSA (home_beta, fitted for each MSA)
    home_beta = constants.parameters_dict[MSA_NAME][1]

    # Load cbg_avg_infect_same, cbg_avg_infect_diff
    if(os.path.exists(os.path.join(root, '3cbg_avg_infect_same_%s.npy'%MSA_NAME))):
        #print('cbg_avg_infect_same, cbg_avg_infect_diff: Load existing file.')
        cbg_avg_infect_same = np.load(os.path.join(root, '3cbg_avg_infect_same_%s.npy'%MSA_NAME))
        cbg_avg_infect_diff = np.load(os.path.join(root, '3cbg_avg_infect_diff_%s.npy'%MSA_NAME))
        pdb.set_trace()
    else:
        print('cbg_avg_infect_same, cbg_avg_infect_diff: File not found. Please check.')
        pdb.set_trace()
    #print('cbg_avg_infect_same.shape:',cbg_avg_infect_same.shape)

    SEIR_at_30d = np.load(os.path.join(root, 'SEIR_at_30d.npy'),allow_pickle=True).item()
    S_ratio = SEIR_at_30d[MSA_NAME]['S'] / (cbg_sizes.sum())
    I_ratio = SEIR_at_30d[MSA_NAME]['I'] / (cbg_sizes.sum())
    #print('S_ratio:',S_ratio,'I_ratio:',I_ratio)

    # Deal with nan and inf (https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
    cbg_avg_infect_same = np.nan_to_num(cbg_avg_infect_same,nan=0,posinf=0,neginf=0)
    cbg_avg_infect_diff = np.nan_to_num(cbg_avg_infect_diff,nan=0,posinf=0,neginf=0)
    cbg_age_msa['Infect'] = cbg_avg_infect_same + cbg_avg_infect_diff
    # Check whether there is NaN in cbg_tables
    #print('Any NaN in cbg_age_msa[\'Infect\']?', cbg_age_msa['Infect'].isnull().any().any())

    # Normalize by cbg population
    cbg_avg_infect_same_norm = cbg_avg_infect_same / cbg_sizes
    cbg_avg_infect_diff_norm = cbg_avg_infect_diff / cbg_sizes
    cbg_avg_infect_all_norm = cbg_avg_infect_same_norm + cbg_avg_infect_diff_norm

    # Compute the average death rate (alpha_bar) for the whole MSA: perform another weighted average over all CBGs
    avg_death_rates_scaled = np.matmul(cbg_sizes.T, cbg_death_rates_scaled) / np.sum(cbg_sizes)
    #print('avg_death_rates_scaled.shape:',avg_death_rates_scaled.shape) # shape: (), because it is a scalar

    # Compute vulnerability and damage for each cbg
    # New new method # 20210619
    cbg_vulnerability = cbg_avg_infect_all_norm * cbg_death_rates_scaled 
    cbg_secondary_damage = cbg_avg_infect_all_norm * (cbg_avg_infect_all_norm*(S_ratio/I_ratio)) * avg_death_rates_scaled
    cbg_damage = cbg_vulnerability + cbg_secondary_damage
    cbg_age_msa['Vulnerability'] = cbg_vulnerability.copy()
    cbg_age_msa['Damage'] = cbg_damage.copy()
    cbg_age_msa['Vulner_Rank'] = cbg_age_msa['Vulnerability'].rank(ascending=False,method='first') 
    cbg_age_msa['Damage_Rank'] = cbg_age_msa['Damage'].rank(ascending=False,method='first')
    # Only those belonging to the MSA (according to nyt) is valid for vaccination.
    # This is to prevent overlapping of CBGs across MSAs.
    cbg_age_msa['Vulner_Rank'] = cbg_age_msa.apply(lambda x :  x['Vulner_Rank'] if x['NYT_Included']==1 else M+1, axis=1)
    cbg_age_msa['Vulner_Rank_New'] = cbg_age_msa['Vulner_Rank'].rank(ascending=True,method='first')
    cbg_age_msa['Damage_Rank'] = cbg_age_msa.apply(lambda x :  x['Damage_Rank'] if x['NYT_Included']==1 else M+1, axis=1)
    cbg_age_msa['Damage_Rank_New'] = cbg_age_msa['Damage_Rank'].rank(ascending=True,method='first')

    # Collect data together
    data_msa = pd.DataFrame()
    data_msa['Vulnerability'] = cbg_age_msa['Vulnerability'].copy()
    data_msa['Damage'] = cbg_age_msa['Damage'].copy()

    # Ranking (dense, percentile)
    data_msa['Vulnerability'] = data_msa['Vulnerability'].rank(method='dense',pct=True)
    data_msa['Damage'] = data_msa['Damage'].rank(method='dense',pct=True)
    data = data.append(data_msa, ignore_index=True)
    #print('len(data): ',len(data))
    vd_corr_list.append(data_msa['Vulnerability'].corr(data['Damage']))    
#print('Correlation between Vulnerability and Damage: ', vd_corr_list)

###############################################################################
# Preprocessing: Binning (数据分箱)

print('Discretization, ', args.num_groups)
enc = KBinsDiscretizer(n_bins=args.num_groups, encode="ordinal",strategy='uniform') #strategy='kmeans''uniform'
for column in data.columns:
    data[column] = enc.fit_transform(np.array(data[column]).reshape(-1,1))
    data[column] = enc.inverse_transform(np.array(data[column]).reshape(-1,1))

# Scatter plot with density
savepath = os.path.join(saveroot, 'figures', 'fig3d.png') #'fig3d.pdf'
scatter_kde(data, 'Vulnerability', 'Damage', savepath, args.color_map)
