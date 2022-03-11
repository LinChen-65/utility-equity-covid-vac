# python plot_corr_with_mobility.py --num_groups 50 --colormap hot

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.offsetbox import AnchoredText
import argparse

import constants
import functions

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/data/chenlin/COVID-19/Data',
                    help='Root to retrieve data. data for dl3, home for rl4.')  
parser.add_argument('--saveroot', default='/data/chenlin/utility-equity-covid-vac/results/figures',
                    help='Root to save generated figures.')
parser.add_argument('--num_groups', type=int, default=50,
                    help='Num of groups to divide CBGs into (for quantization).')   
parser.add_argument('--colormap', default='hot',
                    help='Color map for graph.')                      
args = parser.parse_args()

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
def scatter_kde(df, col_x, col_y, savepath, colormap='Spectral_r', print_corr=False, corr=None): 
    fig=plt.figure()
    ax=fig.add_subplot(111)
    # Calculate the point density
    xystack = np.vstack([df[col_x],df[col_y]])
    z = gaussian_kde(xystack)(xystack)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = df[col_x][idx], df[col_y][idx], z[idx]
    plt.scatter(x, y, c=z, s=40,cmap=colormap)
    cbar = plt.colorbar()
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(13)
    if(col_x=='Elder_Ratio'):
        label_x = 'Older adult ratio' #'Norm. older adult percentage' 
    elif(col_x=='Essential_Worker_Ratio'):
        label_x = 'Essential worker ratio' #'Norm. essential worker percentage' 
    elif(col_x=='Mean_Household_Income'):
        label_x = 'Average household income' #'Norm. average household income'
    elif(col_x=='Minority_Ratio'):
        label_x = 'Minority ratio' 
    else:
        label_x = col_x #'Norm. '+col_x
          
    if(col_y=='Mobility'):
        label_y = 'Per capita mobility' #'Norm. per capita mobility'  #'Normalized mobility'  
    else:
        label_y = col_y #'Norm. '+col_y
    plt.xlabel(label_x.replace("_", " "),fontsize=21) #fontsize=19
    plt.ylabel(label_y.replace("_", " "),fontsize=21) #fontsize=19
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if(print_corr==True):
        if(corr==None):
            plt.text(0.4, 0.9, 'Corr: %s'%(np.round(df[col_x].corr(df[col_y]), 2)), fontsize=18, transform=ax.transAxes)
        else:
            plt.text(0.4, 0.9, 'Corr: %s'%(np.round(corr, 2)), fontsize=18, transform=ax.transAxes)
    plt.savefig(savepath,bbox_inches = 'tight')
    print('Figure saved. Path: ', savepath)
    

############################################################
# Load Data

# Load ACS Data for matching with NYT Data
acs_data = pd.read_csv(os.path.join(args.root,'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
# Load SafeGraph data to obtain CBG sizes (i.e., populations)
filepath = os.path.join(args.root,"safegraph_open_census_data/data/cbg_b01.csv")
cbg_agesex = pd.read_csv(filepath)
# cbg_c24.csv: Occupation
filepath = os.path.join(args.root,"safegraph_open_census_data/data/cbg_c24.csv")
cbg_occupation = pd.read_csv(filepath)
# Load ACS 5-year (2013-2017) Data: Mean Household Income
filepath = os.path.join(args.root,"ACS_5years_Income_Filtered_Summary.csv")
cbg_income = pd.read_csv(filepath)
# Drop duplicate column 'Unnamed:0'
cbg_income.drop(['Unnamed: 0'],axis=1, inplace=True)
# cbg_b02.csv: Race #20220302
filepath = os.path.join(args.root,"safegraph_open_census_data/data/cbg_b02.csv")
cbg_race = pd.read_csv(filepath)
# cbg_b03.csv: Ethnic #20220302
filepath = os.path.join(args.root,"safegraph_open_census_data/data/cbg_b03.csv")
cbg_ethnic = pd.read_csv(filepath)

data = pd.DataFrame()
msa_count = 0
corr_age_mobility_list = []
corr_income_mobility_list = []
corr_occupation_mobility_list = []
corr_minority_mobility_list = [] #20220302

for msa_idx in range(len(constants.MSA_NAME_LIST)):
    MSA_NAME = constants.MSA_NAME_LIST[msa_idx]
    MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME]
    if(MSA_NAME=='NewYorkCity'):continue
    print('MSA_NAME: ',MSA_NAME)
    
    # Extract data specific to one msa, according to ACS data
    msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
    msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
    msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
    good_list = list(msa_data['FIPS Code'].values)

    # Load CBG ids belonging to a specific metro area
    cbg_ids_msa = pd.read_csv(os.path.join(args.root,MSA_NAME,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
    cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
    M = len(cbg_ids_msa)
    # Mapping from cbg_ids to columns in hourly visiting matrices
    cbgs_to_idxs = dict(zip(cbg_ids_msa['census_block_group'].values, range(M)))
    x = {}
    for i in cbgs_to_idxs: x[str(i)] = cbgs_to_idxs[i]

    # Select counties belonging to the MSA
    y = []
    for i in x:
        if((len(i)==12) & (int(i[0:5])in good_list)): y.append(x[i])
        if((len(i)==11) & (int(i[0:4])in good_list)): y.append(x[i])          
    idxs_msa_all = list(x.values())
    idxs_msa_nyt = y

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
    cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
    # Extract columns of interest
    columns_of_interest = ['census_block_group','Sum'] + constants.DETAILED_AGE_LIST
    cbg_age_msa = cbg_age_msa[columns_of_interest].copy()
    # Deal with CBGs with 0 populations
    cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
    # Calculate elder ratios
    cbg_age_msa['Elder_Absolute'] = cbg_age_msa.apply(lambda x : x['70 To 74 Years']+x['75 To 79 Years']+x['80 To 84 Years']+x['85 Years And Over'],axis=1)
    cbg_age_msa['Elder_Ratio'] = cbg_age_msa['Elder_Absolute'] / cbg_age_msa['Sum']
    # Deal with NaN values
    cbg_age_msa.fillna(0,inplace=True)
    # Obtain cbg sizes (populations)
    cbg_sizes = cbg_age_msa['Sum'].values
    cbg_sizes = np.array(cbg_sizes,dtype='int32'); #print('Total population: ',np.sum(cbg_sizes))

    # Load other Safegraph demographic data
    # Occupation
    cbg_occupation_msa = pd.merge(cbg_ids_msa, cbg_occupation, on='census_block_group', how='left')
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

    # Income
    cbg_income_msa = pd.merge(cbg_ids_msa, cbg_income, on='census_block_group', how='left')
    cbg_income_msa['Sum'] = cbg_age_msa['Sum'].copy()
    cbg_income_msa.rename(columns = {'total_households':'Total_Households',
                                     'mean_household_income':'Mean_Household_Income'},inplace=True)
    # Deal with NaN values
    cbg_income_msa.fillna(0,inplace=True)

    # Minority
    # Race
    cbg_race_msa = pd.merge(cbg_ids_msa, cbg_race, on='census_block_group', how='left')
    # Add information of cbg populations, from cbg_age_msa
    cbg_race_msa['Sum'] = cbg_sizes.copy()
    cbg_race_msa.rename(columns={'B02001e2':'White_Absolute'},inplace=True)
    # Extract columns of interest
    columns_of_interest = ['census_block_group', 'Sum', 'White_Absolute']
    cbg_race_msa = cbg_race_msa[columns_of_interest].copy()
    # Ethnicity
    cbg_ethnic_msa = pd.merge(cbg_ids_msa, cbg_ethnic, on='census_block_group', how='left')
    cbg_ethnic_msa.rename(columns={'B03002e13':'Hispanic_White_Absolute',
                                   'B03002e3': 'NH_White'},inplace=True) #经验证，下方计算方式与直接取这个e3结果一致
    # Combine to calculate minority
    cbg_race_msa['Minority_Absolute'] = cbg_race_msa['Sum'] - (cbg_race_msa['White_Absolute'] - cbg_ethnic_msa['Hispanic_White_Absolute'])
    cbg_race_msa['Minority_Ratio'] = cbg_race_msa['Minority_Absolute'] / cbg_race_msa['Sum']
    #print((cbg_ethnic_msa['NH_White'] == (cbg_race_msa['White_Absolute'] - cbg_ethnic_msa['Hispanic_White_Absolute'])).all())

    # Deal with NaN values
    cbg_race_msa.fillna(0,inplace=True)
    # Check whether there is NaN in cbg_tables
    if(cbg_race_msa.isnull().any().any()):
        print('NaN exists in cbg_race_msa. Please check.')
        pdb.set_trace()
    
    '''
    print(np.round((cbg_race_msa['Minority_Absolute'].sum()) / (cbg_race_msa['Sum'].sum()),3), 
           '\n', np.round(cbg_race_msa['Minority_Ratio'].mean(),3),
           '\n', np.round(cbg_race_msa['Minority_Ratio'].std(),3),
           '\n', np.round(cbg_race_msa['Minority_Ratio'].median(),3), 
           '\n', np.round(cbg_race_msa['Minority_Ratio'].max(),3), 
           '\n', np.round(cbg_race_msa['Minority_Ratio'].min(),3))
    pdb.set_trace()
    '''

    ###############################################################################
    # Load and scale age-aware CBG-specific attack/death rates (original)
    
    cbg_death_rates_original = np.loadtxt(os.path.join(args.root, MSA_NAME, 'cbg_death_rates_original_'+MSA_NAME))
    cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape) 
    # Fix attack_scale
    attack_scale = 1
    cbg_attack_rates_scaled = cbg_attack_rates_original * attack_scale
    # Scale death rates
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
    if(os.path.exists(os.path.join(args.root, '3cbg_avg_infect_same_%s.npy'%MSA_NAME))):
        print('cbg_avg_infect_same, cbg_avg_infect_diff: Load existing file.')
        cbg_avg_infect_same = np.load(os.path.join(args.root, '3cbg_avg_infect_same_%s.npy'%MSA_NAME))
        cbg_avg_infect_diff = np.load(os.path.join(args.root, '3cbg_avg_infect_diff_%s.npy'%MSA_NAME))
    else:
        print('cbg_avg_infect_same, cbg_avg_infect_diff: File not found. Please check.')
        pdb.set_trace()

    SEIR_at_30d = np.load(os.path.join(args.root, 'SEIR_at_30d.npy'),allow_pickle=True).item()
    S_ratio = SEIR_at_30d[MSA_NAME]['S'] / (cbg_sizes.sum())
    I_ratio = SEIR_at_30d[MSA_NAME]['I'] / (cbg_sizes.sum())

    # Deal with nan and inf (https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
    cbg_avg_infect_same = np.nan_to_num(cbg_avg_infect_same,nan=0,posinf=0,neginf=0)
    cbg_avg_infect_diff = np.nan_to_num(cbg_avg_infect_diff,nan=0,posinf=0,neginf=0)
    cbg_age_msa['Infect'] = cbg_avg_infect_same + cbg_avg_infect_diff

    # Normalize by cbg population
    cbg_avg_infect_same_norm = cbg_avg_infect_same / cbg_sizes
    cbg_avg_infect_diff_norm = cbg_avg_infect_diff / cbg_sizes
    cbg_avg_infect_all_norm = cbg_avg_infect_same_norm + cbg_avg_infect_diff_norm

    # Compute the average death rate (alpha_bar) for the whole MSA: perform another weighted average over all CBGs
    avg_death_rates_scaled = np.matmul(cbg_sizes.T, cbg_death_rates_scaled) / np.sum(cbg_sizes)
    cbg_age_msa['Mobility'] = cbg_avg_infect_diff_norm.copy()

    # No_Vaccination & Age_Agnostic, accumulated results # 20210802
    deaths_cbg_no_vaccination = np.load(os.path.join(args.root,MSA_NAME,'20210206_deaths_cbg_no_vaccination_%s.npy'%MSA_NAME))
    deaths_cbg_age_agnostic = np.load(os.path.join(args.root,MSA_NAME,'20210206_deaths_cbg_age_agnostic_%s.npy'%MSA_NAME))
    
    # Collect data together
    data_msa = pd.DataFrame()
    data_msa['Sum'] = cbg_age_msa['Sum'].copy()
    data_msa['Elder_Ratio'] = cbg_age_msa['Elder_Ratio'].copy()
    data_msa['Mean_Household_Income'] = cbg_income_msa['Mean_Household_Income'].copy()
    data_msa['Essential_Worker_Ratio'] = cbg_occupation_msa['Essential_Worker_Ratio'].copy()
    data_msa['Minority_Ratio'] = cbg_race_msa['Minority_Ratio'] #20220302
    data_msa['Mobility'] = cbg_age_msa['Mobility'].copy()
    
    data_msa['Valid'] = data_msa.apply(lambda x : 1 if x['Essential_Worker_Ratio']>0.2 else 0, axis=1)
    print('Before filtering, len(data_msa):', len(data_msa))
    data_msa = data_msa[data_msa['Valid']==1]
    print('After filtering, len(data_msa):', len(data_msa))
    data_msa = data_msa.reset_index()
    
    ###############################################################################
    # Grouping: 按args.num_groups分位数，将全体CBG分为args.num_groups个组，将分割点存储在separators中
    
    separators = functions.get_separators(data_msa, args.num_groups, 'Elder_Ratio','Sum', normalized=True)
    data_msa['Elder_Ratio_Quantile'] =  data_msa['Elder_Ratio'].apply(lambda x : functions.assign_group(x, separators))

    separators = functions.get_separators(data_msa, args.num_groups, 'Essential_Worker_Ratio','Sum', normalized=True)
    data_msa['Essential_Worker_Ratio_Quantile'] =  data_msa['Essential_Worker_Ratio'].apply(lambda x : functions.assign_group(x, separators))

    separators = functions.get_separators(data_msa, args.num_groups, 'Mean_Household_Income','Sum', normalized=False)
    data_msa['Mean_Household_Income_Quantile'] =  data_msa['Mean_Household_Income'].apply(lambda x : functions.assign_group(x, separators))

    # Grouping: 按args.num_groups分位数，将全体CBG分为args.num_groups个组，将分割点存储在separators中
    separators = functions.get_separators(data_msa, args.num_groups, 'Minority_Ratio','Sum', normalized=False)
    data_msa['Minority_Ratio_Quantile'] =  data_msa['Minority_Ratio'].apply(lambda x : functions.assign_group(x, separators))

    ###############################################################################
    print('Before ranking:')
    print('Correlation between Elder_Ratio and Mobility: ', np.corrcoef(data_msa['Elder_Ratio'], data_msa['Mobility'])[0][1]) 
    print('Correlation between Income and Mobility: ', np.corrcoef(data_msa['Mean_Household_Income'],data_msa['Mobility'])[0][1])
    print('Correlation between EW_Ratio and Mobility: ', np.corrcoef(data_msa['Essential_Worker_Ratio'], data_msa['Mobility'])[0][1])
    print('Correlation between Minority_Ratio and Mobility: ', np.corrcoef(data_msa['Minority_Ratio'], data_msa['Mobility'])[0][1])
    
    elder_ratio_list = []
    mobility_age_list = []
    for group_idx in range(args.num_groups):
        selected_cbgs = data_msa[data_msa['Elder_Ratio_Quantile']==group_idx].index
        elder_ratio_list.append(get_avg_feat(selected_cbgs,data_msa, 'Elder_Ratio'))
        mobility_age_list.append(get_avg_feat(selected_cbgs,data_msa, 'Mobility'))
    print(np.corrcoef(elder_ratio_list,mobility_age_list)[0][1])
    corr_age_mobility_list.append(np.corrcoef(elder_ratio_list,mobility_age_list)[0][1])
    # Normalization by MSA max #20210927
    mobility_age_list /= np.max(np.array(mobility_age_list))
    elder_ratio_list /= np.max(np.array(elder_ratio_list))
    # Stack to array
    if(msa_idx==0):
        mobility_age_all_msa_array = np.array(mobility_age_list)
        elder_ratio_all_msa_array = np.array(elder_ratio_list)
    else:
        mobility_age_all_msa_array = np.vstack((mobility_age_all_msa_array, np.array(mobility_age_list)))
        elder_ratio_all_msa_array = np.vstack((elder_ratio_all_msa_array , np.array(elder_ratio_list)))
    

    income_list = []
    mobility_income_list = []
    for group_idx in range(args.num_groups):
        selected_cbgs = data_msa[data_msa['Mean_Household_Income_Quantile']==group_idx].index
        income_list.append(get_avg_feat(selected_cbgs,data_msa, 'Mean_Household_Income'))
        mobility_income_list.append(get_avg_feat(selected_cbgs,data_msa, 'Mobility'))
    print(np.corrcoef(income_list,mobility_income_list)[0][1])
    corr_income_mobility_list.append(np.corrcoef(income_list,mobility_income_list)[0][1])
    # Normalization by MSA max #20210927
    mobility_income_list /= np.max(np.array(mobility_income_list))
    income_list /= np.max(np.array(income_list))
    # Stack to array
    if(msa_idx==0):
        mobility_income_all_msa_array = np.array(mobility_income_list)
        income_all_msa_array = np.array(income_list)
    else:
        mobility_income_all_msa_array = np.vstack((mobility_income_all_msa_array, np.array(mobility_income_list)))
        income_all_msa_array = np.vstack((income_all_msa_array, np.array(income_list)))
    
    
    ew_ratio_list = []
    mobility_occupation_list = []
    for group_idx in range(args.num_groups):
        selected_cbgs = data_msa[data_msa['Essential_Worker_Ratio_Quantile']==group_idx].index
        ew_ratio_list.append(get_avg_feat(selected_cbgs,data_msa, 'Essential_Worker_Ratio'))
        mobility_occupation_list.append(get_avg_feat(selected_cbgs,data_msa, 'Mobility'))
    print(np.corrcoef(ew_ratio_list,mobility_occupation_list)[0][1])
    corr_occupation_mobility_list.append(np.corrcoef(ew_ratio_list,mobility_occupation_list)[0][1])
    # Normalization by MSA max #20210927
    mobility_occupation_list /= np.max(np.array(mobility_occupation_list))
    ew_ratio_list /= np.max(np.array(ew_ratio_list))
    # Stack to array
    if(msa_idx==0):
        mobility_occupation_all_msa_array = np.array(mobility_occupation_list)
        ew_ratio_all_msa_array = np.array(ew_ratio_list)
    else:
        mobility_occupation_all_msa_array = np.vstack((mobility_occupation_all_msa_array, np.array(mobility_occupation_list)))
        ew_ratio_all_msa_array = np.vstack((ew_ratio_all_msa_array, np.array(ew_ratio_list)))


    minority_ratio_list = []
    mobility_minority_list = []
    for group_idx in range(args.num_groups):
        selected_cbgs = data_msa[data_msa['Minority_Ratio_Quantile']==group_idx].index
        minority_ratio_list.append(get_avg_feat(selected_cbgs,data_msa, 'Minority_Ratio'))
        mobility_minority_list.append(get_avg_feat(selected_cbgs,data_msa, 'Mobility'))
    print(np.corrcoef(minority_ratio_list,mobility_minority_list)[0][1])
    corr_minority_mobility_list.append(np.corrcoef(minority_ratio_list,mobility_minority_list)[0][1])
    # Normalization by MSA max #20210927
    mobility_minority_list /= np.max(np.array(mobility_minority_list))
    minority_ratio_list /= np.max(np.array(minority_ratio_list))
    # Stack to array
    if(msa_idx==0):
        mobility_minority_all_msa_array = np.array(mobility_minority_list)
        minority_ratio_all_msa_array = np.array(minority_ratio_list)
        
    else:
        mobility_minority_all_msa_array = np.vstack((mobility_minority_all_msa_array, np.array(mobility_minority_list)))
        minority_ratio_all_msa_array = np.vstack((minority_ratio_all_msa_array, np.array(minority_ratio_list)))

    ###############################################################################
    # Ranking (dense, percentile)
    
    data_msa['Elder_Ratio'] = data_msa['Elder_Ratio'].rank(method='dense',pct=True)
    data_msa['Mean_Household_Income'] = data_msa['Mean_Household_Income'].rank(method='dense',pct=True)
    data_msa['Essential_Worker_Ratio'] = data_msa['Essential_Worker_Ratio'].rank(method='dense',pct=True)
    data_msa['Minority_Ratio'] = data_msa['Minority_Ratio'].rank(method='dense',pct=True) #20220302
    data_msa['Mobility'] = data_msa['Mobility'].rank(method='dense',pct=True)

    print('After ranking:')
    print('Correlation between Elder_Ratio and Mobility: ', np.corrcoef(data_msa['Elder_Ratio'], data_msa['Mobility'])[0][1]) 
    print('Correlation between Income and Mobility: ', np.corrcoef(data_msa['Mean_Household_Income'],data_msa['Mobility'])[0][1])
    print('Correlation between EW_Ratio and Mobility: ', np.corrcoef(data_msa['Essential_Worker_Ratio'], data_msa['Mobility'])[0][1])
    print('Correlation between Minority_Ratio and Mobility: ', np.corrcoef(data_msa['Minority_Ratio'], data_msa['Mobility'])[0][1])

    data = data.append(data_msa, ignore_index=True)
    print('len(data): ',len(data))
    

print('corr_age_mobility_list: ', corr_age_mobility_list, 'Mean: ', np.mean(np.array(corr_age_mobility_list)))
print('corr_income_mobility_list: ', corr_income_mobility_list, 'Mean: ', np.mean(np.array(corr_income_mobility_list)))
print('corr_occupation_mobility_list: ', corr_occupation_mobility_list, 'Mean: ', np.mean(np.array(corr_occupation_mobility_list)))
print('corr_minority_mobility_list: ', corr_minority_mobility_list, 'Mean: ', np.mean(np.array(corr_minority_mobility_list)))

print('mobility_age_all_msa_array.shape:',mobility_age_all_msa_array.shape)

data_temp = pd.DataFrame(columns=['Elder_Ratio','Mobility'])
data_temp['Elder_Ratio'] = elder_ratio_all_msa_array.flatten()
data_temp['Mobility'] = mobility_age_all_msa_array.flatten()
#savepath = os.path.join(args.saveroot, '20220302_grouping_%s_%squant_rank_age_mobility_normbygroupmax.jpg'%(args.colormap,args.num_groups))
savepath = os.path.join(args.saveroot, 'fig1e_age.pdf') #20220310
scatter_kde(data_temp, 'Elder_Ratio', 'Mobility', savepath, args.colormap, print_corr=False,corr=np.mean(np.array(corr_age_mobility_list)))

data_temp = pd.DataFrame(columns=['Mean_Household_Income','Mobility'])
data_temp['Mean_Household_Income'] = income_all_msa_array.flatten()
data_temp['Mobility'] = mobility_income_all_msa_array.flatten()
#savepath = os.path.join(args.saveroot, '20220302_grouping_%s_%squant_rank_income_mobility_normbygroupmax.jpg'%(args.colormap,args.num_groups))
savepath = os.path.join(args.saveroot, 'fig1e_income.pdf') #20220310
scatter_kde(data_temp, 'Mean_Household_Income', 'Mobility', savepath, args.colormap, print_corr=False,corr=np.mean(np.array(corr_income_mobility_list)))

data_temp = pd.DataFrame(columns=['Essential_Worker_Ratio','Mobility'])
data_temp['Essential_Worker_Ratio'] = ew_ratio_all_msa_array.flatten()
data_temp['Mobility'] = mobility_occupation_all_msa_array.flatten()
#savepath = os.path.join(args.saveroot, '20220302_grouping_%s_%squant_rank_occupation_mobility_filt0.2_normbygroupmax.jpg'%(args.colormap,args.num_groups))
savepath = os.path.join(args.saveroot, 'fig1e_occupation.pdf') #20220310
scatter_kde(data_temp, 'Essential_Worker_Ratio', 'Mobility', savepath, args.colormap, print_corr=False,corr=np.mean(np.array(corr_occupation_mobility_list)))

data_temp = pd.DataFrame(columns=['Minority_Ratio','Mobility'])
data_temp['Minority_Ratio'] = minority_ratio_all_msa_array.flatten()
data_temp['Mobility'] = mobility_minority_all_msa_array.flatten()
#savepath = os.path.join(args.saveroot, '20220302_grouping_%s_%squant_rank_minority_mobility_normbygroupmax.jpg'%(args.colormap,args.num_groups))
savepath = os.path.join(args.saveroot, 'fig1e_minority.pdf') #20220310
scatter_kde(data_temp, 'Minority_Ratio', 'Mobility', savepath, args.colormap, print_corr=False,corr=np.mean(np.array(corr_minority_mobility_list)))

pdb.set_trace()

###############################################################################
# Preprocessing: Binning (数据分箱)

print('Discretization, ', args.num_groups)
enc = KBinsDiscretizer(n_bins=args.num_groups, encode="ordinal",strategy='uniform') #strategy='kmeans''uniform'
for column in data.columns:
    data[column] = enc.fit_transform(np.array(data[column]).reshape(-1,1))
    data[column] = enc.inverse_transform(np.array(data[column]).reshape(-1,1))

print('Correlation between Elder_Ratio and Mobility: ', np.corrcoef(data['Elder_Ratio'], data['Mobility'])[0][1]) 
print('Correlation between Income and Mobility: ', np.corrcoef(data['Mean_Household_Income'],data['Mobility'])[0][1])
print('Correlation between EW_Ratio and Mobility: ', np.corrcoef(data['Essential_Worker_Ratio'], data['Mobility'])[0][1])

pdb.set_trace()


