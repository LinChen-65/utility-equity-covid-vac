# python plot_groupwise_death_rate.py

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

import constants
import functions

import pdb

#root = '/data/chenlin/COVID-19/Data'
#saveroot = '/data/chenlin/utility-equity-covid-vac/results/figures'
root = os.getcwd()
dataroot = os.path.join(root, 'data')
resultroot = os.path.join(root, 'results')
fig_save_root = os.path.join(root, 'figures')

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--safegraph_root', default=dataroot, #'/data/chenlin/COVID-19/Data',
                    help='Safegraph data root.') 
args = parser.parse_args()

policy_list = ['Age_Agnostic','No_Vaccination']
demo_feat_list = ['Age', 'Income', 'Occupation', 'Race']
NUM_GROUPS = 10

# Drawing settings
alpha=1
markersize=11

################################################################################
# Functions

def get_cbg_ids_sizes(MSA_NAME): #20220227
    # Load CBG ids for the MSA
    cbg_ids_msa = pd.read_csv(os.path.join(dataroot,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
    cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)

    # Extract CBGs belonging to the MSA 
    cbg_age_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
    cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
    cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
    # Deal with NaN values
    cbg_age_msa.fillna(0,inplace=True)
    cbg_sizes = cbg_age_msa['Sum']

    return cbg_ids_msa, cbg_sizes
    

def get_msa_result(cbg_table, demo_feat, num_groups): #20220227
    results = {}
    for policy in policy_list:
        for group_idx in range(num_groups):
            eval('final_deaths_rate_'+ policy.lower())[group_idx] = cbg_table[cbg_table[demo_feat + '_Quantile']==group_idx]['Final_Deaths_' + policy].sum()
            eval('final_deaths_rate_'+ policy.lower())[group_idx] /= cbg_table[cbg_table[demo_feat + '_Quantile']==group_idx]['Sum'].sum()
        MSA_deaths_rate = cbg_table['Final_Deaths_' + policy].sum()
        MSA_deaths_rate /= cbg_table['Sum'].sum()
        
        # Normalization by MSA mean # 20210613
        for i in range(num_groups):
            eval('final_deaths_rate_'+ policy.lower())[i] /= MSA_deaths_rate 
            
        results[policy] = {'deaths':eval('final_deaths_rate_'+ policy.lower()),
                           'MSA_deaths':MSA_deaths_rate.copy()}
    return results


def get_avg_upper_lower(result_dict, num_groups): #20220227
    temp_agnostic = np.zeros(9); temp_aware = np.zeros(9)
    avg_agnostic = np.zeros(num_groups); avg_aware = np.zeros(num_groups)
    upper_agnostic = np.zeros(num_groups); lower_agnostic = np.zeros(num_groups)
    upper_aware = np.zeros(num_groups); lower_aware = np.zeros(num_groups)

    for group_idx in range(num_groups):
        #print('group_idx:',group_idx)
        count=0
        for msa_idx in range(len(constants.MSA_NAME_LIST)):
            if(constants.MSA_NAME_LIST[msa_idx]=='NewYorkCity'):continue
            MSA_NAME = constants.MSA_NAME_LIST[msa_idx]
            temp_agnostic[count] = result_dict[MSA_NAME]['Age_Agnostic']['deaths'][group_idx]
            temp_aware[count] = result_dict[MSA_NAME]['No_Vaccination']['deaths'][group_idx]
            count += 1
        
        avg_agnostic[group_idx] = np.mean(temp_agnostic)
        avg_aware[group_idx] = np.mean(temp_aware)
        upper_agnostic[group_idx] = np.percentile(temp_agnostic, 75, interpolation='nearest')
        upper_aware[group_idx] = np.percentile(temp_aware, 75, interpolation='nearest')
        lower_agnostic[group_idx] = np.percentile(temp_agnostic, 25, interpolation='nearest')
        lower_aware[group_idx] = np.percentile(temp_aware, 25, interpolation='nearest')

    error_upper_agnostic = upper_agnostic-avg_agnostic
    error_lower_agnostic = avg_agnostic-lower_agnostic
    error_agnostic = [error_lower_agnostic,error_upper_agnostic]
    error_upper_aware = upper_aware-avg_aware
    error_lower_aware = avg_aware-lower_aware
    error_aware = [error_lower_aware,error_upper_aware]
    return avg_agnostic, avg_aware, upper_agnostic, upper_aware, lower_agnostic, lower_aware, error_agnostic, error_aware
    

def plot_groupwise_death_rate(demo_feat, num_groups, alpha, markersize, save_figure, savepath, show_legend=True): #20220227
    demo_feat_label_dict = dict()
    demo_feat_label_dict['Elder_Ratio'] = 'Older adult ratio' #percentage #20220309
    demo_feat_label_dict['Mean_Household_Income'] = 'Average household income'
    demo_feat_label_dict['Essential_Worker_Ratio'] = 'Essential worker ratio'
    demo_feat_label_dict['Black_Ratio'] = 'Black resident ratio'
    demo_feat_label_dict['White_Ratio'] = 'White resident ratio'
    demo_feat_label_dict['Hispanic_Ratio'] = 'Hispanic resident ratio'
    demo_feat_label_dict['Minority_Ratio'] = 'Minority ratio' #20220302
    
    #plt.figure(figsize=((9,4)))
    plt.figure(figsize=((8,4))) #20220309
    plt.plot(np.arange(NUM_GROUPS),np.ones(NUM_GROUPS),
            label='SEIR model',marker='o',markersize=markersize,
            color='grey',alpha=0.6,linewidth=3)
    plt.plot(np.arange(NUM_GROUPS),avg_agnostic,
            label = 'Meta-population model',
            marker='s',markersize=markersize,color='C0')
    plt.plot(np.arange(NUM_GROUPS),avg_aware,
            label= 'BD model',
            marker='^',markersize=markersize+1,color='C1')

    error_upper_agnostic = upper_agnostic-avg_agnostic
    error_lower_agnostic = avg_agnostic-lower_agnostic
    error_agnostic=[error_lower_agnostic,error_upper_agnostic]
    plt.errorbar(x=np.arange(NUM_GROUPS), y=avg_agnostic,
                yerr=error_agnostic,
                capsize=6,elinewidth=2,linewidth=3) 

    error_upper_aware = upper_aware-avg_aware
    error_lower_aware = avg_aware-lower_aware
    error_aware=[error_lower_aware,error_upper_aware]
    plt.errorbar(x=np.arange(NUM_GROUPS), y=avg_aware,
                yerr=error_aware,
                capsize=6,elinewidth=3,linewidth=3) 

    x = np.arange(num_groups)
    plt.xticks(x,np.arange(NUM_GROUPS)+1,fontsize=14)
    if(show_legend): #20220310
        plt.legend(fontsize=23,loc='upper center',bbox_to_anchor=(0.56,1.3), ncol=3)
    if(NUM_GROUPS==5):
        plt.xlabel(f'Quintile of {demo_feat_label_dict[demo_feat]}',fontsize=25)
    elif(NUM_GROUPS==10):
        #plt.xlabel(f'Decile of {demo_feat_label_dict[demo_feat]}',fontsize=25)
        plt.xlabel(f'{demo_feat_label_dict[demo_feat]} (decile)',fontsize=25)
    plt.ylabel('Relative risks', fontsize=25)

    if(save_figure):
        plt.savefig(savepath,bbox_inches = 'tight')
        print('Figure saved. Path: ', savepath)

################################################################################
# Predicted disparities among demographic groups, average version, Age

temp_agnostic = np.zeros(9); temp_aware = np.zeros(9)
avg_agnostic = np.zeros(NUM_GROUPS); avg_aware = np.zeros(NUM_GROUPS)
upper_agnostic = np.zeros(NUM_GROUPS); lower_agnostic = np.zeros(NUM_GROUPS)
upper_aware = np.zeros(NUM_GROUPS); lower_aware = np.zeros(NUM_GROUPS)

result_dict = dict()
final_deaths_rate_age_agnostic = np.zeros(NUM_GROUPS)
final_deaths_rate_no_vaccination = np.zeros(NUM_GROUPS)

# Load SafeGraph data to obtain CBG sizes (i.e., populations)
filepath = os.path.join(args.safegraph_root,"safegraph_open_census_data/data/cbg_b01.csv")
cbg_agesex = pd.read_csv(filepath)

count=0
for msa_idx in range(len(constants.MSA_NAME_LIST)):
    if(constants.MSA_NAME_LIST[msa_idx]=='NewYorkCity'):continue
    MSA_NAME = constants.MSA_NAME_LIST[msa_idx]
    MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME]
    
    # Load CBG ids for the MSA
    cbg_ids_msa = pd.read_csv(os.path.join(dataroot,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
    cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
    M = len(cbg_ids_msa)
    
    # Extract CBGs belonging to the MSA 
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
    # Deal with NaN values
    cbg_age_msa.fillna(0,inplace=True)

    # Grouping: 按NUM_GROUPS分位数，将全体CBG分为NUM_GROUPS个组，将分割点存储在separators中
    separators = functions.get_separators(cbg_age_msa, NUM_GROUPS, 'Elder_Ratio','Sum', normalized=True)
    cbg_age_msa['Age_Quantile'] =  cbg_age_msa['Elder_Ratio'].apply(lambda x : functions.assign_group(x, separators))

    # No_Vaccination & Age_Agnostic, accumulated results
    deaths_cbg_no_vaccination = np.load(os.path.join(resultroot,r'20210206_deaths_cbg_no_vaccination_%s.npy'%MSA_NAME))
    deaths_cbg_age_agnostic = np.load(os.path.join(resultroot,r'20210206_deaths_cbg_age_agnostic_%s.npy'%MSA_NAME))

    # Add simulation results to grouping table
    cbg_age_msa['Final_Deaths_No_Vaccination'] = deaths_cbg_no_vaccination[-1,:]
    cbg_age_msa['Final_Deaths_Age_Agnostic'] = deaths_cbg_age_agnostic[-1,:]
    # Check whether there is NaN in cbg_tables
    if(cbg_age_msa.isnull().any().any()):
        print('NaN exists in cbg_age_msa. Please check.')
        pdb.set_trace()
    
    results = get_msa_result(cbg_table=cbg_age_msa, demo_feat='Age', num_groups=NUM_GROUPS) #20220227
    count += 1
    result_dict[MSA_NAME] = copy.deepcopy(results)

return_values = get_avg_upper_lower(result_dict, num_groups=NUM_GROUPS) #20220227
avg_agnostic, avg_aware, upper_agnostic, upper_aware, lower_agnostic, lower_aware, error_agnostic, error_aware = return_values #20220227

#savepath = os.path.join(saveroot, 'groupwise_death_rate_age.png')
savepath = os.path.join(fig_save_root, 'fig1d_age_withlegend.pdf') #20220309
plot_groupwise_death_rate(demo_feat='Elder_Ratio', num_groups=NUM_GROUPS, alpha=alpha, markersize=markersize, save_figure=True, savepath=savepath, show_legend=True)
savepath = os.path.join(fig_save_root, 'fig1d_age.pdf') #20220309
plot_groupwise_death_rate(demo_feat='Elder_Ratio', num_groups=NUM_GROUPS, alpha=alpha, markersize=markersize, save_figure=True, savepath=savepath, show_legend=False)

################################################################################
# Predicted disparities among demographic groups, average version, Income

if('Income' in demo_feat_list):
    result_dict = dict()
    final_deaths_rate_age_agnostic = np.zeros(NUM_GROUPS)
    final_deaths_rate_no_vaccination = np.zeros(NUM_GROUPS)

    # Load ACS 5-year (2013-2017) Data: Mean Household Income
    filepath = os.path.join(dataroot,"ACS_5years_Income_Filtered_Summary.csv")
    cbg_income = pd.read_csv(filepath)
    # Drop duplicate column 'Unnamed:0'
    cbg_income.drop(['Unnamed: 0'],axis=1, inplace=True)

    count=0
    for msa_idx in range(len(constants.MSA_NAME_LIST)):
        if(constants.MSA_NAME_LIST[msa_idx]=='NewYorkCity'):continue
        MSA_NAME = constants.MSA_NAME_LIST[msa_idx]
        MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME]
        
        # Load CBG ids and sizes (population) for the MSA
        cbg_ids_msa, cbg_sizes = get_cbg_ids_sizes(MSA_NAME) #20220227

        # Extract pois corresponding to the metro area (Philadelphia), by merging dataframes
        cbg_income_msa = pd.merge(cbg_ids_msa, cbg_income, on='census_block_group', how='left')
        # Add information of cbg populations, from cbg_age_Phi(cbg_b01.csv)
        cbg_income_msa['Sum'] = cbg_sizes.copy()
        # Rename
        cbg_income_msa.rename(columns = {'total_households':'Total_Households',
                                        'mean_household_income':'Mean_Household_Income'},inplace=True)

        # Grouping: 按NUM_GROUPS分位数，将全体CBG分为NUM_GROUPS个组，将分割点存储在separators中
        separators = functions.get_separators(cbg_income_msa, NUM_GROUPS, 'Mean_Household_Income','Sum', normalized=False)
        cbg_income_msa['Mean_Household_Income_Quantile'] =  cbg_income_msa['Mean_Household_Income'].apply(lambda x : functions.assign_group(x, separators))

        # No_Vaccination & Age_Agnostic, accumulated results
        deaths_cbg_no_vaccination = np.load(os.path.join(resultroot,r'20210206_deaths_cbg_no_vaccination_%s.npy'%MSA_NAME))
        deaths_cbg_age_agnostic = np.load(os.path.join(resultroot,r'20210206_deaths_cbg_age_agnostic_%s.npy'%MSA_NAME))

        # Add simulation results to grouping table
        cbg_income_msa['Final_Deaths_No_Vaccination'] = deaths_cbg_no_vaccination[-1,:]
        cbg_income_msa['Final_Deaths_Age_Agnostic'] = deaths_cbg_age_agnostic[-1,:]
        # Deal with NaN values
        cbg_income_msa.fillna(0,inplace=True)
        # Check whether there is NaN in cbg_tables
        if(cbg_income_msa.isnull().any().any()):
            print('NaN exists in cbg_income_msa. Please check.')
            pdb.set_trace()
        
        results = get_msa_result(cbg_table=cbg_income_msa, demo_feat='Mean_Household_Income', num_groups=NUM_GROUPS) #20220227
        count += 1
        result_dict[MSA_NAME] = copy.deepcopy(results)

    return_values = get_avg_upper_lower(result_dict, num_groups=NUM_GROUPS) #20220227
    avg_agnostic, avg_aware, upper_agnostic, upper_aware, lower_agnostic, lower_aware, error_agnostic, error_aware = return_values #20220227
    #savepath = os.path.join(saveroot, 'groupwise_death_rate_income.png')
    savepath = os.path.join(fig_save_root, 'fig1d_income.pdf')
    plot_groupwise_death_rate(demo_feat='Mean_Household_Income', num_groups=NUM_GROUPS, alpha=alpha, markersize=markersize, save_figure=True, savepath=savepath, show_legend=False)


################################################################################
# Predicted disparities among demographic groups, average version, Occupation

if('Occupation' in demo_feat_list):
    # cbg_c24.csv: Occupation
    filepath = os.path.join(args.safegraph_root,"safegraph_open_census_data/data/cbg_c24.csv")
    cbg_occupation = pd.read_csv(filepath)

    result_dict = dict()
    data_all = pd.DataFrame(columns=['Final_Deaths_No_Vaccination','Final_Deaths_Age_Agnostic',
                                    'Essential_Worker_Quantile','Essential_Worker_Ratio', 
                                    'Death_Rate_Age_Agnostic', 'Death_Rate_Age_Aware',
                                    'Sum','Valid'])
    count=0
    cbg_num = 0
    for msa_idx in range(len(constants.MSA_NAME_LIST)):
        if(constants.MSA_NAME_LIST[msa_idx]=='NewYorkCity'):continue
        MSA_NAME = constants.MSA_NAME_LIST[msa_idx]
        MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME]
        print(MSA_NAME)
        
        final_deaths_rate_age_agnostic = np.zeros(NUM_GROUPS)
        final_deaths_rate_no_vaccination = np.zeros(NUM_GROUPS)
        
        # Load CBG ids and sizes (population) for the MSA
        cbg_ids_msa, cbg_sizes = get_cbg_ids_sizes(MSA_NAME) #20220227

        cbg_occupation_msa = pd.merge(cbg_ids_msa, cbg_occupation, on='census_block_group', how='left')
        columns_of_essential_workers = list(constants.ew_rate_dict.keys())
        for column in columns_of_essential_workers:
            cbg_occupation_msa[column] = cbg_occupation_msa[column].apply(lambda x : x*constants.ew_rate_dict[column])
        cbg_occupation_msa['Essential_Worker_Absolute'] = cbg_occupation_msa.apply(lambda x : x[columns_of_essential_workers].sum(), axis=1)
        cbg_occupation_msa['Sum'] = cbg_sizes.copy()
        cbg_occupation_msa['Employed_Absolute'] = cbg_occupation_msa['C24030e1'] #20220227
        cbg_occupation_msa['Employed_Ratio'] = cbg_occupation_msa['Employed_Absolute'] / cbg_occupation_msa['Sum'] #20220227
        cbg_occupation_msa['Essential_Worker_Ratio'] = cbg_occupation_msa['Essential_Worker_Absolute'] / cbg_occupation_msa['Sum']
        columns_of_interest = ['census_block_group','Sum','Employed_Absolute','Employed_Ratio','Essential_Worker_Absolute','Essential_Worker_Ratio'] #20220227
        cbg_occupation_msa = cbg_occupation_msa[columns_of_interest].copy()
        # Deal with NaN values
        cbg_occupation_msa.fillna(0,inplace=True)

        # 先filter再分组
        #cbg_occupation_msa['Valid'] = cbg_occupation_msa.apply(lambda x : 1 if (x['Essential_Worker_Ratio']>0.2) else 0, axis=1)
        cbg_occupation_msa['Valid'] = cbg_occupation_msa.apply(lambda x : 1 if (True) else 0, axis=1) #20220227, 先不filter
        
        # Grouping: 按NUM_GROUPS分位数，将全体CBG分为NUM_GROUPS个组，将分割点存储在separators中
        separators = functions.get_separators(cbg_occupation_msa[cbg_occupation_msa['Valid']==1], NUM_GROUPS, 
                                            'Essential_Worker_Ratio','Sum', normalized=True)
        cbg_occupation_msa['Essential_Worker_Quantile'] =  cbg_occupation_msa['Essential_Worker_Ratio'].apply(lambda x : functions.assign_group(x, separators))
        cbg_occupation_msa['Essential_Worker_Quantile'] =  cbg_occupation_msa.apply(lambda x : x['Essential_Worker_Quantile'] if x['Valid'] 
                                                                                    else NUM_GROUPS, axis=1)
        
        #print(cbg_occupation_msa['Essential_Worker_Ratio'].mean(), cbg_occupation_msa['Essential_Worker_Ratio'].max())
        #print(len(cbg_occupation_msa),len(cbg_occupation_msa[cbg_occupation_msa['Essential_Worker_Quantile']==NUM_GROUPS]))
        
        # No_Vaccination & Age_Agnostic, accumulated results
        deaths_cbg_no_vaccination = np.load(os.path.join(resultroot,r'20210206_deaths_cbg_no_vaccination_%s.npy'%MSA_NAME))
        deaths_cbg_age_agnostic = np.load(os.path.join(resultroot,r'20210206_deaths_cbg_age_agnostic_%s.npy'%MSA_NAME))
        # Add simulation results to grouping table
        cbg_occupation_msa['Final_Deaths_No_Vaccination'] = deaths_cbg_no_vaccination[-1,:]
        cbg_occupation_msa['Final_Deaths_Age_Agnostic'] = deaths_cbg_age_agnostic[-1,:]
        cbg_occupation_msa['Death_Rate_Age_Aware'] = cbg_occupation_msa['Final_Deaths_No_Vaccination']/cbg_occupation_msa['Sum']
        cbg_occupation_msa['Death_Rate_Age_Agnostic'] = cbg_occupation_msa['Final_Deaths_Age_Agnostic']/cbg_occupation_msa['Sum']
        
        # Check whether there is NaN in cbg_tables
        if(cbg_occupation_msa.isnull().any().any()):
            print('NaN exists in cbg_occupation_msa. Please check.')
            pdb.set_trace()
        data_all = data_all.append(cbg_occupation_msa)
        #print('len(data_all):', len(data_all))
        
        results = get_msa_result(cbg_table=cbg_occupation_msa, demo_feat='Essential_Worker', num_groups=NUM_GROUPS) #20220227
        count += 1
        result_dict[MSA_NAME] = copy.deepcopy(results)
        cbg_num += len(cbg_occupation_msa[cbg_occupation_msa['Valid']==1])
        #print(cbg_num)

    return_values = get_avg_upper_lower(result_dict, num_groups=NUM_GROUPS) #20220227
    avg_agnostic, avg_aware, upper_agnostic, upper_aware, lower_agnostic, lower_aware, error_agnostic, error_aware = return_values #20220227
    #savepath = os.path.join(saveroot, 'groupwise_death_rate_occupation.png')
    savepath = os.path.join(fig_save_root, 'fig1d_occupation.pdf')
    plot_groupwise_death_rate(demo_feat='Essential_Worker_Ratio', num_groups=NUM_GROUPS, alpha=alpha, markersize=markersize, save_figure=True, savepath=savepath, show_legend=False)


################################################################################
# Predicted disparities among demographic groups, average version, Minority

if('Race' in demo_feat_list):
    # cbg_b02.csv: Race #20220225
    filepath = os.path.join(args.safegraph_root,"safegraph_open_census_data/data/cbg_b02.csv")
    cbg_race = pd.read_csv(filepath)
    # cbg_b03.csv: Ethnic #20220226
    filepath = os.path.join(args.safegraph_root,"safegraph_open_census_data/data/cbg_b03.csv")
    cbg_ethnic = pd.read_csv(filepath)

    result_dict = dict()
    final_deaths_rate_age_agnostic = np.zeros(NUM_GROUPS)
    final_deaths_rate_no_vaccination = np.zeros(NUM_GROUPS)

    count = 0
    for msa_idx in range(len(constants.MSA_NAME_LIST)):
        if(constants.MSA_NAME_LIST[msa_idx]=='NewYorkCity'):continue
        MSA_NAME = constants.MSA_NAME_LIST[msa_idx]
        MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME]
        print(MSA_NAME)
        
        # Load CBG ids and sizes (population) for the MSA
        cbg_ids_msa, cbg_sizes = get_cbg_ids_sizes(MSA_NAME) #20220227

        # Extract cbgs corresponding to the metro area, by merging dataframes
        cbg_race_msa = pd.merge(cbg_ids_msa, cbg_race, on='census_block_group', how='left')
        # Add information of cbg populations, from cbg_age_Phi(cbg_b01.csv)
        cbg_race_msa['Sum'] = cbg_sizes.copy()
        # Rename
        cbg_race_msa.rename(columns={'B02001e2':'White_Absolute'},inplace=True)
        # Extract columns of interest
        columns_of_interest = ['census_block_group', 'Sum', 'White_Absolute']
        cbg_race_msa = cbg_race_msa[columns_of_interest].copy()

    
        # Extract cbgs corresponding to the metro area, by merging dataframes
        cbg_ethnic_msa = pd.merge(cbg_ids_msa, cbg_ethnic, on='census_block_group', how='left')
        # Rename
        cbg_ethnic_msa.rename(columns={'B03002e13':'Hispanic_White_Absolute'},inplace=True)
        
        cbg_race_msa['Minority_Absolute'] = cbg_race_msa['Sum'] - (cbg_race_msa['White_Absolute'] - cbg_ethnic_msa['Hispanic_White_Absolute'])
        cbg_race_msa['Minority_Ratio'] = cbg_race_msa['Minority_Absolute'] / cbg_race_msa['Sum']

        # Grouping: 按NUM_GROUPS分位数，将全体CBG分为NUM_GROUPS个组，将分割点存储在separators中
        separators = functions.get_separators(cbg_race_msa, NUM_GROUPS, 'Minority_Ratio','Sum', normalized=False)
        cbg_race_msa['Minority_Ratio_Quantile'] =  cbg_race_msa['Minority_Ratio'].apply(lambda x : functions.assign_group(x, separators))
        print(cbg_race_msa[cbg_race_msa['Minority_Ratio']==0]['Sum'].sum())
        
        ##################### below are tests #####################
        # Examine the average elder_ratio in each ethnic decile
        '''
        # Extract CBGs belonging to the MSA
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
        # Deal with NaN values
        cbg_age_msa.fillna(0,inplace=True)

        cbg_race_msa['Elder_Ratio'] = cbg_age_msa['Elder_Ratio'].copy()

        for i in range(NUM_GROUPS):
            print(np.round(cbg_race_msa[cbg_race_msa['Minority_Ratio_Quantile']==i]['Elder_Ratio'].mean(), 4)) 
        pdb.set_trace()
        '''
        ##################### above are tests #####################

        # No_Vaccination & Age_Agnostic, accumulated results
        deaths_cbg_no_vaccination = np.load(os.path.join(resultroot,r'20210206_deaths_cbg_no_vaccination_%s.npy'%MSA_NAME))
        deaths_cbg_age_agnostic = np.load(os.path.join(resultroot,r'20210206_deaths_cbg_age_agnostic_%s.npy'%MSA_NAME))

        # Add simulation results to grouping table
        cbg_race_msa['Final_Deaths_No_Vaccination'] = deaths_cbg_no_vaccination[-1,:]
        cbg_race_msa['Final_Deaths_Age_Agnostic'] = deaths_cbg_age_agnostic[-1,:]
        # Deal with NaN values
        cbg_race_msa.fillna(0,inplace=True)
        # Check whether there is NaN in cbg_tables
        if(cbg_race_msa.isnull().any().any()):
            print('NaN exists in cbg_race_msa. Please check.')
            pdb.set_trace()
        
        results = get_msa_result(cbg_table=cbg_race_msa, demo_feat='Minority_Ratio', num_groups=NUM_GROUPS) #20220227
        count += 1
        result_dict[MSA_NAME] = copy.deepcopy(results)

    return_values = get_avg_upper_lower(result_dict, num_groups=NUM_GROUPS) #20220227
    avg_agnostic, avg_aware, upper_agnostic, upper_aware, lower_agnostic, lower_aware, error_agnostic, error_aware = return_values #20220227
    #savepath = os.path.join(saveroot, 'groupwise_death_rate_minority.png')
    savepath = os.path.join(fig_save_root, 'fig1d_minority.pdf')
    plot_groupwise_death_rate(demo_feat='Minority_Ratio', num_groups=NUM_GROUPS, alpha=alpha, markersize=markersize, save_figure=True, savepath=savepath, show_legend=False)

