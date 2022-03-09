# regression_feat_death.py -msa_name SanFrancisco

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt

import functions
import constants

import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--msa_name', 
                    help='MSA name.')
parser.add_argument('--root', default='/data/chenlin/COVID-19/Data',
                    help='Root to retrieve data.')
parser.add_argument('--num_groups', type=int, default=10,
                    help='Num of groups to divide CBGs into.')                   
parser.add_argument('--saveroot', default='/data/chenlin/utility-equity-covid-vac/results',
                    help='Root to save generated figures.')
args = parser.parse_args()
print(f'msa_name: {args.msa_name}')

MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[args.msa_name]

# Load CBG ids for the MSA
cbg_ids_msa = pd.read_csv(os.path.join(args.root,args.msa_name,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
M = len(cbg_ids_msa)

# Data for all MSAs
# Load cbg_agesex
filepath = os.path.join(args.root,"safegraph_open_census_data/data/cbg_b01.csv")
cbg_agesex = pd.read_csv(filepath)
# Load cbg_income
filepath = os.path.join(args.root,"ACS_5years_Income_Filtered_Summary.csv")
cbg_income = pd.read_csv(filepath)
cbg_income.drop(['Unnamed: 0'],axis=1, inplace=True)
# Load cbg_occupation
filepath = os.path.join(args.root,"safegraph_open_census_data/data/cbg_c24.csv")
cbg_occupation = pd.read_csv(filepath)
# Load cbg_race
filepath = os.path.join(args.root,"safegraph_open_census_data/data/cbg_b02.csv")
cbg_race = pd.read_csv(filepath)
# Load cbg_ethnic
filepath = os.path.join(args.root,"safegraph_open_census_data/data/cbg_b03.csv")
cbg_ethnic = pd.read_csv(filepath)

# Data for one MSA
# Get cbg_age_msa
cbg_age_msa = functions.load_cbg_age_msa(cbg_agesex, cbg_ids_msa)
cbg_sizes = cbg_age_msa['Sum'].copy()
# Get cbg_income_msa
cbg_income_msa = functions.load_cbg_income_msa(cbg_income, cbg_ids_msa)
# Get cbg_occupation_msa
cbg_occupation_msa = functions.load_cbg_occupation_msa(cbg_occupation, cbg_ids_msa, cbg_sizes)
# Get cbg_race_msa
cbg_race_msa = functions.load_cbg_race_msa(cbg_race, cbg_ids_msa, cbg_sizes)
# Get cbg_ethnic_msa
cbg_ethnic_msa = functions.load_cbg_ethnic_msa(cbg_ethnic, cbg_ids_msa, cbg_sizes)

# Load age-determined fatality risks 
cbg_death_rates_original = np.loadtxt(os.path.join(args.root, args.msa_name, 'cbg_death_rates_original_'+args.msa_name))
# Load simulated death rates
deaths_cbg_no_vaccination = np.load(os.path.join(args.root, args.msa_name,'20210206_deaths_cbg_no_vaccination_%s.npy'%args.msa_name))
final_deaths = deaths_cbg_no_vaccination[-1,:]


# Collect data together
data_msa = pd.DataFrame()
data_msa['Sum'] = cbg_age_msa['Sum'].copy()
data_msa['Elder_Ratio'] = cbg_age_msa['Elder_Ratio'].copy()
data_msa['Mean_Household_Income'] = cbg_income_msa['Mean_Household_Income'].copy()
data_msa['Essential_Worker_Ratio'] = cbg_occupation_msa['Essential_Worker_Ratio'].copy()
data_msa['White_Ratio'] = cbg_race_msa['White_Ratio'].copy() #20220227
data_msa['Hispanic_Ratio'] = cbg_ethnic_msa['Hispanic_Ratio'].copy() #20220226
data_msa['Death_Absolute'] = final_deaths #20220228
data_msa['Death_Ratio'] = data_msa['Death_Absolute'] / data_msa['Sum'] #20220228
data_msa['Fatality_Risk'] = cbg_death_rates_original #20220228


# Grouping: 按NUM_GROUPS分位数，将全体CBG分为NUM_GROUPS个组，将分割点存储在separators中
for column in data_msa.columns:
    separators = functions.get_separators(data_msa, args.num_groups, 'Elder_Ratio','Sum', normalized=False) #True
    data_msa['Elder_Ratio_Quantile'] =  data_msa['Elder_Ratio'].apply(lambda x : functions.assign_group(x, separators))
    separators = functions.get_separators(data_msa, args.num_groups, 'Mean_Household_Income','Sum', normalized=False)
    data_msa['Mean_Household_Income_Quantile'] =  data_msa['Mean_Household_Income'].apply(lambda x : functions.assign_group(x, separators))
    separators = functions.get_separators(data_msa, args.num_groups, 'Essential_Worker_Ratio','Sum', normalized=False) #True
    data_msa['Essential_Worker_Ratio_Quantile'] =  data_msa['Essential_Worker_Ratio'].apply(lambda x : functions.assign_group(x, separators))
    separators = functions.get_separators(data_msa, args.num_groups, 'White_Ratio','Sum', normalized=False)
    data_msa['White_Ratio_Quantile'] = data_msa['White_Ratio'].apply(lambda x : functions.assign_group(x, separators))
    separators = functions.get_separators(data_msa, args.num_groups, 'Hispanic_Ratio','Sum', normalized=False)
    data_msa['Hispanic_Ratio_Quantile'] = data_msa['Hispanic_Ratio'].apply(lambda x : functions.assign_group(x, separators))
    separators = functions.get_separators(data_msa, args.num_groups, 'Death_Ratio','Sum', normalized=False)
    data_msa['Death_Ratio_Quantile'] = data_msa['Death_Ratio'].apply(lambda x : functions.assign_group(x, separators))

fatality_risk_array = np.zeros(args.num_groups)
death_ratio_array = np.zeros(args.num_groups)
for i in range(args.num_groups):  
    avg_fatality_risk = data_msa[data_msa['Elder_Ratio_Quantile']==i]['Fatality_Risk'].mean()
    avg_death_ratio = data_msa[data_msa['Elder_Ratio_Quantile']==i]['Death_Ratio'].mean()
    fatality_risk_array[i] = avg_fatality_risk
    death_ratio_array[i] = avg_death_ratio
    print(avg_fatality_risk, avg_death_ratio)
# Normalize w.r.t. the smallest value
min_fatality_risk = np.min(fatality_risk_array)
fatality_risk_array /= min_fatality_risk
min_death_ratio = np.min(death_ratio_array)
death_ratio_array /= min_death_ratio
plt.figure(figsize=(14,6))
plt.plot(fatality_risk_array, marker='o',label='Age-determined fatality risk')
plt.plot(death_ratio_array, marker='o',label='Simulated death rate')
plt.xticks(np.arange(args.num_groups))
plt.legend()
savepath = os.path.join(args.saveroot, f'comparison_fatalityrisk_deathrate_{args.msa_name}.png')
plt.savefig(savepath,bbox_inches = 'tight')
print('Figure saved. savepath: ', savepath)

pdb.set_trace()

'''    
# Hybrid Grouping
def assign_hybrid_group(data):
    return (data['Elder_Ratio_Quantile']*9 + data['Mean_Household_Income_Quantile']*3 + data['Essential_Worker_Ratio_Quantile'])
    
data_msa['Hybrid_Group'] = data_msa.apply(lambda x : assign_hybrid_group(x), axis=1)  
print(data_msa[data_msa['Hybrid_Group']==0]['White_Ratio'].max())
print(data_msa[data_msa['Hybrid_Group']==0]['White_Ratio'].min())
pdb.set_trace()
'''   

# Preprocessing: scaling
scaler = StandardScaler() #StandardScaler()
for column in data_msa.columns:
    if(column in ['Death_Ratio','White_Ratio_Quantile', 'Hispanic_Ratio_Quantile']):
        continue
    data_msa[column] = scaler.fit_transform(np.array(data_msa[column]).reshape(-1,1))
print('Standardized.')

'''
# Regression
y = data_msa['White_Ratio_Quantile'] #'White_Ratio'
#demo_feat_list = ['Elder_Ratio', 'Mean_Household_Income', 'Essential_Worker_Ratio']
demo_feat_list = ['Elder_Ratio_Quantile', 'Mean_Household_Income_Quantile', 'Essential_Worker_Ratio_Quantile']
X = data_msa[demo_feat_list]
X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
print(results.summary())

y = data_msa['Hispanic_Ratio_Quantile'] #'Hispanic_Ratio'
#demo_feat_list = ['Elder_Ratio', 'Mean_Household_Income', 'Essential_Worker_Ratio']
demo_feat_list = ['Elder_Ratio_Quantile', 'Mean_Household_Income_Quantile', 'Essential_Worker_Ratio_Quantile']
X = data_msa[demo_feat_list]
X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
print(results.summary())
'''

y = data_msa['Death_Ratio'] #'Death_Ratio' #'Death_Absolute' #'Final_Deaths_No_Vaccination'
demo_feat_list = ['Elder_Ratio', 'Mean_Household_Income', 'Essential_Worker_Ratio']
X = data_msa[demo_feat_list]
X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
print(results.summary())

y = data_msa['Death_Ratio']
demo_feat_list = ['Elder_Ratio', 'Mean_Household_Income', 'Essential_Worker_Ratio','White_Ratio','Hispanic_Ratio']
X = data_msa[demo_feat_list]
X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
print(results.summary())

print('##########################################################################################')
y = data_msa['Death_Ratio_Quantile'] #'Death_Ratio' #'Death_Absolute' #'Final_Deaths_No_Vaccination'
demo_feat_list = ['Elder_Ratio', 'Mean_Household_Income', 'Essential_Worker_Ratio']
X = data_msa[demo_feat_list]
X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
print(results.summary())

y = data_msa['Death_Ratio_Quantile']
demo_feat_list = ['Elder_Ratio', 'Mean_Household_Income', 'Essential_Worker_Ratio','White_Ratio','Hispanic_Ratio']
X = data_msa[demo_feat_list]
X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
print(results.summary())

pdb.set_trace()

