#!/usr/bin/env python
# coding: utf-8

# python calculate_age_specific_death_risk.py MSA_NAME
# python calculate_age_specific_death_risk.py Atlanta

import os
import sys
import datetime
import pandas as pd
import numpy as np
import pickle

import constants



#root = 'F:\\deeplearning\\Jupyter_project\\Data\\COVID-19'
#root = 'G:\\COVID19 Backup (20201219)\Data\COVID-19'
root = '/data/chenlin/COVID-19/Data'


MSA_NAME = sys.argv[1]; print('MSA_NAME: ',MSA_NAME) #MSA_NAME = 'SanFrancisco'
MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME] #MSA_NAME_FULL = 'San_Francisco_Oakland_Hayward_CA'

# # Load CBG ids belonging to a specific metro area
cbg_ids_msa = pd.read_csv(os.path.join(root, MSA_NAME, 'Atlanta_Sandy_Springs_Roswell_GA_cbg_ids.csv'))

cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
M = len(cbg_ids_msa)

# Mapping from cbg_ids to columns in hourly visiting matrices
cbgs_to_idxs = dict(zip(cbg_ids_msa['census_block_group'].values, range(M)))

x = {}
for i in cbgs_to_idxs:
    x[str(i)] = cbgs_to_idxs[i]
print('Number of CBGs in this metro area:', M)


# # Load SafeGraph Open Census Data
# ## cbg_b01.csv
filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_b01.csv")
cbg_agesex = pd.read_csv(filepath)
print("Dataframe Shape: ", cbg_agesex.shape)


# Extract CBGs belonging to the metro area
# https://covid-mobility.stanford.edu//datasets/
cbg_agesex_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
print("Dataframe Shape: ", cbg_agesex_msa.shape)

print(M)  # M: 该metro area中包含的CBG个数
print(cbg_agesex.shape)
print(cbg_agesex_msa.shape)

cbg_age_msa = cbg_agesex_msa.copy()
print(len(cbg_age_msa))

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
print("Dataframe Shape: ", cbg_age_msa.shape)



# Deal with CBGs with 0 populations, if any
print(cbg_age_msa[cbg_age_msa['Sum']==0]['census_block_group'].values)
#cbg_age_msa = cbg_age_msa[cbg_age_msa['Sum']!=0].copy()
cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
print(cbg_age_msa.shape)


# # Age-aware CBG-specific Death Rates
# Age-specific death rates
# Taking from literature
age_death_rates = np.array([0.003, 0.001, 0.001, 0.003, 0.006, 0.013, 0.024, 0.040, 0.075, 0.121, 0.207, 0.323, 0.456, 1.075, 1.674, 3.203, 8.292])/100

# Population within age-sex groups of CBGs
# Further grouping, according to AGE_GROUPS_FOR_DEATH_RATES,
# which is defined in constants.py
num_age_groups_for_death = len(constants.AGE_GROUPS_FOR_DEATH_RATES) # 17

for i in range(num_age_groups_for_death): # 0~16
    cbg_age_msa['Death_Group'+str(i)+'_Absolute'] = cbg_age_msa.apply(lambda x : x[constants.AGE_GROUPS_FOR_DEATH_RATES[i]].sum(),axis=1)
    cbg_age_msa['Death_Group'+str(i)+'_Ratio'] = cbg_age_msa.apply(lambda x : x['Death_Group'+str(i)+'_Absolute']/x['Sum'],axis=1)

columns_of_interest = ['Death_Group'+str(i)+'_Ratio' for i in range(num_age_groups_for_death)]
cbg_age_msa['Death_Rate_Original'] = cbg_age_msa.apply(lambda x : np.sum(x[columns_of_interest] * age_death_rates), axis=1)

cbg_death_rates_original = cbg_age_msa['Death_Rate_Original'].copy().values
print('cbg_death_rates_original: ',cbg_death_rates_original.shape,cbg_death_rates_original)

print('Dataframe shape: ', cbg_age_msa.shape)
print(np.isnan(cbg_death_rates_original).any())


# Save/Load age-aware CBG-specific death rates, 
np.savetxt(os.path.join(root, MSA_NAME, 'cbg_death_rates_original_'+MSA_NAME),cbg_death_rates_original)
# test loading
cbg_death_rates_original = np.loadtxt(os.path.join(root, MSA_NAME, 'cbg_death_rates_original_'+MSA_NAME))
print('Age-aware CBG-specific death rates (original): ',cbg_death_rates_original.shape,cbg_death_rates_original)
