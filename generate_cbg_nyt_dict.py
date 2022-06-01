# python generate_cbg_nyt_dict.py

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

import pdb

root = os.getcwd()
dataroot = os.path.join(root, 'data')
resultroot = os.path.join(root, 'results')

parser = argparse.ArgumentParser()
parser.add_argument('--safegraph_root', default=dataroot, #'/data/chenlin/COVID-19/Data', 
                    help='Safegraph data root.')                     
args = parser.parse_args()     


# Constants
datestring = 20210206 
MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
min_datetime = MIN_DATETIME
max_datetime = MAX_DATETIME
NUM_DAYS = 63

dict_savepath = os.path.join(dataroot, 'cbg_nyt_dict.npy')
if(not os.path.exists(dict_savepath)):
    cbg_dict = {}
    for msa_idx in range(len(constants.MSA_NAME_LIST)):
        MSA_NAME = constants.MSA_NAME_LIST[msa_idx]; print('MSA_NAME: ',MSA_NAME)
        MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_NAME]
        ###############################################################################
        # Load Data

        # Load POI-CBG visiting matrices
        #f = open(os.path.join(dataroot, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
        f = open(os.path.join(args.safegraph_root, MSA_NAME, '%s_2020-03-01_to_2020-05-02.pkl'%MSA_NAME_FULL), 'rb') 
        poi_cbg_visits_list = pickle.load(f)
        f.close()

        # Load precomputed parameters to adjust(clip) POI dwell times
        d = pd.read_csv(os.path.join(dataroot, 'parameters_%s.csv' % MSA_NAME)) 
        all_hours = functions.list_hours_in_range(min_datetime, max_datetime)
        poi_areas = d['feet'].values#面积
        poi_dwell_times = d['median'].values#平均逗留时间
        poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
        del d

        # Load ACS Data for MSA-county matching
        acs_data = pd.read_csv(os.path.join(dataroot,'list1.csv'),header=2)
        acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
        msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
        msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
        msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
        good_list = list(msa_data['FIPS Code'].values)
        print('CBG included: ', good_list)
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
        # Rename
        cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
        # Extract columns of interest
        columns_of_interest = ['census_block_group','Sum']
        cbg_age_msa = cbg_age_msa[columns_of_interest].copy()
        # Deal with CBGs with 0 populations
        cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
        M = len(cbg_age_msa)

        # Obtain cbg sizes (populations)
        cbg_sizes = cbg_age_msa['Sum'].values
        cbg_sizes = np.array(cbg_sizes,dtype='int32')
        print('Total population: ',np.sum(cbg_sizes))
        del cbg_age_msa

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

        cbg_dict[MSA_NAME] = idxs_msa_nyt

    np.save(dict_savepath, cbg_dict)

test_dict = np.load(dict_savepath, allow_pickle=True).item()
print(test_dict.keys())