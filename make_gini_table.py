# python make_gini_table.py --msa_name SanFrancisco --vaccination_time 31 --vaccination_ratio 0.1 --rel_to Baseline

# pylint: disable=invalid-name,trailing-whitespace,superfluous-parens,line-too-long

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import socket
import os
import pandas as pd
import numpy as np
import argparse

import constants
import functions

import pdb

# root
'''
hostname = socket.gethostname()
print('hostname: ', hostname)
if(hostname=='fib-dl3'):
    root = '/data/chenlin/COVID-19/Data' #dl3
    saveroot = '/data/chenlin/utility-equity-covid-vac/results'
elif(hostname=='rl4'):
    root = '/home/chenlin/COVID-19/Data' #rl4
    saveroot = '/home/chenlin/utility-equity-covid-vac/results'
'''
root = os.getcwd()
dataroot = os.path.join(root, 'data')
saveroot = os.path.join(root, 'results')

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--msa_name', 
                    help='MSA name. If \'all\', then iterate over all MSAs.')
parser.add_argument('--vaccination_time', type=int, default=31,
                    help='Time to distribute vaccines.')
parser.add_argument('--vaccination_ratio' , type=float, default=0.1,
                    help='Vaccination ratio relative to MSA population.')
parser.add_argument('--num_seeds', type=int, default=30,
                    help='Num of seeds. Used to identify which files to load.')
parser.add_argument('--num_groups', type=int, default=5,
                    help='Num of groups to divide CBGs into.') 
#parser.add_argument('--acceptance_scenario', 
#                    help='Scenario of vaccine hesitancy (fully/real/cf18/cf13/cf17/ALL). Only useful when consider_hesitancy is True.')
parser.add_argument('--recheck_interval', type=float, default = 0.01,
                    help='Recheck interval (After distributing some portion of vaccines, recheck the most vulnerable demographic group).')                             
parser.add_argument('--rel_to', 
                    help='Relative to which strategy (either No_Vaccination or Baseline).')
parser.add_argument('--safegraph_root', default=dataroot, #'/data/chenlin/COVID-19/Data',
                    help='Safegraph data root.') 
args = parser.parse_args()
print('args.msa_name:',args.msa_name)
print('Vaccination Time:',str(args.vaccination_time))
print('Vaccination ratio: ', args.vaccination_ratio)
print('Number of groups: ',args.num_groups)
#print('Vaccine acceptance scenario: ', args.acceptance_scenario)
print('Relative to: ', args.rel_to)



# Derived variables
if(args.msa_name=='all'):
    msa_name_list = ['Atlanta', 'Chicago', 'Dallas', 'Houston', 'LosAngeles', 'Miami', 'Philadelphia', 'SanFrancisco', 'WashingtonDC']
else:
    msa_name_list = [args.msa_name]

demo_policy_list = ['Age', 'Age_Reverse', 'Income', 'Income_Reverse', 'Occupation', 'Occupation_Reverse','Minority', 'Minority_Reverse'] 
print('Single_demo_feat based policy list: ', demo_policy_list)
demo_feat_list = ['Elder_Ratio', 'Mean_Household_Income', 'EW_Ratio', 'Minority_Ratio'] #20220302
print('Demographic feature list: ', demo_feat_list)

if(args.rel_to=='No_Vaccination'):
    policy_list = ['No_Vaccination','Baseline'] + demo_policy_list
elif(args.rel_to=='Baseline'):
    policy_list = ['Baseline','No_Vaccination'] + demo_policy_list
else:
    print('Invalid args.rel_to.')
print('policy list:', policy_list)    


############################################################
# Functions

# Analyze results and produce graphs: All policies
def output_result(cbg_table, demo_feat, policy_list, num_groups, rel_to):
    results = {}
    
    for policy in policy_list:
        exec("final_deaths_rate_%s_total = cbg_table['Final_Deaths_%s'].sum()/cbg_table['Sum'].sum()" % (policy.lower(),policy))
        cbg_table['Final_Deaths_' + policy] = eval('avg_final_deaths_' + policy.lower())
        exec("%s = np.zeros(num_groups)" % ('final_deaths_rate_'+ policy.lower()))
        deaths_total_abs = eval('final_deaths_rate_%s_total'%(policy.lower()))
        #deaths_total_abs = np.round(deaths_total_abs,6) #20220305注释
        
        for i in range(num_groups):
            eval('final_deaths_rate_'+ policy.lower())[i] = cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Final_Deaths_' + policy].sum()
            eval('final_deaths_rate_'+ policy.lower())[i] /= cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Sum'].sum()
        deaths_gini_abs = functions.gini(eval('final_deaths_rate_'+ policy.lower()))
        #deaths_gini_abs = np.round(deaths_gini_abs,6) #20220305注释
       
        if(rel_to=='No_Vaccination'): # compared to No_Vaccination
            if(policy=='No_Vaccination'):
                deaths_total_no_vaccination = deaths_total_abs
                deaths_gini_no_vaccination = deaths_gini_abs
                deaths_total_rel = 0
                deaths_gini_rel = 0
                results[policy] = {'deaths_total_abs':'%.6f'% deaths_total_abs, 
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.6f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.6f'% deaths_gini_rel}   
            else:
                #deaths_total_rel = (np.round(eval('final_deaths_rate_%s_total'%(policy.lower())),6) - deaths_total_no_vaccination) / deaths_total_no_vaccination
                #deaths_gini_rel = (np.round(functions.gini(eval('final_deaths_rate_'+ policy.lower())),6) - deaths_gini_no_vaccination) / deaths_gini_no_vaccination
                deaths_total_rel = (eval('final_deaths_rate_%s_total'%(policy.lower())) - deaths_total_no_vaccination) / deaths_total_no_vaccination #20220305
                deaths_gini_rel = (functions.gini(eval('final_deaths_rate_'+ policy.lower())) - deaths_gini_no_vaccination) / deaths_gini_no_vaccination #20220305
                results[policy] = {'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.6f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.6f'% deaths_gini_rel}    
        
        elif(rel_to=='Baseline'): # compared to Baseline
            if(policy=='Baseline'):
                deaths_total_baseline = deaths_total_abs
                deaths_gini_baseline = deaths_gini_abs
                deaths_total_rel = 0
                deaths_gini_rel = 0
                results[policy] = {'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.6f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.6f'% deaths_gini_rel}   
            else:
                #deaths_total_rel = (np.round(eval('final_deaths_rate_%s_total'%(policy.lower())),6) - deaths_total_baseline) / deaths_total_baseline
                #deaths_gini_rel = (np.round(functions.gini(eval('final_deaths_rate_'+ policy.lower())),6) - deaths_gini_baseline) / deaths_gini_baseline
                deaths_total_rel = (eval('final_deaths_rate_%s_total'%(policy.lower())) - deaths_total_baseline) / deaths_total_baseline #20220305
                deaths_gini_rel = (functions.gini(eval('final_deaths_rate_'+ policy.lower())) - deaths_gini_baseline) / deaths_gini_baseline #20220305
                #pdb.set_trace()
                results[policy] = {'deaths_total_abs':'%.6f'% deaths_total_abs,
                                   'deaths_total_rel':'%.6f'% deaths_total_rel,
                                   'deaths_gini_abs':'%.6f'% deaths_gini_abs,
                                   'deaths_gini_rel':'%.6f'% deaths_gini_rel}                        
    return results


def make_gini_table(policy_list, demo_feat_list, save_result=False, save_path=None):
    cbg_table_name_dict=dict()
    cbg_table_name_dict['Elder_Ratio'] = cbg_age_msa
    cbg_table_name_dict['Mean_Household_Income'] = cbg_income_msa
    cbg_table_name_dict['EW_Ratio'] = cbg_occupation_msa
    cbg_table_name_dict['Minority_Ratio'] = cbg_ethnic_msa #20220302
    
    print('Policy list: ', policy_list)
    print('Demographic feature list: ', demo_feat_list)

    gini_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('All','deaths_total_abs'),('All','deaths_total_rel')]))
    gini_df['Policy'] = policy_list
        
    for demo_feat in demo_feat_list:
        results = output_result(cbg_table_name_dict[demo_feat], 
                                demo_feat, policy_list, num_groups=args.num_groups,
                                rel_to=args.rel_to)
       
        for i in range(len(policy_list)):
            policy = policy_list[i]
            gini_df.loc[i,('All','deaths_total_abs')] = results[policy]['deaths_total_abs']
            gini_df.loc[i,('All','deaths_total_rel')] = results[policy]['deaths_total_rel'] 
            gini_df.loc[i,(demo_feat,'deaths_gini_abs')] = results[policy]['deaths_gini_abs']
            gini_df.loc[i,(demo_feat,'deaths_gini_rel')] = results[policy]['deaths_gini_rel'] 

    gini_df.set_index(['Policy'],inplace=True)
    # Transpose
    gini_df_trans = pd.DataFrame(gini_df.values.T, index=gini_df.columns, columns=gini_df.index)#转置
    # Save .csv
    if(save_result==True):
        gini_df_trans.to_csv(save_path)
        
    return gini_df_trans



############################################################

# MSA list
# Load ACS Data for matching with NYT Data
acs_data = pd.read_csv(os.path.join(dataroot,'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
# Load SafeGraph data to obtain CBG sizes (i.e., populations)
filepath = os.path.join(args.safegraph_root,"safegraph_open_census_data/data/cbg_b01.csv")
cbg_agesex = pd.read_csv(filepath)
# Load ACS 5-year (2013-2017) Data: Mean Household Income
filepath = os.path.join(args.safegraph_root,"ACS_5years_Income_Filtered_Summary.csv")
cbg_income = pd.read_csv(filepath)
cbg_income.drop(['Unnamed: 0'],axis=1, inplace=True)
# cbg_c24.csv: Occupation
filepath = os.path.join(args.safegraph_root,"safegraph_open_census_data/data/cbg_c24.csv")
cbg_occupation = pd.read_csv(filepath)
# cbg_b03.csv: Ethnic #20220225
filepath = os.path.join(args.safegraph_root,"safegraph_open_census_data/data/cbg_b03.csv")
cbg_ethnic = pd.read_csv(filepath)

for this_msa in msa_name_list:
    # Extract data specific to one msa, according to ACS data
    MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[this_msa]
    msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas) #; print('\nMatching MSA_NAME_FULL to MSAs in ACS Data: ',msa_match)
    msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy() #; print('Number of counties matched: ',len(msa_data))
    msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
    good_list = list(msa_data['FIPS Code'].values);#print('Indices of counties matched: ',good_list)

    # Load CBG ids belonging to a specific metro area
    cbg_ids_msa = pd.read_csv(os.path.join(dataroot,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
    cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
    M = len(cbg_ids_msa);#print('Number of CBGs in this metro area:', M)
    # Mapping from cbg_ids to columns in hourly visiting matrices
    cbgs_to_idxs = dict(zip(cbg_ids_msa['census_block_group'].values, range(M)))
    x = {}
    for i in cbgs_to_idxs:
        x[str(i)] = cbgs_to_idxs[i]

    # Select counties belonging to the MSA
    y = []
    for i in x:
        if((len(i)==12) & (int(i[0:5])in good_list)): y.append(x[i])
        if((len(i)==11) & (int(i[0:4])in good_list)): y.append(x[i])
            
    idxs_msa_all = list(x.values());#print('Number of CBGs in this metro area:', len(idxs_msa_all))
    idxs_msa_nyt = y; #print('Number of CBGs in to compare with NYT data:', len(idxs_msa_nyt))

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

    # Obtain cbg sizes (populations)
    cbg_sizes = cbg_age_msa['Sum'].values
    cbg_sizes = np.array(cbg_sizes,dtype='int32')
    print('Total population: ',np.sum(cbg_sizes))

    # Load other Safegraph demographic data
    # Income
    cbg_income_msa = functions.load_cbg_income_msa(cbg_income, cbg_ids_msa) #20220302
    cbg_income_msa['Sum'] = cbg_age_msa['Sum'].copy()
    # Deal with NaN values
    cbg_income_msa.fillna(0,inplace=True)

    # Occupation
    cbg_occupation_msa = pd.merge(cbg_ids_msa, cbg_occupation, on='census_block_group', how='left')
    cbg_occupation_msa = functions.load_cbg_occupation_msa(cbg_occupation, cbg_ids_msa, cbg_sizes) #20220302
    cbg_occupation_msa.rename(columns={'Essential_Worker_Ratio':'EW_Ratio'},inplace=True)
    # Deal with NaN values
    cbg_occupation_msa.fillna(0,inplace=True)

    # Minority 
    cbg_ethnic_msa = pd.merge(cbg_ids_msa, cbg_ethnic, on='census_block_group', how='left')
    cbg_ethnic_msa['Sum'] = cbg_age_msa['Sum']
    # Rename
    cbg_ethnic_msa.rename(columns={'B03002e3':'NH_White'},inplace=True)
    cbg_ethnic_msa['Minority_Absolute'] = cbg_ethnic_msa['NH_White'].copy()
    cbg_ethnic_msa['Minority_Ratio'] = cbg_ethnic_msa['Minority_Absolute'] / cbg_ethnic_msa['Sum']
    # Deal with NaN values
    cbg_ethnic_msa.fillna(0,inplace=True)


    ###############################################################################
    # Grouping: 按args.num_groups分位数，将全体CBG分为args.num_groups个组，将分割点存储在separators中

    separators = functions.get_separators(cbg_age_msa, args.num_groups, 'Elder_Ratio','Sum', normalized=True)
    cbg_age_msa['Elder_Ratio_Quantile'] =  cbg_age_msa['Elder_Ratio'].apply(lambda x : functions.assign_group(x, separators))

    separators = functions.get_separators(cbg_occupation_msa, args.num_groups, 'EW_Ratio','Sum', normalized=True)
    cbg_occupation_msa['EW_Ratio_Quantile'] =  cbg_occupation_msa['EW_Ratio'].apply(lambda x : functions.assign_group(x, separators))

    separators = functions.get_separators(cbg_income_msa, args.num_groups, 'Mean_Household_Income','Sum', normalized=False)
    cbg_income_msa['Mean_Household_Income_Quantile'] =  cbg_income_msa['Mean_Household_Income'].apply(lambda x : functions.assign_group(x, separators))

    separators = functions.get_separators(cbg_ethnic_msa, args.num_groups, 'Minority_Ratio','Sum', normalized=True) #20220225
    cbg_ethnic_msa['Minority_Ratio_Quantile'] =  cbg_ethnic_msa['Minority_Ratio'].apply(lambda x : functions.assign_group(x, separators))

    ###############################################################################
    # Preprocessing: from history_D2_xxx to final_deaths_xxx #20220304

    file_name_dict = {'baseline': 'baseline',
                    'age': 'age_flood', 
                    'age_reverse': 'age_flood', 
                    'income': 'income_flood',
                    'income_reverse': 'income_flood', 
                    'occupation': 'jue_ew_flood',
                    'occupation_reverse': 'jue_ew_flood'
                    }

    for policy in policy_list:
        print(policy)
        policy = policy.lower()
        final_deaths_result_path = os.path.join(saveroot, f'vac_results_{args.vaccination_time}d_{args.vaccination_ratio}_{args.recheck_interval}', f'final_deaths_{policy}_{args.vaccination_time}d_{args.vaccination_ratio}_{args.recheck_interval}_30seeds_{this_msa}')
        if(os.path.exists(final_deaths_result_path)):
            print('Final deaths file already exists.')
            exec(f'final_deaths_{policy} = np.fromfile(\'{final_deaths_result_path}\')')
            exec(f'final_deaths_{policy} = np.reshape(final_deaths_{policy},(args.num_seeds,M))')   
        else: # Generate final_deaths_xx files
            if policy in ['minority', 'minority_reverse']:
                history_D2_result_path = os.path.join(saveroot, f'vac_results_{args.vaccination_time}d_{args.vaccination_ratio}_{args.recheck_interval}', f'history_D2_{policy}_{args.vaccination_time}d_{args.vaccination_ratio}_{args.recheck_interval}_30seeds_{this_msa}')
            elif(policy=='no_vaccination'):
                history_D2_result_path = os.path.join(saveroot,'vaccination_results_adaptive_31d_0.1_0.01',rf'20210206_history_D2_no_vaccination_adaptive_0.1_0.01_30seeds_{this_msa}')
            #elif(policy=='baseline'):
            #    history_D2_result_path = os.path.join(root,this_msa,f'vaccination_results_adaptive_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}', f'test_history_D2_{file_name_dict[policy]}_adaptive_{args.vaccination_ratio}_{args.recheck_interval}_30seeds_{this_msa}') #20220304
            else:
                #history_D2_result_path = os.path.join(root,this_msa,f'vaccination_results_adaptive_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}', f'test_history_D2_{file_name_dict[policy]}_adaptive_{args.vaccination_time}d_{args.vaccination_ratio}_{args.recheck_interval}_30seeds_{this_msa}') #20220304
                if('reverse' not in policy):
                    history_D2_result_path = os.path.join(saveroot,f'vaccination_results_adaptive_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}', rf'20210206_history_D2_{file_name_dict[policy]}_adaptive_{args.vaccination_ratio}_{args.recheck_interval}_30seeds_{this_msa}') #20220304
                else:
                    history_D2_result_path = os.path.join(saveroot,f'vaccination_results_adaptive_reverse_{str(args.vaccination_time)}d_{args.vaccination_ratio}_{args.recheck_interval}', rf'20210206_history_D2_{file_name_dict[policy]}_adaptive_reverse_{args.vaccination_ratio}_{args.recheck_interval}_30seeds_{this_msa}') #20220304
            exec(f'history_D2_{policy} = np.fromfile(\'{history_D2_result_path}\')')
            exec(f'history_D2_{policy} = np.reshape(history_D2_{policy},(63,args.num_seeds,M))')   
            exec(f'final_deaths_{policy} = np.array(history_D2_{policy}[-1,:,:])')
            exec(f'final_deaths_{policy}.tofile(final_deaths_result_path)')
            print(f'Final deaths file saved? {os.path.exists(final_deaths_result_path)}')
        exec(f'avg_final_deaths_{policy} = final_deaths_{policy}.mean(axis=0)')

    ###############################################################################
    # Add simulation results to grouping tables

    for policy in policy_list:
        exec("cbg_age_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
        exec("cbg_income_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
        exec("cbg_occupation_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower()))
        exec("cbg_ethnic_msa['Final_Deaths_%s'] = avg_final_deaths_%s" % (policy,policy.lower())) #20220225

    # Check whether there is NaN in cbg_tables
    print('Any NaN in cbg_age_msa?', cbg_age_msa.isnull().any().any())
    print('Any NaN in cbg_income_msa?', cbg_income_msa.isnull().any().any())
    print('Any NaN in cbg_occupation_msa?', cbg_occupation_msa.isnull().any().any())
    print('Any NaN in cbg_ethnic_msa?', cbg_ethnic_msa.isnull().any().any()) #20220225

    save_path = os.path.join(saveroot, f'gini_table/gini_table_{str(args.vaccination_time)}_{args.recheck_interval}_{args.num_groups}_{this_msa}_rel2{args.rel_to}.csv')
    print('save_path: ',save_path)

    gini_df = make_gini_table(policy_list, demo_feat_list, save_result=True, save_path=save_path)

    print(gini_df['Minority'])