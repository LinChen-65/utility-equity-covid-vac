import numpy as np
import pandas as pd
import datetime
import constants
import pdb

def list_hours_in_range(min_hour, max_hour):
    """
    Return a list of datetimes in a range from min_hour to max_hour, inclusive. Increment is one hour. 
    """
    assert(min_hour <= max_hour)
    hours = []
    while min_hour <= max_hour:
        hours.append(min_hour)
        min_hour = min_hour + datetime.timedelta(hours=1)
    return hours


def match_msa_name_to_msas_in_acs_data(msa_name, acs_msas):
    '''
    Matches the MSA name from our annotated SafeGraph data to the
    MSA name in the external datasource in MSA_COUNTY_MAPPING
    '''
    msa_pieces = msa_name.split('_')
    query_states = set()
    i = len(msa_pieces) - 1
    while True:
        piece = msa_pieces[i]
        if len(piece) == 2 and piece.upper() == piece:
            query_states.add(piece)
            i -= 1
        else:
            break
    query_cities = set(msa_pieces[:i+1])

    for msa in acs_msas:
        if ', ' in msa:
            city_string, state_string = msa.split(', ')
            states = set(state_string.split('-'))
            if states == query_states:
                cities = city_string.split('-')
                overlap = set(cities).intersection(query_cities)
                if len(overlap) > 0:  # same states and at least one city matched
                    return msa
    return None
    

def get_fips_codes_from_state_and_county_fp(state, county):
    state = str(int(state))
    county = str(int(county))
    if len(state) == 1:
        state = '0' + state
    if len(county) == 1:
        county = '00' + county
    elif len(county) == 2:
        county = '0' + county
    return int(state + county)
    

# Average history records across random seeds
def average_across_random_seeds(history_C2, history_D2, num_cbgs, cbg_idxs, print_results=False, draw_results=False):
    num_days = len(history_C2)
    
    # Average history records across random seeds
    avg_history_C2 = np.zeros((num_days,num_cbgs))
    avg_history_D2 = np.zeros((num_days,num_cbgs))
    for i in range(num_days):
        avg_history_C2[i] = np.mean(history_C2[i],axis=0)
        avg_history_D2[i] = np.mean(history_D2[i],axis=0)
    
    # Extract lines corresponding to CBGs in the metro area/county
    cases_msa = np.zeros(num_days)
    deaths_msa = np.zeros(num_days)
    for i in range(num_days):
        for j in cbg_idxs:
            cases_msa[i] += avg_history_C2[i][j]
            deaths_msa[i] += avg_history_D2[i][j]
            
    if(print_results==True):
        print('Cases: ',cases_msa)
        print('Deaths: ',deaths_msa)
    
    return avg_history_C2, avg_history_D2,cases_msa,deaths_msa
    

# Average history records across random seeds, only deaths
def average_across_random_seeds_only_death(history_D2, num_cbgs, cbg_idxs, print_results=False):
    num_days = len(history_D2)
    
    # Average history records across random seeds
    avg_history_D2 = np.zeros((num_days,num_cbgs))
    for i in range(num_days):
        avg_history_D2[i] = np.mean(history_D2[i],axis=0)
    
    # Extract lines corresponding to CBGs in the metro area/county
    deaths_msa = np.zeros(num_days)
    for i in range(num_days):
        for j in cbg_idxs:
            deaths_msa[i] += avg_history_D2[i][j]
            
    if(print_results==True):
        print('Deaths: ',deaths_msa)
    
    return avg_history_D2, deaths_msa



    
def apply_smoothing(x, agg_func=np.mean, before=3, after=3):
    new_x = []
    for i, x_point in enumerate(x):
        before_idx = max(0, i-before)
        after_idx = min(len(x), i+after+1)
        new_x.append(agg_func(x[before_idx:after_idx]))
    return np.array(new_x)
    
    
# 水位法分配疫苗
# Adjustable execution ratio
def vaccine_distribution_flood(cbg_table, vaccination_ratio, demo_feat, ascending, execution_ratio):
    cbg_table_sorted = cbg_table.copy()
    
    # Calculate number of available vaccines
    num_vaccines = cbg_table['Sum'].sum() * vaccination_ratio * execution_ratio
    print('Total num of vaccines: ',cbg_table_sorted['Sum'].sum() * vaccination_ratio)
    #print('Num of vaccines distributed according to the policy:',num_vaccines)
    
    # Rank CBGs according to some demographic feature
    cbg_table_sorted = cbg_table_sorted.sort_values(by=demo_feat, axis=0, ascending=ascending)
    
    # Distribute vaccines according to the ranking
    vaccination_vector = np.zeros(len(cbg_table))
    
    # First, find the CBG before which all CBGs can be covered
    for i in range(len(cbg_table)):
        if ((cbg_table_sorted['Sum'].iloc[:i].sum()<=num_vaccines) & (cbg_table_sorted['Sum'].iloc[:i+1].sum()>num_vaccines)):
            num_fully_covered = i
            break
    
    # For fully-covered CBGs, the num of vaccines distributed = CBG population.
    for i in range(num_fully_covered):
        vaccination_vector[i] = cbg_table_sorted['Sum'].iloc[i]
    
    # For the last CBG, it is partially covered, with vaccines left.
    already_distributed = vaccination_vector.sum()
    vaccination_vector[num_fully_covered] = num_vaccines - already_distributed
    
    # Re-order the vaccination vector by original indices
    cbg_table_sorted['Vaccination_Vector'] = vaccination_vector
    cbg_table_original = cbg_table_sorted.sort_index()
    vaccination_vector = cbg_table_original['Vaccination_Vector'].values
    
    # Vaccines left to be distributed randomly (baseline)
    num_vaccines_left = cbg_table['Sum'].sum() * vaccination_ratio - vaccination_vector.sum()
    #print('Num of vaccines left:', num_vaccines_left)
    
    # Random permutation
    random_permutation = np.arange(len(cbg_table))
    np.random.seed(42)
    np.random.shuffle(random_permutation)
    
    for i in range(len(cbg_table)):
        if(vaccination_vector[random_permutation[i]]==0):
            if(cbg_table.loc[random_permutation[i],'Sum']<=num_vaccines_left):
                vaccination_vector[random_permutation[i]] = cbg_table.loc[random_permutation[i],'Sum']
            else:
                vaccination_vector[random_permutation[i]] = num_vaccines_left
            num_vaccines_left -= vaccination_vector[random_permutation[i]]
    
    del cbg_table_sorted
    print('Final check of distributed vaccines:',vaccination_vector.sum())
    return vaccination_vector
    

def get_separators(cbg_table, num_groups, col_indicator, col_sum, normalized): #20220302
    separators = np.zeros(num_groups+1)
    total = cbg_table[col_sum].sum()
    group_size = total / num_groups
    cbg_table_work = cbg_table.copy()
    cbg_table_work.sort_values(col_indicator, inplace=True)

    separators[0] = -0.1 # to prevent making the first group [0,0] (which results in an empty group)
    separators[-1] = 1 if normalized else cbg_table[col_indicator].max()
    max_total = cbg_table_work[cbg_table_work[col_indicator]==separators[-1]][col_sum].sum()
    if(max_total>group_size):
        rest_group_size = (total - max_total) / (num_groups - 1)
        rest_num_groups = num_groups - 1
    else:
        rest_group_size = group_size
        rest_num_groups = num_groups

    last_position = 0
    for i in range(rest_num_groups):
        for j in range(last_position, len(cbg_table_work)):
            if (cbg_table_work.head(j)[col_sum].sum() <= rest_group_size*(i+1)) & (cbg_table_work.head(j+1)[col_sum].sum() >= rest_group_size*(i+1)):
                separators[i+1] = cbg_table_work.iloc[j][col_indicator]
                last_position = j
                break 
    
    return separators

    
# Assign CBGs to groups, which are defined w.r.t certain demographic features
def assign_group(x, separators,reverse=False):
    """
        reverse: 控制是否需要反向，以保证most disadvantaged is assigned the largest group number.
        是largest而不是smallest的原因是，mortality rate是一个negative index，值越大越糟糕。
        把最脆弱的组放在最右边，这样如果真的存在预想的inequality，那么斜率就是正的。
    """
    num_groups = len(separators)-1
    for i in range(num_groups):
        #if((x>=separators[i]) & (x<separators[i+1])):
        if((x>separators[i]) & (x<=separators[i+1])):
            if(reverse==False):
                return i
            else:
                return num_groups-1-i
    if(reverse==False):
        return(num_groups-1)
    else:
        return 0
              
        
# 水位法分配疫苗 # Adjustable execution ratio.
# Only 'flooding' in the most vulnerable demographic group.
# Check whether a CBG has been covered, before vaccinating it.
def vaccine_distribution_flood_new(cbg_table, vaccination_ratio, demo_feat, ascending, execution_ratio, leftover, is_last):
    cbg_table_sorted = cbg_table.copy()
    
    # Rank CBGs: (1)Put the uncovered ones (Covered=0) on the top; (2)Rank CBGs according to some demographic feature
    #cbg_table_sorted['Covered'] = cbg_table_sorted.apply(lambda x : 1 if x['Vaccination_Vector']==x['Sum'] else 0, axis=1)
    cbg_table_sorted['Covered'] = cbg_table_sorted.apply(lambda x : 1 if abs(x['Vaccination_Vector']-x['Sum'])<2 else 0, axis=1) #20220305
    
    #print('Num of covered cbgs:', len(cbg_table_sorted[cbg_table_sorted['Covered']==1]))
    # Rank CBGs according to some demographic feature
    #cbg_table_sorted.sort_values(by=['Most_Vulnerable','Covered',demo_feat],ascending=[False, True, ascending],inplace = True)
    cbg_table_sorted.sort_values(by=['Covered','Most_Vulnerable',demo_feat],ascending=[ True,False, ascending],inplace = True)#20220305
    
    # Calculate total number of available vaccines
    num_vaccines = cbg_table_sorted['Sum'].sum() * vaccination_ratio * execution_ratio + leftover
    
    # Distribute vaccines according to the ranking
    vaccination_vector = np.zeros(len(cbg_table))
    
    # First, find the CBG before which all CBGs can be covered
    for i in range(len(cbg_table)):
        if ((cbg_table_sorted['Sum'].iloc[:i].sum()<=num_vaccines) & (cbg_table_sorted['Sum'].iloc[:i+1].sum()>num_vaccines)):
            num_fully_covered = i
            break
    if(i==len(cbg_table)-1): num_fully_covered=i # 20210224
    
    # For fully-covered CBGs, the num of vaccines distributed = CBG population.
    for i in range(num_fully_covered):
        vaccination_vector[i] = cbg_table_sorted['Sum'].iloc[i]
    
    # For the last CBG, it is partially covered, with vaccines left.
    if(is_last==True):
        already_distributed = vaccination_vector.sum()
        vaccination_vector[num_fully_covered] = num_vaccines - already_distributed
    
    # Re-order the vaccination vector by original indices
    cbg_table_sorted['Vaccination_Vector'] = vaccination_vector
    cbg_table_original = cbg_table_sorted.sort_index()
    vaccination_vector = cbg_table_original['Vaccination_Vector'].values
    '''
    # if executation_ratio<1, vaccines left to be distributed randomly (baseline)
    num_vaccines_left = cbg_table['Sum'].sum() * vaccination_ratio - vaccination_vector.sum()
    #print('Num of vaccines left:', num_vaccines_left)
    
    # Random permutation
    random_permutation = np.arange(len(cbg_table))
    np.random.seed(42)
    np.random.shuffle(random_permutation)
    
    for i in range(len(cbg_table)):
        if(vaccination_vector[random_permutation[i]]==0):
            if(cbg_table.loc[random_permutation[i],'Sum']<=num_vaccines_left):
                vaccination_vector[random_permutation[i]] = cbg_table.loc[random_permutation[i],'Sum']
            else:
                vaccination_vector[random_permutation[i]] = num_vaccines_left
            num_vaccines_left -= vaccination_vector[random_permutation[i]]
    '''
    del cbg_table_sorted
    #print('Final check of distributed vaccines:',vaccination_vector.sum())
    return vaccination_vector


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) 


def assign_acceptance_absolute(income, acceptance_scenario): # vaccine_acceptance = 1 - vaccine_hesitancy
    # Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7778842/pdf/10900_2020_Article_958.pdf
    if(acceptance_scenario=='real'):
        if(income<=30000): return 0.72
        if((income>30000)&(income<=60000)): return 0.74
        if((income>60000)&(income<=99999)): return 0.81
        if(income>99999): return 0.86
    elif(acceptance_scenario=='cf1'):
        if(income<=30000): return 0.576 #0.72*0.5
        if((income>30000)&(income<=60000)): return 0.592 #0.74*0.5 
        if((income>60000)&(income<=99999)): return 0.81
        if(income>99999): return 0.86
    elif(acceptance_scenario=='cf2'):
        if(income<=30000): return 0.3
        if((income>30000)&(income<=60000)): return 0.6
        if((income>60000)&(income<=99999)): return 1
        if(income>99999): return 1
    elif(acceptance_scenario=='cf3'):
        if(income<=30000): return 0.3
        if((income>30000)&(income<=60000)): return 0.3
        if((income>60000)&(income<=99999)): return 1
        if(income>99999): return 1
    elif(acceptance_scenario=='cf4'):
        if(income<=30000): return 0.2
        if((income>30000)&(income<=60000)): return 0.2
        if((income>60000)&(income<=99999)): return 1
        if(income>99999): return 1
    elif(acceptance_scenario=='cf5'):
        if(income<=30000): return 0.1
        if((income>30000)&(income<=60000)): return 0.1
        if((income>60000)&(income<=99999)): return 1
        if(income>99999): return 1
    elif(acceptance_scenario=='cf6'):
        if(income<=30000): return 0.1
        if((income>30000)&(income<=60000)): return 0.5
        if((income>60000)&(income<=99999)): return 1
        if(income>99999): return 1
    elif(acceptance_scenario=='cf7'):
        if(income<=30000): return 0.1
        if((income>30000)&(income<=60000)): return 0.8
        if((income>60000)&(income<=99999)): return 1
        if(income>99999): return 1
    elif(acceptance_scenario=='cf8'):
        if(income<=30000): return 0
        if((income>30000)&(income<=60000)): return 0
        if((income>60000)&(income<=99999)): return 1
        if(income>99999): return 1
    else:
        print('Invalid scenario. Please check.')
        pdb.set_trace()


def assign_acceptance_quantile(quantile, acceptance_scenario=None):
    if(acceptance_scenario=='cf9'):
        if(quantile==0): return 0
        if(quantile==1): return 0
        if(quantile==2): return 0.5
        if(quantile==3): return 1
        if(quantile==4): return 1
    elif(acceptance_scenario=='cf10'):
        if(quantile==0): return 0.3
        if(quantile==1): return 0.3
        if(quantile==2): return 0.3
        if(quantile==3): return 1
        if(quantile==4): return 1
    elif(acceptance_scenario=='cf11'):
        if(quantile==0): return 0.3
        if(quantile==1): return 0.3
        if(quantile==2): return 1
        if(quantile==3): return 1
        if(quantile==4): return 1
    elif(acceptance_scenario=='cf12'):
        if(quantile==0): return 0.3
        if(quantile==1): return 1
        if(quantile==2): return 1
        if(quantile==3): return 1
        if(quantile==4): return 1    
    elif(acceptance_scenario=='cf13'):
        if(quantile==0): return 0.2
        if(quantile==1): return 0.4
        if(quantile==2): return 0.6
        if(quantile==3): return 0.8
        if(quantile==4): return 1    
    elif(acceptance_scenario=='cf14'):
        if(quantile==0): return 0.2
        if(quantile==1): return 0.2
        if(quantile==2): return 1
        if(quantile==3): return 1
        if(quantile==4): return 1
    elif(acceptance_scenario=='cf15'):
        if(quantile==0): return 0.1
        if(quantile==1): return 0.1
        if(quantile==2): return 1
        if(quantile==3): return 1
        if(quantile==4): return 1
    elif(acceptance_scenario=='cf16'):
        if(quantile==0): return 0.1
        if(quantile==1): return 1
        if(quantile==2): return 1
        if(quantile==3): return 1
        if(quantile==4): return 1
    elif(acceptance_scenario=='cf17'):
        if(quantile==0): return 0.1
        if(quantile==1): return 0.3
        if(quantile==2): return 0.5
        if(quantile==3): return 0.7
        if(quantile==4): return 1
    elif(acceptance_scenario=='cf18'):
        if(quantile==0): return 0.6
        if(quantile==1): return 0.7
        if(quantile==2): return 0.8
        if(quantile==3): return 0.9
        if(quantile==4): return 1
    else:
        print('Invalid scenario. Please check.')
        pdb.set_trace()


def load_cbg_age_msa(cbg_agesex, cbg_ids_msa): #20220228
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
    # Deal with NaN values
    cbg_age_msa.fillna(0,inplace=True)

    return cbg_age_msa


def load_cbg_income_msa(cbg_income, cbg_ids_msa): #20220228
    # Extract cbgs corresponding to the metro area, by merging dataframes
    cbg_income_msa = pd.merge(cbg_ids_msa, cbg_income, on='census_block_group', how='left')
    # Rename
    cbg_income_msa.rename(columns = {'total_households':'Total_Households',
                                     'mean_household_income':'Mean_Household_Income'},inplace=True)
    # Deal with NaN values
    cbg_income_msa.fillna(0,inplace=True)
    
    return cbg_income_msa


def load_cbg_occupation_msa(cbg_occupation, cbg_ids_msa, cbg_sizes): #20220228
    cbg_occupation_msa = pd.merge(cbg_ids_msa, cbg_occupation, on='census_block_group', how='left')
    cbg_occupation_msa['Sum'] = cbg_sizes.copy()
    columns_of_essential_workers = list(constants.ew_rate_dict.keys())
    for column in columns_of_essential_workers:
        cbg_occupation_msa[column] = cbg_occupation_msa[column].apply(lambda x : x*constants.ew_rate_dict[column])
    cbg_occupation_msa['Essential_Worker_Absolute'] = cbg_occupation_msa.apply(lambda x : x[columns_of_essential_workers].sum(), axis=1)
    cbg_occupation_msa['Employed_Absolute'] = cbg_occupation_msa['C24030e1'] #20220227
    cbg_occupation_msa['Employed_Ratio'] = cbg_occupation_msa['Employed_Absolute'] / cbg_occupation_msa['Sum'] #20220227
    cbg_occupation_msa['Essential_Worker_Ratio'] = cbg_occupation_msa['Essential_Worker_Absolute'] / cbg_occupation_msa['Sum']
    columns_of_interest = ['census_block_group','Sum','Employed_Absolute','Employed_Ratio','Essential_Worker_Absolute','Essential_Worker_Ratio'] #20220227
    cbg_occupation_msa = cbg_occupation_msa[columns_of_interest].copy()
    # Deal with NaN values
    cbg_occupation_msa.fillna(0,inplace=True)

    return cbg_occupation_msa

'''
def load_cbg_race_msa(cbg_race, cbg_ids_msa, cbg_sizes): #20220228 #20220306注释
    # Extract cbgs corresponding to the metro area, by merging dataframes
    cbg_race_msa = pd.merge(cbg_ids_msa, cbg_race, on='census_block_group', how='left')
    cbg_race_msa['Sum'] = cbg_sizes.copy()
    # Rename
    cbg_race_msa.rename(columns={'B02001e2':'White_Absolute'},inplace=True)
    cbg_race_msa.rename(columns={'B02001e3':'Black_Absolute'},inplace=True)
    cbg_race_msa['White_Ratio'] = cbg_race_msa['White_Absolute'] / cbg_race_msa['Sum']
    cbg_race_msa['Black_Ratio'] = cbg_race_msa['Black_Absolute'] / cbg_race_msa['Sum']
    # Extract columns of interest
    columns_of_interest = ['census_block_group', 'Sum', 'White_Absolute', 'Black_Absolute','White_Ratio', 'Black_Ratio']
    cbg_race_msa = cbg_race_msa[columns_of_interest].copy()
    # Deal with NaN values
    cbg_race_msa.fillna(0,inplace=True)

    return cbg_race_msa
'''

def load_cbg_ethnic_msa(cbg_ethnic, cbg_ids_msa, cbg_sizes): #20220306
    # Extract cbgs corresponding to the metro area, by merging dataframes
    cbg_ethnic_msa = pd.merge(cbg_ids_msa, cbg_ethnic, on='census_block_group', how='left')
    cbg_ethnic_msa['Sum'] = cbg_sizes.copy()
    cbg_ethnic_msa.rename(columns={'B03002e1':'Sum',
                                   'B03002e2':'NH_Total',
                                   'B03002e3':'NH_White',
                                   'B03002e4':'NH_Black',
                                   'B03002e5':'NH_Indian',
                                   'B03002e6':'NH_Asian',
                                   'B03002e7':'NH_Hawaiian',
                                   'B03002e12':'Hispanic'}, inplace=True)
    # Extract columns of interest
    columns_of_interest = ['census_block_group','Sum','NH_Total','NH_White','NH_Black','NH_Indian','NH_Asian','NH_Hawaiian','Hispanic']
    cbg_ethnic_msa = cbg_ethnic_msa[columns_of_interest].copy()
    # Deal with NaN values
    cbg_ethnic_msa.fillna(0,inplace=True)
    # Deal with CBGs with 0 populations
    cbg_ethnic_msa['Sum'] = cbg_ethnic_msa['Sum'].apply(lambda x : x if x!=0 else 1)

    cbg_ethnic_msa['Minority_Absolute'] = cbg_ethnic_msa['NH_White'].copy()
    cbg_ethnic_msa['Minority_Ratio'] = cbg_ethnic_msa['Minority_Absolute'] / cbg_ethnic_msa['Sum']
    columns_of_interest = ['census_block_group','Sum', 'Minority_Absolute', 'Minority_Ratio']
    cbg_ethnic_msa = cbg_ethnic_msa[columns_of_interest].copy()
    '''
    # Rename
    cbg_ethnic_msa.rename(columns={'B03002e12':'Hispanic_Absolute'},inplace=True)
    cbg_ethnic_msa['Hispanic_Ratio'] = cbg_ethnic_msa['Hispanic_Absolute'] / cbg_ethnic_msa['Sum']
    # Extract columns of interest
    columns_of_interest = ['census_block_group', 'Sum', 'Hispanic_Absolute', 'Hispanic_Ratio']
    cbg_ethnic_msa = cbg_ethnic_msa[columns_of_interest].copy()
    '''
    # Deal with NaN values
    cbg_ethnic_msa.fillna(0,inplace=True) 
    # Check whether there is NaN in cbg_tables
    if(cbg_ethnic_msa.isnull().any().any()):
        print('NaN exists in cbg_ethnic_msa. Please check.')
        pdb.set_trace()

    return cbg_ethnic_msa


def obtain_vulner_damage(cbg_age_msa, msa_name, root): #20220306
    '''Obtain vulnerability and damage, according to theoretical analysis.'''
    nyt_included = np.zeros(len(idxs_msa_all))
    for i in range(len(nyt_included)):
        if(i in idxs_msa_nyt):
            nyt_included[i] = 1
    cbg_age_msa['NYT_Included'] = nyt_included.copy()

    # Retrieve the attack rate for the whole MSA (home_beta, fitted for each MSA)
    home_beta = constants.parameters_dict[msa_name][1]

    # Get cbg_avg_infect_same, cbg_avg_infect_diff
    if(os.path.exists(os.path.join(root, f'3cbg_avg_infect_same_{msa_name}.npy'))):
        #print('cbg_avg_infect_same, cbg_avg_infect_diff: Load existing file.')
        cbg_avg_infect_same = np.load(os.path.join(root, f'3cbg_avg_infect_same_{msa_name}.npy'))
        cbg_avg_infect_diff = np.load(os.path.join(root, f'3cbg_avg_infect_diff_{msa_name}.npy'))
    else:
        print('cbg_avg_infect_same, cbg_avg_infect_diff: File not found. Please check.')
        pdb.set_trace()
    #print('cbg_avg_infect_same.shape:',cbg_avg_infect_same.shape)

    SEIR_at_30d = np.load(os.path.join(root, 'SEIR_at_30d.npy'),allow_pickle=True).item()
    S_ratio = SEIR_at_30d[msa_name]['S'] / (cbg_sizes.sum())
    I_ratio = SEIR_at_30d[msa_name]['I'] / (cbg_sizes.sum())
    #print('S_ratio:',S_ratio,'I_ratio:',I_ratio)

    # Deal with nan and inf (https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
    cbg_avg_infect_same = np.nan_to_num(cbg_avg_infect_same,nan=0,posinf=0,neginf=0)
    cbg_avg_infect_diff = np.nan_to_num(cbg_avg_infect_diff,nan=0,posinf=0,neginf=0)
    cbg_age_msa['Infect'] = cbg_avg_infect_same + cbg_avg_infect_diff
    # Check whether there is NaN in cbg_tables
    if(cbg_age_msa['Infect'].isnull().any().any()):
        print('There are NaNs in cbg_age_msa[\'Infect\']. Please check.')
        pdb.set_trace()

    # Normalize by cbg population
    cbg_avg_infect_same_norm = cbg_avg_infect_same / cbg_sizes
    cbg_avg_infect_diff_norm = cbg_avg_infect_diff / cbg_sizes
    cbg_avg_infect_all_norm = cbg_avg_infect_same_norm + cbg_avg_infect_diff_norm

    # Compute the average death rate for the whole MSA: perform another weighted average over all CBGs
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

    return cbg_age_msa

    
def annotate_group_vulnerability(demo_feat, cbg_table, num_groups): #20220306
    '''The smaller the group number, the more vulnerable the group.'''
    final_deaths_rate_current = np.zeros(num_groups)
    for group_id in range(num_groups):
        final_deaths_rate_current[group_id] = cbg_table[cbg_table[f'{demo_feat}_Quantile']==group_id]['Final_Deaths_Current'].sum()
        final_deaths_rate_current[group_id] /= cbg_table[cbg_table[f'{demo_feat}_Quantile']==group_id]['Sum'].sum()
    # Sort groups according to vulnerability
    group_vulnerability = np.argsort(-final_deaths_rate_current) # 死亡率从大到小排序
    group_vulner_dict = dict()
    for i in range(num_groups):
        for j in range(num_groups):
            if(final_deaths_rate_current[i]==final_deaths_rate_current[group_vulnerability[j]]):
                group_vulner_dict[i] = j
    # Annotate the CBGs according to the corresponding group vulnerability
    cbg_table['Group_Vulnerability'] = cbg_table.apply(lambda x : group_vulner_dict[x[f'{demo_feat}_Quantile']], axis=1)
    
    return cbg_table['Group_Vulnerability']

