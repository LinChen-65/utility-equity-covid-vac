import numpy as np
import pandas as pd

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
def average_across_random_seeds_only_death(history_D2, num_cbgs, cbg_idxs, print_results=False, draw_results=False):
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
    
'''    
def to_percent(temp, position):
    #return '%1.0f'%(5*(temp+1)) + '%' 
    #return '%1.0f'%(5*(temp+1))
    return '%1.0f'%((100/)*(temp+1))
'''

def get_separators(cbg_table, num_groups, col_indicator, col_sum, normalized):
    separators = np.zeros(num_groups+1)
    
    total = cbg_table[col_sum].sum()
    group_size = total / num_groups
    
    cbg_table_work = cbg_table.copy()
    cbg_table_work.sort_values(col_indicator, inplace=True)
    last_position = 0
    for i in range(num_groups):
        for j in range(last_position, len(cbg_table_work)):
            if (cbg_table_work.head(j)[col_sum].sum() <= group_size*(i+1)) & (cbg_table_work.head(j+1)[col_sum].sum() >= group_size*(i+1)):
                separators[i+1] = cbg_table_work.iloc[j][col_indicator]
                last_position = j
                break
    
    #separators[0] = 0
    separators[0] = -0.1 # to prevent making the first group [0,0] (which results in an empty group)
    separators[-1] = 1 if normalized else cbg_table[col_indicator].max()
    
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
    cbg_table_sorted['Covered'] = cbg_table_sorted.apply(lambda x : 1 if x['Vaccination_Vector']==x['Sum'] else 0, axis=1)
    #print('Num of covered cbgs:', len(cbg_table_sorted[cbg_table_sorted['Covered']==1]))
    # Rank CBGs according to some demographic feature
    cbg_table_sorted.sort_values(by=['Most_Vulnerable','Covered',demo_feat],ascending=[False, True, ascending],inplace = True)
    
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
    