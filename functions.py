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
'''
def get_separators(cbg_table, num_groups, col_indicator, col_sum, normalized):
    separators = np.zeros(num_groups+1)
    
    total = cbg_table[col_sum].sum()
    group_size = total / num_groups
    
    cbg_table_work = cbg_table.copy()
    cbg_table_work.sort_values(col_indicator, inplace=True)
    
    for j in range(last_position, len(cbg_table_work)):
        if (cbg_table_work.head(j)[col_sum].sum() <= group_size*(i+1)) & (cbg_table_work.head(j+1)[col_sum].sum() >= group_size*(i+1)):
            first_separator = cbg_table_work.iloc[j][col_indicator]
            break
    if(first_separator==0):
        
    else:
        separators[0] = 0
        last_position = 0
        for i in range(num_groups):
            for j in range(last_position, len(cbg_table_work)):
                if (cbg_table_work.head(j)[col_sum].sum() <= group_size*(i+1)) & (cbg_table_work.head(j+1)[col_sum].sum() >= group_size*(i+1)):
                    separators[i+1] = cbg_table_work.iloc[j][col_indicator]
                    last_position = j
                    break
    
    
    separators[-1] = 1 if normalized else cbg_table[col_indicator].max()
    
    return separators
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
'''
def assign_group(x, separators):
    num_groups = len(separators)-1
    for i in range(num_groups):
        #if((x>=separators[i]) & (x<separators[i+1])):
        if((x>separators[i]) & (x<=separators[i+1])):
            return i
    return(num_groups-1)
'''
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

# Analyze results and produce graphs
# Comparison between Dem-Struct-Agnostic and Dem-Struct-Aware            
'''            
def output_result_0_only_deaths(cbg_table, demo_feat, policy_list, num_groups, 
                                print_result=True,draw_result=True,
                                store_result=False, timestring='yyyymmdd'):
    
    #demo_feat: demographic feature of interest: 'Age','Poverty'
    #num_groups: number of quantiles that all CBGs are divided into
    
    
    color_list = ['k','r','b','y','c']
    
    label = dict()
    for policy in range(len(policy_list)):
        label[policy] = policy
        if('Age_Agnostic' in policy_list):
            label['Age_Agnostic'] = 'Dem-Struct-Agnostic'
            label['No_Vaccination'] = 'Dem-Struct-Aware'
    
    demo_feat_show = {'Age':'Elder Ratio',
                      'Mean_Household_Income':'Mean Household Income',
                      'Median_Household_Income':'Median Household Income',
                      'Poverty':'Poverty Ratio',
                      'Mobility':'Mobility Reduction Level',
                      'Race':'White Ratio',
                      'Population_Density':'Population Density',
                      'Essential_Worker':'Essential Worker Ratio'
                     }

    reverse=False
    
    results = {}
    for policy in policy_list:
        
        exec("final_deaths_rate_%s_total = cbg_table['Final_Deaths_%s'].sum()/cbg_table['Sum'].sum()" % (policy.lower(),policy))
        cbg_table['Final_Deaths_' + policy] = eval('avg_final_deaths_' + policy.lower())
        exec("%s = np.zeros(num_groups)" % ('final_deaths_rate_'+ policy.lower()))
        
        for i in range(num_groups):
            eval('final_deaths_rate_'+ policy.lower())[i] = cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Final_Deaths_' + policy].sum()
            eval('final_deaths_rate_'+ policy.lower())[i] /= cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Sum'].sum()
            
        results[policy] = {'deaths':eval('final_deaths_rate_'+ policy.lower())}
        
        if(print_result==True):
            print('Policy: ', policy)
            print('final_deaths_rate_'+policy.lower()+': ',eval('final_deaths_rate_'+ policy.lower()))
            print('Standard deviance from total: ', np.std(eval('final_deaths_rate_'+ policy.lower())))

    if(draw_result==True):
        fig, ax = plt.subplots(1,1,figsize=(8,3))
        
        for policy in policy_list:
            if(reverse==True):
                group_deaths = np.zeros(num_groups)
                for i in range(num_groups):
                    group_deaths[i] = eval('final_deaths_rate_'+policy.lower())[-i-1]
                ax[0].plot(np.arange(num_groups),
                         group_deaths,
                         label=label[policy],
                         color=color_list[policy_list.index(policy)],
                         marker='o'
                        )
            else:
                ax.plot(np.arange(num_groups),
                         eval('final_deaths_rate_'+policy.lower()),
                         label=label[policy],
                         color=color_list[policy_list.index(policy)],
                         marker='o')
            ax.plot([0,num_groups-1], 
                     [eval('final_deaths_rate_'+policy.lower()+'_total'), eval('final_deaths_rate_'+policy.lower()+'_total')], 
                     color=color_list[policy_list.index(policy)],
                     linestyle='--')
        #for i in range(eval('num_groups_by_'+demo_feat.lower())):
        #    plt.plot([i,i],[final_cases_rate_baseline[i], final_cases_rate_extreme_1[i]],color='g')
        ax.set_xticks(np.arange(num_groups))
        ax.legend()
        ax.xaxis.set_major_formatter(FuncFormatter(functions.to_percent))
        ax.set_xlabel(demo_feat_show[demo_feat]+' Percentile(%)')
        ax.set_title('Predicted Death Rates in %s' % MSA_NAME)
        
        if store_result == True:
            path = os.path.join(root,
                                '%s_%s_disparities_%s'
                                % (timestring,demo_feat.lower(), MSA_NAME))
            # Save graphs
            f=plt.gcf()
            f.savefig(path,bbox_inches='tight')
            plt.show()
            f.clear()
    
    return results
'''    


# Analyze results and produce graphs
# All policies
'''
def output_result(cbg_table, demo_feat, policy_list, num_groups, print_result=True,draw_result=True):
    
    #demo_feat: demographic feature of interest: 'Age','Poverty'
    #num_groups: number of quantiles that all CBGs are divided into
    
    print('Observation dimension: ', demo_feat)
    reverse = False
    color_list = ['k','b','r','y','c','peru']

    results = {}
    
    for policy in policy_list:
        exec("final_cases_rate_%s_total = cbg_table['Final_Cases_%s'].sum()/cbg_table['Sum'].sum()" % (policy.lower(),policy))
        exec("final_deaths_rate_%s_total = cbg_table['Final_Deaths_%s'].sum()/cbg_table['Sum'].sum()" % (policy.lower(),policy))
        
        cbg_table['Final_Cases_' + policy] = eval('avg_final_cases_' + policy.lower())
        cbg_table['Final_Deaths_' + policy] = eval('avg_final_deaths_' + policy.lower())
        
        exec("final_cases_rate_%s = np.zeros(num_groups)" % (policy.lower()))
        exec("%s = np.zeros(num_groups)" % ('final_deaths_rate_'+ policy.lower()))
        
        for i in range(num_groups):
            eval('final_cases_rate_'+ policy.lower())[i] = cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Final_Cases_' + policy].sum()
            eval('final_cases_rate_'+ policy.lower())[i] /= cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Sum'].sum()
            eval('final_deaths_rate_'+ policy.lower())[i] = cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Final_Deaths_' + policy].sum()
            eval('final_deaths_rate_'+ policy.lower())[i] /= cbg_table[cbg_table[demo_feat + '_Quantile']==i]['Sum'].sum()
            
        cases_total_abs = eval('final_cases_rate_%s_total'%(policy.lower()))
        deaths_total_abs = eval('final_deaths_rate_%s_total'%(policy.lower()))
        cases_gini_abs =  gini.gini(eval('final_cases_rate_'+ policy.lower()))
        deaths_gini_abs = gini.gini(eval('final_deaths_rate_'+ policy.lower()))
        
        if(policy=='Baseline'):
            cases_total_baseline = cases_total_abs;print('cases_total_baseline:',cases_total_baseline)
            deaths_total_baseline = deaths_total_abs;print('deaths_total_baseline:',deaths_total_baseline)
            cases_gini_baseline = cases_gini_abs
            deaths_gini_baseline = deaths_gini_abs
                
            cases_total_rel = 0
            deaths_total_rel = 0
            cases_gini_rel = 0
            deaths_gini_rel = 0
                                         
            results[policy] = {'cases_total_abs':'%.5f'% cases_total_abs,
                               'cases_total_rel':'%.5f'% cases_total_rel,
                               'cases_gini_abs':'%.5f'% cases_gini_abs,
                               'cases_gini_rel':'%.5f'% cases_gini_rel,
                           
                               'deaths_total_abs':'%.5f'% deaths_total_abs,
                               'deaths_total_rel':'%.5f'% deaths_total_rel,
                               'deaths_gini_abs':'%.5f'% deaths_gini_abs,
                               'deaths_gini_rel':'%.5f'% deaths_gini_rel}   
        else:
            cases_total_rel = (eval('final_cases_rate_%s_total'%(policy.lower())) - cases_total_baseline) / cases_total_baseline
            deaths_total_rel = (eval('final_deaths_rate_%s_total'%(policy.lower())) - deaths_total_baseline) / deaths_total_baseline
            cases_gini_rel =  (gini.gini(eval('final_cases_rate_'+ policy.lower())) - cases_gini_baseline) / cases_gini_baseline
            deaths_gini_rel = (gini.gini(eval('final_deaths_rate_'+ policy.lower())) - deaths_gini_baseline) / deaths_gini_baseline
        
            results[policy] = {'cases_total_abs':'%.5f'% cases_total_abs,
                               'cases_total_rel':'%.5f'% cases_total_rel,
                               'cases_gini_abs':'%.5f'% cases_gini_abs,
                               'cases_gini_rel':'%.5f'% cases_gini_rel,
                           
                               'deaths_total_abs':'%.5f'% deaths_total_abs,
                               'deaths_total_rel':'%.5f'% deaths_total_rel,
                               'deaths_gini_abs':'%.5f'% deaths_gini_abs,
                               'deaths_gini_rel':'%.5f'% deaths_gini_rel}                        
        
                                    
        if(print_result==True):
            print('Policy: ', policy)
            print('final_cases_rate_'+ policy.lower(), eval('final_cases_rate_%s_total'%(policy.lower())))
            print('Cases, Gini Index: ',gini.gini(eval('final_cases_rate_'+ policy.lower())))
            print('Deaths, Gini Index: ',gini.gini(eval('final_deaths_rate_'+ policy.lower())))
            
            if(policy=='Baseline'):
                cases_total_baseline = eval('final_cases_rate_%s_total'%(policy.lower()))
                deaths_total_baseline = eval('final_deaths_rate_%s_total'%(policy.lower()))
                cases_gini_baseline = gini.gini(eval('final_cases_rate_'+ policy.lower()))
                deaths_gini_baseline = gini.gini(eval('final_deaths_rate_'+ policy.lower()))
                
            if(policy!='Baseline' and policy!='No_Vaccination'):
                print('Compared to baseline:')
                
                print('Cases total: ', (eval('final_cases_rate_%s_total'%(policy.lower())) - cases_total_baseline) / cases_total_baseline)
                print('Deaths total: ', (eval('final_deaths_rate_%s_total'%(policy.lower())) - deaths_total_baseline) / deaths_total_baseline)
                print('Cases gini: ', (gini.gini(eval('final_cases_rate_'+ policy.lower())) - cases_gini_baseline) / cases_gini_baseline)
                print('Deaths gini: ', (gini.gini(eval('final_deaths_rate_'+ policy.lower())) - deaths_gini_baseline) / deaths_gini_baseline)

    if(draw_result==True):
        plt.figure(figsize=(8,3))
        if(demo_feat=='Age'):
            plt.title('Predicted case rates for CBGs with different elder ratios')
        elif(demo_feat=='Mobility'):
            plt.title('Predicted case rates for CBGs with different mobility reduction levels')
        elif(demo_feat=='Mean_Household_Income'):
            plt.title('Predicted case rates for CBGs with different mean household incomes')
        else:
            plt.title('Predicted case rates for CBGs with different %s ratios' % (demo_feat.lower()))
        
        for policy in policy_list:
            if(reverse==True):
                group_cases = np.zeros(num_groups)
                for i in range(num_groups):
                    group_cases[i] = eval('final_cases_rate_'+policy.lower())[-i-1]
                plt.plot(np.arange(num_groups),
                         group_cases,
                         label=policy,
                         color=color_list[policy_list.index(policy)],
                         marker='o'
                        )
            else:
                plt.plot(np.arange(num_groups),
                         eval('final_cases_rate_'+policy.lower()),
                         label=policy,
                         color=color_list[policy_list.index(policy)],
                         marker='o')
            plt.plot([0,num_groups-1], 
                     [eval('final_cases_rate_'+policy.lower()+'_total'), eval('final_cases_rate_'+policy.lower()+'_total')], 
                     color=color_list[policy_list.index(policy)],
                     linestyle='--')
        plt.xticks(np.arange(num_groups))
        plt.legend()

        plt.figure(figsize=(8,3))
        if(demo_feat=='Age'):
            plt.title('Predicted death rates for CBGs with different elder ratios')
        elif(demo_feat=='Mobility'):
            plt.title('Predicted death rates for CBGs with different mobility reduction levels')
        elif(demo_feat=='Mean_Household_Income'):
            plt.title('Predicted death rates for CBGs with different mean household incomes')
        else:
            plt.title('Predicted death rates for CBGs with different %s ratios' % (demo_feat.lower()))
        for policy in policy_list:
            if(reverse==True):
                group_deaths = np.zeros(num_groups)
                for i in range(num_groups):
                    group_deaths[i] = eval('final_deaths_rate_'+policy.lower())[-i-1]
                plt.plot(np.arange(num_groups),
                         group_deaths,
                         label=policy,
                         color=color_list[policy_list.index(policy)],
                         marker='o'
                        )
            else:
                plt.plot(np.arange(num_groups),
                         eval('final_deaths_rate_'+policy.lower()),
                         label=policy,
                         color=color_list[policy_list.index(policy)],
                         marker='o')
            plt.plot([0,num_groups-1], 
                     [eval('final_deaths_rate_'+policy.lower()+'_total'), eval('final_deaths_rate_'+policy.lower()+'_total')], 
                     color=color_list[policy_list.index(policy)],
                     linestyle='--')
        plt.xticks(np.arange(num_groups))
        plt.legend()

    return results
''' 

def make_gini_table(policy_list, demo_feat_list, num_groups, show_option, 
                    save_path, save_result=False):
    
    cbg_table_name_dict=dict()
    cbg_table_name_dict['Age'] = cbg_age_msa
    cbg_table_name_dict['Mean_Household_Income'] = cbg_income_msa
    cbg_table_name_dict['Race'] = cbg_race_msa
    cbg_table_name_dict['Mobility'] = cbg_mobility_msa
    cbg_table_name_dict['Essential_Worker'] = cbg_occupation_msa
    cbg_table_name_dict['Population_Density'] = cbg_density_msa
    
    print('Policy list: ', policy_list)
    print('Demographic feature list: ', demo_feat_list)
    print('Show Option: ', show_option)

    if(show_option=='cases'):
        gini_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('All','cases_total_abs'),('All','cases_total_rel')]))
    if(show_option=='deaths'):
        gini_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('All','deaths_total_abs'),('All','deaths_total_rel')]))
    
    gini_df['Policy'] = policy_list

    for demo_feat in demo_feat_list:
        print(demo_feat)
        results = functions.output_result(cbg_table_name_dict[demo_feat], 
                                          demo_feat, policy_list, num_groups=NUM_GROUPS,
                                          print_result=False, draw_result=False)
        if(show_option=='cases'):
            for i in range(len(policy_list)):
                policy = policy_list[i]
                gini_df.loc[i,('All','cases_total_abs')] = results[policy]['cases_total_abs']
                gini_df.loc[i,('All','cases_total_rel')] = results[policy]['cases_total_rel']
                gini_df.loc[i,(demo_feat,'cases_gini_abs')] = results[policy]['cases_gini_abs']
                gini_df.loc[i,(demo_feat,'cases_gini_rel')] = results[policy]['cases_gini_rel']
        elif(show_option=='deaths'):
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

        
        
        
        
        
# 水位法分配疫苗
# Adjustable execution ratio.
# Only 'flooding' in the most vulnerable demographic group.
# Check whether a CBG has been covered, before vaccinating it.
def vaccine_distribution_flood_new(cbg_table, vaccination_ratio, demo_feat, ascending, execution_ratio, leftover, is_last):
    cbg_table_sorted = cbg_table.copy()
    
    # Rank CBGs: (1)Put the uncovered ones (Covered=0) on the top; (2)Rank CBGs according to some demographic feature
    cbg_table_sorted['Covered'] = cbg_table_sorted.apply(lambda x : 1 if x['Vaccination_Vector']==x['Sum'] else 0, axis=1)
    #print('Num of covered cbgs:', len(cbg_table_sorted[cbg_table_sorted['Covered']==1]))
    # Rank CBGs according to some demographic feature
    cbg_table_sorted.sort_values(by=['Most_Vulnerable','Covered',demo_feat],ascending=[False, True, ascending],inplace = True)
    
    # Calculate number of available vaccines
    #leftover = cbg_table_sorted['Sum'].sum() * vaccination_ratio * execution_ratio - cbg_table_sorted[cbg_table_sorted['Covered']==1]['Sum'].sum()
    #print('leftover:',leftover)
    num_vaccines = cbg_table_sorted['Sum'].sum() * vaccination_ratio * execution_ratio + leftover
    #print('Total num of vaccines: ',num_vaccines)
    #print('Num of vaccines distributed according to the policy:',num_vaccines)
    
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
    '''
    del cbg_table_sorted
    #print('Final check of distributed vaccines:',vaccination_vector.sum())
    return vaccination_vector
    