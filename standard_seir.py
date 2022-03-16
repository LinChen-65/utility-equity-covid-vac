# python standard_seir.py MSA_NAME
# Ref: https://github.com/silpara/simulators/blob/master/compartmental_models/SEIR%20Simulator%20in%20Python.ipynb

import setproctitle
setproctitle.setproctitle("covid-19-vac@chenlin")

import socket
import sys
import os
import datetime
import numpy as np
import pandas as pd
import constants
import functions
from sklearn.metrics import mean_squared_error
from math import sqrt
import pdb
import constants

#import SEIR
from scipy.integrate import odeint
from bayes_opt import BayesianOptimization
#from lmfit import minimize, Parameters, Parameter, report_fit
from hyperopt import hp, tpe, Trials, fmin


# root
hostname = socket.gethostname()
print('hostname: ', hostname)
if(hostname in ['fib-dl3','rl3','rl2']):
    root = '/data/chenlin/COVID-19/Data' #dl3
    saveroot = '/data/chenlin/utility-equity-covid-vac/results/'
elif(hostname=='rl4'):
    root = '/home/chenlin/COVID-19/Data' #rl4
    saveroot = '/home/chenlin/utility-equity-covid-vac/results/'
# subroot
subroot = 'seir'
if(not os.path.exists(os.path.join(saveroot, subroot))): # if folder does not exist, create one.
    os.makedirs(os.path.join(saveroot, subroot))

MIN_DATETIME = datetime.datetime(2020, 3, 1, 0)
MAX_DATETIME = datetime.datetime(2020, 5, 2, 23)
NUM_DAYS = 63

BETA_AND_PSI_PLAUSIBLE_RANGE = {"min_home_beta": 0.0011982272027079982,
                                "max_home_beta": 0.023964544054159966,
                                "max_poi_psi": 4886.41659532027,
                                "min_poi_psi": 515.4024854336667}
                                
###############################################################################
# Main variables

MSA_NAME = sys.argv[1]; print('MSA_NAME: ',MSA_NAME)
if(MSA_NAME == 'all'):
    msa_name_list = constants.MSA_NAME_LIST
else:
    msa_name_list = [MSA_NAME]


###############################################################################
# Functions

def ode_model(z, t, beta, sigma, gamma):
    """
    Reference https://www.idmod.org/docs/hiv/model-seir.html
    """
    S, E, I, R, D = z
    N = S + E + I + R
    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - sigma*E
    dIdt = sigma*E - gamma*I
    dRdt = gamma*I
    dDdt = IFR*I # 202101807
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def ode_solver(t, initial_conditions, params):
    initE, initI, initR, initN, initD = initial_conditions
    beta, sigma, gamma = params
    initS = initN - (initE + initI + initR)
    res = odeint(ode_model, [initS, initE, initI, initR, initD], t, args=(beta, sigma, gamma))
    return res

def Standard_SEIR(initE, initI, initR, initN, initD, beta, sigma, gamma, days):
    initial_conditions = [initE, initI, initR, initN, initD]
    params = [beta, sigma, gamma]
    tspan = np.arange(0, days, 1)
    sol = ode_solver(tspan, initial_conditions, params)
    S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]
    '''
    print('S: ', S)
    print('E: ', E)
    print('I: ', I)
    print('R: ', R)
    '''
    return S, E, I, R, D
    

def target_function(beta):
    beta = np.round(beta, 4)
    """
    Function with unknown internals we wish to maximize.
    This is just serving as an example, for all intents and purposes think of the internals of this function, 
    i.e.: the process which generates its output values, as unknown.
    """
    S, E, I, R, D = Standard_SEIR(initE, initI, initR, initN, initD, beta, sigma, gamma, days)
    predicted_cases_accumulated = I + R
    predicted_cases_daily = [0]
    for i in range(1,NUM_DAYS):
        predicted_cases_daily.append(predicted_cases_accumulated[i]-predicted_cases_accumulated[i-1])
    #print('predicted_cases_daily: ', predicted_cases_daily)
    
    #return -sqrt(mean_squared_error(cases_daily_smooth, predicted_cases_daily)) # 取负号，这样就变成max的目标函数
    loss = sqrt(mean_squared_error(cases_daily_smooth, predicted_cases_daily))
    #print('loss: ', loss)
    return loss # 取负号，这样就变成max的目标函数
    
###############################################################################
# Load Data

# Load ACS Data for MSA-county matching
acs_data = pd.read_csv(os.path.join(root,'list1.csv'),header=2)
acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
# Load SafeGraph data to obtain CBG sizes (i.e., populations)
filepath = os.path.join(root,"safegraph_open_census_data/data/cbg_b01.csv")
cbg_agesex = pd.read_csv(filepath)
# Load ground truth: NYT Data
nyt_data = pd.read_csv(os.path.join(root, 'us-counties.csv'))

for this_msa in msa_name_list:
    MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[this_msa]

    msa_match = functions.match_msa_name_to_msas_in_acs_data(MSA_NAME_FULL, acs_msas)
    msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
    msa_data['FIPS Code'] = msa_data.apply(lambda x : functions.get_fips_codes_from_state_and_county_fp((x['FIPS State Code']),x['FIPS County Code']), axis=1)
    good_list = list(msa_data['FIPS Code'].values)
    #print('County included: ',good_list)

    # Load CBG ids for the MSA
    cbg_ids_msa = pd.read_csv(os.path.join(root,this_msa,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
    cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)
    M = len(cbg_ids_msa)

    cbg_age_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
    cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
    # Deal with CBGs with 0 populations
    print(cbg_age_msa[cbg_age_msa['Sum']==0]['census_block_group'])
    cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)
    M = len(cbg_age_msa)
    cbg_sizes = cbg_age_msa['Sum'].values
    cbg_sizes = np.array(cbg_sizes,dtype='int32')
    print('Total population: ',np.sum(cbg_sizes))
    del cbg_age_msa

    nyt_data['in_msa'] = nyt_data.apply(lambda x : x['fips'] in good_list , axis=1)
    nyt_data_msa = nyt_data[nyt_data['in_msa']==True].copy()
    # Extract data according to simulation time range
    nyt_data_msa['in_simu_period'] = nyt_data_msa['date'].apply(lambda x : True if (x<'2020-05-10') & (x>'2020-03-07') else False)
    nyt_data_msa_in_simu_period = nyt_data_msa[nyt_data_msa['in_simu_period']==True].copy() 
    nyt_data_msa_in_simu_period.reset_index(inplace=True)
    del nyt_data_msa
    # Group by date
    nyt_data_group = nyt_data_msa_in_simu_period.groupby(nyt_data_msa_in_simu_period["date"])
    # Sum up cases/deaths from different counties
    # Cumulative
    nyt_data_cumulative = nyt_data_group.sum()[['cases','deaths']]

    # NYT data: Accumulated cases and deaths
    cases = nyt_data_cumulative['cases'].values
    if(len(cases)<NUM_DAYS):
        cases = [0]*(NUM_DAYS-len(cases)) + list(cases)
    cases_smooth = functions.apply_smoothing(cases, agg_func=np.mean, before=3, after=3)

    deaths = nyt_data_cumulative['deaths'].values
    if(len(deaths)<NUM_DAYS):
        deaths = [0]*(NUM_DAYS-len(deaths)) + list(deaths)
    deaths_smooth = functions.apply_smoothing(deaths, agg_func=np.mean, before=3, after=3)

    # NYT data: From cumulative to daily
    # Cases
    cases_daily = [0]
    for i in range(1,len(nyt_data_cumulative)):
        cases_daily.append(nyt_data_cumulative['cases'].values[i]-nyt_data_cumulative['cases'].values[i-1])
    if(len(cases_daily)<NUM_DAYS):
        cases_daily = [0]*(NUM_DAYS-len(cases_daily)) + list(cases_daily)
    # Smoothed ground truth
    cases_daily_smooth = functions.apply_smoothing(cases_daily, agg_func=np.mean, before=3, after=3)

    # Deaths
    deaths_daily = [0]
    for i in range(1,len(nyt_data_cumulative)):
        deaths_daily.append(nyt_data_cumulative['deaths'].values[i]-nyt_data_cumulative['deaths'].values[i-1])
    if(len(deaths_daily)<NUM_DAYS):
        deaths_daily = [0]*(NUM_DAYS-len(deaths_daily)) + list(deaths_daily)
    # Smoothed ground truth
    deaths_daily_smooth = functions.apply_smoothing(deaths_daily, agg_func=np.mean, before=3, after=3)

    print('Data loaded.')

    ###############################################################################
    # Fixed parameters for SEIR

    # ref: https://www.medrxiv.org/content/10.1101/2020.04.01.20049825v1.full.pdf
    initN = cbg_sizes.sum() #1380000000
    # S0 = 966000000
    initE = 0 #initN * (1 - 1e-4) #1
    #initI = initN * 1e-5 #1
    initR = 0
    initD = 0
    sigma = 1/4 #1/5.2
    gamma = 1/3.5 #1/2.9
    IFR = 0.0066
    # beta is learnable #R0 = 4 #beta = R0 * gamma
    days = NUM_DAYS #150

      
    # Bayesian optimization with lib: Hyperopt
    p_sicks_list = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
    #p_sicks_list.append(constants.parameters_dict[MSA_NAME][0]) #20220315
    for i in range(len(p_sicks_list)):
        p_sicks = p_sicks_list[i]
        print('p_sicks: ', p_sicks)
        #initI = initN * p_sicks
        initI = 0
        initE = initN * p_sicks
        # Create the domain space
        #space = hp.uniform('beta', 0, 1)
        space = hp.quniform('beta', 0, 1, 0.001)
        # Create the algorithm
        tpe_algo = tpe.suggest
        # Create a trials object
        tpe_trials = Trials()
        # Run 2000 evals with the tpe algorithm
        tpe_best = fmin(fn=target_function, space=space, 
                        algo=tpe_algo, trials=tpe_trials, 
                        #max_evals=10)
                        max_evals=300)

        print(tpe_best)
        S, E, I, R, D = Standard_SEIR(initE, initI, initR, initN, initD, tpe_best['beta'], sigma, gamma, days)
        predicted_cases_accumulated = I + R
        predicted_cases_daily = [0]
        for j in range(1,NUM_DAYS):
            predicted_cases_daily.append(predicted_cases_accumulated[j]-predicted_cases_accumulated[j-1])
        this_opt = sqrt(mean_squared_error(cases_daily_smooth, predicted_cases_daily))
        #print('this_opt: ', this_opt)
        
        
        if(i==0):
            beta_opt = tpe_best['beta']
            result_opt = this_opt
            p_sicks_opt = p_sicks
        elif(this_opt < result_opt):
            beta_opt = tpe_best['beta']
            result_opt = this_opt
            p_sicks_opt = p_sicks
            
    print('p_sicks_opt: ', p_sicks_opt)
    print('beta_opt: ', beta_opt)
    print('result_opt: ', result_opt)


    #initI = initN * p_sicks_opt
    initI = 0
    initE = initN * p_sicks_opt
    S, E, I, R, D = Standard_SEIR(initE, initI, initR, initN, initD, beta_opt, sigma, gamma, days)

    print('D: ', D)
    predicted_deaths_accumulated = D
    predicted_deaths_daily = [0]
    for i in range(1,NUM_DAYS):
        predicted_deaths_daily.append(predicted_deaths_accumulated[i]-predicted_deaths_accumulated[i-1])
    print('predicted_deaths_daily: ', predicted_deaths_daily)
    predicted_deaths_daily = np.array(predicted_deaths_daily) #20220315
    RMSE = sqrt(mean_squared_error(deaths_daily_smooth, predicted_deaths_daily))
    normalizer = deaths_daily_smooth.mean()
    RMSE_norm = RMSE / normalizer
    print('RMSE: ', RMSE, 'RMSE_norm: ', RMSE_norm)

    '''
    predicted_cases_accumulated = I + R
    #predicted_cases_accumulated = I
    predicted_cases_daily = [0]
    for i in range(1,NUM_DAYS):
        predicted_cases_daily.append(predicted_cases_accumulated[i]-predicted_cases_accumulated[i-1])
    print('predicted_cases_daily: ', predicted_cases_daily)
    print(sqrt(mean_squared_error(cases_daily_smooth, predicted_cases_daily)))
    '''
    print('p_sicks_opt: ', p_sicks_opt)
    print('beta_opt: ', beta_opt)
    print('result_opt: ', result_opt)

    filepath = os.path.join(saveroot, subroot, f'seir_daily_deaths_{this_msa}') #20220315
    predicted_deaths_daily.tofile(filepath)
    print(f'{this_msa}, file saved at: {filepath}')

pdb.set_trace()