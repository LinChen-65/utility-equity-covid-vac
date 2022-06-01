import numpy as np
import time

class Model:
    def __init__(self,
                 starting_seed=0,
                 num_seeds=1,
                 debug=False,
                 clip_poisson_approximation=True,
                 ipf_final_match='poi',
                 ipf_num_iter=100):

        self.starting_seed = starting_seed
        self.num_seeds = num_seeds
        self.debug = debug
        self.ipf_final_match = ipf_final_match
        assert ipf_final_match in ['cbg', 'poi']
        self.ipf_num_iter = ipf_num_iter
        self.clip_poisson_approximation = clip_poisson_approximation

        np.random.seed(self.starting_seed)

    def init_exogenous_variables(self,                             
                                 poi_areas,
                                 cbg_sizes,                                 
                                 all_hours,
                                 p_sick_at_t0,
                                 vaccination_time, # when to apply vaccination (which hour)
                                 vaccination_vector, # num of vaccines each CBG receives 
                                 protection_rate,
                                 poi_psi,
                                 home_beta,
                                 cbg_attack_rates_original,
                                 cbg_death_rates_original,
                                 poi_cbg_visits_list=None,
                                 poi_dwell_time_correction_factors=None,
                                 just_compute_r0=False,
                                 latency_period=96,  # 4 days
                                 infectious_period=84,  # 3.5 days
                                 confirmation_rate=.1,
                                 confirmation_lag=168,  # 7 days
                                 death_lag=432,  # 18 days
                                 no_print=False,
                                 ):
        self.M = len(poi_areas)#POI的数量
        self.N = len(cbg_sizes)#cbg的数量
        self.T = len(all_hours)#时间长度1512个小时
        self.PSI = poi_psi#ψ
        self.POI_AREAS = poi_areas#poi面积apj
        self.DWELL_TIME_CORRECTION_FACTORS = poi_dwell_time_correction_factors#dpj^2
        self.POI_FACTORS = self.PSI / poi_areas#ψ/apj
        if poi_dwell_time_correction_factors is not None:
            self.POI_FACTORS = poi_dwell_time_correction_factors * self.POI_FACTORS#dpj^2*ψ/apj
            #print('Adjusted POI transmission rates with dwell time correction factors')
            self.included_dwell_time_correction_factors = True
        else:
            self.included_dwell_time_correction_factors = False
        self.POI_CBG_VISITS_LIST = poi_cbg_visits_list#导入访问矩阵
        assert len(self.POI_CBG_VISITS_LIST) == self.T
        assert self.POI_CBG_VISITS_LIST[0].shape == (self.M, self.N)

        
        # CBG variables
        self.CBG_SIZES = cbg_sizes  #cbg人口数量
         # assume constant transmission rate, irrespective of how many people per square mile are in CBG.
        self.HOME_BETA = home_beta#β0
        self.CBG_ATTACK_RATES_ORIGINAL = cbg_attack_rates_original
        self.CBG_DEATH_RATES_ORIGINAL = cbg_death_rates_original
        self.LATENCY_PERIOD = latency_period
        self.INFECTIOUS_PERIOD = infectious_period
        self.all_hours = all_hours#全部的时间
        self.P_SICK_AT_T0 = p_sick_at_t0  # p0
        self.VACCINATION_TIME = vaccination_time
        self.VACCINATION_VECTOR = vaccination_vector
        self.PROTECTION_RATE = protection_rate
        self.just_compute_r0 = just_compute_r0
        self.confirmation_rate = confirmation_rate
        self.confirmation_lag = confirmation_lag
        self.death_lag = death_lag

        self.CBG_ATTACK_RATES_NEW = self.CBG_ATTACK_RATES_ORIGINAL * (1*(1-self.VACCINATION_VECTOR/self.CBG_SIZES)+(1-self.PROTECTION_RATE)*self.VACCINATION_VECTOR/self.CBG_SIZES)
        self.CBG_DEATH_RATES_NEW = self.CBG_DEATH_RATES_ORIGINAL * (1*(1-self.VACCINATION_VECTOR/self.CBG_SIZES)+(1-self.PROTECTION_RATE)*self.VACCINATION_VECTOR/self.CBG_SIZES)
        self.CBG_ATTACK_RATES_NEW = np.clip(self.CBG_ATTACK_RATES_NEW, 0, None) 
        self.CBG_DEATH_RATES_NEW = np.clip(self.CBG_DEATH_RATES_NEW, 0, None) 
        self.CBG_DEATH_RATES_NEW = np.clip(self.CBG_DEATH_RATES_NEW, None, 1) 
        assert((self.CBG_DEATH_RATES_NEW>=0).all())
        assert((self.CBG_DEATH_RATES_NEW<=1).all())

    def init_endogenous_variables(self):
        # Initialize exposed/latent individuals
        self.P0 = np.random.binomial(
            self.CBG_SIZES,
            self.P_SICK_AT_T0,
            size=(self.num_seeds, self.N))
        self.cbg_latent = self.P0
        self.cbg_infected = np.zeros((self.num_seeds, self.N))
        self.cbg_removed = np.zeros((self.num_seeds, self.N))
        self.cases_to_confirm = np.zeros((self.num_seeds, self.N))
        self.new_confirmed_cases = np.zeros((self.num_seeds, self.N))
        self.deaths_to_happen = np.zeros((self.num_seeds, self.N))
        self.new_deaths = np.zeros((self.num_seeds, self.N))
        self.C2=np.zeros((self.num_seeds, self.N))
        self.D2=np.zeros((self.num_seeds, self.N))
    
    def get_new_infectious(self):
        new_infectious = np.random.binomial(self.cbg_latent.astype(int), 1 / self.LATENCY_PERIOD)
        return new_infectious
    
    def get_new_removed(self):
        new_removed = np.random.binomial(self.cbg_infected.astype(int), 1 / self.INFECTIOUS_PERIOD)
        return new_removed
    
    def format_floats(self, arr):
        return [int(round(x)) for x in arr]
    
    def simulate_disease_spread(self,verbosity=24,no_print=False): 
        L_1=[]
        I_1=[]
        R_1=[]
        C_1=[]
        D_1=[]
        T1=[]
        t = 0
        C=[0]
        D=[0]
        
        history_C2 = []
        history_D2 = []
        
        epidemic_over = False
        
        while t < self.T:#仿真时间之内
            iter_t0 = time.time()
            if (verbosity > 0) and (t % verbosity == 0):
                L = np.sum(self.cbg_latent, axis=1) # 获得msa内所有cbg的L态人数
                I = np.sum(self.cbg_infected, axis=1) # 获得msa内所有cbg的I态人数
                R = np.sum(self.cbg_removed, axis=1) # 获得msa内所有cbg的R态人数
                
                T1.append(t)
                L_1.append(L)
                I_1.append(I)
                R_1.append(R)
                C_1.append(C)
                D_1.append(D)
                
                history_C2.append(self.C2) # Save history for cases
                history_D2.append(self.D2) # Save history for deaths
            
            if(epidemic_over == False):
                assert((self.cbg_latent>=0).all())
                assert((self.cbg_infected>=0).all())
                assert((self.cbg_removed>=0).all())
                
                assert((self.cases_to_confirm>=0).all())
                assert((self.new_confirmed_cases>=0).all())
                assert((self.deaths_to_happen>=0).all())
                assert((self.new_deaths>=0).all())
                
                self.update_states(t)
                C1 = np.sum(self.new_confirmed_cases,axis=1)
                self.C2=self.C2+self.new_confirmed_cases
                C[0]=C[0]+C1
                D1 = np.sum(self.new_deaths,axis=1)
                self.D2=self.D2+self.new_deaths
                D[0]=D[0]+D1
                if self.debug and verbosity > 0 and t % verbosity == 0:
                    print('Num active POIs: %d. Num with infection rates clipped: %d' % (self.num_active_pois, self.num_poi_infection_rates_clipped))
                    print('Num CBGs active at POIs: %d. Num with clipped num cases from POIs: %d' % (self.num_cbgs_active_at_pois, self.num_cbgs_with_clipped_poi_cases))
                if self.debug:
                    print("Time for iteration %i: %2.3f seconds" % (t, time.time() - iter_t0))

                if np.max(self.cbg_latent + self.cbg_infected) < 1:
                    epidemic_over = True # epidemic is over
                    print('Disease died off after t=%d. Stopping experiment.' % t)
                    #if t < self.T-1:
                        # need to fill in trailing 0's in self.history
                        #self.fill_remaining_history(t)
                    #break
                t += 1
                
            else: 
                t += 1
        cbg_all_affected = self.cbg_latent + self.cbg_infected + self.cbg_removed
        
        if self.N <= 10:
            print('Final state after %d rounds: L+I+R=%s' % (t, self.format_floats(cbg_all_affected)))
        total_affected = np.sum(cbg_all_affected, axis=1)
     
        return T1,L_1,I_1,R_1,self.C2,self.D2, total_affected, history_C2, history_D2, cbg_all_affected
    
    def update_states(self, t):
        self.get_new_cases(t)
        new_infectious = self.get_new_infectious()
        new_removed = self.get_new_removed()
        if not self.just_compute_r0:
            # normal case.
            self.cbg_latent = self.cbg_latent + self.cbg_new_cases - new_infectious
            self.cbg_infected = self.cbg_infected + new_infectious - new_removed
            self.cbg_removed = self.cbg_removed + new_removed
            self.new_confirmed_cases = np.random.binomial(self.cases_to_confirm.astype(int), 1/self.confirmation_lag)
            new_cases_to_confirm = np.random.binomial(new_infectious.astype(int), self.confirmation_rate)
            self.cases_to_confirm = self.cases_to_confirm + new_cases_to_confirm - self.new_confirmed_cases
            self.new_deaths = np.random.binomial(self.deaths_to_happen.astype(int), 1/self.death_lag)
            if t<self.VACCINATION_TIME:
                new_deaths_to_happen = np.random.binomial(new_infectious.astype(int), self.CBG_DEATH_RATES_ORIGINAL)
            else:
                new_deaths_to_happen = np.random.binomial(new_infectious.astype(int), self.CBG_DEATH_RATES_NEW)
            self.deaths_to_happen = self.deaths_to_happen + new_deaths_to_happen - self.new_deaths
        else:
            self.cbg_latent = self.cbg_latent - new_infectious
            self.cbg_infected = self.cbg_infected + new_infectious - new_removed
            self.cbg_removed = self.cbg_removed + new_removed + self.cbg_new_cases
           

    def get_new_cases(self, t):
        ### Compute CBG densities and infection rates
        cbg_densities = self.cbg_infected / self.CBG_SIZES  # S x N,Ici/Nci
        num_sus = np.clip(self.CBG_SIZES - self.cbg_latent - self.cbg_infected - self.cbg_removed, 0, None)  # S x N，易感人数即普通人人数维度是1×N
        sus_frac = num_sus / self.CBG_SIZES  # S x N，普通人比例

        if t<self.VACCINATION_TIME:
            cbg_base_infection_rates = self.HOME_BETA * self.CBG_ATTACK_RATES_ORIGINAL * cbg_densities  # S x N，得到λtcbg
        else:
            cbg_base_infection_rates = self.HOME_BETA * self.CBG_ATTACK_RATES_NEW * cbg_densities  # S x N，得到λtcbg
        cbg_base_infection_rates=np.nan_to_num(cbg_base_infection_rates) 
        self.num_base_infection_rates_clipped = np.sum(cbg_base_infection_rates > 1)
        cbg_base_infection_rates = np.clip(cbg_base_infection_rates, None, 1.0)#限制感染率，大于1就取1。

        ### Load or compute POI x CBG matrix
        if self.POI_CBG_VISITS_LIST is not None:  # try to load
            poi_cbg_visits = self.POI_CBG_VISITS_LIST[t]  # M x N导入lpf
            poi_visits = poi_cbg_visits @ np.ones(poi_cbg_visits.shape[1]) #@表示做内积，得到的是每个poi的所有cbg的访问数之和。Vpjt

        self.num_active_pois = np.sum(poi_visits > 0)#访问人数大于0的poi数量
        col_sums = np.squeeze(np.array(poi_cbg_visits.sum(axis=0)))#Ucit从cbg ci出来的人数 1xN
        self.cbg_num_out = col_sums#每个cbg的人数，U
        # S x M = (M) * ((M x N) @ (S x N).T ).T
        poi_infection_rates = self.POI_FACTORS * (poi_cbg_visits @ cbg_densities.T).T#λpoit，poi内的感染率
        self.num_poi_infection_rates_clipped = np.sum(poi_infection_rates > 1)
        if self.clip_poisson_approximation:
            poi_infection_rates = np.clip(poi_infection_rates, None, 1.0)#修正poi内的感染率

        # S x N = (S x N) * ((S x M) @ (M x N))
        if t<self.VACCINATION_TIME:
            cbg_mean_new_cases_from_poi = self.CBG_ATTACK_RATES_ORIGINAL * sus_frac * (poi_infection_rates @ poi_cbg_visits)
        else:
            cbg_mean_new_cases_from_poi = self.CBG_ATTACK_RATES_NEW * sus_frac * (poi_infection_rates @ poi_cbg_visits)
        cbg_mean_new_cases_from_poi=np.nan_to_num(cbg_mean_new_cases_from_poi) 
        cbg_mean_new_cases_from_poi = cbg_mean_new_cases_from_poi.astype(np.float64) 
        num_cases_from_poi = np.random.poisson(cbg_mean_new_cases_from_poi)
        self.num_cbgs_active_at_pois = np.sum(cbg_mean_new_cases_from_poi > 0)

        self.num_cbgs_with_clipped_poi_cases = np.sum(num_cases_from_poi > num_sus)
        self.cbg_new_cases_from_poi = np.clip(num_cases_from_poi, None, num_sus)
        num_sus_remaining = num_sus - self.cbg_new_cases_from_poi

        self.cbg_new_cases_from_base = np.random.binomial(
            num_sus_remaining.astype(int),
            cbg_base_infection_rates)
        self.cbg_new_cases = self.cbg_new_cases_from_poi + self.cbg_new_cases_from_base

        assert (self.cbg_new_cases <= num_sus).all()

