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
                                 poi_psi,
                                 home_beta,
                                 poi_cbg_visits_list=None,
                                 poi_dwell_time_correction_factors=None,
                                 just_compute_r0=False,
                                 latency_period=96,  # 4 days
                                 infectious_period=84,  # 3.5 days
                                 confirmation_rate=.1,
                                 confirmation_lag=168,  # 7 days
                                 death_rate=.0066,
                                 death_lag=432,  # 18 days
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
        if self.POI_CBG_VISITS_LIST is not None:
            #print('Received POI_CBG_VISITS_LIST, will NOT be computing hourly matrices on the fly')
            assert len(self.POI_CBG_VISITS_LIST) == self.T
            assert self.POI_CBG_VISITS_LIST[0].shape == (self.M, self.N)
        else:
            # will use this matrix to compute hourly counts; must match all_hours length
            assert self.POI_TIME_COUNTS.shape[1] == self.T
        self.clipping_monitor = {
        'num_base_infection_rates_clipped':[],
        'num_active_pois':[],
        'num_poi_infection_rates_clipped':[],
        'num_cbgs_active_at_pois':[],
        'num_cbgs_with_clipped_poi_cases':[]}
        # CBG variables
        self.CBG_SIZES = cbg_sizes  #cbg人口数量
         # assume constant transmission rate, irrespective of how many people per square mile are in CBG.
        self.HOME_BETA = home_beta#β0
        self.LATENCY_PERIOD = latency_period
        self.INFECTIOUS_PERIOD = infectious_period
        self.all_hours = all_hours#全部的时间
        self.P_SICK_AT_T0 = p_sick_at_t0  # p0
        self.just_compute_r0 = just_compute_r0
        self.confirmation_rate = confirmation_rate
        self.confirmation_lag = confirmation_lag
        self.death_rate = death_rate
        self.death_lag = death_lag

    
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
    
    def simulate_disease_spread(self,verbosity=24): 
        start_time = time.time() 
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
        
        while t < self.T:#仿真时间之内
            iter_t0 = time.time()
            if (verbosity > 0) and (t % verbosity == 0):
                L = np.sum(self.cbg_latent, axis=1)#获得msa内所有cbg的L态人数
                I = np.sum(self.cbg_infected, axis=1)#获得msa内所有cbg的I态人数
                R = np.sum(self.cbg_removed, axis=1)#获得msa内所有cbg的R态人数
                T1.append(t)
                L_1.append(L)
                I_1.append(I)
                R_1.append(R)
                C_1.append(C)
                D_1.append(D)
                
                history_C2.append(self.C2) # Save history for cases
                history_D2.append(self.D2) # Save history for deaths
                
                #print('t:',t,'L:',L,'I:',I,'R',R,'C',C,'D',D)
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
                print('Disease died off after t=%d. Stopping experiment.' % t)
                if t < self.T-1:
                    # need to fill in trailing 0's in self.history
                    self.fill_remaining_history(t)
                break
            t += 1
        all_infected = self.cbg_latent + self.cbg_infected + self.cbg_removed
        if self.N <= 10:
            print('Final state after %d rounds: L+I+R=%s' % (t, self.format_floats(all_infected)))
        total = np.sum(all_infected, axis=1)
        print(f'Average number of people infected across random seeds: {np.mean(total):.3f}')
        if self.just_compute_r0:
            assert self.cbg_latent.sum() == 0
            assert self.cbg_infected.sum() == 0

            initial_cases = self.P0.sum(axis=1)
            self.estimated_R0 = {'R0':1.*(total - initial_cases) / initial_cases}
            assert self.estimated_R0['R0'].shape  == total.shape == initial_cases.shape
            print("Mean initial cases across seeds: %2.3f; new cases from initial: %2.3f; estimated R0: %2.3f" %
                (initial_cases.mean(), (total - initial_cases).mean(), self.estimated_R0['R0'].mean()))

            total_base = self.history['all']['new_cases_from_base'].sum(axis=1)
            total_poi = self.history['all']['new_cases_from_poi'].sum(axis=1)
            assert total_base.shape == total_poi.shape == initial_cases.shape
            self.estimated_R0['R0_base'] = 1.*total_base / initial_cases
            self.estimated_R0['R0_POI'] = 1.*total_poi / initial_cases
            assert np.allclose(self.estimated_R0['R0_base'] + self.estimated_R0['R0_POI'], self.estimated_R0['R0'])
        end_time = time.time()
        print('Simulation time = %.3fs -> %.3fs per iteration' %
            (end_time - start_time, (end_time - start_time)/t))
        return T1,L_1,I_1,R_1,self.C2,self.D2, history_C2, history_D2, all_infected


    def update_states(self, t):
        self.get_new_cases(t)
        new_infectious = self.get_new_infectious()
        new_removed = self.get_new_removed()
        if not self.just_compute_r0:
            # normal case.
            #print('normal case')
            self.cbg_latent = self.cbg_latent + self.cbg_new_cases - new_infectious
            self.cbg_infected = self.cbg_infected + new_infectious - new_removed
            self.cbg_removed = self.cbg_removed + new_removed
            self.new_confirmed_cases = np.random.binomial(self.cases_to_confirm.astype(int), 1/self.confirmation_lag)
            new_cases_to_confirm = np.random.binomial(new_infectious.astype(int), self.confirmation_rate)
            self.cases_to_confirm = self.cases_to_confirm + new_cases_to_confirm - self.new_confirmed_cases
            #print(np.sum(new_cases_to_confirm + self.new_confirmed_cases,axis=1))
            self.new_deaths = np.random.binomial(self.deaths_to_happen.astype(int), 1/self.death_lag)
            new_deaths_to_happen = np.random.binomial(new_infectious.astype(int), self.death_rate)
            self.deaths_to_happen = self.deaths_to_happen + new_deaths_to_happen - self.new_deaths
        else:
            # if we want to calibrate R0, don't allow anyone new to become infected - just put new_cases in removed.
            self.cbg_latent = self.cbg_latent - new_infectious
            self.cbg_infected = self.cbg_infected + new_infectious - new_removed
            self.cbg_removed = self.cbg_removed + new_removed + self.cbg_new_cases
           

    def get_new_cases(self, t):
        ### Compute CBG densities and infection rates
        cbg_densities = self.cbg_infected / self.CBG_SIZES  # S x N,Ici/Nci
        overall_densities = (np.sum(self.cbg_infected, axis=1) / np.sum(self.CBG_SIZES)).reshape(-1, 1)  # S x 1#总感染率，全部cbg的感染数除以总人数
        num_sus = np.clip(self.CBG_SIZES - self.cbg_latent - self.cbg_infected - self.cbg_removed, 0, None)  # S x N，易感人数即普通人人数维度是1×N
        sus_frac = num_sus / self.CBG_SIZES  # S x N，普通人比例
        #assert (cbg_densities >= 0).all()
        #assert (cbg_densities <= 1).all()
        #assert (sus_frac >= 0).all()
        #assert (sus_frac <= 1).all()

        if self.PSI > 0:
            # Our model: can only be infected by people in your home CBG.
                cbg_base_infection_rates = self.HOME_BETA * cbg_densities  # S x N，得到λtcbg
                cbg_base_infection_rates=np.nan_to_num(cbg_base_infection_rates)
        else:
            # Ablation: standard model with uniform mixing.
            cbg_base_infection_rates = np.tile(overall_densities, self.N) * self.HOME_BETA  # S x N
        self.num_base_infection_rates_clipped = np.sum(cbg_base_infection_rates > 1)
        cbg_base_infection_rates = np.clip(cbg_base_infection_rates, None, 1.0)#限制感染率，大于1就取1。


        ### Load or compute POI x CBG matrix
        if self.POI_CBG_VISITS_LIST is not None:  # try to load
            poi_cbg_visits = self.POI_CBG_VISITS_LIST[t]  # M x N导入lpf
            poi_visits = poi_cbg_visits @ np.ones(poi_cbg_visits.shape[1]) #@表示做内积，得到的是每个poi的所有cbg的访问数之和。Vpjt
   
        if not self.just_compute_r0:#凑数用的，并非这个意思
          # use network data
            self.num_active_pois = np.sum(poi_visits > 0)#访问人数大于0的poi数量
            col_sums = np.squeeze(np.array(poi_cbg_visits.sum(axis=0)))#Ucit从cbg ci出来的人数 1xN
            self.cbg_num_out = col_sums#每个cbg的人数，U
            # S x M = (M) * ((M x N) @ (S x N).T ).T
            poi_infection_rates = self.POI_FACTORS * (poi_cbg_visits @ cbg_densities.T).T#λpoit，poi内的感染率
            self.num_poi_infection_rates_clipped = np.sum(poi_infection_rates > 1)
            if self.clip_poisson_approximation:
                poi_infection_rates = np.clip(poi_infection_rates, None, 1.0)#修正poi内的感染率

            # S x N = (S x N) * ((S x M) @ (M x N))
            cbg_mean_new_cases_from_poi = sus_frac * (poi_infection_rates @ poi_cbg_visits)#
            cbg_mean_new_cases_from_poi=np.nan_to_num(cbg_mean_new_cases_from_poi)
            num_cases_from_poi = np.random.poisson(cbg_mean_new_cases_from_poi)
            self.num_cbgs_active_at_pois = np.sum(cbg_mean_new_cases_from_poi > 0)

        self.num_cbgs_with_clipped_poi_cases = np.sum(num_cases_from_poi > num_sus)
        self.cbg_new_cases_from_poi = np.clip(num_cases_from_poi, None, num_sus)
        num_sus_remaining = num_sus - self.cbg_new_cases_from_poi

        self.cbg_new_cases_from_base = np.random.binomial(
            num_sus_remaining.astype(int),
            cbg_base_infection_rates)
        self.cbg_new_cases = self.cbg_new_cases_from_poi + self.cbg_new_cases_from_base

        # Keep track of clipping
        self.clipping_monitor['num_base_infection_rates_clipped'].append(self.num_base_infection_rates_clipped)
        self.clipping_monitor['num_active_pois'].append(self.num_active_pois)
        self.clipping_monitor['num_poi_infection_rates_clipped'].append(self.num_poi_infection_rates_clipped)
        self.clipping_monitor['num_cbgs_active_at_pois'].append(self.num_cbgs_active_at_pois)
        self.clipping_monitor['num_cbgs_with_clipped_poi_cases'].append(self.num_cbgs_with_clipped_poi_cases)
        assert (self.cbg_new_cases <= num_sus).all()

