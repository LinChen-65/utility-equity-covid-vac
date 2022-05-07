# Strategic COVID-19 vaccine distribution can simultaneously elevate social utility and equity

## System environments
Operating system: Ubuntu 16.04.7 LTS

python packages: numpy==1.21.2 matplotlib==3.4.3 datetime==4.3 pandas==1.1.5 scipy==1.5.4  statsmodels==0.12.1 scikit-learn==0.23.2 hyperopt==0.2.7 bayesian-optimization==1.2.0 adjustText==0.7.3

## Installation guide
Download and install Anaconda (e.g., through command line): 
```
wget -c https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
./Anaconda3-2021.05-Linux-x86_64.sh
```

#### Method 1: Use the provided .yml environment file
Set up the pre-configured virtual conda environment: `conda env create -f ./covid-utility-equity.yml`

Activate the environment: `conda activate covid-utility-equity`

#### Method 2: Create an environment yourself
```
conda create -n covid python==3.7
pip install numpy pandas matplotlib setproctitle datetime scipy statsmodels scikit-learn hyperopt bayesian-optimization adjustText
```

## Dataset
We put the datasets in a zenodo repo (https://sandbox.zenodo.org/record/1058859). 
**Please download them from this link and unzip them into 'data' and 'results' folders in the current directory, respectively.**

Sources of raw data:
- COVID-19 daily death data are available at the New York Times (https://github.com/nytimes/covid-19-data). 
- Mobile phone mobility data and demographic data for census block groups are available at SafeGraph (https://www.safegraph.com/academics). Specifically, Safegraph demographic data can be freely downloaded (https://docs.safegraph.com/docs/open-census-data). When you download the data, remember to change the parameter '--safegraph_root' in the code.
- Estimated mobility networks (ending with '*.pkl') should be retrieved from http://covid-mobility.stanford.edu, so we do not include it in the repository.
- Social vulnerability indices for communities are available at the website of U.S. Agency for Toxic Substances and Disease Registry (https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html).
- Vaccination data are available at the website of U.S. CDC (https://covid.cdc.gov/covid-data-tracker/##vaccination-demographic).

## Running the code
Note: Parameters should be specified to generate corrsponding results. Example values are provided in the python files. The following code uses Atlanta as an example. If you would like to test other MSAs, you can change specify the name via the '--msa_name' argument.

#### 1. Fit epidemic curves (Fig.1, Supplementary Fig.1)
```
python grid_search_parameters.py --msa_name Atlanta
python adjust_scaling_factors.py --msa_name Atlanta
# Obtain simulation results on BD model and meta-population model
python simulation_on_disease_model.py --msa_name Atlanta
# Determine upper and lower bounds
python get_upper_lower_bound_of_models_wider.py --msa_name Atlanta --direction lower --tolerance 1.5
python get_upper_lower_bound_of_models_wider.py --msa_name Atlanta --direction upper --tolerance 1.5
# Simulate with standard SEIR models, used in Fig. 1(b)(c)(d)
python standard_seir.py --msa_name Atlanta --save_result
# Fig. 1(b)(c), Supplementary Fig.1
python plot_curves.py
```

Expected outcomes:

<img src='https://github.com/LinChen-65/utility-equity-covid-vac/blob/main/figures/%5Bplot_curves%5D_result_13.png' height=480>

```
# Fig. 1(d)
python plot_groupwise_death_rate.py
# Note: To generate this figure, you need to first run the code file named 'get_upper_lower_bound_of_models_wider.py', as mentioned above.
```

Expected outcomes:

<img src='https://github.com/LinChen-65/utility-equity-covid-vac/blob/main/figures/%5Bplot_groupwise_death_rate%5D_result_horizontal.png' height=150>

```
# Fig. 1(e)
python plot_corr_with_mobility.py
# Note: To generate this figure, you need to first run the code file named 'generate_infect_same_diff.py', which is illustrated in Part 4.
```

Expected outcomes:

<img src='https://github.com/LinChen-65/utility-equity-covid-vac/blob/main/figures/%5Bplot_corr_with_mobility%5D_result.png' height=150>

#### 2. Correlation analysis of demographic features (Supplementary Fig.2)
```
python plot_correlation_demo_feats.py
```

Expected outcomes:

<img src='https://github.com/LinChen-65/utility-equity-covid-vac/blob/main/figures/%5Bplot_correlation_demo_feats%5D_result.png' height=300>


#### 3. Simulate vaccine distribution strategies (Fig.2)
```
python vaccination_singledemo.py --msa_name Atlanta
python make_gini_table.py --msa_name Atlanta --vaccination_time 31 --vaccination_ratio 0.1 --rel_to Baseline
# Fig. 2(a)
python plot_singledemo_outcome.py
```

Expected outcomes:

<img src='https://github.com/LinChen-65/utility-equity-covid-vac/blob/main/figures/%5Bplot_singledemo_outcome%5D_result.png' height=150>

```
# Fig. 2(b)
python plot_hesitancy_scenarios.py
```

Expected outcomes:

<img src='https://github.com/LinChen-65/utility-equity-covid-vac/blob/main/figures/%5Bplot_hesitancy_scenarios%5D_result.png' height=150>

```
# Perform double-sided t tests
python hypothesis_test.py --msa_name Atlanta
# Simulate middle policies (Supplementary Table 1)
python vaccination_singledemo_middle_policies.py --msa_name Atlanta
```

#### 4. Calculate community risk and societal risk (Fig.3(d))
```
# Calculate the susceptible-infectious ratio
python get_s_i_ratio_at_vaccination_moment.py
# Estimate the infection risk for people in the same CBG/in other CBGs
python generate_infect_same_diff.py --msa_name Atlanta
# Calculate the correlation between community risk and societal risk, Fig. 3(d)
python plot_corr_cr_sr.py
```

Expected outcomes:

<img src='https://github.com/LinChen-65/utility-equity-covid-vac/blob/main/figures/%5Bplot_corr_cr_sr%5D_result.png' height=150>

#### 5. Regression analysis with/without community risk and societal risk (Fig.3)
```
# Generate random bags of vaccination results
python vaccination_randombag.py --msa_name Atlanta --random_seed 66
# Regression, Fig. 3(b)(c)
python plot_regression_randombag_sample_average.py --msa_name all
```

Expected outcomes:

<img src='https://github.com/LinChen-65/utility-equity-covid-vac/blob/main/figures/%5Bplot_regression_randombag_sample_average%5D_result.png' height=150>

#### 6. All-round vaccination strategies (Fig.4, Supplementary Fig.18)
```
python plot_comprehensive_utility_equity.py --with_supplementary
```

Expected outcomes (example):

<img src='https://github.com/LinChen-65/utility-equity-covid-vac/blob/main/figures/%5Bplot_comprehensive_utility_equity%5D_result.png' height=300>

```
# Supplementary Fig.18
python plot_curves.py --with_vac
```

Expected outcomes: 

<img src='https://github.com/LinChen-65/utility-equity-covid-vac/blob/main/figures/%5Bplot_curves%5D_result_2.png' height=300>
