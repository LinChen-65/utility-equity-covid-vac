# Strategic COVID-19 vaccine distribution can simultaneously elevate social utility and equity

## System environments
Operating system: Ubuntu 16.04.7 LTS

python packages: numpy==1.21.2 matplotlib==3.4.3 datetime==4.3 pandas==1.1.5 scipy==1.5.4  statsmodels==0.12.1 scikit-learn==0.23.2

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
pip install numpy pandas matplotlib setproctitle datetime scipy statsmodels scikit-learn
```

## Datasets
- COVID-19 daily death data are available at the New York Times (https://github.com/nytimes/covid-19-data). 
- Mobile phone mobility data and demographic data for census block groups are available at SafeGraph (https://www.safegraph.com/academics).
- Estimated mobility networks are retrieved from http://covid-mobility.stanford.edu.
- Social vulnerability indices for communities are available at the website of U.S. Agency for Toxic Substances and Disease Registry (https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html).
- Vaccination data are available at the website of U.S. CDC (https://covid.cdc.gov/covid-data-tracker/##vaccination-demographic).

## Running the code
Note: Parameters inside square brackets should be specified to generate corrsponding results. Example values are provided in the python files.

1. **Fit epidemic curves (Fig.1)**
```
python grid_search_parameters.py [MSA_NAME] [quick_test] [p_sick_at_t0]
python adjust_scaling_factors.py [MSA_NAM] [quick_test]
# Plot range within 150% of the best RMSE
python get_upper_lower_bound_of_models_wider.py [MSA_NAME] [quick_test] [direction] [tolerance]
```

2. **Correlation analysis of demographic features (Fig.1)**
```
python correlation_demo_feats_new.py [NUM_GROUPS] [colormap]
```
3. **Simulate vaccine distribution strategies (Fig.2)**
```
python vaccination_adaptive_singledemo_svi_hesitancy_test.py [MSA_NAME] [VACCINATION_TIME] [VACCINATION_RATIO] [consider_hesitancy] [ACCEPTANCE_SCENARIO] [quick_test]
# Hypothesis test for the significance of changes
python hypothesis_test_fig2.py [MSA_NAME]
```
4. **Calculate community risk and societal harm (Fig.3)**
```
# Calculate the susceptible-infectious ratio
python get_s_i_ratio_at_vaccination_moment.py [quick_test]
# Estimate the infection risk for people in the same CBG/in other CBGs 
python generate_infect_same_diff.py [MSA_NAME]
# Correlation between community risk and societal harm
python correlation_cm_sh.py [NUM_GROUPS] [colormap]
```
5. **Regression analysis with/without community risk and societal harm (Fig.3)**
```
# Generate random bags of vaccination results
python vaccination_randombag_newdamage_gini.py [MSA_NAME] [RANDOM_SEED] [quick_test]
# Regression
python regression_randombag_sample.py [MSA_NAME] [LEN_SEEDS] [NUM_SAMPLE] [SAMPLE_FRAC]
```
6. **Auto-search all-round vaccination strategies (Fig.4)**
```
python vaccination_adaptive_hybrid_autosearch_conform.py [MSA_NAME] [VACCINATION_TIME] [VACCINATION_RATIO] [consider_hesitancy] [ACCEPTANCE_SCENARIO] [w1] [w2] [w3] [w4] [w5] [quick_test] 
```
