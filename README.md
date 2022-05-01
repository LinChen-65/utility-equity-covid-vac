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
- Mobile phone mobility data and demographic data for census block groups are available at SafeGraph (https://www.safegraph.com/academics). Safegraph demographic data can be freely downloaded (https://docs.safegraph.com/docs/open-census-data), so we do not include it in the repository. When you download the data, remember to change the parameter '--safegraph_root' in the code.
- Estimated mobility networks (ending with '*.pkl') should be retrieved from http://covid-mobility.stanford.edu, so we do not include it in the repository.
- Social vulnerability indices for communities are available at the website of U.S. Agency for Toxic Substances and Disease Registry (https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html).
- Vaccination data are available at the website of U.S. CDC (https://covid.cdc.gov/covid-data-tracker/##vaccination-demographic).
- Since the data files and intermediate results needed to test the code are too large, we put it in zenodo (https://sandbox.zenodo.org/record/1056829). Please download them from this link and replace the 'data' and 'results' folder with these downloaded folders.

## Running the code
Note: Parameters should be specified to generate corrsponding results. Example values are provided in the python files.

1. **Fit epidemic curves (Fig.1, Supplementary Fig.1, Supplementary Fig.18)**
```
python grid_search_parameters.py --msa_name Atlanta --quick_test
python adjust_scaling_factors.py --msa_name Atlanta --quick_test
# Simulate with standard SEIR models, used in Fig. 1(b)(c)(d)
python standard_seir.py --msa_name Atlanta
# Fig. 1(b)(c), Supplementary Fig.1
python plot_curves.py
```

Generated figures:

<img src='https://github.com/LinChen-65/utility-equity-covid-vac/blob/main/figures/%5Bplot_curves%5D_result_1.png' height=480>
<img src='https://github.com/LinChen-65/utility-equity-covid-vac/blob/main/figures/%5Bplot_curves%5D_result_2.png' height=450>
<img src='https://github.com/LinChen-65/utility-equity-covid-vac/blob/main/figures/%5Bplot_curves%5D_result_3.png' height=350>

```
# Fig. 1(d)
python plot_groupwise_death_rate.py
```

Generated figures:

<img src='https://github.com/LinChen-65/utility-equity-covid-vac/blob/main/figures/%5Bplot_groupwise_death_rate%5D_result.png' height=280>

```
# Fig. 1(e)
python plot_corr_with_mobility.py
```

Generated figures:


2. **Correlation analysis of demographic features (Supplementary Fig.2)**
```
python plot_correlation_demo_feats.py
```

Generated figures:


3. **Simulate vaccine distribution strategies (Fig.2)**
```
python vaccination_singledemo.py --msa_name Atlanta
python make_gini_table.py --msa_name Atlanta --vaccination_time 31 --vaccination_ratio 0.1 --rel_to Baseline
# Fig. 2(a)
python plot_singledemo_outcome.py
```

Generated figures:

```
# Fig. 2(b)
python hesitancy_scenarios.py
```

Generated figures:

```
# Fig. 2(c)
python hypothesis_test.py --msa_name Atlanta
```

Generated figures:

```
# Simulate middle policies (Supplementary Table 1)
python vaccination_singledemo_middle_policies.py --msa_name Atlanta
```

4. **Calculate community risk and societal harm (Fig.3(d))**
```
# Calculate the susceptible-infectious ratio
python get_s_i_ratio_at_vaccination_moment.py
# Estimate the infection risk for people in the same CBG/in other CBGs
python generate_infect_same_diff.py --msa_name Atlanta
# Calculate the correlation between community risk and societal risk, Fig. 3(d)
python correlation_cr_sr.py
```

Generated figures:

5. **Regression analysis with/without community risk and societal harm (Fig.3)**
```
# Generate random bags of vaccination results
python vaccination_randombag.py --msa_name Atlanta --random_seed 66
# Regression, Fig. 3(b)(c)
python plot_regression_randombag_sample_average.py --msa_name all
```

Generated figures:

6. **Auto-search all-round vaccination strategies (Fig.4)**
```
python vaccination_comprehensive_autosearch.py --msa_name Atlanta --vaccination_time 31 --vaccination_ratio 0.1
python plot_comprehensive_utility_equity.py --with_supplementary
```

Generated figures (example):
