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
1. **Fit epidemic curves (Fig.1)**
```
python xxx.py
```

2. **Correlation analysis of demographic features (Fig.1)**
```
python xxx.py
```
3. **Simulate vaccine distribution strategies (Fig.2)**
```
python xxx.py
```
4. **Calculate community risk and societal harm (Fig.3)**
```
python xxx.py
```
5. **Regression analysis with/without community risk and societal harm (Fig.3)**
```
python xxx.py
```
6. **Construct all-round vaccination strategies (Fig.4)**
```
python xxx.py
```
