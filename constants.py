NUM_AGE_GROUP_FOR_ATTACK_RATES = 9
NUM_AGE_GROUP_FOR_DEATH_RATES = 17

DETAILED_AGE_LIST =['Under 5 Years','5 To 9 Years','10 To 14 Years','15 To 17 Years','18 To 19 Years','20 Years','21 Years',
                    '22 To 24 Years','25 To 29 Years','30 To 34 Years','35 To 39 Years','40 To 44 Years','45 To 49 Years',
                    '50 To 54 Years','55 To 59 Years', '60 To 61 Years','62 To 64 Years','65 To 66 Years','67 To 69 Years',
                    '70 To 74 Years','75 To 79 Years','80 To 84 Years','85 Years And Over']


AGE_GROUPS_FOR_ATTACK_RATES = {
                                0:['Under 5 Years','5 To 9 Years'],
                                1:['10 To 14 Years','15 To 17 Years','18 To 19 Years'],
                                2:['20 Years','21 Years', '22 To 24 Years','25 To 29 Years'],
                                3:['30 To 34 Years','35 To 39 Years'],
                                4:['40 To 44 Years','45 To 49 Years'],
                                5:['50 To 54 Years','55 To 59 Years'],
                                6:['60 To 61 Years','62 To 64 Years','65 To 66 Years','67 To 69 Years'],
                                7:['70 To 74 Years','75 To 79 Years'],
                                8:['80 To 84 Years','85 Years And Over']
                              }

AGE_GROUPS_FOR_DEATH_RATES = {
                                0:['Under 5 Years'],
                                1:['5 To 9 Years'],
                                2:['10 To 14 Years'],
                                3:['15 To 17 Years','18 To 19 Years'],
                                4:['20 Years','21 Years', '22 To 24 Years'],
                                5:['25 To 29 Years'],
                                6:['30 To 34 Years'],
                                7:['35 To 39 Years'],
                                8:['40 To 44 Years'],
                                9:['45 To 49 Years'],
                                10:['50 To 54 Years'],
                                11:['55 To 59 Years'],
                                12:['60 To 61 Years','62 To 64 Years'],
                                13:['65 To 66 Years','67 To 69 Years'],
                                14:['70 To 74 Years'],
                                15:['75 To 79 Years'],
                                16:['80 To 84 Years','85 Years And Over']
                             }
                             

FIPS_CODES_FOR_50_STATES_PLUS_DC = { # https://gist.github.com/wavded/1250983/bf7c1c08f7b1596ca10822baeb8049d7350b0a4b
    "10": "Delaware",
    "11": "Washington, D.C.",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    }     


MSA_NAME_LIST = ['Atlanta','Chicago','Dallas','Houston', 'LosAngeles','Miami','Philadelphia','SanFrancisco','WashingtonDC']
MSA_NAME_FULL_DICT = {
    'Atlanta':'Atlanta_Sandy_Springs_Roswell_GA',
    'Chicago':'Chicago_Naperville_Elgin_IL_IN_WI',
    'Dallas':'Dallas_Fort_Worth_Arlington_TX',
    'Houston':'Houston_The_Woodlands_Sugar_Land_TX',
    'LosAngeles':'Los_Angeles_Long_Beach_Anaheim_CA',
    'Miami':'Miami_Fort_Lauderdale_West_Palm_Beach_FL',
    'Philadelphia':'Philadelphia_Camden_Wilmington_PA_NJ_DE_MD',
    'SanFrancisco':'San_Francisco_Oakland_Hayward_CA',
    'WashingtonDC':'Washington_Arlington_Alexandria_DC_VA_MD_WV'
}

# parameters:[p_sick_at_t0, home_beta, poi_psi]
parameters_dict = {'Atlanta':[2e-4, 0.0037, 2388],
                   'Chicago': [1e-4,0.0063,2076],
                   'Dallas':[2e-4, 0.0063, 1452],
                   'Houston': [5e-4, 0.0037,1139],
                   'LosAngeles': [2e-4,0.0088,1452],
                   'Miami': [5e-4, 0.0012, 1764],
                   'Philadelphia': [0.001, 0.0037, 827],
                   'SanFrancisco': [5e-4, 0.0037, 1139],
                   'WashingtonDC': [5e-5, 0.0037, 2700]}

# fit to daily smooth deaths
death_scale_dict = {'Atlanta':[1.20],
                    'Chicago':[1.30],
                    'Dallas':[1.03],
                    'Houston':[0.83],
                    'LosAngeles':[1.52],
                    'Miami':[0.78],
                    'Philadelphia':[2.08],
                    'SanFrancisco':[0.64],
                    'WashingtonDC':[1.40]
                    }    

# Upper lower bound of fitted scales
upper_lower_death_scales = {1.2: {'Atlanta':[1.01,1.36],
                                    'Chicago':[1.15,1.45],
                                    'Dallas':[0.9,1.15],
                                    'Houston':[0.68,0.97],
                                    'LosAngeles':[1.34,1.68],
                                    'Miami':[0.65,0.93],
                                    'Philadelphia':[1.48,2.63],
                                    'SanFrancisco':[0.54,0.78],
                                    'WashingtonDC':[1.23,1.54]
                                    },
                            1.5: {'Atlanta':[0.9,1.48],
                                    'Chicago':[1.05,1.55],
                                    'Dallas':[0.82,1.24],
                                    'Houston':[0.58,1.07],
                                    'LosAngeles':[1.23,1.79],
                                    'Miami':[0.55,1.03],
                                    'Philadelphia':[1.1,3.02],
                                    'SanFrancisco':[0.45,0.86],
                                    'WashingtonDC':[1.14,1.63]
                                    }
                            }

# Essential Worker Rates in each work type (Ref: JUE)                   
ew_rate_dict = {
    'C24030e4' : 1,
    'C24030e31': 1,
    'C24030e5': 1,  
    'C24030e32': 1,
    'C24030e12': 1,
    'C24030e39': 1,
    'C24030e6': 1,
    'C24030e33': 1,
    'C24030e7': 1,
    'C24030e34': 1,
    'C24030e8': 0.842,
    'C24030e35': 0.842,
    'C24030e9': 0.444,
    'C24030e36': 0.444,
    'C24030e11': 0.821,
    'C24030e38': 0.821,
    'C24030e13': 0.545,
    'C24030e40': 0.545,
    'C24030e15': 1,
    'C24030e42': 1,
    'C24030e16': 0.5,
    'C24030e43': 0.5,
    'C24030e18': 0.778,
    'C24030e45': 0.778,
    'C24030e19': 1,
    'C24030e46': 1,
    'C24030e20': 0.636,
    'C24030e47': 0.636,
    'C24030e22': 0,
    'C24030e49': 0,
    'C24030e23': 1,
    'C24030e50': 1,
    'C24030e25': 0,
    'C24030e52': 0,
    'C24030e26': 0.667,
    'C24030e53': 0.667,
    'C24030e27': 0.643,
    'C24030e54': 0.643
}