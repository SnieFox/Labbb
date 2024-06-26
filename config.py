# environment parameters
env = 'test'
THREADS = 16

# run parameters
N = 30 if env == 'test' else 100
G = 2500 if env == 'test' else 10000
NR = 10 if env == 'test' else 100

# convergence parameters
EPS = 0.0001
N_LAST_GENS = 10
DELTA = SIGMA = 0.01

# algorithm parameters
get_p_m = lambda l: 0.1 / l / N
get_pop_seed = lambda run_i: 1381*run_i + 5912826

# output parameters
DISTRIBUTIONS_TO_PLOT = 5
RUNS_TO_PLOT = 5
OUTPUT_FOLDER = 'out_test' if env == 'test' else 'out'
RUN_STATS_NAMES = [
    'NI', 'F_found', 'F_avg',
    'RR_min', 'NI_RR_min', 'RR_max', 'NI_RR_max', 'RR_avg',
    'Teta_min', 'NI_Teta_min', 'Teta_max', 'NI_Teta_max', 'Teta_avg',
    'I_min', 'NI_I_min', 'I_max', 'NI_I_max', 'I_avg',
    's_min', 'NI_s_min', 's_max', 'NI_s_max', 's_avg',
    'GR_early', 'GR_late', 'NI_GR_late', 'GR_avg',
    'NI_loose', 'Num_loose', 'nonSuc', 'nonMin_NI', 'nonMax_NI', 'nonAvg_NI', 'nonSigma_NI',
    'nonAvg_F_found', 'nonSigma_F_found', 'nonMax_F_found'
]
EXP_STATS_NAMES = [
    'Suc', 'N_Suc', 'Min_NI', 'Max_NI', 'Avg_NI', 'Sigma_NI',
    
    'Min_RR_min', 'NI_RR_min', 'Max_RR_max', 'NI_RR_max',
    'Avg_RR_min', 'Avg_RR_max', 'Avg_RR_avg',
    'Sigma_RR_min', 'Sigma_RR_max', 'Sigma_RR_avg',

    'Min_Teta_min', 'NI_Teta_min', 'Max_Teta_max', 'NI_Teta_max',
    'Avg_Teta_min', 'Avg_Teta_max', 'Avg_Teta_avg',
    'Sigma_Teta_min', 'Sigma_Teta_max', 'Sigma_Teta_avg',
    
    'Min_I_min', 'NI_I_min', 'Max_I_max', 'NI_I_max',
    'Avg_I_min', 'Avg_I_max', 'Avg_I_avg',
    'Sigma_I_min', 'Sigma_I_max', 'Sigma_I_avg',
    
    'Min_s_min', 'NI_s_min', 'Max_s_max', 'NI_s_max',
    'Avg_s_min', 'Avg_s_max', 'Avg_s_avg',
    
    'Min_GR_early', 'Max_GR_early', 'Avg_GR_early',
    'Min_GR_late', 'Max_GR_late', 'Avg_GR_late',
    'Min_GR_avg', 'Max_GR_avg', 'Avg_GR_avg',

    'nonSuc', 'nonMin_NI', 'nonMax_NI', 'nonAvg_NI', 'nonSigma_NI',
    'nonAvg_F_found', 'nonSigma_F_found', 'nonMax_F_found',
    'Avg_NI_loose', 'Sigma_NI_loose', 'Avg_Num_loose', 'Sigma_Num_loose'
]
FCONSTALL_RUN_STATS_NAMES = [
    'NI',
    'RR_min', 'NI_RR_min', 'RR_max', 'NI_RR_max', 'RR_avg',
    'Teta_min', 'NI_Teta_min', 'Teta_max', 'NI_Teta_max', 'Teta_avg'
]
FCONSTALL_EXP_STATS_NAMES = [
    'Suc', 'N_Suc', 'Min_NI', 'Max_NI', 'Avg_NI', 'Sigma_NI',
    
    'Min_RR_min', 'NI_RR_min', 'Max_RR_max', 'NI_RR_max',
    'Avg_RR_min', 'Avg_RR_max', 'Avg_RR_avg',
    'Sigma_RR_min', 'Sigma_RR_max', 'Sigma_RR_avg',

    'Min_Teta_min', 'NI_Teta_min', 'Max_Teta_max', 'NI_Teta_max',
    'Avg_Teta_min', 'Avg_Teta_max', 'Avg_Teta_avg',
    'Sigma_Teta_min', 'Sigma_Teta_max', 'Sigma_Teta_avg'
]
c_values = {
    100: [0.9801, 0.970299, 0.96059601, 0.95099005],
    200: [0.990025, 0.985074875, 0.980149501, 0.975248753],
    300: [0.993344444, 0.990033296, 0.986733185, 0.983444075],
    400: [0.99500625, 0.992518734, 0.990037438, 0.987562344],
    500: [0.996004, 0.994011992, 0.992023968, 0.99003992],
    1000: [0.998001, 0.997002999, 0.996005996, 0.99500999],
}