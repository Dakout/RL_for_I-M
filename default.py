import os
import numpy as np
import parameters as pars
import environment as env

''' environment params '''
MEASUREMENT_ERROR = 0.5

''' network params '''
N_ACTIONS = 4
N_OBS = 1
N_LSTM_LAYERS = 1
LSTM_INPUT_DIM = 50
N_HIDDEN = 80
LRELU_SLOPE = 0.3

''' optimizer params '''
LEARNING_RATE = 1e-3
BETAS = (0.9, 0.999)
OPT_EPS = 1e-8
WEIGHT_DECAY = 0
AMSGRAD = True
MAXIMIZE = False
LR_SCHEDULER = True
LR_STEP_SIZE = 10
LR_GAMMA = 0.8
NORMALIZE_OBS = False
MU_NORM  = pars.PARS.MU_D_0 + pars.PARS.MU_K_0
STD_NORM = (pars.PARS.SIGMA_D_0**2 + pars.PARS.SIGMA_K_0**2)**(1/2)

''' storage params '''
CHECKPOINT_DIR  = os.path.join(os.path.abspath(os.getcwd()), './saved_networks')

''' agent params '''
EPOCHS = int(2e2)
BATCH_SIZE = 20
TARGET_UPDATE = 3
MAX_EPSILON = 0.6
DEC_EPSILON = 1e-4
MIN_EPSILON = 0.01

''' tree params '''
PRIOR_MU_D_1, PRIOR_MU_K_1 = env.action_prior_update(np.zeros(1), pars.PARS.MU_D_0, pars.PARS.MU_K_0, pars.PARS)
INITIAL_COUNT = -1
INITIAL_NODES = {}
N_OBS_BUCKETS = 30
# we define it as the 10 percentile of the initial deterioration distribution
FLOOR_QUANTILE = 0.1
# we define it as the 80 percentile of the final deterioration distribution
# when performing action A0 at every timestep
CEIL_QUANTILE  = 0.8

''' pompc params '''
DEPTH_THRESHOLD = 20
EXPLORATION_C = 1
N_POMCP_RUNS = 100
N_ROLLOUT_RUNS = 3500
USE_UCB = True
