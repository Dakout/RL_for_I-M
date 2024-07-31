import numpy as np
import scipy as sp
from scipy import stats

class PARS:

    # cost parameters for actions
    C_0 = 0
    C_1 = 1
    C_2 = 5
    C_3 = 100
    C_F = 150
    R = 0.02
    GAMMA = 1 / (1+R)
    D_CR = 0

    # initial model parameters in the log space
    SIGMA_D_0 = 20.85
    SIGMA_K_0 = 1
    MU_D_0 = -132.64
    MU_K_0 = 6.4
    DELTA_D = 10.5
    DELTA_K = 0.2


    T_END = 21

    # evolution of prior and posterior model parameters
    SIGMA_K_t_PRIOR = np.full(shape=T_END+1, fill_value=SIGMA_K_0)
    SIGMA_K_t_POST  = np.zeros(T_END+1)
    SIGMA_D_t_PRIOR = np.zeros(T_END+1)
    SIGMA_D_t_POST  = np.zeros(T_END+1)
    RHO_t_PRIOR     = np.zeros(T_END+1)
    RHO_t_POST      = np.zeros(T_END+1)

    # at timestep 1: first observation -> posterior values at 0 equal to prior
    SIGMA_K_t_POST[0]  = SIGMA_K_0
    SIGMA_D_t_PRIOR[0] = SIGMA_D_0
    SIGMA_D_t_POST[0]  = SIGMA_D_0


    def __init__(self, SIGMA_E):
        self.SIGMA_E = SIGMA_E
        for k in range(1, self.T_END+1):
            self.SIGMA_D_t_PRIOR[k] = np.sqrt(self.SIGMA_K_t_POST[k-1]**2 + self.SIGMA_D_t_POST[k-1]**2 +
                                              2*self.RHO_t_POST[k-1]*self.SIGMA_K_t_POST[k-1]*self.SIGMA_D_t_POST[k-1])
            self.SIGMA_D_t_POST[k]  = self.SIGMA_E*self.SIGMA_D_t_PRIOR[k] / np.sqrt(self.SIGMA_E**2 + self.SIGMA_D_t_PRIOR[k]**2)
            self.RHO_t_PRIOR[k]     = (self.RHO_t_POST[k-1]*self.SIGMA_D_t_POST[k-1] + self.SIGMA_K_t_POST[k-1]) / self.SIGMA_D_t_PRIOR[k]
            self.RHO_t_POST[k]      = self.RHO_t_PRIOR[k]*self.SIGMA_E / np.sqrt(self.SIGMA_E**2 + (1-self.RHO_t_PRIOR[k]**2)*self.SIGMA_D_t_PRIOR[k]**2)
            self.SIGMA_K_t_POST[k]  = self.SIGMA_K_t_PRIOR[k]*np.sqrt(self.SIGMA_E**2 + (1-self.RHO_t_PRIOR[k]**2)*self.SIGMA_D_t_PRIOR[k]**2) / \
                                                              np.sqrt(self.SIGMA_E**2 + self.SIGMA_D_t_PRIOR[k]**2)
        
        return


# -------------------------------------------------------------------
""" Additional functions which calculate some analytical solutions"""
# -------------------------------------------------------------------

# function which calculates the analytical LCC for action 0
def A0_ana(e=0.1):
    J = PARS(e) # measurement error does not really matter here

    mean_vec = J.MU_D_0 + np.linspace(0,21,22)*J.MU_K_0
    std_vec = np.sqrt(J.SIGMA_D_0**2 + (np.linspace(0,21,22)*J.SIGMA_K_0)**2)
    prob_vec = sp.stats.norm.cdf(mean_vec/std_vec)
    cost_vec = np.array([J.C_F*J.GAMMA**k for k in range(22)])
    k_mean = np.full(len(mean_vec),J.MU_K_0)

    LCC_A0 = np.sum(prob_vec*cost_vec)

    # cumulative cost
    c_vec_cum = np.zeros(len(cost_vec))
    for k in range(len(cost_vec)):
        c_vec_cum[k] = np.sum((cost_vec*prob_vec)[0:k+1])

    return LCC_A0, mean_vec, std_vec, cost_vec, c_vec_cum, k_mean


# function which calculates the analytical LCC for action 1:
def A1_ana(e=0.1):
    J = PARS(e) # measurement error does not really matter here

    mean_vec1 = J.MU_D_0 + np.linspace(0,21,22)*(J.MU_K_0)
    mean_vec2 = np.insert(np.linspace(1,21,21)*np.linspace(0,20,21) / 2, 0, 0)*J.DELTA_K
    mean_vec = mean_vec1 - mean_vec2
    std_vec = np.sqrt(J.SIGMA_D_0**2 + (np.linspace(0,21,22)*J.SIGMA_K_0)**2)
    prob_vec = sp.stats.norm.cdf(mean_vec/std_vec)
    cost_vec1 = np.array([J.C_F*J.GAMMA**k for k in range(22)])
    cost_vec2 = np.array([0] + [J.C_1*J.GAMMA**k for k in range(1, 21)] + [0])
    cost_vec = prob_vec*cost_vec1 + cost_vec2
    k_mean = J.MU_K_0 - np.insert(np.linspace(0,20,21),0,0)*J.DELTA_K

    LCC_A1 = np.sum(prob_vec*cost_vec1)  + np.sum(cost_vec2)

    # cumulative cost
    c_vec_cum = np.zeros(len(cost_vec1))
    for k in range(len(cost_vec1)):
        c_vec_cum[k] = np.sum(cost_vec[0:k+1])

    return LCC_A1, mean_vec, std_vec, cost_vec, c_vec_cum, k_mean


# function which calculates the analytical LCC for action 2:
def A2_ana(e=0.1):
    J = PARS(e) # measurement error does not really matter here

    mean_vec = J.MU_D_0 + np.linspace(0,21,22)*J.MU_K_0 - np.insert(np.insert(np.linspace(1,20,20), 0, 0),0,0)*J.DELTA_D
    std_vec = np.sqrt(J.SIGMA_D_0**2 + (np.linspace(0,21,22)*J.SIGMA_K_0)**2)
    prob_vec = sp.stats.norm.cdf(mean_vec/std_vec)
    cost_vec1 = np.array([J.C_F*J.GAMMA**k for k in range(22)])
    cost_vec2 = np.array([0] + [J.C_2*J.GAMMA**k for k in range(1, 21)] + [0])
    cost_vec = prob_vec*cost_vec1 + cost_vec2
    k_mean = np.full(len(mean_vec), J.MU_K_0)

    LCC_A2 = np.sum(prob_vec*cost_vec1 + cost_vec2)

    # cumulative cost
    c_vec_cum = np.zeros(len(cost_vec1))
    for k in range(len(cost_vec1)):
        c_vec_cum[k] = np.sum(cost_vec[0:k+1])

    return LCC_A2, mean_vec, std_vec, cost_vec, c_vec_cum, k_mean


# function which analytically calculates a lower bound for the LCC_A3
def A3_ana(e=1e3):
    if e<1e2:
        print("Warning: this analytical solution is based on low correlation between D & K -> we need high sigma_e!!")

    J = PARS(e) # measurement error does not really matter here
    mean_vec = np.array([J.MU_D_0] + [J.MU_D_0+J.MU_K_0]*21)
    std_vec = np.sqrt(J.SIGMA_D_0**2 + (np.linspace(0,21,22)*J.SIGMA_K_0)**2)
    k_mean = np.full(len(mean_vec), J.MU_K_0)

    prob = sp.stats.norm.cdf(mean_vec/std_vec)
    # cost of doing action 3 at timesteps 1 to 20
    # cost of failure at each timestep 0 to 21
    cost_vec1 = J.C_F*J.GAMMA**(np.linspace(0,21,22))
    cost_vec2 = np.array([0] + [J.C_3*J.GAMMA**k for k in range(1, 21)] + [0])
    cost_vec = prob*cost_vec1 + cost_vec2

    LCC_A3 = np.sum(prob*cost_vec1) + np.sum(cost_vec2)

    # cumulative cost
    c_vec_cum = np.zeros(len(cost_vec1))
    for k in range(len(cost_vec1)):
        c_vec_cum[k] = np.sum(cost_vec[0:k+1])

    return LCC_A3, mean_vec, std_vec, cost_vec, c_vec_cum, k_mean
