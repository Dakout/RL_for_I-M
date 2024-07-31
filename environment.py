import numpy as np
import scipy as sp
from scipy import stats
import parameters as pars


# function which generates initial sample vectors for D and K
def draw_initial_samples(p, n):
    # get initial deterioration and rate from inital distributions
    d0 = sp.stats.norm.rvs(loc=p.MU_D_0, scale=p.SIGMA_D_0, size=n)
    k0 = sp.stats.norm.rvs(loc=p.MU_K_0, scale=p.SIGMA_K_0, size=n)
    return d0, k0


# function which generates the sample vectors for D and K after taken actions &
# adds the corresponding costs for each taken action
def draw_next(d, k, a, p, t):
    r = np.zeros(d.shape)

    # action 0: do nothing -> K sample gets added to each corresponding D sample, K and R remain unchanged
    d[a == 0] += k[a == 0]

    # action 1: slow down deterioration -> subtract delta K from each D and K sample
    d[a == 1] += k[a == 1] - p.DELTA_K
    k[a == 1] -= p.DELTA_K
    r[a == 1] += p.C_1

    # action 2: improve state -> subtract delta D from each D sample, K remains unchanged
    d[a == 2] += k[a == 2] - p.DELTA_D
    r[a == 2] += p.C_2

    # action 3:
    if len(d[a==3]) > 0:
        a3_vec = sp.stats.multivariate_normal.rvs(
                 [p.MU_K_0, p.MU_D_0+p.MU_K_0],
                 [[p.SIGMA_K_t_PRIOR[t+1]**2, p.RHO_t_PRIOR[t+1]*p.SIGMA_K_t_PRIOR[t+1]*p.SIGMA_D_t_PRIOR[t+1]],
                  [p.RHO_t_PRIOR[t+1]*p.SIGMA_K_t_PRIOR[t+1]*p.SIGMA_D_t_PRIOR[t+1], p.SIGMA_D_t_PRIOR[t+1]**2]],
                 size=len(d[a==3])
                 )

        if a3_vec.shape == (2,):
            k[a == 3] = a3_vec[0]
            d[a == 3] = a3_vec[1]
        else:
            k[a == 3] = a3_vec[:,0]
            d[a == 3] = a3_vec[:,1]
        r[a == 3] += p.C_3

    r_disc = r*(p.GAMMA**t)

    return d, k, r, r_disc

# function which returns the cost of failure
def failure_cost(d, p, t):
    r = np.zeros(d.shape)
    # add occured failure cost to each sample
    r[d > 0] += p.C_F
    # get discounted reward for current time step
    r_disc = r*(p.GAMMA**t)
    return r, r_disc


# function that generates observation samples based on the current state of the system samples
def get_obs(d_samples, p):
    return sp.stats.norm.rvs(loc=d_samples, scale=p.SIGMA_E)


# function which updates the means of vectorized D & K after an action vector has been taken
def action_prior_update(action, post_mu_D_prev, post_mu_K_prev, p):
    if action == 0:
        prior_mu_D_next = post_mu_D_prev + post_mu_K_prev
        prior_mu_K_next = post_mu_K_prev
    elif action == 1:
        prior_mu_D_next = post_mu_D_prev + post_mu_K_prev - p.DELTA_K
        prior_mu_K_next = post_mu_K_prev - p.DELTA_K
    elif action == 2:
        prior_mu_D_next = post_mu_D_prev + post_mu_K_prev - p.DELTA_D
        prior_mu_K_next = post_mu_K_prev
    elif action == 3:
        prior_mu_D_next = p.MU_D_0 + p.MU_K_0
        prior_mu_K_next = p.MU_K_0

    return prior_mu_D_next, prior_mu_K_next


# function which updates the means of D & K after an action has been taken
def action_prior_update_batch(action, post_mu_D_prev, post_mu_K_prev, p):
    assert len(action) == len(post_mu_D_prev) == len(post_mu_K_prev)

    action_0 = (action == 0)
    action_1 = (action == 1)
    action_2 = (action == 2)
    action_3 = (action == 3)

    # preallocate
    prior_mu_D_next = np.full(shape=len(action), fill_value=np.inf)
    prior_mu_K_next = np.full(shape=len(action), fill_value=np.inf)

    prior_mu_D_next[action_0] = post_mu_D_prev[action_0] + post_mu_K_prev[action_0]
    prior_mu_K_next[action_0] = post_mu_K_prev[action_0]

    prior_mu_D_next[action_1] = post_mu_D_prev[action_1] + post_mu_K_prev[action_1] - p.DELTA_K
    prior_mu_K_next[action_1] = post_mu_K_prev[action_1] - p.DELTA_K

    prior_mu_D_next[action_2] = post_mu_D_prev[action_2] + post_mu_K_prev[action_2] - p.DELTA_D
    prior_mu_K_next[action_2] = post_mu_K_prev[action_2]

    prior_mu_D_next[action_3] = p.MU_D_0 + p.MU_K_0
    prior_mu_K_next[action_3] = p.MU_K_0

    if (prior_mu_D_next == np.inf).any() or (prior_mu_K_next == np.inf).any():
        raise RuntimeError('Belief not updated!')

    return prior_mu_D_next, prior_mu_K_next


# function that updates the means of D & K after an observation
def observation_posterior_update(observation, prior_mu_D_t, prior_mu_K_t, p, t):
    posterior_mu_D_t = observation*(p.SIGMA_D_t_POST[t]/p.SIGMA_E)**2 + \
                       prior_mu_D_t*(p.SIGMA_D_t_POST[t]/p.SIGMA_D_t_PRIOR[t])**2

    posterior_mu_K_t = (observation - prior_mu_D_t)/(p.SIGMA_E**2 + p.SIGMA_D_t_PRIOR[t]**2)* \
                       (p.RHO_t_PRIOR[t]*p.SIGMA_D_t_PRIOR[t]*p.SIGMA_K_t_PRIOR[t]) + \
                       prior_mu_K_t
    return posterior_mu_D_t, posterior_mu_K_t


# function which returns to functions which generate samples of d & k according
# to the current belief which incoroporates previous actions and observations
def observation_belief_generator_independent(post_mu_d, post_mu_k, p, t):
    mu_d_belief_gen = lambda n: sp.stats.norm.rvs(loc=post_mu_d, scale=p.SIGMA_D_t_POST[t], size=n)
    mu_k_belief_gen = lambda n: sp.stats.norm.rvs(loc=post_mu_k, scale=p.SIGMA_K_t_POST[t], size=n)
    return mu_d_belief_gen, mu_k_belief_gen


# function which returns a function which generate samples of d & k according
# to the current belief which incoroporates previous actions and observations
def observation_belief_generator(post_mu_d, post_mu_k, p, t):
    belief_sample_gen = lambda n: sp.stats.multivariate_normal.rvs(mean=[post_mu_d, post_mu_k],
                                                                   cov=[[p.SIGMA_D_t_POST[t], p.RHO_t_POST[t]],
                                                                        [p.RHO_t_POST[t], p.SIGMA_K_t_POST[t]]],
                                                                   size=n)
    return belief_sample_gen


# function which calculates the mean, std, and cost with MC simulation with inputs:
# N: number of samples, action: action to perform at each timestep, error: list of observation errors
def benchmark_calc(N, action, error=0.1):
    if action not in [0,1,2,3]:
        print("Action not implemented, please select 0, 1, 2 or 3")
        return

    # calculate the mean, std and LCC for action 1
    m_d = list()
    m_k = list()
    std_d = list()
    m_c = list()

    for e in error:
        P = pars.PARS(e)
        D, K = draw_initial_samples(P, int(N))
        a = np.ones(len(D))*action

        m_d.append(np.mean(D))
        m_k.append(np.mean(K))
        std_d.append(np.std(D))
        C = np.zeros(len(D))
        # check if bridge is already broken in the beginning
        _, R_Disc = failure_cost(D, P, 0)
        C += R_Disc
        # first timestep: do nothing
        D, _, _, R_Disc = draw_next(D, K, np.zeros(len(D)), P, t=0)
        C += R_Disc

        m_d.append(np.mean(D))
        m_k.append(np.mean(K))
        std_d.append(np.std(D))
        m_c.append(np.mean(C))

        # from 1 to 20
        for T in range(1, P.T_END):
            _, R_Disc = failure_cost(D, P, T)
            C += R_Disc
            D, K, _, R_Disc = draw_next(D, K, a, P, T)
            C += R_Disc

            m_d.append(np.mean(D))
            m_k.append(np.mean(K))
            std_d.append(np.std(D))
            m_c.append(np.mean(C))


        # last time step: only add possible failure cost and do not do any actions
        _, R_Disc = failure_cost(D, P, P.T_END)
        C += R_Disc
        m_c.append(np.mean(C))

        m_d = np.array(m_d)
        m_k = np.array(m_k)
        m_c = np.array(m_c)
        std_d = np.array(std_d)
        return np.mean(C), m_d, std_d, m_k, m_c
