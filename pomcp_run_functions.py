import numpy as np
import scipy as sp
from scipy import stats
import os
import time
from tqdm.notebook import tqdm_notebook
import pathos.multiprocessing as pmp
from pathos.multiprocessing import ProcessPool as Pool
import copy

from pomcp import POMCP
import parameters as pars
import environment as env
import visualization as vis


"""
##############################################################################
############################ Simulation functions ############################
##############################################################################
"""

# function which computes a single trajectory with the monte carlo tree method
def single_run(P, pomcp_params: dict):
    # list of actions taken throughout the search
    action_list = [0]
    
    # draw intial samples
    d, k = env.draw_initial_samples(p=P, n=1)
    c = np.zeros(len(d))

    # check for failure
    _, r_disc = env.failure_cost(d, p=P, t=0)
    c += r_disc

    # perform action 0 and draw next state
    d, k, _, r_disc = env.draw_next(d, k, a=np.zeros(len(d)), p=P, t=0)
    c += r_disc

    # Step 0a: get initial observation
    o = env.get_obs(d, P)

    # Step 0b: initialize Pomcp, where inital beliefs are set
    A = POMCP(P, initial_observation=o, **pomcp_params)

    # Calculate policy in a loop
    for T in range(1, 21):
        # Step 1: perform monte carlo search for best next action from current observation
        next_a = A.Search(T)
        action_list.append(next_a.tolist())

        # Step 2: check for failure
        _, r_disc = env.failure_cost(d=d, p=P, t=T)
        c += r_disc

        # Step 2: Apply found next action, generate next d & k states and collect discounted action reward
        d, k, _, r_disc = env.draw_next(d=d, k=k, a=next_a, p=P, t=T)
        c += r_disc

        # at last time step (when t=20, draw_next yields d at t=21), no observation, pruning and belief updating necessary
        if T < 20: 
            # Step 3: get next observation from next d state
            o = env.get_obs(d, P)

            # Step 4: prune the tree after the taken action and subsequent observation made
            A.prune_after_action(next_a, o)

            # Step 5: update the belief state based on the action taken and the observation made
            A.update_belief(next_a, o, t=T)

    # T=T_END: only add possible failure cost and do not do any actions
    _, r_disc = env.failure_cost(d=d, p=P, t=P.T_END)
    c += r_disc
    
    assert len(action_list) == 21
    return c, np.asarray(action_list)


def single_run_parallel_initializer(P, pomcp_params: dict, n: int):
    d_vec = list()
    k_vec = list()
    c_vec = list()
    A_vec = list()
    
    for i in range(n):
        # draw intial samples
        d, k = env.draw_initial_samples(p=P, n=1)
        c = np.zeros(len(d))
    
        # check for failure
        _, r_disc = env.failure_cost(d, p=P, t=0)
        c += r_disc
    
        # perform action 0 and draw next state
        d, k, _, r_disc = env.draw_next(d, k, a=np.zeros(len(d)), p=P, t=0)
        c += r_disc
    
        # Step 0a: get initial observation
        o = env.get_obs(d, P)
    
        # Step 0b: initialize Pomcp, where inital beliefs are set
        A = POMCP(P, initial_observation=o, **pomcp_params)
        
        d_vec.append(d)
        k_vec.append(k)
        c_vec.append(c)
        A_vec.append(A)
        
        assert len(d_vec) == len(k_vec) == len(c_vec) == len(A_vec)

    return d_vec, k_vec, c_vec, A_vec



# function which computes a single trajectory with the monte carlo tree method
def single_run_parallel(d, k, c, A, P):
    # list of actions taken throughout the search
    action_list = [0]

    # Calculate policy in a loop
    for T in range(1, 21):
        # Step 1: perform monte carlo search for best next action from current observation
        next_a = A.Search(T)
        action_list.append(next_a.tolist())

        # Step 2: check for failure
        _, r_disc = env.failure_cost(d=d, p=P, t=T)
        c += r_disc

        # Step 2: Apply found next action, generate next d & k states and collect discounted action reward
        d, k, _, r_disc = env.draw_next(d=d, k=k, a=next_a, p=P, t=T)
        c += r_disc

        # at last time step (when t=20, draw_next yields d at t=21), no observation, pruning and belief updating necessary
        if T < 20: 
            # Step 3: get next observation from next d state
            o = env.get_obs(d, P)

            # Step 4: prune the tree after the taken action and subsequent observation made
            A.prune_after_action(next_a, o)

            # Step 5: update the belief state based on the action taken and the observation made
            A.update_belief(next_a, o, t=T)

    # T=T_END: only add possible failure cost and do not do any actions
    _, r_disc = env.failure_cost(d, P, P.T_END)
    c += r_disc
    
    assert len(action_list) == 21
    return c, np.asarray(action_list)


# function which summarizes the parallel initializer and single_run_parallel 
# -> final function which yields a cost vector with desired length: n and 
# and an action matrix with shape: nx21
def parallel_runner(P, pomcp_params, n, rounds_chooser=True):  
    costs = np.zeros(n)
    actions = np.zeros((n, P.T_END), dtype=np.int8)
    
    
    n_cpus = pmp.cpu_count()
    print(f"\nParallel computing with {n_cpus} cpus\n")
    pool = Pool(n_cpus)
    
    if rounds_chooser:
    
        rounds = int(np.ceil(n / n_cpus))

        assert n % n_cpus == 0 

        for k in tqdm_notebook(range(rounds)):

            ds, ks, cs, As = single_run_parallel_initializer(P, pomcp_params, n_cpus)

            try:
                result = pool.amap(single_run_parallel, ds, ks, cs, As, [copy.deepcopy(P) for _ in range(n_cpus)])
            except ValueError:
                print("Restarting pool")
                pool.restart()
                print("Restarting done")
                result = pool.amap(single_run_parallel, ds, ks, cs, As, [copy.deepcopy(P) for _ in range(n_cpus)])

            if result._success:
                r = result.get()
                costs[k*n_cpus:(k+1)*n_cpus] = np.array([x[0] for x in r]).flatten()
                actions[k*n_cpus:(k+1)*n_cpus, :] = np.array([x[1] for x in r])
            else:
                print("\nParallel run was not succesfull!")
    
    else:
        print('Here in the non-rounds programm')
        ds, ks, cs, As = single_run_parallel_initializer(P, pomcp_params, n)
        #print(ds, ks, cs, As, [id(_) for _ in As])
        try:
            result = pool.amap(single_run_parallel, ds, ks, cs, As, [copy.deepcopy(P) for _ in range(n)])
        except ValueError:
            print("Restarting pool")
            pool.restart()
            print("Restarting done")
            result = pool.amap(single_run_parallel, ds, ks, cs, As, [copy.deepcopy(P) for _ in range(n)])

        if result._success:
            r = result.get()
            costs = np.array([x[0] for x in r]).flatten()
            actions = np.array([x[1] for x in r])
        else:
            print("\nParallel run was not succesfull!")
         
    pool.close()
    pool.join()
    pool.clear()
    
    return costs, actions


"""
##############################################################################
############################## Timing functions ##############################
##############################################################################
"""

# function which runs the single_run function n times and returns a vector
# times it took to execute the program
def time_single_run(n, P, pomcp_params: dict):
    time_vec = np.zeros(n)
    LCC_vec = np.zeros(n)
    
    for k in tqdm_notebook(range(n)):
        time_before = time.time()
        cost, action_list = single_run(P, pomcp_params)
        time_vec[k] = time.time() - time_before
        LCC_vec[k] = cost
    
    # check that all values have been filled
    assert not (time_vec == np.zeros(n)).any()
    assert not (LCC_vec == np.zeros(n)).any()
    
    return time_vec, LCC_vec


# function which runs the parallel_run function n times and returns a vector
# times it took to execute the program
def time_parallel_runs(n, P, pomcp_params):
    time_vec = np.zeros(n)
    LCC_vec = np.zeros(n)
    
    for k in tqdm_notebook(range(n)):
        time_before = time.time()
        cost, action_list = single_run(P, pomcp_params)
        time_vec[k] = time.time() - time_before
        LCC_vec[k] = cost
    
    # check that all values have been filled
    assert not (time_vec == np.zeros(n)).any()
    assert not (LCC_vec == np.zeros(n)).any()
    
    return time_vec, LCC_vec


# function which tests the rollout function for different values of rollout #
def rollout_timer(rollout_run_vec, n, P, pomcp_runs, plot=False):

    print(f"\nTesting the rollout run time {n} times!\n")
    mean_roll_time = np.zeros(len(rollout_run_vec))
    cov_roll_time = np.zeros(len(rollout_run_vec))
    mean_tree_time = np.zeros(len(rollout_run_vec))
    cov_tree_time = np.zeros(len(rollout_run_vec))
    
    for i, r in enumerate(rollout_run_vec):
        print(f"Run number: {r}\n")
        #roll_times = np.zeros(n)
        tree_times = np.zeros(n)
        
        pomcp_params = {'floor_quantile': 0.1,
                        'ceil_quantile': 0.8,
                        'n_obs_buckets': 30,
                        'n_pomcp_runs': pomcp_runs,
                        'n_rollout_runs': r}

        # initialize it with whatever observation, result not important, only execution time
        A = POMCP(P, initial_observation=-140, **pomcp_params)
        
        # time the whole tree which incorporates the rollout function
        for k in tqdm_notebook(range(n)):
            ti = time.time()
            _, _ = single_run(P, pomcp_params)
            ta = time.time() - ti
            tree_times[k] = ta
            
        mean_roll_time = None
        cov_roll_time = None
        mean_tree_time[i] = np.mean(tree_times)
        cov_tree_time[i]  = np.std(tree_times) / np.mean(tree_times)
    
    
    if plot:
        vis.time_rollout_run_plot(rollout_run_vec, mean_roll_time, cov_roll_time, 
                                  mean_tree_time, cov_tree_time, save=False)
    
    return mean_roll_time, cov_roll_time, mean_tree_time, cov_tree_time


# function which tests the tree time for different pomcp #
def pomcp_timer(pomcp_run_vec, n, P, rollout_runs, plot=False):
    
    print(f"\nTesting the pomcp run time {n} times!\n")
    mean_pomcp_time = np.zeros(len(pomcp_run_vec))
    cov_pomcp_time = np.zeros(len(pomcp_run_vec))
    
    for i, r in enumerate(pomcp_run_vec):
        print(f"Run number: {r}\n")
        t_vec = np.zeros(n)
        
        pomcp_params = {'floor_quantile': 0.1,
                        'ceil_quantile': 0.8,
                        'n_obs_buckets': 30,
                        'n_pomcp_runs': r,
                        'n_rollout_runs': rollout_runs}
        
        # time how long it takes for the tree to traverse a trajectory for a given pomcp depth
        for k in tqdm_notebook(range(n)):
            ti = time.time()
            _, _ = single_run(P, pomcp_params)
            ta = time.time() - ti
            t_vec[k] = ta
            
        mean_pomcp_time[i] = np.mean(t_vec)
        cov_pomcp_time[i] = np.std(t_vec) / np.mean(t_vec)

    if plot:
        vis.time_pomcp_run_plot(pomcp_run_vec, mean_pomcp_time, cov_pomcp_time)
    
    return mean_pomcp_time, cov_pomcp_time


"""
##############################################################################
############################### Cost functions ###############################
##############################################################################
"""

# function which tests the tree LCC for different pomcp #
def pomcp_runs_vs_LCC_parallel(pomcp_run_vec, n, P, n_rollouts, rounds_chooser=True, plot=False):
    
    print(f"\nTesting the pomcp run time {n} times!\n")
    mean_LCC = np.zeros(len(pomcp_run_vec))
    cov_LCC = np.zeros(len(pomcp_run_vec))
    
    for i, r in enumerate(pomcp_run_vec):
        print(f"Run number: {r}\n")
        
        pomcp_params = {'n_pomcp_runs': r,
                        'n_rollout_runs': n_rollouts}
        
        costs, _ = parallel_runner(P, pomcp_params, n, rounds_chooser)
            
        mean_LCC[i] = np.mean(costs)
        cov_LCC[i]  = np.std(costs) / np.mean(costs)
    
    return mean_LCC, cov_LCC


# function which tests the tree LCC for different rollout #
def rollout_runs_vs_LCC_parallel(rollout_run_vec, n, P, n_pomcp, rounds_chooser=True, plot=False):
    
    print(f"\nTesting the rollout run time {n} times!\n")
    mean_LCC = np.zeros(len(rollout_run_vec))
    cov_LCC = np.zeros(len(rollout_run_vec))
    
    for i, r in enumerate(rollout_run_vec):
        print(f"Run number: {r}\n")
        
        pomcp_params = {'n_pomcp_runs': n_pomcp,
                        'n_rollout_runs': r}
        
        costs, _ = parallel_runner(P, pomcp_params, n, rounds_chooser)
            
        mean_LCC[i] = np.mean(costs)
        cov_LCC[i]  = np.std(costs) / np.mean(costs)
    
    return mean_LCC, cov_LCC


# function which computes multiple trajectories of the monte carlo tree search and yields:
# - the mean LCC of these runs
# - the distribution of actions taken at each time step
# - evolution plot of the action distribution
def mean_LCC_and_action_statistics(P, pomcp_params: dict, n=100, plot=True, parallel=False):
    distr = np.zeros((P.T_END,4))
    
    if parallel:
        costs, actions = parallel_runner(P, pomcp_params, n)
        
        for act in actions:
            distr[np.arange(act.size), act] += 1
    
    else:
        costs = np.zeros(n)
        for k in tqdm_notebook(range(n)):
            costs, actions = single_run(P, pomcp_params)
            distr[np.arange(actions.size), actions] += 1
        
    # check if everything went right
    assert (np.sum(distr, axis=1) == n).all()

    # normalize distribution
    distr = distr / n
    if plot:
        vis.action_statistics_visualization(distr)
    
    return costs, distr


# function which evaluates the mean LCC for various amounts of discretization
# buckets of the observation in order to find the optimal number of buckets
# for a given number of pomcp and rollout runs
def num_buckets_vs_LCC_parallel(bucket_vec, P, params, n, plot=False):
    mean_LCCs = np.zeros(len(bucket_vec))
    cov_LCCs  = np.zeros(len(bucket_vec))
    
    for k, b in enumerate(bucket_vec):
        params['n_obs_buckets'] = b
        
        costs, _ = parallel_runner(P, params, n)
        
        mean_LCCs[k] = np.mean(costs)
        cov_LCCs[k]  = np.std(costs) / np.mean(costs)
        
    assert not (mean_LCCs == np.zeros(len(bucket_vec))).any()
    assert not (cov_LCCs == np.zeros(len(bucket_vec))).any()
    
    if plot:
        vis.num_buckets_vs_LCC_plot(bucket_vec, mean_LCCs, cov_LCCs, save=False)
    
    return mean_LCCs, cov_LCCs


"""
##############################################################################
############################## Belief functions ##############################
##############################################################################
"""

# function which computes the bucket bounds with quantile inputs
# d bounds are calculated with the quantile values for taking A0 at every timestep
# k bounds are calculated with the quantile values for taking A1 at every timestep
def get_bounds_by_quantiles(d_floor_quantile: float, d_ceil_quantile: float, 
                            k_floor_quantile: float, k_ceil_quantile: float,
                            p):
    # get the analytical solutions for the propagation of D when taking A0 at every t
    _, d_mean_vec, d_std_vec, _, _, _ = pars.A0_ana()
    
    d_floor_bound = p.MU_D_0 - p.SIGMA_D_0*sp.stats.norm.ppf(1-d_floor_quantile)
    d_ceil_bound  = d_mean_vec[-1] - d_std_vec[-1]*sp.stats.norm.ppf(1-d_ceil_quantile)
    
    # get the analytical solutions for the propagation of K when taking A1 at every t
    _, _, _, _, _, k_mean = pars.A1_ana()
    
    k_ceil_bound = p.MU_K_0 + p.SIGMA_K_0*sp.stats.norm.ppf(k_ceil_quantile)
    k_floor_bound  = k_mean[-1] + p.SIGMA_K_0*sp.stats.norm.ppf(k_floor_quantile)
    
    return d_floor_bound, d_ceil_bound, k_floor_bound, k_ceil_bound
    
    
    
# function which returns the buckets by specifying:
# - the amount of buckets
# - the upper and lower bucket bounds for d & k, respectively
def get_belief_buckets(n_d_buckets: int, n_k_buckets: int, d_lower: float, 
                       d_upper: float, k_lower: float, k_upper: float):
    
    assert n_d_buckets > 1 and n_k_buckets > 1
    
    # get the bucket bounds
    d_bucket_bounds = np.linspace(d_lower, d_upper, n_d_buckets+1)
    k_bucket_bounds = np.linspace(k_lower, k_upper, n_k_buckets+1)
    
    # preallocate vectors for the buckets
    d_buckets = np.zeros(shape=(n_d_buckets, 2))
    k_buckets = np.zeros(shape=(n_k_buckets, 2))
    
    # fill the vectors with the bounds together with -inf and +inf
    d_buckets[:,0] = d_bucket_bounds[:-1]
    d_buckets[:,1] = d_bucket_bounds[1:]
    k_buckets[:,0] = k_bucket_bounds[:-1]
    k_buckets[:,1] = k_bucket_bounds[1:]

    return d_bucket_bounds, k_bucket_bounds, d_buckets, k_buckets


# function which returns a list of grids, where the nodes are the midpoints of the d & k buckets
def get_belief_grids(d_bucket_bounds, k_bucket_bounds, d_buckets=None, k_buckets=None):

    if (d_bucket_bounds is not None) and (k_bucket_bounds is not None):
        outer_grid_D, outer_grid_K = np.meshgrid(d_bucket_bounds, k_bucket_bounds)

    if (d_buckets is not None) and (k_buckets is not None):
        midpoint_grid_D, midpoint_grid_K = np.meshgrid(np.mean(d_buckets, axis=1), np.mean(k_buckets, axis=1))
    
    return outer_grid_D, outer_grid_K, midpoint_grid_D, midpoint_grid_K


# function which returns a grid containing which represents the bounds of the input buckets
def get_outer_grid_by_buckets(bucket):
    bound_arr = np.append(bucket[:,0], bucket[-1,1])

    # belief bucket should not contain infinte bounds, bad for sampling
    if bound_arr[0] == -np.inf:
        bound_arr[0] = 2*bound_arr[1] - bound_arr[2]
    
    if bound_arr[-1] == np.inf:
        bound_arr[-1] = 2*bound_arr[-1] + bound_arr[-2]
    
    return np.meshgrid(bucket, bucket)


# function which summarizes the initialization, belief update and search into one function so that 
# it can be passed to the parallelization mapping function
def search_helper(d_grid_val, k_grid_val, A, P, t):
    # initialize tree with random observation (observation not needed, just a formality to delete old tree)
    A.initialize_tree(initial_obs=-130.)
    # set the posterior means of d & k according to the bucket midpoint and the desired time
    A.tree.nodes[-1][4].observation_belief_generator_update(d_grid_val, k_grid_val, P, t)
    # search best action
    return A.Search(t)
    

# computes the next action for the whole belief grid for a given time and returns the action grid
#D_mid, K_mid, A, P, t=1, parallel=True
def belief_trier(d_grid, k_grid, A, P, t, parallel=False):

    assert d_grid.shape == k_grid.shape

    height, width = d_grid.shape

    # allocate vector for action grid
    Z = np.full(shape=d_grid.shape, fill_value=np.inf)

    if parallel:
        # get pool of cpus for multiprocessing
        pool = Pool(pmp.cpu_count())
        
        for h_ in tqdm_notebook(range(height)):
            try:
                result = pool.amap(search_helper, d_grid[h_, :], k_grid[h_, :], [copy.deepcopy(A) for k in range(width)], [copy.deepcopy(P) for k in range(width)], [t]*width)
            except ValueError:
                print("Restarting pool")
                pool.restart()
                print("Restarting done")
                result = pool.amap(search_helper, d_grid[h_, :], k_grid[h_, :], [copy.deepcopy(A) for k in range(width)], [copy.deepcopy(P) for k in range(width)], [t]*width)
            
            if result._success:
                Z[h_, :] = result.get()
        
        pool.close()
        pool.join()
    
    else:
        for h_ in tqdm_notebook(range(height)):
            for w_ in range(width):
                # initialize tree with random observation (not needed)
                A.initialize_tree(initial_obs=-130)
                # set the posterior means of d & k according to the bucket midpoint and the desired time
                A.tree.nodes[-1][4].observation_belief_generator_update(d_grid[h_, w_], k_grid[h_, w_], p=P, t=t)
                # search best action
                Z[h_, w_] = A.Search(t)

    return Z

