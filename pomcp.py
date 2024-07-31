import numpy as np
import scipy as sp

import environment as env
import default
from auxiliary import Tree, UCB

#POMCP solver
class POMCP():

    def __init__(self, pars_instance, initial_observation, **kwargs):

        """
        POMCP class defining all the functionality

        -param pars: object specifying the environment parameters

        -param n_actions: number of actions available
        -param n_obs_buckets: number of discrete observation buckets (ranges of deterioration)
        -param floor quantile: quantile of intial distribution used for the calculation of the upper boundary of the floor bucket
        -param ceil quantile: quantile of initial distribution used for the calculation of the lower boundary of the ceiling bucket

        -param depth_threshold: threshold of searching depth before making the next step
        -param exploration_c: parameter for UCB exploration (in paper: 1)
        -param n_pomcp_runs: number of times the tree is constructed (for statistics)
        -param n_rollout_runs: number of times a trajectory is sampled to get an
                               estimate for the value of an action node
        -param use_ucb: boolean flag whether the UCB value or the pure value is used

        -param kwargs: Kwargs.

        """

        self.pars            = pars_instance

        self.n_actions       = kwargs.get('n_actions', default.N_ACTIONS)
        self.n_obs_buckets   = kwargs.get('n_obs_buckets', default.N_OBS_BUCKETS)
        self.floor_quantile  = kwargs.get('floor_quantile', default.FLOOR_QUANTILE)
        self.ceil_quantile   = kwargs.get('ceil_quantile', default.CEIL_QUANTILE)

        self.depth_threshold = kwargs.get('depth_threshold', default.DEPTH_THRESHOLD)
        self.exploration_c   = kwargs.get('exploration_c', default.EXPLORATION_C)
        self.n_pomcp_runs    = kwargs.get('n_pomcp_runs', default.N_POMCP_RUNS)
        self.n_rollout_runs  = kwargs.get('n_rollout_runs', default.N_ROLLOUT_RUNS)
        self.use_ucb         = kwargs.get('use_UCB', default.USE_UCB)

        self.initialize()
        self.initialize_tree(initial_observation)
        self.update_belief(act=0, obs=initial_observation, t=1)


    # give state, action, and observation space
    def initialize(self):
        self.actions = np.array([i for i in range(self.n_actions)])
        self.observations = np.array([i for i in range(self.n_obs_buckets)])
        self.floor_bucket = self.pars.MU_D_0 - self.pars.SIGMA_D_0*sp.stats.norm.ppf(1-self.floor_quantile)
        self.ceil_bucket = (self.pars.MU_D_0 + self.pars.T_END*self.pars.MU_K_0) - \
                            np.sqrt(self.pars.SIGMA_D_0**2 + (self.pars.T_END*self.pars.SIGMA_K_0)**2) * \
                            sp.stats.norm.ppf(1-self.ceil_quantile)
        self.bucket_bounds = np.linspace(self.floor_bucket, self.ceil_bucket, self.n_obs_buckets-1)
        self.buckets = np.zeros(shape=(self.n_obs_buckets, 2))
        self.buckets[:,0] = [-np.inf] + list(self.bucket_bounds)
        self.buckets[:,1] = list(self.bucket_bounds) + [np.inf]

        self.post_mu_d_now = self.pars.MU_D_0
        self.post_mu_k_now = self.pars.MU_K_0
        return


    # function which initializes the tree (for manual updating from outside)
    def initialize_tree(self, initial_obs):
        self.tree = Tree(initial_obs, self.get_observation_bucket(initial_obs))
        return


    # function that updates the current prior and posterior beliefs based on a
    # new action taken and a new observation made
    def update_belief(self, act, obs, t):
        # update the belief parameters: prior and posterior mean of d & k
        self.prior_mu_d_now, self.prior_mu_k_now = env.action_prior_update(action=act, post_mu_D_prev=self.post_mu_d_now,
                                                                           post_mu_K_prev=self.post_mu_k_now, p=self.pars)
        self.post_mu_d_now, self.post_mu_k_now = env.observation_posterior_update(observation=obs,
                                                                                  prior_mu_D_t=self.prior_mu_d_now,
                                                                                  prior_mu_K_t=self.prior_mu_k_now,
                                                                                  p=self.pars, t=t)
        # update the generator functions of the root node
        self.tree.nodes[-1][4].observation_belief_generator_update(self.post_mu_d_now, self.post_mu_k_now, self.pars, t=1)
        return



    # Search module
    def Search(self, time):
        d_gen = self.tree.nodes[-1][4].d_sample_generator
        k_gen = self.tree.nodes[-1][4].k_sample_generator
        mu_d_and_k_belief_generator = self.tree.nodes[-1][4].d_and_k_sample_generator
        # Repeat Simulations n_pomcp_runs amount of times
        for _ in range(self.n_pomcp_runs):
            # get best action
            new = mu_d_and_k_belief_generator(n=1)
            self.Simulate(current_d=np.array([new[0]]), current_k=np.array([new[1]]), h=-1, depth=time)
            #new_d = d_gen(n=1)
            #new_k = k_gen(n=1)
            #self.Simulate(current_d=new_d, current_k=new_k, h=-1, depth=time)
        # Get best action
        action, _ = self.SearchBest(h=-1)
        return action


    def Simulate(self, current_d, current_k, h, depth: int):
        # Check significance of update
        if depth > self.depth_threshold:
            failure_r, _ = env.failure_cost(d=current_d, p=self.pars, t=depth)
            return failure_r

        # If leaf node and not the last time time step (no action)
        if self.tree.isLeafNode(h):
            for action in self.actions:
                self.tree.ExpandTreeFrom(h, action, IsAction=True)

            # do a rollout n_rollout_runs number of times and then average
            new_value = np.sum(self.Rollout(np.full(shape=self.n_rollout_runs, fill_value=current_d),
                                            np.full(shape=self.n_rollout_runs, fill_value=current_k), depth)) / self.n_rollout_runs
            # update number of visits and values
            self.tree.nodes[h][2] += 1
            self.tree.nodes[h][3] = new_value
            return new_value

        cum_reward = 0
        # check for failure in current state
        failure_reward, _ = env.failure_cost(d=current_d, p=self.pars, t=depth)
        # Searches best action and the corresponding action node
        next_action, next_action_node = self.SearchBest(h)
        # Apply best action, generate next d & k states and collect action reward
        d_next, k_next, action_reward, _ = env.draw_next(d=current_d, k=current_k, a=next_action, p=self.pars, t=depth)
        # get next observation from next d
        sample_observation = env.get_obs(d_samples=current_d, p=self.pars)
        # Get resulting node index
        next_obs_node = self.getObservationNode(next_action_node, sample_observation)
        # Estimate node Value
        cum_reward += failure_reward + action_reward + self.pars.GAMMA*self.Simulate(d_next, k_next, next_obs_node, depth + 1)
        # Backtrack
        self.tree.nodes[h][2] += 1
        self.tree.nodes[next_action_node][2] += 1
        self.tree.nodes[next_action_node][3] += (cum_reward - self.tree.nodes[next_action_node][3])/self.tree.nodes[next_action_node][2]
        return cum_reward


    def Rollout(self, s_d, s_k, depth: int):
        # Check significance of update
        if depth > self.depth_threshold:
            failure_r, _ = env.failure_cost(d=s_d, p=self.pars, t=depth)
            return failure_r

        cum_rollout_reward = np.zeros(self.n_rollout_runs)

        # check for failure
        r, _ = env.failure_cost(d=s_d, p=self.pars, t=depth)
        cum_rollout_reward += r

        # Pick random action
        action = np.random.choice(self.actions, size=self.n_rollout_runs, replace=True)
        # Generate states and observations
        s_d_next, s_k_next, s_r, _ = env.draw_next(d=s_d, k=s_k, a=action, p=self.pars, t=depth)
        cum_rollout_reward += s_r + self.pars.GAMMA*self.Rollout(s_d_next, s_k_next, depth + 1)
        return cum_rollout_reward

    # searchBest action to take
    def SearchBest(self, h):
        min_value = None
        result = None
        result_action = None
        if self.use_ucb:
            if self.tree.nodes[h][4].node_type != 'action_node':
                children = self.tree.nodes[h][1]
                # UCB for each child node
                for action, child in children.items():
                    # if node is unvisited return it
                    if self.tree.nodes[child][2] == 0:
                        return action, child
                    ucb = UCB(self.tree.nodes[h][2], self.tree.nodes[child][2],
                              self.tree.nodes[child][3], self.exploration_c)

                    # Min is kept
                    if min_value is None or ucb < min_value:
                        min_value = ucb
                        result = child
                        result_action = action
            #return action-child_id values
            return result_action, result
        else:
            if self.tree.nodes[h][4].node_type != 'action_node':
                children = self.tree.nodes[h][1]
                # pick optimal value node for termination
                for action, child in children.items():
                    node_value = self.tree.nodes[child][3]
                    # keep min
                    if min_value is None or node_value < min_value:
                        min_value = node_value
                        result = child
                        result_action = action
            return result_action, result


    # Checks that an observation was already made before moving
    def getObservationNode(self, h_action_node, sample_obs):
        # get observation bucket
        obs_bucket = self.get_observation_bucket(sample_obs)

        # Check if a given observation node has been visited
        if obs_bucket not in list(self.tree.nodes[h_action_node][1].keys()):
            # If not create the node
            self.tree.ExpandTreeFrom(h_action_node, (sample_obs, obs_bucket), IsAction=False)

        # Get the index of the next observation node and return it
        next_observation_node_index = self.tree.nodes[h_action_node][1][obs_bucket]
        return next_observation_node_index


    # transforms an observation from the continuous space into discrete buckets
    def get_observation_bucket(self, o_sample):
        return np.where((o_sample > self.buckets[:,0]) & (o_sample < self.buckets[:,1]))[0].item()


    # Prune tree after action and observation were made
    def prune_after_action(self, action, observation):
        # get observation bucket
        o_bucket = self.get_observation_bucket(observation)

        # Get node after action
        action_node = self.tree.nodes[-1][1][action]

        # Get new root (after observation)
        new_root = self.getObservationNode(action_node, observation)
        # remove new_root from parent's children to avoid deletion
        del self.tree.nodes[action_node][1][o_bucket]

        # prune unnesecary nodes
        self.tree.prune(-1)

        # set new_root as root (key = -1)
        self.tree.make_new_root(new_root)
        return
