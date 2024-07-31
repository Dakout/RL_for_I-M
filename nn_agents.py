import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from tqdm.notebook import tqdm_notebook
import time
from datetime import datetime
import pickle

import parameters as pars
import environment as env
import default
from networks import Dueling_ADRQN
import visualization as vis

""" Note: loading stored network into default agent will yield different results, if 
           measurement_error, pars and lrelu_slope are not known !!"""

class Agent:
    def __init__(self, **kwargs):

        """
        Master class defining the basics all RL-agents will need.

        :param measurement_error: measurement error of observations
        :param epochs: number of epochs to train the network
        :param pars: parameters specifying the environment
        :param batch_size: Batch size to sample trajectories.
        :param target_update: After how many iterations to update target network.

        :param max_epsilon: Maximum value of epsilon. (Starting value)
        :param dec_epsilon: Epsilon decay factor.
        :param min_epsilon: Minimum value of epsilon parameter.
        :param epsilon: current value of epsilon

        :param n_actions: number of actions available
        :param NN_parameters: Parameters for the Q-network.
        :param kwargs: Kwargs.
        """

        # device: cpu or gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.measurement_error = kwargs.get('measurement_error', default.MEASUREMENT_ERROR)
        self.epochs            = kwargs.get('epochs', default.EPOCHS)
        self.pars              = pars.PARS(self.measurement_error)
        self.batch_size        = kwargs.get('batch_size', default.BATCH_SIZE)
        self.target_update     = kwargs.get('target_update', default.TARGET_UPDATE)

        self.max_epsilon = kwargs.get('max_epsilon', default.MAX_EPSILON)
        self.dec_epsilon = kwargs.get('dec_epsilon', default.DEC_EPSILON)
        self.min_epsilon = kwargs.get('min_epsilon', default.MIN_EPSILON)
        self.epsilon     = self.max_epsilon

        self.n_actions      = kwargs.get('n_actions', default.N_ACTIONS)
        self.NN_parameters  = kwargs.get('NN_parameters', {})
        self.network        = None
        self.target_network = None

        self.normalize_obs  = kwargs.get('normalize_obs', default.NORMALIZE_OBS)
        self.mu_norm        = kwargs.get('mu_norm', default.MU_NORM)
        self.std_norm       = kwargs.get('std_norm', default.STD_NORM) 

        self.loss_history = None

        self.initialize()


    def decrement_epsilon(self):
        # Decrements the epsilon after each step till it reaches minimum epsilon (0.1)
        # epsilon = epsilon - decrement (default is 0.99e-6)
        self.epsilon = self.epsilon - self.dec_epsilon if self.epsilon > self.min_epsilon \
            else self.min_epsilon

    def action_one_hot_batch(self, action):
        a = F.one_hot(torch.tensor(action).long(), self.n_actions)
        a = a.type(torch.FloatTensor).to(self.device)
        return a
    
    def observation_normalization(self, Obs):
        return (Obs - self.mu_norm) / self.std_norm


class Dueling_ADRQN_Agent(Agent):
    def __init__(self, **kwargs):

        """
        RL-Agent using a Recurrent Neural Network to calculate the Q values, from both the state and the last action
        performed, and chose the best action at each time step.
        """
        super().__init__(**kwargs)


    def initialize(self):
        self.network = Dueling_ADRQN(name='network_' + str(self.measurement_error) + '_' + datetime.now().strftime("%d-%b-%Y_(%H-%M-%S)"), **self.NN_parameters)
        self.target_network = Dueling_ADRQN(name='target_network_' + str(self.measurement_error) + '_' + datetime.now().strftime("%d-%b-%Y_(%H-%M-%S)"),  **self.NN_parameters)


    def learn(self):
        loss_vec = np.zeros(self.epochs)
        best_loss = np.inf

        self.target_network.optimizer.zero_grad()
        self.replace_target_network()

        for ep in tqdm_notebook(range(self.epochs), mininterval=2, desc='Train loop: '):
            loss = 0
            h_s = c_s = t_h_s = t_c_s = None
            # get life cycle trajectories with current weights
            trajects = self.sample_trajectory_batch(self.batch_size)

            for t in range(1, self.pars.T_END - 1):
                # propagate batch of trajectories through both networks
                val, adv, h_s, c_s = self.network.forward(trajects[t]['previous_actions'],
                                                          trajects[t]['observations'],
                                                          h_s,
                                                          c_s)
                t_val, t_adv, t_h_s, t_c_s = self.target_network.forward(trajects[t+1]['previous_actions'],
                                                                         trajects[t+1]['observations'],
                                                                         t_h_s,
                                                                         t_c_s)

                # compute q-values at each time step
                q   = torch.min(torch.add(val, adv - adv.mean(dim=1, keepdim=True)), dim=1)[0].view(-1,1)
                t_q = torch.min(torch.add(t_val, t_adv - t_adv.mean(dim=1, keepdim=True)), dim=1)[0].view(-1,1)
                # compute the loss at each time step and accumulate
                loss += self.network.loss(q, self.pars.GAMMA*t_q + trajects[t]['rewards']).to(self.network.device)

            val, adv, h_s, c_s = self.network.forward(trajects[self.pars.T_END-1]['previous_actions'],
                                                      trajects[self.pars.T_END-1]['observations'],
                                                      h_s,
                                                      c_s)

            q   = torch.min(torch.add(val, adv - adv.mean(dim=1, keepdim=True)), dim=1)[0].view(-1,1)
            # at last time step no action -> sum last two rewards
            loss += self.network.loss(q, trajects[-2]['rewards'] + self.pars.GAMMA*trajects[-1]['rewards']).to(self.network.device)
            loss_vec[ep] = loss
            loss.backward()

            # update weights, learning rate, epsilon and step count
            self.network.optimizer.step()
            self.network.lr_scheduler.step()
            self.decrement_epsilon()
            
            # early stopping
            if ep > 60:
                if loss > loss_vec[ep-30]:
                    loss_vec = loss_vec[0:ep-30+1]
                    break

            # 'burnin period'
            if ep > 4:
                if loss < best_loss:
                    best_loss = loss
                    self.network.save_checkpoint()


            # periodically update the target network
            if ep % self.target_update == 0:
                self.replace_target_network()

            # periodically test network
            #if (ep % 80 == 0) or (ep == self.epochs-1):
            #    LCC = self.test(int(1e4), int(1e4), False)
            #    if LCC < best_LCC:
            #        best_LCC = LCC
            #        print("Best LCC of current model: ", LCC)
            #        self.network.save_checkpoint()

        self.loss_history = loss_vec

        # plot histogram of costs
        vis.loss_plot(loss_vec)

        return


    def test(self, n_t, batch, plot=True):
        assert (n_t > 0) and (n_t % batch == 0)
        distr = np.zeros((21, 4))
        _eps = self.epsilon
        self.epsilon = 0
        self.network.optimizer.zero_grad()

        cost_vec = torch.zeros(1).to(self.device)
        rounds = int(np.ceil(n_t / batch))
        for _ in tqdm_notebook(range(rounds), mininterval=2, desc='Test loop: '):
            rewards = torch.zeros(batch).view(-1, 1).to(self.device)
            traj = self.sample_trajectory_batch(batch)

            for j in range(self.pars.T_END + 1):
                rewards += (self.pars.GAMMA**j)*traj[j]['rewards']
            for j in range(1, self.pars.T_END + 1):
                A_prev = self.action_one_hot_batch(torch.argmin(traj[j]['previous_actions'], dim=1))
                distr[j-1, :] += torch.sum(traj[j]['previous_actions'], dim=0).cpu().numpy()

            cost_vec = torch.cat((cost_vec, rewards.squeeze()), -1)

        cost_vec = cost_vec[1:]
        LCC_mean = cost_vec.mean()
        LCC_std  = cost_vec.std()
        LCC_cov  = LCC_std / LCC_mean

        print("Mean life cycle cost: ", LCC_mean)
        print("LCC standard deviation: ", LCC_std)
        print("LCC coefficient of variation: ", LCC_cov)

        if plot:
            vis.test_plot(cost_vec.cpu().numpy(), n_t)

        # reset epsilon to value before
        self.epsilon = _eps

        return LCC_mean, distr


    def sample_trajectory_batch(self, n_trajectories):
        P = self.pars
        # get skeleton of trajectories
        trjcts = self.get_trajectory_batch_template(n_trajectories)

        _nhs = _ncs = None
        O_arr = np.zeros((P.T_END + 1, n_trajectories))
        R_arr = np.zeros((P.T_END + 1, n_trajectories))
        C_arr = np.zeros((P.T_END + 1, n_trajectories))
        D_arr = np.zeros((P.T_END + 1, n_trajectories))

        D, K = env.draw_initial_samples(P, n_trajectories)
        D_arr[0] = D

        C = np.zeros(len(D))

        A_prev = self.action_one_hot_batch(np.zeros(len(D)))
        # check if bridge is already broken in the beginning
        R, _ = env.failure_cost(D, P, 0)
        C += R
        # T=0: do nothing
        D, _, R, _ = env.draw_next(D, K, np.zeros(len(D)), P, t=0)
        C += R
        D_arr[1] = D

        ## T=1:
        C_arr[0] = C
        R_arr[0] = C
        trjcts[1]['previous_actions'] = A_prev

        # from 1 to 20
        for T in range(1, P.T_END):
            # Step 1: get observation
            O = env.get_obs(D, P)
            O = torch.tensor([O]).view(-1,1)
            O = O.type(torch.FloatTensor).to(self.device)

            # normalize for better convergence
            if self.normalize_obs:
                for k in range(1, len(trjcts)-1):
                    O = self.observation_normalization(O)

            with torch.no_grad():
                # Step 2: propagate through network
                _val, _adv, _nhs, _ncs = self.network.forward(A_prev, O, _nhs, _ncs)

                # Step 3: Select action
                if np.random.random() > self.epsilon:
                    A_prev = self.action_one_hot_batch(torch.argmin(_adv, dim=1))
                else:
                    A_prev = self.action_one_hot_batch(np.random.choice(self.n_actions, size=n_trajectories))

                # Step 4: check for failure
                R, _ = env.failure_cost(D, P, T)
                C += R

                # Step 5: Draw next state
                D, K, R, _ = env.draw_next(D, K, torch.argmax(A_prev.cpu(), dim=1), P, T)
                C += R

                C_arr[T] = C
                R_arr[T] = C - C_arr[T-1]
                O_arr[T] = O.cpu().squeeze()
                D_arr[T+1] = D
                trjcts[T]['rewards'] = torch.tensor(C - C_arr[T-1], dtype=torch.float32).view(-1,1).to(self.device)
                trjcts[T+1]['previous_actions'] = A_prev
                trjcts[T]['observations'] = O


        # T=T_END: only add possible failure cost and do not do any actions
        R, _ = env.failure_cost(D, P, P.T_END)
        C += R
        C_arr[-1] = C
        R_arr[-1] = C - C_arr[-2]
        trjcts[-1]['rewards'] = torch.tensor(C - C_arr[-2], dtype=torch.float32).view(-1,1).to(self.device)
        return trjcts


    # function which gets the template of the sampled trajectories:
    # list of dicts, where at each timestep you have a dict containing the
    # previous action, the current observation and the resulting reward
    def get_trajectory_batch_template(self, n_trajects):
        ts = list()
        # 22 entries, one for each timestep, first entry corresponding to T=0 is
        # empty because you start taking actions at T=1
        for _ in range(self.pars.T_END + 1):
            ts.append(dict())
        ts[0]['rewards'] = torch.zeros((n_trajects, 1)).to(self.device)
        return ts
    
    def replace_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())
        #self.target_network = self.network.clone().detach().requires_grad_(False)

    def save_model(self):
        # Saves the values of q_eval and q_next at the checkpoint
        self.network.save_checkpoint()
        self.target_network.save_checkpoint()

    def load_model(self):
        # Loads the values of q_eval and q_next at the checkpoint
        self.network.load_checkpoint()
        self.target_network.load_checkpoint()


    """
    ################################################################################
    ################################################################################
                                Code for visualization
    ################################################################################
    ################################################################################
    """

    def batch_action_statistics(self, n_trajectories):
        P = self.pars
        distr = np.zeros((21, 4))
        _nhs = _ncs = None

        D, K = env.draw_initial_samples(P, n_trajectories)

        A_prev = self.action_one_hot_batch(np.zeros(len(D)))
        distr[0, :] += torch.sum(A_prev, dim=0).cpu().numpy()

        D, _, R, _ = env.draw_next(D, K, np.zeros(len(D)), P, t=0)

        # from 1 to 20
        for T in range(1, P.T_END):
            #print("\nTimestep: ", T)
            # Step 1: get observation
            O = env.get_obs(D, P)
            O = torch.tensor([O]).view(-1,1)
            O = O.type(torch.FloatTensor).to(self.device)
            if self.normalize_obs:
                O = self.observation_normalization(O)

            with torch.no_grad():
                _val, _adv, _nhs, _ncs = self.network.forward(A_prev, O, _nhs, _ncs)
                A_prev = self.action_one_hot_batch(torch.argmin(_adv, dim=1))
                distr[T, :] += torch.sum(A_prev, dim=0).cpu().numpy()
                D, _, R, _ = env.draw_next(D, K, torch.argmax(A_prev.cpu(), dim=1), P, T)

        # check if everything went right
        assert (np.sum(distr, axis=1) == n_trajectories).all()

        # normalize distribution
        distr = distr / n_trajectories

        vis.action_statistics_visualization(distr)

        return distr
