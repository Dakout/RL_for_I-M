import os
import numpy as np
import pickle
from datetime import datetime
import copy

import visualization as vis
import environment as env
import parameters as pars


def LCC_calculator(distr, P):
    L_vec = np.zeros(4)
    C_vec = np.array([0, 1, 5, 100])
    for t in range(1, len(distr)):
        L_vec += distr[t,:] * P.GAMMA**t * C_vec

    return L_vec, L_vec/np.sum(distr[0,:])


# function that stores an agent later on
def save_agent(agent):
    if not os.path.exists('./best_agents'):
        os.mkdir('./best_agents')

    storage_string = 'agent_' + str(agent.measurement_error) + '_' + datetime.now().strftime("%d-%b-%Y_(%H-%M-%S)")
    agent.network.update_names('best_network_' + storage_string)
    agent.target_network = None

    with open('./best_agents/' + storage_string + '.pkl', 'wb') as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)

    return

        
def load_agent(agent_name=None):
    # if no specific agent was given load the first one in the best_agents list
    if agent_name is None:
        agent_list = os.listdir('./best_agents')
        agent_name = agent_list[0]
        assert 'agent' in agent_name
    
    if not '.pkl' in agent_name:
        agent_name = agent_name + '.pkl'
        
    with open('./best_agents/' + agent_name, 'rb') as f:
        b_a = pickle.load(f)
    
    # make sure the parameters are correct (are often not loaded with error)
    msrm_err = float(agent_name.split("_")[1])
    p1 = pars.PARS(msrm_err)
    p2 = pars.PARS(msrm_err)

    # check if both 
    assert (p1.SIGMA_K_t_PRIOR == p2.SIGMA_K_t_PRIOR).all()
    assert (p1.SIGMA_K_t_POST == p2.SIGMA_K_t_POST).all()
    assert (p1.SIGMA_D_t_PRIOR == p2.SIGMA_D_t_PRIOR).all()
    assert (p1.SIGMA_K_t_POST == p2.SIGMA_K_t_POST).all()
    assert (p1.RHO_t_PRIOR == p2.RHO_t_PRIOR).all()
    assert (p1.RHO_t_POST == p2.RHO_t_POST).all()

    if not ((b_a.pars.SIGMA_K_t_PRIOR == p1.SIGMA_K_t_PRIOR).all() and 
            (b_a.pars.SIGMA_K_t_POST == p1.SIGMA_K_t_POST).all() and
            (b_a.pars.SIGMA_D_t_PRIOR == p1.SIGMA_D_t_PRIOR).all() and 
            (b_a.pars.SIGMA_D_t_POST == p1.SIGMA_D_t_POST).all() and
            (b_a.pars.RHO_t_PRIOR == p1.RHO_t_PRIOR).all() and 
            (b_a.pars.RHO_t_POST == p1.RHO_t_POST).all()):
        
        print("Had to change parameters!")
        print("Old pars: ")
        print(b_a.pars.SIGMA_K_t_PRIOR, b_a.pars.SIGMA_K_t_POST, b_a.pars.SIGMA_D_t_PRIOR, b_a.pars.SIGMA_D_t_POST, b_a.pars.RHO_t_PRIOR, b_a.pars.RHO_t_POST)
        print("new pars: ")
        print(p1.SIGMA_K_t_PRIOR, p1.SIGMA_K_t_POST, p1.SIGMA_D_t_PRIOR, p1.SIGMA_D_t_POST, p1.RHO_t_PRIOR, p1.RHO_t_POST)
        b_a.pars = copy.deepcopy(p1)

    return b_a


def test_all_in_dir(directory, test_tuple=(int(5e5), int(1e4)), action_statistics=int(1e5), 
                    plot_action_statistics=False, network_plots=False):

    file_list = os.listdir(directory)
    LCCs = np.zeros(len(file_list))
    names = np.zeros(len(file_list))
    
    for k, f in enumerate(file_list):
        if not os.path.isdir(os.path.join(directory, f)):
            # if the directory is full of agents, test the networks of the agents
            if 'agent' in f:
                print("Testing following agent: ", f)
                a = load_agent(f)
                t = a.test(test_tuple[0], test_tuple[1])

            # if the directory is full of networks, load a default agent, reload the networks and test them
            elif 'network' in f:
                # load some agent
                a = load_agent()
                a.network.checkpoint_dir = os.path.join(os.path.abspath(os.getcwd()), directory)
                a.network.update_names(f)
                a.network.load_checkpoint()
                t = a.test(test_tuple[0], test_tuple[0])
                if plot_action_statistics:
                    a.batch_action_statistics(action_statistics)
                
            LCCs[k]  = t.cpu().numpy()
            names[k] = float(f.split('_')[1])
    
    if network_plots:
        vis.measurement_error_evolution_plot(LCCs, names)
    
    return LCCs, names


def belief_progress(t_batch, p, plot=False):
    beliefs = list()

    mu_d = np.full(shape=len(t_batch[1]['observations']), fill_value=p.MU_D_0)
    mu_k = np.full(shape=len(t_batch[1]['observations']), fill_value=p.MU_K_0)

    beliefs.append(dict())
    beliefs[0]['T'] = 0
    beliefs[0]['mu_d'] = mu_d
    beliefs[0]['mu_k'] = mu_k
    beliefs[0]['act'] = 0


    for t in range(1, len(t_batch)-1):
        # retrieve clean action and observation from trajectory batch
        act = np.argmax(t_batch[t]['previous_actions'].cpu().numpy(), axis=1)
        #act = t_batch[t]['previous_actions'].cpu().numpy()
        obs = t_batch[t]['observations'].cpu().numpy().squeeze()
        # update beliefs with taken actions and observations
        mu_d, mu_k = env.action_prior_update_batch(action=act, post_mu_D_prev=mu_d, post_mu_K_prev=mu_k, p=p)
        mu_d, mu_k = env.observation_posterior_update(observation=obs, prior_mu_D_t=mu_d, prior_mu_K_t=mu_k, p=p, t=t)
        
        # store posterior beliefs at each timestep
        beliefs.append(dict())
        beliefs[t]['T'] = t
        beliefs[t]['mu_d'] = mu_d
        beliefs[t]['mu_k'] = mu_k
        beliefs[t]['act'] = act
        #raise RuntimeError()
    
    act = np.argmax(t_batch[t+1]['previous_actions'].cpu().numpy(), axis=1)
    beliefs.append(dict())
    beliefs[t+1]['T'] = t+1
    beliefs[t+1]['act'] = act

    if plot:
        vis.belief_progress(beliefs)

    return beliefs