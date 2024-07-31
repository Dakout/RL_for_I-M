import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp

import environment as env


def action_statistics_solution(NN_sol, MCTS_sol, save=False):
    
    if type(NN_sol) == dict:
        NN = [NN_sol['actions'][k] for k in range(len(NN_sol['actions']))]
        NN.append(z['LCC'].cpu().numpy().tolist())
    else:
        NN = NN_sol
    
    MCTS = MCTS_sol
    
    print(NN[4]-np.sum(NN[0:4]))
    
    x = np.arange(4)
    plt.figure(figsize=(10,6))
    plt.bar(x[0] - 0.17, height=NN[1], width=0.32, bottom=0, color='tab:blue', label='$a_1$')
    plt.bar(x[1] - 0.17, height=NN[2], width=0.32, bottom=0, color='tab:green', label='$a_2$')
    plt.bar(x[2] - 0.17, height=NN[4]-np.sum(NN[0:4]), width=0.32, bottom=0, color='xkcd:crimson', label='$F$')
    plt.bar(x[3] - 0.17, height=NN[1], width=0.32, bottom=0, color='tab:blue')
    plt.bar(x[3] - 0.17, height=NN[2], width=0.32, bottom=NN[1], color='tab:green')
    plt.bar(x[3] - 0.17, height=NN[4]-np.sum(NN[0:4]), width=0.32, bottom=np.sum(NN[1:4]), color='xkcd:crimson')
    
    plt.bar(x[0] + 0.17, height=MCTS[1], width=0.32, bottom=0, color='tab:blue')
    plt.bar(x[1] + 0.17, height=MCTS[2], width=0.32, bottom=0, color='tab:green')
    plt.bar(x[2] + 0.17, height=0, width=0.32, bottom=0, color='xkcd:crimson')
    plt.bar(x[3] + 0.17, height=MCTS[1], width=0.32, bottom=0, color='tab:blue')
    plt.bar(x[3] + 0.17, height=MCTS[2], width=0.32, bottom=MCTS[1], color='tab:green')
    
    #plt.text(-0.08, 2, 'NN', fontsize=16)
    #plt.bar(x[3] + 0.33, height=np.sum(MCTS[1:4]), width=0.33, bottom=np.sum(MCTS[1:4]), color='xkcd:crimson')
            
    plt.xticks(ticks=x, labels=['NN     MCTS\n$a_1$', 'NN     MCTS\n$a_2$', 'NN     MCTS\nF', 'NN     MCTS\nLCC'], fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Cost split', fontsize=17, labelpad=15)
    plt.ylabel("Mean cost", fontsize=17, labelpad=15)
    #plt.gca().set_position([0, 0, 1, 1])
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
    
    
    if save:
        plt.savefig("costsplit2.svg")
        
    plt.show()
    return


# function that plots the analytical solutions of taking a specific action at 
# every time step and compares it with the results from the simulation
def analytical_plot(p, save=False):
    x = np.linspace(0,21,22)
    _,y_mean_0,_,_,_,_ = p.A0_ana()
    _,y_mean_1,_,_,_,_ = p.A1_ana()
    _,y_mean_2,_,_,_,_ = p.A2_ana()
    _,y_mean_3,_,_,_,_ = p.A3_ana()

    # simulated solutions
    mean_list = list()
    for k in range(4):
        _, m_d, _, _, _ = env.benchmark_calc(1e5, k, [1e3])
        mean_list.append(m_d)

    plt.figure(figsize=(10,6))
    line1, = plt.plot(x,y_mean_0,linewidth=3, label='Ana. $\\mu_{D,A_0}$', c='tab:orange')
    scatter1 = plt.scatter(x, mean_list[0], s=60, c='tab:blue', marker='o', label='MC $\\mu_{D,A_0}$')
    line2, = plt.plot(x,y_mean_1,linewidth=3, label='Ana. $\\mu_{D,A_1}$', c='tab:blue')
    scatter2 = plt.scatter(x, mean_list[1], s=60, c='tab:orange', marker='o', label='MC $\\mu_{D,A_1}$')
    line3, = plt.plot(x,y_mean_2,linewidth=3, label='Ana. $\\mu_{D,A_2}$', c='tab:green')
    scatter3 = plt.scatter(x, mean_list[2], s=60, c='tab:gray', marker='o', label='MC $\\mu_{D,A_2}$')
    line4, = plt.plot(x,y_mean_3,linewidth=3, label='Ana. $\\mu_{D,A_3}$', c='tab:gray')
    scatter4 = plt.scatter(x, mean_list[3], s=60, c='tab:green', marker='o', label='MC $\\mu_{D,A_3}$')

    legend1 = plt.legend(handles=[line1, scatter1], loc='upper right', bbox_to_anchor=(0.73, 0.96), fontsize=14)
    legend2 = plt.legend(handles=[line2, scatter2], loc='upper right', bbox_to_anchor=(0.99, 0.72), fontsize=14)
    legend3 = plt.legend(handles=[line3, scatter3], loc='upper right', bbox_to_anchor=(0.73, 0.15), fontsize=14)
    legend4 = plt.legend(handles=[line4, scatter4], loc='upper right', bbox_to_anchor=(0.99, 0.40), fontsize=14)

    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.gca().add_artist(legend3)
    plt.gca().add_artist(legend4)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('t', fontsize=17, labelpad=15)
    plt.ylabel("$\\mu_{D,A_i}$", fontsize=17, rotation=0, labelpad=25)
    plt.gca().set_position([0, 0, 1, 1])
    #plt.legend(fontsize=14)
    plt.title("Evolution of $\\mu_{D,A_0},~\\mu_{D,A_1},~\\mu_{D,A_2}$ and $\\mu_{D,A_3}$ over time", fontsize=20)
    plt.tight_layout()
    if save:
        plt.savefig("meanevolutions.svg")
        
    plt.show()
    return



# function that plots the evolution of the network loss over the epochs
def loss_plot(loss_vec, save=False):
    plt.figure(figsize=(6,5))
    plt.plot(loss_vec)
    plt.yscale("log")
    plt.grid()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Training epochs', fontsize=17, labelpad=15)
    plt.ylabel('Training loss', fontsize=17, labelpad=15)
    plt.tight_layout()
    if save:
        plt.savefig("loss_curve.svg")
    
    plt.show()
    return


# function that plots a histogram of achieved LCCs
def test_plot(cost_vec: np.array, n_t: int, save=False):
    plt.figure(figsize=(4,5))
    ax = plt.gca()
    plt.hist(cost_vec, bins=int(min(0.1*n_t, 300)))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('LCC', fontsize=17, labelpad=10)
    plt.ylabel('occurence', fontsize=17, labelpad=10)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    #plt.axis([0, 50, 0, 200000,])
    plt.tight_layout()
    
    if save:
        plt.savefig("cost_hist.svg")
    
    plt.show()
    return


# function that plots a statistical distribution of actions taken at every
# timestep as an evolving barplot
def action_statistics_visualization(distr, save=False):
    x = np.linspace(0, distr.shape[0]-1, distr.shape[0])

    plt.figure(figsize=(10,6))
    plt.bar(x, height=distr[:,0], width=0.95, bottom=0, color='tab:orange', label='$a_0$')
    plt.bar(x, height=distr[:,1] + distr[:,0], width=0.95, bottom=distr[:,0], color='tab:blue', label='$a_1$')
    plt.bar(x, height=distr[:,2] + distr[:,1] + distr[:,0], width=0.95, bottom=distr[:,1] + distr[:,0], color='tab:green', label='$a_2$')
    plt.bar(x, height=distr[:,3] + distr[:,2] + distr[:,1] + distr[:,0], width=0.95, bottom=distr[:,2] + distr[:,1] + distr[:,0], color='tab:gray', label='$a_3$')
    plt.xticks(x, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('t', fontsize=17, labelpad=10)
    plt.axis([-0.55, 20.55, 0, 1])
    plt.ylabel('Action proportion', fontsize=17)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig("action_statistics_mcts_350.svg")
    
    plt.show()
    return


"""
##############################################################################
################################# LCC plots ##################################
##############################################################################
"""

# function that plots the evolution of LCCs vs the increase of measurement errors
def measurement_error_evolution_plot(LCCs, names, save=False, mcts=None):
    assert len(LCCs) == len(names)

    # sort arrays according to ascending measurement error
    a = np.array([names, LCCs]).T
    a = a[a[:, 0].argsort()]
    names = a[:,0]
    LCCs = a[:,1]

    plt.figure(figsize=(10,6))
    ax = plt.gca()
    plt.plot(names, LCCs, 'o--', markersize=10, linewidth=2.5, label='NN')
    plt.xscale("log")
    plt.grid()
    plt.xlabel(r"$\sigma_E$", labelpad=15, fontsize=17)
    plt.ylabel("Mean LCC", labelpad=15, fontsize=17)
    
    if mcts:
        ax.scatter(mcts['error'], mcts['LCC'], s=100, color='tab:red', label='MCTS')
        #plt.plot(mcts['error'], mcts['LCC'], '--', color='black)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1], loc='upper left', fontsize=14)

    locmaj = mpl.ticker.LogLocator(base=10,numticks=10) 
    ax.xaxis.set_major_locator(locmaj)
    plt.xticks(fontsize=14)
    locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=10)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    plt.yticks(fontsize=14)
    plt.tight_layout()
    #plt.legend(loc='upper left', fontsize=14)
    #plt.hlines(29.45, xmin=0, xmax=100000)
    if save:
        plt.savefig("measurementevolution_tree_and_NN2.svg")
    
    plt.show()
    return


def rollout_run_LCC_plots(rollout_vec, rollout_LCCs, pomcp_vec, save=False):
    # pomcp_LCCs & rollout_LCCs should be a list of arrays/lists 
 
    plt.figure(figsize=(10,6))
    ax = plt.gca()
            
    for k in range(len(pomcp_vec)):
        shapes = ['o--', 's--', 'v--', '*--', 'p--']
        string = r'N_{T}'
        plt.plot(rollout_vec, rollout_LCCs[k], shapes[k], color='xkcd:turquoise', markersize=10, linewidth=2.5, label=rf'${string} = {pomcp_vec[k]}$')
     
    plt.xscale("log")
    plt.grid()
    plt.xlabel("Rollout runs", labelpad=15, fontsize=17)
    #plt.ylabel(r"$\mu_{LCC, 100}$", rotation=0 , labelpad=15, fontsize=17)
    plt.ylabel("Mean LCC", labelpad=10, fontsize=17)
    

    locmaj = mpl.ticker.LogLocator(base=10,numticks=10) 
    ax.xaxis.set_major_locator(locmaj)
    plt.xticks(fontsize=14)
    locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=10)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    
    if save:
        plt.savefig("rolloutrunsvsLCC3.svg")
    
    plt.show()
    return


def pomcp_run_LCC_plots(pomcp_vec, pomcp_LCCs, rollout_vec, save=False):
    # pomcp_LCCs & rollout_LCCs should be a list of arrays/lists 
 
    plt.figure(figsize=(10,6))
    ax = plt.gca()
    
    for k in range(len(rollout_vec)):
        shapes = ['o--', 's--', 'v--', '*--', 'p--']
        string = r'N_{R}'
        plt.plot(pomcp_vec, pomcp_LCCs[k], shapes[k], color='xkcd:magenta', markersize=10, linewidth=2.5, label=rf'${string} = {rollout_vec[k]}$')
     
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.xlabel("Tree iterations", labelpad=15, fontsize=17)
    #plt.ylabel(r"$\mu_{LCC, 100}$", rotation=0 , labelpad=15, fontsize=17)
    plt.ylabel("Mean LCC", labelpad=10, fontsize=17)
    

    locmaj = mpl.ticker.LogLocator(base=10,numticks=10) 
    ax.xaxis.set_major_locator(locmaj)
    plt.xticks(fontsize=14)
    locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=10)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    plt.yticks(fontsize=14)
    
    plt.legend(fontsize=14)
    plt.tight_layout()
    
    if save:
        plt.savefig("pomcprunvsLCC3.svg")
    
    plt.show()
    return


def num_buckets_vs_LCC_plot(buckets, LCCs, covs=None, save=False):
     
    plt.figure(figsize=(10,6))
    ax = plt.gca()
    
    if type(LCCs) == list:
        assert sum([len(buckets) == len(LCCs[k]) for k in range(len(LCCs))]) == len(LCCs)
        linestyle_tuple = {'loosely dotted': (0, (1, 10)),
                           'dotted': (0, (1, 1)),
                           'densely dotted': (0, (1, 1)),

                           'loosely dashed': (0, (5, 10)),
                           'dashed': (0, (5, 5)),
                           'densely dashed': (0, (5, 1)),

                           'loosely dashdotted': (0, (3, 10, 1, 10)),
                           'dashdotted': (0, (3, 5, 1, 5)),
                           'densely dashdotted': (0, (3, 1, 1, 1)),

                           'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
                           'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
                           'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}
       
        selected_linestyles = ['solid', 'loosely dotted', 'loosely dashdotted']
        assert len(selected_linestyles) == len(LCCs)
        
        # scatter plot points
        ax.scatter(buckets, LCCs[0], s=100, color='darkblue')
        ax.scatter(buckets, LCCs[1], s=100, color='xkcd:azure')
        ax.scatter(buckets, LCCs[2], s=100, color='cyan')
           
        # lines 
        line1, = plt.plot(buckets, LCCs[0], linestyle='-', linewidth=2.5, color='darkblue', label=r'$\sigma_E = 0.5$')
        line2, = plt.plot(buckets, LCCs[1], linestyle='--', linewidth=2.5, color='xkcd:azure', label=r'$\sigma_E = 50$')
        line3, = plt.plot(buckets, LCCs[2], linestyle=':', linewidth=2.5, color='cyan', label=r'$\sigma_E = 350$')
        
        plt.legend(handles=[line3, line2, line1], loc='right', fontsize=14)
        
        if covs:
            for k in range(len(covs)):
                ci = sp.stats.norm.ppf(0.99) * covs[k] / np.sqrt(1000)
                ax.fill_between(buckets, LCCs[k]-ci, LCCs[k]+ci, color='b', alpha=.9)
        
    else:
        assert len(buckets) == len(LCCs)
    
        plt.plot(buckets, LCCs, 'o--', markersize=10, linewidth=2.5)
    
    plt.xscale("log")
    plt.grid()
    plt.xlabel("# buckets", labelpad=15, fontsize=17)
    plt.ylabel("Mean LCC", labelpad=20, fontsize=17)

    locmaj = mpl.ticker.LogLocator(base=10,numticks=10) 
    ax.xaxis.set_major_locator(locmaj)
    plt.xticks(fontsize=14)
    locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=10)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    plt.yticks(fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig("bucketsvsLCC.svg")
    
    plt.show()
    return
    


"""
##############################################################################
################################ Time plots ##################################
##############################################################################
"""

# function that plots the increase in computation time when increasing the 
# number of rollout runs: can supply pure Rollout run data and Tree run data
def time_rollout_run_plot(rr, rmu_t, rcov_t=None, tmu_t=None, tcov_t=None, pomcp_runs=None, save=False):

    plt.figure(figsize=(10,6))
    ax = plt.gca()
    
    # check if multiple vectors are given in list
    if type(rmu_t) == list:
        #assert [len(rr) == k.shape[0] for k in rmu_t]
        shapes = ['o--', 's--', 'v--', '*--', 'p--']
        
        if rmu_t[0] is not None:
            for k in range(len(rmu_t)):
                string = r'N_{P,R}'
                plt.plot(rr, rmu_t[k], shapes[k], color='xkcd:sienna', markersize=10, linewidth=2.5, label=rf'${string} = {pomcp_runs[k]}$')

            order = np.arange(len(rmu_t))
        
        if tmu_t is not None:
            assert pomcp_runs is not None
            for k in range(len(tmu_t)):
                string = r'N_{T}'
                plt.plot(rr, tmu_t[k], shapes[k], color='xkcd:turquoise', markersize=10, linewidth=2.5, label=rf'${string} = {pomcp_runs[k]}$')
            
            # specify order of items in list
            #order = np.flip((np.arange(len(rmu_t)) + len(rmu_t))).tolist() + np.arange(len(rmu_t)).tolist()
            order = np.flip(np.arange(max(2*len(rmu_t), len(tmu_t))))

        handles, labels = plt.gca().get_legend_handles_labels()
        #plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=14)
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper left', fontsize=14)
        #plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=14)
            
    else:
        assert len(rr) == len(rmu_t)
        plt.plot(rr, rmu_t, 'o--', color='xkcd:magenta', markersize=10, linewidth=2.5, label=r'Rollout $\mu_t$')
        if rcov_t is not None:
            plt.plot(rr, rcov_t, 'o--', color='salmon', markersize=6, linewidth=1.5, label=r'Rollout cov$_t$')
        
        if tmu_t is not None:
            plt.plot(rr, tmu_t, 'o--', color='xkcd:turquoise', markersize=10, linewidth=2.5, label=r'Tree $\mu_t$')
            if tcov_t is not None:
                plt.plot(rr, tcov_t, 'o--', color='xkcd:yellowgreen', markersize=6, linewidth=1.5, label=r'Tree cov$_t$')
        
        if (rcov_t is not None) or (tmu_t is not None):
            plt.legend(fontsize=14)
        
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.xlabel("Rollout runs", labelpad=15, fontsize=17)
    #plt.ylabel(r'$\mu_{t,10}$', rotation=0, labelpad=15, fontsize=17)
    plt.ylabel("Mean comp. time [s]", labelpad=15, fontsize=17)

    locmaj = mpl.ticker.LogLocator(base=10,numticks=10) 
    ax.xaxis.set_major_locator(locmaj)
    plt.xticks(fontsize=14)
    locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=10)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    plt.yticks(fontsize=14)
    #ax.set(xlim=(0, 4e4), ylim=(1e-3, 1e3))
    #plt.axis([0, 4e4, 1e-3, 1, 1e2])
    plt.tight_layout()
    
    if save:
        plt.savefig("timedrolloutruns8.svg")
    
    plt.show()
    return


# function that plots the increase in computation time when increasing the 
# number of rollout runs: can supply pure Rollout run data and Tree run data
def time_pomcp_run_plot(pr, pmu_t, pcov_t=None, rollout_runs=None, save=False):
    
    plt.figure(figsize=(10,6))
    ax = plt.gca()
    
    if type(pmu_t) == list:
        shapes = ['o--', 's--', 'v--', '*--', 'p--']
        for k in range(len(pmu_t)):
            string = r'N_{R}'
            plt.plot(pr, pmu_t[k], shapes[k], color='xkcd:magenta', markersize=10, linewidth=2.5, label=rf'${string} = {rollout_runs[k]}$')
        
        order = np.flip(np.arange(len(pmu_t)))

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=14)
        
    else: 
        plt.plot(pr, pmu_t, 'o--', color='xkcd:sienna', markersize=10, linewidth=2.5, label=r'Rollout $\mu_t$')
        if pcov_t is not None:
            plt.plot(pr, pcov_t, 'o--', color='xkcd:tan', markersize=6, linewidth=1.5, label=r'Rollout cov$_t$')

    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.xlabel("Tree iterations", labelpad=15, fontsize=17)
    #plt.ylabel(r"$\mu_{t,10}$", rotation=0, labelpad=15, fontsize=17)
    plt.ylabel("Mean comp. time [s]", labelpad=15, fontsize=17)

    locmaj = mpl.ticker.LogLocator(base=10,numticks=10) 
    ax.xaxis.set_major_locator(locmaj)
    ax.yaxis.set_major_locator(locmaj)
    plt.xticks(fontsize=14)
    locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=10)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    plt.yticks(fontsize=14)
    
    if pcov_t is not None:
        plt.legend(fontsize=14)
    
    plt.tight_layout()
    
    if save:
        plt.savefig("timedpomcpruns6.svg")
    
    plt.show()
    return

"""
##############################################################################
################################ Belief plots ################################
##############################################################################
"""

# function that plots the suggested action to take for a belief grid of D & K
def belief_grid_visualization(D, K, Z, colormap='viridis', plot_cbar=True, save=False):
    Z_min = int(np.min(Z))
    Z_max = int(np.max(Z))
    
    if colormap == 'custom':
        cmap = plt.get_cmap('viridis', Z_max - Z_min + 1)
        colors = cmap.colors
        
        colors[0,:] = list(mpl.colors.to_rgba('tab:orange'))
        colors[1,:] = list(mpl.colors.to_rgba('tab:blue'))
        colors[2,:] = list(mpl.colors.to_rgba('tab:green'))
        colors[3,:] = list(mpl.colors.to_rgba('tab:gray'))
       
    else:
        cmap = plt.get_cmap(colormap, Z_max - Z_min + 1)
        if isinstance(cmap, mpl.colors.LinearSegmentedColormap):
            colors = cmap(np.arange(0,cmap.N))
        elif isinstance(cmap, mpl.colors.ListedColormap):
            colors = cmap.colors
    
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', colors, cmap.N)

    # tell the colorbar to tick at integers
    if plot_cbar:
        plt.figure(figsize=(7.3333,6))
        # get discrete colormap
        #cmap = plt.get_cmap(colormap, Z_max - Z_min + 1)
        # set limits .5 outside true range
        p = plt.pcolormesh(D, K, Z, cmap=cmap, vmin=Z_min - 0.5, vmax=Z_max + 0.5)
        cbar = plt.colorbar(p, ticks=np.arange(Z_max - Z_min + 1))
        # ticklabels as individual actions
        if (colormap == 'custom') and (Z_max - Z_min + 1 == 4):
            cbar.ax.set_yticklabels([r'$a_0$', r'$a_1$', r'$a_2$', r'$a_3$'])
        cbar.ax.tick_params(labelsize=14)
    else:
        plt.figure(figsize=(6,6))
        p = plt.pcolormesh(D, K, Z, cmap=cmap, vmin=Z_min - 0.5, vmax=Z_max + 0.5)
    plt.xlabel(r"$\mu_D{''}$", fontsize=17)
    plt.ylabel(r"$\mu_K{''}$", rotation=0, fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.axis([np.min(D), np.max(D), np.max(K), np.min(K)])
    plt.tight_layout()
    if save:
        if plot_cbar:
            plt.savefig("100x100beliefgrid_t_20_e_50_buckets_10_with_bar.svg")
        else:
            plt.savefig("100x100beliefgrid_t_20_e_50_buckets_10.svg")
    plt.show()
    return


# function that plots multiple belief trajectories representing different 
# timesteps as different colors; results in moving point cloud
def belief_progress(beliefs, colormap='viridis', save=False):

    plt.figure(figsize=(10,6))
    
    cmap = plt.get_cmap(colormap, len(beliefs))
    if isinstance(cmap, mpl.colors.LinearSegmentedColormap):
        colors = cmap(np.arange(0,cmap.N))
    elif isinstance(cmap, mpl.colors.ListedColormap):
        colors = cmap.colors
    
    # set starting element to black
    colors[0,:] = [0, 0, 0, 1.]
    # construct new color map with the custom map
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', colors, cmap.N)

    bounds = np.linspace(0,21,22)
    
    # plot first element as large black spot
    plt.scatter(beliefs[0]['mu_d'], beliefs[0]['mu_k'], s=200, c=np.expand_dims(colors[0,:], axis=0))
    for k in range(1, len(beliefs)):
        plt.scatter(beliefs[k]['mu_d'], beliefs[k]['mu_k'], s=50, c=np.expand_dims(colors[k,:], axis=0))
        #im = ax.plot(beliefs[k]['mu_d'], beliefs[k]['mu_k'], 'o', c=colors[k,:])

    #plt.grid()
    plt.xlabel(r"$\mu_D{''}$", fontsize=17)
    plt.ylabel(r"$\mu_K{''}$", rotation=0, fontsize=17, labelpad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    axes = plt.gca()
    print(axes.get_xlim(), axes.get_ylim())
    plt.vlines(0, ymin=2, ymax=10)
    #plt.axis([-197, 39, -0.7, 9.5])
    cax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=1)
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, spacing='proportional', ticks=np.arange(-0.5, 21.5), boundaries=bounds, format='%1i')
    cax.set_ylabel('t', rotation=0, fontsize=15, labelpad=8)
    #plt.tight_layout()
    if save:
        plt.savefig("beliefprogress_s_500_sigma_e_100.svg")
    plt.show()
    return


# function that plots the evolution of the belief of a single trajectory 
def single_belief_trajectory(beliefs, colormap='viridis', index=0, save=False):
    #fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
    fig = plt.figure(figsize=(10,6))
    cmap = plt.get_cmap(colormap, len(beliefs))
    if isinstance(cmap, mpl.colors.LinearSegmentedColormap):
        colors = cmap(np.arange(0,cmap.N))
    elif isinstance(cmap, mpl.colors.ListedColormap):
        colors = cmap.colors
    
    # set starting element to black
    colors[0,:] = [0, 0, 0, 1.]
    # construct new color map with the custom map
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', colors, cmap.N)
    bounds = np.linspace(0,21,22)

    mu_ds = [beliefs[k]['mu_d'][index] for k in range(len(beliefs))]
    mu_ks = [beliefs[k]['mu_k'][index] for k in range(len(beliefs))]

    for k in range(len(mu_ds)-1):
        plt.scatter(mu_ds[k], mu_ks[k], s=100, c=np.expand_dims(colors[k,:], axis=0))
        plt.plot(mu_ds[k:k+2], mu_ks[k:k+2], '--', c=colors[k+1,:])
    
    plt.scatter(mu_ds[-1], mu_ks[-1], s=100, c=np.expand_dims(colors[-1,:], axis=0))

    plt.grid()
    plt.xlabel(r"$\mu_D{''}$", fontsize=17)
    plt.ylabel(r"$\mu_K{''}$", rotation=0, fontsize=17, labelpad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.axis([-150, 10, 4, 9])
    cax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=1)
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, spacing='proportional', ticks=np.arange(-0.5, 21.5), boundaries=bounds, format='%1i')
    cax.set_ylabel('t', rotation=0, fontsize=15, labelpad=8)
    #mpl.tight_layout()
    if save:
        plt.savefig("belieftrajectory.svg")
    
    plt.show()
    return


# function which plots the beliefs at different timesteps and indicates the specific action taken
def belief_and_action_plot(belief, index, colormap='viridis', save=False):
    
    if colormap == 'custom':
        cmap = plt.get_cmap('viridis', 4)
        colors = cmap.colors
        
        colors[0,:] = list(mpl.colors.to_rgba('tab:orange'))
        colors[1,:] = list(mpl.colors.to_rgba('tab:blue'))
        colors[2,:] = list(mpl.colors.to_rgba('tab:green'))
        colors[3,:] = list(mpl.colors.to_rgba('tab:gray'))
        bounds = np.linspace(0,4,5)
    else:
        cmap = plt.get_cmap(colormap, len(beliefs))
        if isinstance(cmap, mpl.colors.LinearSegmentedColormap):
            colors = cmap(np.arange(0,cmap.N))
        elif isinstance(cmap, mpl.colors.ListedColormap):
            colors = cmap.colors

        # set starting element to black
        colors[0,:] = [0, 0, 0, 1.]
        # construct new color map with the custom map
        bounds = np.linspace(0,21,22)
    
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', colors, cmap.N)
    

    plt.figure(figsize=(10,6))
    assert type(index) == list
    for i in index:
        mu_ds = belief[i]['mu_d']
        mu_ks = belief[i]['mu_k']
        acts  = belief[i+1]['act']

        for k in range(len(mu_ds)):
            plt.scatter(mu_ds[k], mu_ks[k], s=10, c=np.expand_dims(colors[acts[k],:], axis=0))

    plt.grid()
    plt.xlabel(r"$\mu_D{''}$", fontsize=17)
    plt.ylabel(r"$\mu_K{''}$", rotation=0, fontsize=17, labelpad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.axis([-197, 39, -0.7, 9.5])
    plt.axis([-160, 20, 1.5, 7])
    #plt.axis([-184.24356806976368, 9.555819340323422, 1.5566793717860714, 10.92837882841855])
    cax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=1)
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, spacing='proportional', ticks=np.arange(-0.5, 21.5), boundaries=bounds, format='%1i')
    cax.set_ylabel('t', rotation=0, fontsize=15, labelpad=8)
    #plt.tight_layout()
    if save:
        plt.savefig("beliefandaction.svg")
    
    plt.show()
    return
