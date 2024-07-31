# Auxilliary code
import numpy as np

import default
import environment as env
import parameters as pars

# builds a tree
class Node():
    def __init__(self, type: str, a_or_o: int):

        """
        Node class defining the node structure and the node operations

        -param node_type: Type of node, either 'action_node' or 'observation_node'
        -param action_or_bucket_observation: integer describing the current state
        -param d_sample_generator: function which generates new d samples based on the posterior belief
        -param k_sample_generator: function which generates new k samples based on the posterior belief

        -param kwargs: Kwargs.
        """

        assert type in ['action_node', 'observation_node']

        # set key node attrubutes which are always set
        self.node_type = type
        self.action_or_bucket_observation = a_or_o


    # function which sets the d & k sample generator as an attribute of the node
    def observation_belief_generator_update(self, posterior_mu_d, posterior_mu_k, p, t):
        assert self.node_type == 'observation_node'
        self.t = t
        #self.mu_d_post = posterior_mu_d
        #self.mu_k_post = posterior_mu_k
        self.d_sample_generator, self.k_sample_generator = env.observation_belief_generator_independent(posterior_mu_d, posterior_mu_k, p, t)
        self.d_and_k_sample_generator = env.observation_belief_generator(posterior_mu_d, posterior_mu_k, p, t)
        return


# builds a tree
class Tree():
    def __init__(self, initial_observation, bucket_observation, **kwargs):

        """
        Tree class defining the tree structure and the tree operations

        -param count: current count of number of nodes (also id of node)
        -param nodes: dictionary of all nodes in the tree

        -param kwargs: Kwargs.
        """

        self.count = kwargs.get('initial_count', default.INITIAL_COUNT) # -1
        self.nodes = kwargs.get('initial_nodes', default.INITIAL_NODES.copy()) # {}
        self.giveParameters = ['isRoot', {}, 0, 0, Node(type='observation_node', a_or_o=bucket_observation)]
        self.nodes[self.count] = self.giveParameters


    # Expand the tree by one node.
    # If the result of an action give IsAction = True
    def ExpandTreeFrom(self, parent, a_or_o, IsAction):
        #print("\n\n\n\n\n\n\n")
        #print("Here!!")
        self.count += 1
        if IsAction: # a_or_o is an integer representing actions 0-3
            # add action node to tree
            self.nodes[self.count] = [parent, {}, 0, 0, Node(type='action_node', a_or_o=a_or_o)]
            # inform parent node: construct connection from observation node to action node
            self.nodes[parent][1][a_or_o] = self.count
            #print(self.nodes)

        else: # a_or_o is a tuple containing: (observation, bucket_observation)
            _, obs_bucket = a_or_o
            # add observation node to the tree
            self.nodes[self.count] = [parent, {}, 0, 0, Node(type='observation_node', a_or_o=obs_bucket)]
            # inform parent node: construct connection from action node to observation node
            self.nodes[parent][1][obs_bucket] = self.count
            #print(self.nodes)

        return

    # Check given nodeindex corresponds to leaf node
    def isLeafNode(self, n):
        if self.nodes[n][2] == 0:
            return True
        else:
            return False


    # Removes a node and all its successors
    def prune(self, node):
        children = self.nodes[node][1]
        del self.nodes[node]
        for _, child in children.items():
            self.prune(child)
        return


    # make new root and update children
    def make_new_root(self, new_root):
        self.nodes[-1] = self.nodes[new_root].copy()
        del self.nodes[new_root]
        self.nodes[-1][0] = 'isRoot'
        # update children
        for _ , child in self.nodes[-1][1].items():
            self.nodes[child][0] = -1
        return


#UCB score calculation
def UCB(N_total, n_local, value, expl_c): #N=Total, n= local, V = value, c = parameter
    return value - expl_c*np.sqrt(np.log(N_total)/n_local)
