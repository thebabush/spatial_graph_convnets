import numpy as np
from . import block
import torch
import scipy.sparse as sp


class variable_size_graph():

    def __init__(self, task_parameters): 

        # parameters
        vocab_size = task_parameters['Voc']
        nb_of_clust = task_parameters['nb_clusters_target']
        clust_size_min = task_parameters['size_min']
        clust_size_max = task_parameters['size_max']
        p = task_parameters['p']
        q = task_parameters['q']
        self_loop = True
        W0 = task_parameters['W0']
        u0 = task_parameters['u0']

        # create block model graph and put random signal on it
        W,c=block.unbalanced_block_model(nb_of_clust,clust_size_min,clust_size_max,p,q)
        u=np.random.randint(vocab_size,size=W.shape[0])
    
        # add the subgraph to be detected
        W,c=block.add_a_block(W0,W,c,nb_of_clust,q)
        u=np.concatenate((u,u0),axis=0)

        # shuffle
        W,c,idx=block.schuffle(W,c)
        u=u[idx]
        u=torch.from_numpy(u)
        u=u.long()

        # add self loop
        if self_loop:
            for i in range(W.shape[0]):
                W[i,i]=1

        # create the target
        target= (c==nb_of_clust).astype(float)
        target=torch.from_numpy(target)
        target=target.long()

        # mapping matrices
        W_coo=sp.coo_matrix(W)
        nb_edges=W_coo.nnz
        nb_vertices=W.shape[0]
        edge_to_starting_vertex=sp.coo_matrix( ( np.ones(nb_edges) ,(np.arange(nb_edges), W_coo.row) ),
                                               shape=(nb_edges, nb_vertices) )
        edge_to_ending_vertex=sp.coo_matrix( ( np.ones(nb_edges) ,(np.arange(nb_edges), W_coo.col) ),
                                               shape=(nb_edges, nb_vertices) )

        self.adj_matrix=W  
        self.edge_to_starting_vertex=edge_to_starting_vertex
        self.edge_to_ending_vertex=edge_to_ending_vertex  
        self.signal=u
        self.target=target

