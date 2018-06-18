import numpy as np

import sgraphnn.util.block
from sgraphnn import util


CONSTANT_SEED = 1


task_parameters = {
    # Size of input vocabulary (indexes into dict)
    'Voc'                :       3,
    'nb_clusters_target' :       2,
    'nb_communities'     :      10,
    'p'                  :       0.5,
    'q'                  :       0.1,
    # Min and max cluster size
    'size_max'           :      25,
    'size_min'           :      15,
    # Subgraph size (duh)
    'size_subgraph'      :      20,
}

# Initial random graph params
task_parameters['W0'] = sgraphnn.util.block.random_graph(task_parameters['size_subgraph'], task_parameters['p'])
task_parameters['u0'] = np.random.randint(task_parameters['Voc'], size=task_parameters['size_subgraph'])

