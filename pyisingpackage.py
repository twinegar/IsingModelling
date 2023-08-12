#%% 
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import isingnetworks as isn
#%% 
k        = 9    # number of nodes
t        = 1000  # number of time points
n        = 1    # number of subjects
density  = 0.5 
graph = np.random.uniform(0, 1, (k,k)) * np.random.binomial(1, density, (k,k)) # make weighted adjacency matrix
np.fill_diagonal(graph, 0)                                                      # nodes are not connected to themselves 
graph = np.tril(graph) + np.triu(graph.T, 1)                                    # make it symmetric
print(np.allclose(graph, graph.T, rtol=1e-05, atol=1e-05))                      # check symmetry
plt.imshow(graph)                                                               # display heatmap of adjacency matrix
thresholds = np.random.uniform(0,1,k)                                           # set thresholds uniformly
eqSteps = 1024 # number of MC sweeps for equilibration
mcSteps = t    # number of MC sweeps for calculation
#%%
def PlotNetwork(graph): # plot estimated network structure
    fig, ax = plt.subplots()
    G = nx.from_numpy_matrix(graph)    # convert graph to nx object
    pos = nx.spring_layout(G)          # positioning
    cor = np.ndarray.flatten(graph)*10 # make edge values a list
    width = np.absolute(cor)           # get magnitude of interactions 
    clr = []
    for i in cor:                      # check if correlations are positive or negative 
        if i < 0: 
            clr.append("red")
        else: 
            clr.append("green")
    nx.draw_networkx_nodes(G, pos, ax= ax, node_color="white", edgecolors="blue" )
    nx.draw_networkx_edges(G, pos, width=width, ax=ax, edge_color= clr)
    _ = nx.draw_networkx_labels(G, pos, ax=ax)
    return G
# %%
g = PlotNetwork(graph)
model = isn.IsingModel(g)
model.set_J(1)
model.set_iterations(1000)
model.set_initial_state(0)
temperature = np.arange(0, 10, 0.05)
model.viz(temperature)

# %%
