#%% 
# Modified by Tomas 20/8/2023 to examine energy states of the model

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
#%% 
# Setup 
k        = 9    #number of nodes
n        = 10  #number of subjects
density  = 0.5  #1 - sparcity 
#%% 
# Functions 
def calcEnergy(config, extf, gr):
    #Calculates the energy of a given configuration
    energy = 0
    for i in range(len(config)):
        S = config[i]
        nb = np.dot(config, gr[i])
        energy += -nb*S - extf[i] * S
    return energy

def stateeng(ext, g):
    #iterate through numbers of active symptoms and record energy
    res = []
    state = 2*np.zeros(k) -1
    for i in range(len(state)): 
        res.append(calcEnergy(state, ext, g))
        state[i] = state[i] * (-1)
    return res

#%% 
# Simulation
def sim(nn): 
    #Takes in n subjects and calculates energy landscape for 
    data = []
    for subjects in range(nn):
        graph = np.random.uniform(0, 1, (k,k)) * np.random.binomial(1, density, (k,k))  #make weighted adjacency matrix
        np.fill_diagonal(graph, 0)                                                      #nodes are not connected to themselves 
        graph = np.tril(graph) + np.triu(graph.T, 1)                                    #make it symmetric
        print(np.allclose(graph, graph.T, rtol=1e-05, atol=1e-05))                      #check symmetry
        extfield = np.zeros(k)                                                          #set external field interactions (zero for default sims)
        plt.plot(np.linspace(0, 9, 9), stateeng(extfield, graph))
    plt.show()
    return data
#%%
dt = sim(100)
#%%
