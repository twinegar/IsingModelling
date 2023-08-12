# Modified by Tomas 17/5/2023 to fit to depression modelling data
# Key modifications: 
# Weighted interactions between all nodes in the system (done)
# External field interactions
# Different classification function (done)
# Simulating timeseries data (done; single subject level)

#%% 
%matplotlib inline
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import networkx as nx

#%%-----------------------------------------------------------------------
#Setup 
#-------------------------------------------------------------------------
k        = 9    # number of nodes
t        = 1000  # number of time points
n        = 2    # number of subjects
beta     = np.linspace(1.53, 3.28, t)
density  = 0.5 
extfield = np.zeros(k) + 5 # strength of external force field at each node
threshold = 4
graph = np.random.uniform(0, 1, (k,k)) * np.random.binomial(1, density, (k,k)) # make weighted adjacency matrix
np.fill_diagonal(graph, 0)                                                      # nodes are not connected to themselves 
graph = np.tril(graph) + np.triu(graph.T, 1)                                    # make it symmetric
print(np.allclose(graph, graph.T, rtol=1e-05, atol=1e-05))                      # check symmetry
plt.imshow(graph)                                                               # display heatmap of adjacency matrix
thresholds = np.random.uniform(0,1,k)                                           # set thresholds uniformly
eqSteps = 1024 # number of MC sweeps for equilibration
mcSteps = t    # number of MC sweeps for calculation
E,M,C,X  = np.zeros(t), np.zeros(t), np.zeros(t), np.zeros(t)
#%%-----------------------------------------------------------------------
#Plotting 
#-------------------------------------------------------------------------
def MakeGraph(var1, var2): # scatter over time  
    f = plt.figure(figsize=(18, 10)); # plot the calculated values    

    sp =  f.add_subplot(2, 2, 1 );
    plt.scatter(np.linspace(0, len(var1), len(var1)), var1, s=50, marker='o', color='IndianRed')
    plt.xlabel("Time", fontsize=20);
    plt.ylabel("Energy ", fontsize=20);     plt.axis('tight');

    sp =  f.add_subplot(2, 2, 2 );
    plt.plot(np.linspace(0, len(var1), len(var1)), abs(var2),color='RoyalBlue')
    plt.xlabel("Time", fontsize=20); 
    plt.ylabel("Magnetization ", fontsize=20);   plt.axis('tight');

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
#%%-----------------------------------------------------------------------
#Functions
#-------------------------------------------------------------------------
def initialstate(k):   
    # generates a random spin configuration for initial condition
    state = 2*np.random.randint(2, size=(k))-1
    return state

def initialstate0(k): 
    # generates a spin configuation with all negative spins
    state = np.zeros(k) - 1
    return state

def calcEnergy(config):
    #Energy of a given configuration
    energy = 0
    for i in range(len(config)):
        S = config[i]
        nb = np.dot(config, graph[i])
        energy += -nb*S - extfield[i] * S
    return energy

def mcmove(config, graph, beta, h):
    # Monte Carlo move using Metropolis algorithm
    for i in range(k):
        a = np.random.randint(0, k)
        S =  config[a]
        nb = np.dot(config, graph[a])
        cost = -2 * S * nb - 2 * h[a] * S
        if cost < 0:
            S *= -1
        elif rand() < np.exp(-cost*beta):
            S *= -1
        config[a] = S
    return config

def calcMag(config):
    #Magnetization of a given configuration
    if np.sum(config) > (threshold - (k-threshold)):
        mag = 2
    else: 
        mag = 1 
    return mag

def mag(config):
    m = 1/len(config) * np.sum(config)
    return m 

# %%----------------------------------------------------------------------
#  MAIN PART OF THE CODE (Per subject random initial condition)
#-------------------------------------------------------------------------
'''Broken ATM
PlotNetwork(graph)
for tt in range(n):
    E1 = M1 = E2 = M2 = 0
    config = initialstate(k)
    iT=beta[tt]
    
    for i in range(eqSteps):         # equilibrate
        mcmove(config, graph, iT)    # Monte Carlo moves

    for i in range(mcSteps):
        mcmove(config, graph, iT)           
        Ene = calcEnergy(config)     # calculate the energy
        Mag = calcMag(config)        # calculate the magnetisation

        E1 += Ene
        M1 = Mag
        M2 += Mag*Mag 
        E2 += Ene*Ene

    E[tt] = E1
    M[tt] = M1

    MakeGraph(E, M)
    plt.show()
'''
# %%
n = 1 
E = []; M = []

for subjects in range(n): 
    extfield = np.zeros(k)-10
    E1 = M1 = 0 
    iT= 0
    config = initialstate0(k)
    
    for i in range(eqSteps):
        mcmove(config, graph, iT, extfield)
    for i in range(mcSteps): 
        if i % 250 == 0: 
            plt.imshow(np.reshape(config, (3,3)), cmap = 'hot')
            print(("Config at", i, str(extfield[1])))
            plt.show()
        pastconfig = config
        mcmove(config, graph, iT, extfield)
        if config.all() == pastconfig.all():
            extfield += 0.05
        Ene = calcEnergy(config)     # calculate the energy
        Mag = calcMag(config)        # calculate the magnetisation

        E.append(Ene)
        M.append(Mag)

    MakeGraph(np.asarray(E), np.asarray(M))
    plt.show()
#%%
#Constant vars 
n = 5 
extfield = np.zeros(k) 

for subjects in range(n): 
    E = []; M = []
    E1 = M1 = 0 
    iT= 0
    config = initialstate(k)
    #for i in range(eqSteps):
        #mcmove(config, graph, iT, extfield)
    #for i in range(mcSteps): 
    for i in range(mcSteps): 
        if i % 250 == 0: 
            plt.imshow(np.reshape(config, (3,3)), cmap = 'hot')
            print(("Config at", i, extfield[1]))
            plt.show()
        pastconfig = config
        mcmove(config, graph, iT, extfield)
#        if config.all() == pastconfig.all():
#            extfield += -1/mcSteps
        Ene = calcEnergy(config)     # calculate the energy
        Mag = calcMag(config)        # calculate the magnetisation

        E.append(Ene)
        M.append(Mag)
    extfield += -0.2
    MakeGraph(np.asarray(E), np.asarray(M))
    plt.show()
#%%
plt.imshow(np.reshape(config, (3,3)), cmap = 'hot') # plot final time point configuration
#%%
