#%% 
# Modified by Tomas 17/5/2023 to fit to depression modelling data
# Key modifications: 
# Weighted interactions between all nodes in the system
# Probabalistic sparse connectivity of symptoms
# External field interactions
# Different classification function
# Simulating timeseries data 
# Within individual symptom networks

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

#%% 
# Setup 
k        = 9    #number of nodes
t        = 3000 #number of time points
n        = 10   #number of subjects
beta     = 0.5  #inverse temperature term
density  = 0.5  #1 - sparcity 
threshold = 5   #how many symptoms have to be present to be classified as depressed                                                                   #initialize counter
#%% 
# Functions 
def initialstate(k):  
    #Generates a random spin configuration for initial condition
    state = 2*np.random.randint(2, size=(k))-1
    return state

def calcEnergy(config, extf, gr):
    #Calculates the energy of a given configuration
    energy = 0
    for i in range(len(config)):
        S = config[i]
        nb = np.dot(config, gr[i])
        energy += -nb*S - extf[i] * S
    return energy

def mcmove(config, cost, ext, g):
    #Runs Markov chain through a Monte Carlo process via the Metropolis algorithm
    old = calcEnergy(config, ext, g) #prior energy
    a = np.random.randint(0, k)   #randomly pick symptom to interact with
    config[a] = config[a] * (-1)  #negate sign
    new = calcEnergy(config, ext, g) #calculate new energy
    if (old > new + cost) or (np.random.rand() < np.exp(new * beta) - np.exp(new * beta) * cost): #conditions for accepting proposal
        return new
    else: 
        config[a] = config[a] * (-1) #reverse proposal if conditions are not met
        return old

def calcMag(config):
    #Subject condition at a given configuration
    if np.sum(config) > (threshold - (k-threshold)): #check if n of active symptoms is greater than the threshold condition
        mag = 8.5
    else: 
        mag = 0.5
    return mag

def switchcost(mlist, clist): 
    #Decaying cost function for state switching
    if len(mlist) < 2:
        return 0
    elif mlist[-1] != mlist[-2]:
        return 100 #somewhat arbitrary currently 
    else: 
        return clist[-1] * 0.99
    
def adderr(data, amount):
    # Measurement Error and Individual differences
    # Sample an individual mean and SD for each participant to simulate individual variability 
    array = (data.copy()).astype('float64')                #copy data and convert to float
    res = []
    for i in range(len(data)):
        m = np.random.normal(0, amount)                    #sample mean for subject
        sd = np.random.uniform(0, amount)                  #sample sd
        for j in range(len(data[i])):
            array[i][j] += np.random.normal(m, sd)         #add error
        res.append(np.append(array[i], calcMag(array[i]))) #create new array including condition assignment
    return res

#%% 
# Simulation
def sim(nn, tt): 
    #Takes in n subjects, t timepoints, and a counter for naming output files
    data = []
    for subjects in range(nn): 
        sample = []                                                                     #list for sampled data
        graph = np.random.uniform(0, 1, (k,k)) * np.random.binomial(1, density, (k,k))  #make weighted adjacency matrix
        np.fill_diagonal(graph, 0)                                                      #nodes are not connected to themselves 
        graph = np.tril(graph) + np.triu(graph.T, 1)                                    #make it symmetric
        print(np.allclose(graph, graph.T, rtol=1e-05, atol=1e-05))                      #check symmetry
        thresholds = np.random.uniform(0,1,k)                                           #set transition thresholds uniformly
        extfield = np.zeros(k)                                                          #set external field interactions (zero for default sims)
        config = initialstate(k)                                                        #generate initial system state
        maglst, energylst, cstlist, sumlst = [], [], [], []                             #initialize lists for tracking vars
        for i in range(tt): 
            cstlist.append(switchcost(maglst, cstlist))
            energylst.append(mcmove(config, cstlist[-1], extfield, graph))
            maglst.append(calcMag(config))
            sumlst.append(sum((config +1)/ 2))
        sample.append(sumlst)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Number of Active Symptoms and Distribution Over Time')
        ax1.plot(np.linspace(0, len(sample[0]), len(sample[0])), sample[0], color= "k")
        ax2.hist(sample, orientation= 'horizontal', color= "k")
        plt.setp((ax1, ax2), ylim=(0,9))
        plt.show()
        data.append(sample)
    return data

#%%
# Run simulation
dt = sim(n, t)

#%%
# Post stats
stats = []
for i in range(len(dt)): 
    stats.append({'mean' : np.mean(dt[i]), 'median' : np.median(dt[i]), 'var' : np.var(dt[i]), 'kurtosis' : sp.stats.kurtosis(dt[i][0])})
# %%
