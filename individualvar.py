#%% 
# Modified by Tomas 15/8/2023 to fit to depression modelling data
# Key modifications: 
# Within individual symptom networks
# Histogram plotting
# Descriptive stats

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp
#%% 
# Setup 
k        = 9     #number of nodes
t        = 10000 #number of time points
n        = 1000  #number of subjects
beta     = 0.5   #inverse temperature term
density  = 0.5   #1 - sparcity 
threshold = 5    #how many symptoms have to be present to be classified as depressed                                                                   #initialize counter
#%% 
# Functions 
def initialState(k):  
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

def mcMove(config, cost, ext, g):
    #Runs Markov chain through a Monte Carlo process via the Metropolis algorithm
    old = calcEnergy(config, ext, g) #prior energy
    a = np.random.randint(0, k)      #randomly pick symptom to interact with
    config[a] = config[a] * (-1)     #negate sign
    new = calcEnergy(config, ext, g) #calculate new energy
    if (old > new + cost) or (np.random.rand() < 0.02): #np.exp(new * beta) - np.exp(new * beta) * cost): #conditions for accepting proposal
        return new
    else: 
        config[a] = config[a] * (-1) #reverse proposal if conditions are not met
        return old

def calcMag(config):
    #Subject condition at a given configuration
    if np.sum(config) > (threshold - (k-threshold)): #check if n of active symptoms is greater than the threshold condition
        mag = 2
    else: 
        mag = 1
    return mag

def switchCost(mlist, clist): 
    #Decaying cost function for state switching (imperfect)
    if len(mlist) < 2:
        return 0
    elif mlist[-1] != mlist[-2]:
        return 15 #somewhat arbitrary currently (play around 15-20 seem reasonable by energy landscape scaling)
    else: 
        return clist[-1] * 0.99
    
def addErr(data, amount):
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
    tax = []
    samplet = np.random.randint(0, tt)
    for subjects in range(nn): 
        graph = np.random.uniform(0, 1, (k,k)) * np.random.binomial(1, density, (k,k))  #make weighted adjacency matrix
        np.fill_diagonal(graph, 0)                                                      #nodes are not connected to themselves 
        graph = np.tril(graph) + np.triu(graph.T, 1)                                    #make it symmetric
        if not (np.allclose(graph, graph.T, rtol=1e-05, atol=1e-05)):
            print('Error: graph not symmetrical')                                       #check symmetry
        extfield = np.zeros(k)                                                          #set external field interactions (zero for default sims)
        config = initialState(k)                                                        #generate initial system state
        maglst, energylst, cstlist, sumlst, sample = [], [], [], [], []                           #initialize lists for tracking vars
        for i in range(samplet + 1): 
            cstlist.append(switchCost(maglst, cstlist))
            energylst.append(mcMove(config, cstlist[-1], extfield, graph))
            maglst.append(calcMag(config))
            sumlst.append(sum((config +1)/ 2))
            if i == samplet:
                tax.append(config)
        sample.append(sumlst)
        '''
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Number of Active Symptoms and Distribution Over Time')
        ax1.plot(np.linspace(0, len(sample[0]), len(sample[0])), sample[0], color= "k")
        ax2.hist(sample, orientation= 'horizontal', color= "k")
        plt.setp((ax1, ax2), ylim=(0,9))
        plt.show()
        '''
        data.append(sample)
    data = np.asarray(data)
    plt.hist(data.flatten())
    plt.title('Group Level Distribution of Symptom Sum Score')
    plt.xlabel('Number of Active Symptoms')
    plt.ylabel('Count')
    plt.show()
    return tax

#%%
# Run simulation and write results
counter = 0
for i in range(10):
    dt = sim(n, t)
    dt = np.asarray(dt)
    MEn = addErr(dt, 0)   #no error
    MEs = addErr(dt, 0.1) #small error 
    MEm = addErr(dt, 0.2) #moderate error
    colnames = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'condition']  #set column names for output data
    # Write data to xlsx for use with R
    MEn = pd.DataFrame(MEn, columns= colnames)
    MEs = pd.DataFrame(MEs, columns= colnames)
    MEm = pd.DataFrame(MEm, columns= colnames)
    MEn.to_excel("FinMEn" + str(counter) + ".xlsx", index=False)
    MEs.to_excel("FinMEs" + str(counter) + ".xlsx", index=False)
    MEm.to_excel("FinMEm" + str(counter) + ".xlsx", index=False)
    counter += 1
#%%
# Post stats
stats = []
for i in range(len(dt)): 
    stats.append({'mean' : np.mean(dt[i]), 'median' : np.median(dt[i]), 'var' : np.var(dt[i]), 'sd' : np.sqrt((np.var(dt[i]))), 'kurtosis' : sp.kurtosis(dt[i][0])})
#(access stats for given subject with following syntax: "stats[subject number: int][stat name : string]")

# %%
