#%% 
# Modified by Tomas 17/5/2023 to fit to depression modelling data
# Key modifications: 
# Weighted interactions between all nodes in the system
# External field interactions
# Different classification function
# Simulating timeseries data 

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% 
# Setup 
k        = 9    #number of nodes
t        = 3000 #number of time points
n        = 10   #number of subjects
beta     = 0.5 
density  = 0.5  #1 - sparcity
threshold = 5   #how many symptoms have to be present to be classified as depressed
graph = np.random.uniform(0, 1, (k,k)) * np.random.binomial(1, density, (k,k))  #make weighted adjacency matrix
np.fill_diagonal(graph, 0)                                                      #nodes are not connected to themselves 
graph = np.tril(graph) + np.triu(graph.T, 1)                                    #make it symmetric
print(np.allclose(graph, graph.T, rtol=1e-05, atol=1e-05))                      #check symmetry
plt.imshow(graph, cmap='Greys')                                                 #display heatmap of adjacency matrix
thresholds = np.random.uniform(0,1,k)                                           #set transition thresholds uniformly
colnames = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'condition']  #set column names for output data
counter = 0                                                                     #initialize counter
#%% 
# Functions 
def initialstate(k):  
    #Generates a random spin configuration for initial condition
    state = 2*np.random.randint(2, size=(k))-1
    return state

def calcEnergy(config, extf):
    #Calculates the energy of a given configuration
    energy = 0
    for i in range(len(config)):
        S = config[i]
        nb = np.dot(config, graph[i])
        energy += -nb*S - extf[i] * S
    return energy

def mcmove(config, cost, ext):
    #Runs Markov chain through a Monte Carlo process via the Metropolis algorithm
    old = calcEnergy(config, ext) #prior energy
    a = np.random.randint(0, k)   #randomly pick symptom to interact with
    config[a] = config[a] * (-1)  #negate sign
    new = calcEnergy(config, ext) #calculate new energy
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
def sim(nn, tt, count): 
    #Takes in n subjects, t timepoints, and a counter for naming output files
    sample = []                            #list for sampled data
    samplet = t-1 #np.random.randint(10, tt, 1) #choose random timepoint for when to sample
    for subjects in range(nn): 
        extfield = np.zeros(k)             #set external field interactions (zero for default sims)
        config = initialstate(k)           #generate initial system state
        maglst, energylst, cstlist, sumlst = [], [], [], [] #initialize lists for tracking vars
        for i in range(tt): 
            cstlist.append(switchcost(maglst, cstlist))
            energylst.append(mcmove(config, cstlist[-1], extfield))
            maglst.append(calcMag(config))
            sumlst.append(sum((config +1)/ 2))
            if i == samplet:               #if we are at the sampling point record to array
                sample.append(config)
                if np.random.rand() < 0.5: #plot a few example timeseries        
                    plt.plot(np.linspace(0, len(maglst), len(maglst)), maglst, linewidth= 5, color= 'r')                    
                    plt.plot(np.linspace(0, len(sumlst), len(sumlst)),sumlst, linewidth= 5, color= 'k')
                    plt.ylim(-0.1,9.1) 
                    plt.xlabel('Time')
                    plt.ylabel('State')
                    plt.title('Individual Subject State Over Time')    
                    y = [-0.1, 9.1]
                    labels = ['Healthy', 'Depressed']
                    plt.yticks(y, labels, rotation= 'vertical')
                    plt.show()
        
    sample = np.asarray(sample)
    
    #Add error
    MEn = adderr(sample, 0)   #no error
    MEs = adderr(sample, 0.1) #small error 
    MEm = adderr(sample, 0.2) #moderate error

    #Write data to xlsx for use with R
    MEn = pd.DataFrame(MEn, columns= colnames)
    MEs = pd.DataFrame(MEs, columns= colnames)
    MEm = pd.DataFrame(MEm, columns= colnames)
    MEn.to_excel("MEn" + str(count)  + ".xlsx", index=False)
    MEs.to_excel("MEs" + str(count)  + ".xlsx", index=False)
    MEm.to_excel("MEm" + str(count)  + ".xlsx", index=False)
    
    return count + 1 #increment counter
#%%
# Run simulation
for i in range(1):
    counter = sim(n, t,  counter)

# %%
