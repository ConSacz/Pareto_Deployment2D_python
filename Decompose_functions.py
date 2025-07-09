import numpy as np
from Domination_functions import get_pareto_front

def normalized(data):
    x_sta = np.min(data, axis = 0)
    x_nad = np.max(data, axis = 0)
    data_normalized = (data - x_sta) / (x_nad - x_sta)
    
    return data_normalized

def weight_assign(pop,RP):
    pop.sort(key=lambda p: p['Cost'][1])
    x_sta = RP[:,0].flatten()
    x_nad = RP[:,1].flatten()
    Npop = len(pop)
    Nf = len(pop[0]['Cost'])
    w = np.zeros((Npop, Nf), dtype=int)
    
    # PF = get_pareto_front(pop)
    # NPF = len(PF)
    # data_PF = np.array([ind['Cost'].flatten() for ind in PF])
    # data_normalized = normalized(data_PF)
    
    # indices = np.array([i // (Npop//NPF+1) for i in range(Npop)])
    # w = data_normalized[indices]
    weights = np.linspace(x_sta,x_nad,10)
    weights[:, 1] = weights[::-1, 1]
    weights[:, [0, 1]] = weights[:, [1, 0]]
    indices = np.array([i // (5) for i in range(Npop)])
    w = weights[indices]
    
    return pop, w
    