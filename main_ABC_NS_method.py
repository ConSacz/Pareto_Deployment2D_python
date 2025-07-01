
import numpy as np
import matplotlib.pyplot as plt
# from scipy.io import savemat
# from scipy.spatial.distance import cdist
# import os
from CostFunction_MOO import CostFunction_MOO
from Graph import Graph
from Connectivity_graph import Connectivity_graph
from Domination_functions import check_domination, get_pareto_front


# %% ------------------------- PARAMETERS --------------------------
np.random.seed(0)
size = 100
MaxIt = 200
nPop = 25
N = 60

rc = 20
rs = np.ones(N, dtype=int) * 10
#rs = np.random.uniform(10, 15, N)
stat = np.zeros((2, N))  # tạo mảng 2xN
stat[0, :] = rs          # dòng 1 là rs
stat[1, 0] = rc          # phần tử đầu dòng 2 = rc

# %% ------------------------- INITIATION --------------------------
Covered_Area = np.zeros((size, size), dtype=int)
#Obstacle_Area = gen_target_area(1000, size)
Obstacle_Area = np.ones((size, size), dtype=int)

sink = np.array([size//2, size//2])
a = 1
L = np.zeros(nPop, dtype=int)

pop = []
for _ in range(nPop):
    alpop = np.random.uniform(sink[0]-rc/2, sink[1]+rc/2, (N, 2))
    alpop[0] = sink
    cov = CostFunction_MOO(alpop, stat, Obstacle_Area, Covered_Area.copy())
    pop.append({'Position': alpop, 'Cost': cov})

Extra_archive = get_pareto_front(pop)
        
# %% ------------------------- MAIN LOOP --------------------------
for it in range(MaxIt):
# %% ------------------------- EXPLORATION LOOP --------------------------
    for i in range(nPop):
        k = np.random.randint(nPop)
        phi = a * np.random.uniform(-1, 1, (N, 2)) * (1 - L[i] / MaxIt)**5
        alpop = pop[i]['Position'] + phi * (pop[i]['Position'] - pop[k]['Position'])
        alpop[:, :2] = np.clip(alpop[:, :2], 0, size - 1)

        if Connectivity_graph(Graph(alpop[:, :2], rc),[]):
            alpop_cost = CostFunction_MOO(alpop, stat, Obstacle_Area, Covered_Area.copy())
            if check_domination(alpop_cost, pop[i]['Cost']) == 1:
                pop[i]['Position'] = alpop
                pop[i]['Cost'] = alpop_cost
            elif check_domination(alpop_cost, pop[i]['Cost']) == -1:
                L[i] += 1
                continue
            else:
                Extra_archive.append({'Position': alpop, 'Cost': alpop_cost})
                Extra_archive = get_pareto_front(Extra_archive)
# %% ------------------------- SELECTION LOOP --------------------------
    w = np.array([0.5, 0.5])
    E = np.array([np.sum(p['Cost'].flatten() * w) for p in pop])
    if np.sum(E) == 0:
        E_ave = np.ones_like(E) / nPop
    else:
        E_ave = E / np.sum(E)
        
# %% ------------------------- EXPLOITATION LOOP --------------------------
    for _ in range(nPop):
        i = np.random.choice(nPop, p=E_ave)
        for k in range(N):
            alpop = pop[i]['Position'].copy()
            h = np.random.randint(N)
            phi = a * np.random.uniform(-1, 1, (1, 2)) * (1 - L[i] / MaxIt)**2
            alpop[k] += phi.flatten() * (pop[i]['Position'][k] - pop[i]['Position'][h])
            alpop[:, :2] = np.clip(alpop[:, :2], 0, size - 1)

            if Connectivity_graph(Graph(alpop[:, :2], rc),[]):
                alpop_cost = CostFunction_MOO(alpop, stat, Obstacle_Area, Covered_Area.copy())
                if check_domination(alpop_cost, pop[i]['Cost']) == 1:
                    pop[i]['Position'] = alpop
                    pop[i]['Cost'] = alpop_cost
                    break
                elif check_domination(alpop_cost, pop[i]['Cost']) == -1:
                    continue
                else:
                    Extra_archive.append({'Position': alpop, 'Cost': alpop_cost})
                    Extra_archive = get_pareto_front(Extra_archive)
                    break

    k = np.random.choice(np.arange(0, nPop-1), size=len(Extra_archive), replace=False)
    for i in sorted(k, reverse=True):
        del pop[i]
    pop = pop + Extra_archive
    print(f"Iter={it}, PF = {len(Extra_archive):.4f}")

    
# %% ------------------------- PLOT --------------------------
# Tạo mảng data từ Cost của Extra_archive
    data = np.array([ind['Cost'] for ind in Extra_archive])  # mỗi ind là dict có key 'Cost'
    data_set = np.array([ind['Cost'] for ind in pop])
    
    # Tạo figure 3D
    fig = plt.figure(1)
    plt.clf()
    
    # Vẽ Pareto front
    plt.plot(data_set[:, 0], data_set[:, 1], 'o', color='g')
    plt.plot(data[:, 0], data[:, 1], 'o', color='r')
    
    plt.xlabel('Non-coverage')
    plt.ylabel('Energy')
    
    # Cập nhật đồ thị theo từng iteration
    plt.pause(0.01)
