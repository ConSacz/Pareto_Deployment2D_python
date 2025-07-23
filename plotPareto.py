try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import numpy as np
import matplotlib.pyplot as plt
from utils.Workspace_functions import load_mat
from utils.Domination_functions import get_pareto_front

folder_name = 'data'
file_name = ''

# %% DATA IMPORT
#  NSGA case 1
NSGA_case1 = []
for i in range(10):
    file_name = f'NSGA_{i}_1.mat'
    data = load_mat(folder_name, file_name)
    pop = data['pop']
    PF = np.array([ind['Cost'].flatten() for ind in get_pareto_front(pop)])
    NSGA_case1.append(PF)
    
#  NSGA case 2
NSGA_case2 = []
for i in range(10):
    file_name = f'NSGA_{i}.2.mat'
    data = load_mat(folder_name, file_name)
    pop = data['pop']
    PF = np.array([ind['Cost'].flatten() for ind in get_pareto_front(pop)])
    NSGA_case2.append(PF)
    
#  NSABC case 1
NSABC_case1 = []
for i in range(10):
    file_name = f'NSABC_{i}.1.mat'
    data = load_mat(folder_name, file_name)
    pop = data['pop']
    PF = np.array([ind['Cost'].flatten() for ind in get_pareto_front(pop)])
    NSABC_case1.append(PF)
    
#  NSABC case 2
NSABC_case2 = []
for i in range(10):
    file_name = f'NSABC_{i}.2.mat'
    data = load_mat(folder_name, file_name)
    pop = data['pop']
    PF = np.array([ind['Cost'].flatten() for ind in get_pareto_front(pop)])
    NSABC_case2.append(PF)

#  DWABC case 1
DWABC_case1 = []
for i in range(3):
    file_name = f'DWABC_20pop_{i}.1.mat'
    data = load_mat(folder_name, file_name)
    pop = data['pop']
    PF = np.array([ind['Cost'].flatten() for ind in get_pareto_front(pop)])
    DWABC_case1.append(PF)  

#  DWABC case 2
DWABC_case2 = []
for i in range(3):
    file_name = f'DWABC_20pop_{i}.2.mat'
    data = load_mat(folder_name, file_name)
    pop = data['pop']
    PF = np.array([ind['Cost'].flatten() for ind in get_pareto_front(pop)])
    DWABC_case2.append(PF)     
del data, pop, PF, i 
# %% PLOT DATA
fig = plt.figure(1)
plt.clf()
a = 5
b = 0
c = 0
# Vẽ Pareto front
plt.plot(NSGA_case2[a][:, 0], NSGA_case2[a][:, 1], 'o', color='g', label = 'NSGA')
plt.plot(NSABC_case2[b][:, 0], NSABC_case2[b][:, 1], 'o', color='b', label = 'NSABC')
plt.plot(DWABC_case2[c][:, 0], DWABC_case2[c][:, 1], 'o', color='r', label = 'DWABC')
#plt.plot(data2[:, 0], data2[:, 1], 'o', color='r', label = 'NSWABC')
#plt.plot(data3[:, 0], data3[:, 1], 'o', color='g', label = 'NSWABC')
# for i in range(len(data_set)):
#     x, y = data_set[i]
#     plt.text(x, y, str(i), fontsize=8, ha='right', va='bottom', color='blue')
plt.legend()
# plt.xlim(RP[0,0], RP[0,1])
# plt.ylim(RP[1,0], RP[1,1])
plt.xlabel('Non-coverage')
plt.ylabel('Energy')
None
# Cập nhật đồ thị theo từng iteration
plt.pause(0.001)
        
        
        