from gurobipy import *
from traceProducer.traceProducer import *
from traceProducer.jobClassDescription import *
from datastructures.jobCollection import *
from simulator.simulator import *
from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

completionTimeOfCoflow = []
cdfOfFDLS = []
cdfOfWeaver = []

numRacks = 4
numJobs = 8
randomSeed = 13

jobClassDescs = [JobClassDescription(1, 4, 1, 10),
                 JobClassDescription(1, 4, 10, 1000),
                 JobClassDescription(4, numRacks, 1, 10),
                 JobClassDescription(4, numRacks, 10, 1000)]

fracsOfClasses = [41,
                  29,
                  9,
                  21]

tr = CustomTraceProducer(numRacks, numJobs, jobClassDescs, fracsOfClasses, randomSeed)
tr.prepareTrace()

sim = Simulator(tr)

K = tr.getNumJobs()
N = tr.getNumRacks()
I = N
J = N
M = 10

li, lj, coflowlist = tr.produceCoflowSizeAndList()

S = tr.produceCoflowSet(coflowlist)

w = tr.obtainJobWeight()

r = tr.obtainJobReleaseTime()

# LP_IDC
mod = Model("LP_IDC")

C = mod.addVars(K, vtype = GRB.CONTINUOUS)

mod.update()

mod.setObjective(quicksum(w[k]*C[k] for k in range(K)), GRB.MINIMIZE)

mod.addConstrs(C[k] >= r[k] + li[k,i]
             for i in range(I)
             for k in range(K))

mod.addConstrs(C[k] >= r[k] + lj[k,j]
             for j in range(J)
             for k in range(K))

for i in range(I):
    for s in S:
        mod.addConstr(quicksum(li[k[3],i] * C[k[3]] for k in s) >= (Utils.sumCoflowSquareSet(s,"i",i) + Utils.sumCoflowSetSquare(s,"i",i)) / (2*M))
        
for j in range(J):
    for s in S:
        mod.addConstr(quicksum(lj[k[3],j] * C[k[3]] for k in s) >= (Utils.sumCoflowSquareSet(s,"j",j) + Utils.sumCoflowSetSquare(s,"j",j)) / (2*M))
            
mod.optimize()
    
EPOCH_IN_MILLIS = Constants.SIMULATION_QUANTA

flowlist = tr.turnCoflowListToFlowList(coflowlist)

for f in flowlist:
    f[4] = C[f[4]].X

# FDLS
loadI = np.zeros((M,I))
loadO = np.zeros((M,J))
A = [[] for h in range(M)]

flowlist.sort(key = lambda f: f[4])

for f in flowlist:
    h_star = -1
    minload = float("inf")
    for h in range(M):
        if loadI[h][f[1]] + loadO[h][f[2]] < minload:
            h_star = h
            minload = loadI[h][f[1]] + loadO[h][f[2]]
            
    A[h_star].append([f[3], f[4]])
    loadI[h_star][f[1]] += f[0]
    loadO[h_star][f[2]] += f[0]

for h in range(M):
    sim.simulate(A[h], EPOCH_IN_MILLIS)

value_FDLS = tr.calculateTotalWeightedCompletionTime()

completionTimeOfCoflowFDLS = tr.getCompletionTimeOfCoflows()
completionTimeOfCoflowFDLS.sort()

print("========================================================")
print('OPT: %f' % mod.objVal)
print('FDLS: %f' % value_FDLS)
print(value_FDLS / mod.objVal)
print("========================================================")

# Initialize the finished time of job and the remaining bytes of flows
tr.initJobFinishedTimeAndFlowRemainingBytes()

# Weaver
loadI = np.zeros((M,I))
loadO = np.zeros((M,J))
L_add = [0 for h in range(M)]
L = [0 for h in range(M)]
A = [[] for h in range(M)]

flowlist.sort(key = lambda f: f[0], reverse = True)

for f in flowlist:
    h_star = -1
    minload = float("inf")
    
    for h in range(M):
        loadI[h][f[1]] += f[0]
        loadO[h][f[2]] += f[0]
        
        if loadI[h][f[1]] > L_add[h]:
            L_add[h] = loadI[h][f[1]]
        if loadO[h][f[2]] > L_add[h]:
            L_add[h] = loadO[h][f[2]]
        
        loadI[h][f[1]] -= f[0]
        loadO[h][f[2]] -= f[0]
        
        if (L_add[h] > L[h]) and (L_add[h] < minload):
            h_star = h
            minload = L_add[h]
    
    if h_star == -1:
        minload = float("inf")
        for h in range(M):
            loadI[h][f[1]] += f[0]
            loadO[h][f[2]] += f[0]
            
            maxload = max(loadI[h][f[1]], loadO[h][f[2]])
            
            loadI[h][f[1]] -= f[0]
            loadO[h][f[2]] -= f[0]
            
            if maxload < minload:
                h_star = h
                minload = maxload
    
    A[h_star].append([f[3], f[4]])
    loadI[h_star][f[1]] += f[0]
    loadO[h_star][f[2]] += f[0]
    
    if loadI[h_star][f[1]] > L[h_star]:
        L[h_star] = loadI[h_star][f[1]]
    if loadO[h_star][f[2]] > L[h_star]:
        L[h_star] = loadO[h_star][f[2]]
    
    L_add = L.copy()

for h in range(M):
    sim.simulate(A[h], EPOCH_IN_MILLIS)

value_Weaver = tr.calculateTotalWeightedCompletionTime()

completionTimeOfCoflowWeaver = tr.getCompletionTimeOfCoflows()
completionTimeOfCoflowWeaver.sort()
        
print("========================================================")
print('OPT: %f' % mod.objVal)
print('Weaver: %f' % value_Weaver)
print(value_Weaver / mod.objVal)
print("========================================================")


completionTimeOfCoflow.extend(completionTimeOfCoflowFDLS)
completionTimeOfCoflow.extend(completionTimeOfCoflowWeaver)

completionTimeOfCoflow.sort()

for i in range(len(completionTimeOfCoflow)):
    j = 0
    while(j < len(completionTimeOfCoflowFDLS)):
        if completionTimeOfCoflowFDLS[j] > completionTimeOfCoflow[i]:
            cdfOfFDLS.append(j/K)
            break
        
        if j == len(completionTimeOfCoflowFDLS) - 1:
            cdfOfFDLS.append((j+1)/K)
        
        j += 1

for i in range(len(completionTimeOfCoflow)):
    j = 0
    while(j < len(completionTimeOfCoflowWeaver)):
        if completionTimeOfCoflowWeaver[j] > completionTimeOfCoflow[i]:
            cdfOfWeaver.append(j/K)
            break
        
        if j == len(completionTimeOfCoflowWeaver) - 1:
            cdfOfWeaver.append((j+1)/K)
        
        j += 1
        
algo = {'cdfOfFDLS': cdfOfFDLS, 'cdfOfWeaver': cdfOfWeaver}

file = open('../result/custom_indivisible_CDF_ex/custom_indivisible_CDF_ex.txt','w')

file.write('completionTimeOfCoflow ' + str(len(completionTimeOfCoflow)))
for c in completionTimeOfCoflow:
    file.write(' ' + str(c))
file.write('\n')

for key, values in algo.items():
    file.write(key + ' ' + str(len(values)))
    
    for value in values:
        file.write(' ' + str(value))
        
    file.write('\n')

file.close()

# ????????????????????????15??????10

plt.figure(figsize=(15,10),dpi=100,linewidth = 2)


plt.plot(completionTimeOfCoflow,cdfOfFDLS,'o-',color = 'g', label="FDLS")


plt.plot(completionTimeOfCoflow,cdfOfWeaver,'^-',color = 'b', label="Weaver")


# ????????????????????????????????????????????????x????????????????????????????????????y????????????????????????

plt.title("Indivisible coflows from custom", size=40, x=0.5, y=1.03)

# ????????????????????????

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

# ???????????????????????????????????????

plt.ticklabel_format(style='plain')

# ??????x???(labelpad????????????????????????)

plt.xlabel("Completion time of coflow (ms)", fontsize=30, labelpad = 15)

# x????????????????????????
plt.gca().xaxis.set_major_locator(MaxNLocator(integer = True))

# ??????y???(labelpad????????????????????????)

plt.ylabel("CDF", fontsize=30, labelpad = 20)

# ???????????????????????????

plt.legend(loc = "best", fontsize=20)

# ????????????

plt.show()