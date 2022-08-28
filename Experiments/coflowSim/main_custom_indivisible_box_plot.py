from gurobipy import *
from traceProducer.traceProducer import *
from traceProducer.jobClassDescription import *
from datastructures.jobCollection import *
from simulator.simulator import *
from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

CDLS = []

rseed = 13
turn = 100
    
while(turn > 0):
    print(turn)
    numRacks = 4
    numJobs = 8
    randomSeed = rseed
    
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
    
    for k in coflowlist:
        k[3] = C[k[3]].X
    
    # CDLS
    loadI = np.zeros((M,I))
    loadO = np.zeros((M,J))
    A = [[] for h in range(M)]
    
    for k in coflowlist:
        h_star = -1
        minload = float("inf")
        for h in range(M):
            maxload = float("-inf")
            for i in range(I):
                for j in range(J):
                    if loadI[h][i] + loadO[h][j] + k[0][i] + k[1][j] > maxload:
                        maxload = loadI[h][i] + loadO[h][j] + k[0][i] + k[1][j]
            if maxload < minload:
                h_star = h
                minload = maxload
                
        for t in k[2].tasks:
            if t.taskType != TaskType.REDUCER:
                continue
            
            for f in t.flows:
                A[h_star].append([f, k[3]])
                
        for i in range(I):
            loadI[h_star][i] += k[0][i]
        for j in range(J):
            loadO[h_star][j] += k[1][j]
    
    for h in range(M):
        sim.simulate(A[h], EPOCH_IN_MILLIS)
    
    value_CDLS = tr.calculateTotalWeightedCompletionTime()
    
    print("========================================================")
    print('OPT: %f' % mod.objVal)
    print('CDLS: %f' % value_CDLS)
    print(value_CDLS / mod.objVal)
    print("========================================================")
    
    CDLS.append(value_CDLS / mod.objVal)
    
    rseed += 1
    turn -= 1

algo = {'CDLS': CDLS}

file = open('../result/custom_indivisible_box_plot/custom_indivisible_box_plot.txt','w')
for key, values in algo.items():
    file.write(key + ' ' + str(len(values)))
    
    for value in values:
        file.write(' ' + str(value))
        
    file.write('\n')

pd_CDLS = pd.Series(CDLS)
file.write('Q1,Q2,Q3,mean ' + str(4) + ' ' + str(pd_CDLS.quantile(0.25)) + ' ' + str(pd_CDLS.quantile(0.5)) + ' ' + str(pd_CDLS.quantile(0.75)) + ' ' + str(pd_CDLS.mean()) + '\n')

file.close()

# 設定圖片大小為長15、寬10

fig, ax = plt.subplots(figsize=(15,10))

ax.boxplot(algo.values())

ax.set_xticklabels(algo.keys())

# 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離

plt.title("Indivisible coflows from custom", size=40, x=0.5, y=1.03)

# 設置刻度字體大小

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

# 標示x軸(labelpad代表與圖片的距離)

plt.xlabel("Algorithms", fontsize=30, labelpad = 15)

# 標示y軸(labelpad代表與圖片的距離)

plt.ylabel("Approximation ratio", fontsize=30, labelpad = 20)

