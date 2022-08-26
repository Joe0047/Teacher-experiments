from gurobipy import *
from traceProducer.traceProducer import *
from datastructures.jobCollection import *
from simulator.simulator import *
from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

thresNumOfFlows = []
CLS = []

curThresNumFlows = 10
lastThresNumFlows = 30
stepThresSize = 5

upperBoundOfJob = 8

while(curThresNumFlows <= lastThresNumFlows):
    pathToCoflowBenchmarkTraceFile = "./coflow-benchmark-master/FB2010-1Hr-150-0.txt"
    tr = CoflowBenchmarkTraceProducer(pathToCoflowBenchmarkTraceFile)
    tr.prepareTrace()
    
    sim = Simulator(tr)
    
    print(curThresNumFlows)
    tr.filterJobsByNumFlows(curThresNumFlows, upperBoundOfJob)
    thresNumOfFlows.append(curThresNumFlows)
    
    K = tr.getNumJobs()
    N = tr.getNumRacks()
    I = N
    J = N
    M = 10
    
    print(K)
    
    li, lj, coflowlist = tr.produceCoflowSizeAndList()
    
    Si, Sj = tr.produceFlowSet(coflowlist)
    
    w = tr.obtainJobWeight()
    
    r = tr.obtainJobReleaseTime()
    
    # LP_DC
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
        for s in Si[i]:
            mod.addConstr(quicksum(d[f[4],f[1],f[2]] * Cf[f[4],f[1],f[2]] for f in s) >= (Utils.sumFlowSetSquare(s) + Utils.sumFlowSquareSet(s)) / (2*M))
                
        
    for j in range(J):
        for s in Sj[j]:
            mod.addConstr(quicksum(d[f[4],f[1],f[2]] * Cf[f[4],f[1],f[2]] for f in s) >= (Utils.sumFlowSetSquare(s) + Utils.sumFlowSquareSet(s)) / (2*M))
                
    mod.optimize()
        
    EPOCH_IN_MILLIS = Constants.SIMULATION_QUANTA
    
    for f in flowlist:
        f[4] = C[f[4]].X
    
    # CLS
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
                A[h_star].append(f)
                
        for i in range(I):
            loadI[h_star][i] += k[0][i]
        for j in range(J):
            loadO[h_star][j] += k[1][j]
    
    for h in range(M):
        sim.simulate(A[h], EPOCH_IN_MILLIS)
    
    value_CLS = tr.calculateTotalWeightedCompletionTime()
    
    print("========================================================")
    print('OPT: %f' % mod.objVal)
    print('CLS: %f' % value_CLS)
    print(value_CLS / mod.objVal)
    print("========================================================")
    
    CLS.append(value_CLS / mod.objVal)
    
    curThresNumFlows += stepThresSize

algo = {'CLS': CLS}

file = open('../result/benchmark_indivisible/benchmark_indivisible.txt','w')
for key, values in algo.items():
    file.write(key + ' ' + str(len(values)))
    
    for value in values:
        file.write(' ' + str(value))
        
    file.write('\n')

file.close()

# 設定圖片大小為長15、寬10

plt.figure(figsize=(15,10),dpi=100,linewidth = 2)


plt.plot(thresNumOfFlows,CLS,'s-',color = 'r', label="CLS")


# 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離

plt.title("Indivisible coflows from benchmark", size=40, x=0.5, y=1.03)

# 設置刻度字體大小

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

# 標示x軸(labelpad代表與圖片的距離)

plt.xlabel("Threshold of the number of flows", fontsize=30, labelpad = 15)

# x軸只顯示整數刻度
plt.gca().xaxis.set_major_locator(MaxNLocator(integer = True))

# 標示y軸(labelpad代表與圖片的距離)

plt.ylabel("Approximation ratio", fontsize=30, labelpad = 20)

# 顯示出線條標記位置

plt.legend(loc = "best", fontsize=20)

# 畫出圖片

plt.show()
