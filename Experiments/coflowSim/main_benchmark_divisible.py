from gurobipy import *
from traceProducer.traceProducer import *
from datastructures.jobCollection import *
from simulator.simulator import *
from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

thresNumOfFlows = []
FDLS = []
Weaver = []

curThresNumFlows = 10
lastThresNumFlows = 30
stepThresSize = 5

upperBoundOfJob = 8
randomSeed = 13

while(curThresNumFlows <= lastThresNumFlows):
    pathToCoflowBenchmarkTraceFile = "./coflow-benchmark-master/FB2010-1Hr-150-0.txt"
    tr = CoflowBenchmarkTraceProducer(pathToCoflowBenchmarkTraceFile, randomSeed)
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
    
    d, flowlist = tr.produceFlowSizeAndList()
    
    Si, Sj = tr.produceFlowSet(flowlist)
    
    w = tr.obtainJobWeight()
    
    r = tr.obtainJobReleaseTime()
    
    # LP_DC
    mod = Model("LP_DC")
    
    C = mod.addVars(K, vtype = GRB.CONTINUOUS)
    Cf = mod.addVars(K, I, J, vtype = GRB.CONTINUOUS)
    
    mod.update()
    
    mod.setObjective(quicksum(w[k]*C[k] for k in range(K)), GRB.MINIMIZE)
    
    mod.addConstrs(C[k] >= Cf[k,i,j]
                 for i in range(I)
                 for j in range(J)
                 for k in range(K))
    
    mod.addConstrs(Cf[k,i,j] >= r[k] + d[k,i,j]
                 for i in range(I)
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
        f[4] = Cf[f[4],f[1],f[2]].X
    
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
    
    print("========================================================")
    print('OPT: %f' % mod.objVal)
    print('FDLS: %f' % value_FDLS)
    print(value_FDLS / mod.objVal)
    print("========================================================")
    
    FDLS.append(value_FDLS / mod.objVal)
    
    # Initialize the finished time of job and the remaining bytes of flows
    tr.initJobFinishedTimeAndFlowRemainingBytes()
    
    # Weaver
    loadI = np.zeros((M,I))
    loadO = np.zeros((M,J))
    L = [0 for h in range(M)]
    A = [[] for h in range(M)]
    
    flowlist.sort(key = lambda f: f[0], reverse = True)
    
    for f in flowlist:
        h_star = -1
        minload = float("inf")
        flag = -1
        for h in range(M):
            if loadI[h][f[1]]+f[0] > L[h]:
                flag = 1
            if loadO[h][f[2]]+f[0] > L[h]:
                flag = 1
        
        if flag == 1:
            for h in range(M):
                maxload = max(max(loadI[h][f[1]]+f[0], loadO[h][f[2]]+f[0]), L[h])
                if maxload < minload:
                    h_star = h
                    minload = maxload
        
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
    
    for h in range(M):
        sim.simulate(A[h], EPOCH_IN_MILLIS)
    
    value_Weaver = tr.calculateTotalWeightedCompletionTime()
            
    print("========================================================")
    print('OPT: %f' % mod.objVal)
    print('Weaver: %f' % value_Weaver)
    print(value_Weaver / mod.objVal)
    print("========================================================")
    
    Weaver.append(value_Weaver / mod.objVal)
    
    curThresNumFlows += stepThresSize

algo = {'FDLS': FDLS, 'Weaver': Weaver}

file = open('../result/benchmark_divisible/benchmark_divisible.txt','w')
for key, values in algo.items():
    file.write(key + ' ' + str(len(values)))
    
    for value in values:
        file.write(' ' + str(value))
        
    file.write('\n')

file.close()

# 設定圖片大小為長15、寬10

plt.figure(figsize=(15,10),dpi=100,linewidth = 2)


plt.plot(thresNumOfFlows,FDLS,'o-',color = 'g', label="FDLS")


plt.plot(thresNumOfFlows,Weaver,'^-',color = 'b', label="Weaver")


# 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離

plt.title("Divisible coflows from benchmark", size=40, x=0.5, y=1.03)

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
