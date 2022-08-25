from gurobipy import *
from traceProducer.traceProducer import *
from traceProducer.jobClassDescription import *
from datastructures.jobCollection import *
from simulator.simulator import *
from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

numOfCoflows = []
rawFDLS = []
rawWeaver = []
FDLS = []
Weaver = []

curNumCoflows = 5
lastNumCoflows = 9
stepSize = 2

while(curNumCoflows <= lastNumCoflows):
    numOfCoflows.append(curNumCoflows)
    
    rseed = 13
    turn = 100
    listOfTurnsFDLS = []
    average_FDLS = 0
    listOfTurnsWeaver = []
    average_Weaver = 0
    
    while(turn > 0):
        print(curNumCoflows)
        print(turn)
        numRacks = 4
        numJobs = curNumCoflows
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
        M = 4
        
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
        
        print("========================================================")
        print('OPT: %f' % mod.objVal)
        print('FDLS: %f' % value_FDLS)
        print(value_FDLS / mod.objVal)
        print("========================================================")
        
        listOfTurnsFDLS.append(value_FDLS / mod.objVal)
        
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
                
        print("========================================================")
        print('OPT: %f' % mod.objVal)
        print('Weaver: %f' % value_Weaver)
        print(value_Weaver / mod.objVal)
        print("========================================================")
        
        listOfTurnsWeaver.append(value_Weaver / mod.objVal)
        
        rseed += 1
        turn -= 1
    
    for f in listOfTurnsFDLS:
        average_FDLS += f
    average_FDLS /= len(listOfTurnsFDLS)
    FDLS.append(average_FDLS)
    
    rawFDLS.append(listOfTurnsFDLS)
    
    for w in listOfTurnsWeaver:
        average_Weaver += w
    average_Weaver /= len(listOfTurnsWeaver)
    Weaver.append(average_Weaver)
    
    rawWeaver.append(listOfTurnsWeaver)
    
    curNumCoflows += stepSize

raw = {'rawFDLS': rawFDLS, 'rawWeaver': rawWeaver}
algo = {'FDLS': FDLS, 'Weaver': Weaver}

file = open('../result/custom_divisible/custom_divisible.txt','w')

for key, values in raw.items():
    file.write(key + ' ' + str(len(values)))
    
    for value in values:
        file.write(' ' + str(len(value)))
        for v in value:
            file.write(' ' + str(v))
        
    file.write('\n')
    
for key, values in algo.items():
    file.write(key + ' ' + str(len(values)))
    
    for value in values:
        file.write(' ' + str(value))
        
    file.write('\n')

file.close()

# 設定圖片大小為長15、寬10

plt.figure(figsize=(15,10),dpi=100,linewidth = 2)


plt.plot(numOfCoflows,FDLS,'o-',color = 'g', label="FDLS")


plt.plot(numOfCoflows,Weaver,'^-',color = 'b', label="Weaver")


# 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離

plt.title("Divisible coflows from custom", size=40, x=0.5, y=1.03)

# 設置刻度字體大小

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

# 標示x軸(labelpad代表與圖片的距離)

plt.xlabel("Number of coflows", fontsize=30, labelpad = 15)

# x軸只顯示整數刻度
plt.gca().xaxis.set_major_locator(MaxNLocator(integer = True))

# 標示y軸(labelpad代表與圖片的距離)

plt.ylabel("Approximation ratio", fontsize=30, labelpad = 20)

# 顯示出線條標記位置

plt.legend(loc = "best", fontsize=20)

# 畫出圖片

plt.show()
