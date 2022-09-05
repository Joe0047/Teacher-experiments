from gurobipy import *
from traceProducer.traceProducer import *
from traceProducer.jobClassDescription import *
from datastructures.jobCollection import *
from simulator.simulator import *
import numpy as np
import matplotlib.pyplot as plt
import time

instanceOfAlgo = [None, "CDLS", None]
rawCDLS = []
CDLS = []

rseed = 13
turn = 100
listOfTurnsCDLS = []
average_CDLS = 0

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
 
    start = time.time()
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
    
    end = time.time()
    
    executionTimeOfCDLS = end - start
    
    print("========================================================")
    print('execution time of CDLS: %f' % executionTimeOfCDLS)
    print("========================================================")
    
    listOfTurnsCDLS.append(executionTimeOfCDLS)
    
    rseed += 1
    turn -= 1

for c in listOfTurnsCDLS:
    average_CDLS += c
average_CDLS /= len(listOfTurnsCDLS)
CDLS.append(average_CDLS)

rawCDLS.append(listOfTurnsCDLS)

raw = {'rawCDLS': rawCDLS}
algo = {'CDLS': CDLS}

file = open('../result/custom_indivisible_time_complexity/custom_indivisible_time_complexity.txt','w')

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

x = np.arange(len(instanceOfAlgo))

width = 0.3

plt.bar(x[0],0,width)

plt.bar(x[1],CDLS,width,color = 'r', label="CDLS")

plt.bar(x[2],0,width)

# 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離

plt.title("Indivisible coflows from custom", size=40, x=0.5, y=1.03)

# 設置刻度字體大小

plt.xticks(x,instanceOfAlgo,fontsize=20)

plt.yticks(fontsize=20)

# 標示x軸(labelpad代表與圖片的距離)

plt.xlabel("Algorithms", fontsize=30, labelpad = 15)

# 標示y軸(labelpad代表與圖片的距離)

plt.ylabel("Execution time (s)", fontsize=30, labelpad = 20)

# 顯示出線條標記位置

plt.legend(loc = "best", fontsize=20)

# 畫出圖片

plt.show()