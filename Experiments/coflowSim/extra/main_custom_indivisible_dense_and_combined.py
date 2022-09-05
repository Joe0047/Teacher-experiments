from gurobipy import *
from traceProducer.traceProducer import *
from traceProducer.jobClassDescription import *
from datastructures.jobCollection import *
from simulator.simulator import *
from utils.utils import *
import math
import numpy as np
import matplotlib.pyplot as plt

instanceOfCoflows = ["Dense", "Combined"]
rawCDLS = []
CDLS = []

rseed = 13
turn = 100
listOfTurnsDenseCDLS = []
average_DenseCDLS = 0
    
while(turn > 0):
    print("Dense")
    print(turn)
    numRacks = 4
    numJobs = 5
    randomSeed = rseed
    
    jobClassDescs = [JobClassDescription(int(math.sqrt(numRacks)), numRacks, 1, 100)]
    
    fracsOfClasses = [1]
    
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
    
    listOfTurnsDenseCDLS.append(value_CDLS / mod.objVal)
    
    rseed += 1
    turn -= 1

for c in listOfTurnsDenseCDLS:
    average_DenseCDLS += c
average_DenseCDLS /= len(listOfTurnsDenseCDLS)
CDLS.append(average_DenseCDLS)

rawCDLS.append(listOfTurnsDenseCDLS)

rseed = 13
turn = 100
listOfTurnsCombinedCDLS = []
average_CombinedCDLS = 0
    
while(turn > 0):
    print("Combined")
    print(turn)
    numRacks = 4
    numJobs = 5
    randomSeed = rseed
    random.seed(randomSeed)
    
    jobClassDescs = []
    for i in range(numJobs):
        coin = random.randint(0, 1)
        if coin == 0:
            jobClassDescs.append(JobClassDescription(int(math.sqrt(numRacks)), numRacks, 1, 100))
        else:
            jobClassDescs.append(JobClassDescription(1, int(math.sqrt(numRacks)), 1, 100))
    
    fracsOfClasses = []
    for i in range(numJobs):
        fracsOfClasses.append(1)
    
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
    
    listOfTurnsCombinedCDLS.append(value_CDLS / mod.objVal)
    
    rseed += 1
    turn -= 1

for f in listOfTurnsCombinedCDLS:
    average_CombinedCDLS += f
average_CombinedCDLS /= len(listOfTurnsCombinedCDLS)
CDLS.append(average_CombinedCDLS)

rawCDLS.append(listOfTurnsCombinedCDLS)

raw = {'rawCDLS': rawCDLS}
algo = {'CDLS': CDLS}

file = open('../result/custom_indivisible_dense_and_combined/custom_indivisible_dense_and_combined.txt','w')

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

x = np.arange(len(instanceOfCoflows))

width = 0.3

plt.bar(x,CDLS,width,color = 'r', label="CDLS")

# 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離

plt.title("Indivisible coflows from custom", size=40, x=0.5, y=1.03)

# 設置刻度字體大小

plt.xticks(x,instanceOfCoflows,fontsize=20)

plt.yticks(fontsize=20)

# 標示x軸(labelpad代表與圖片的距離)

plt.xlabel("Coflow instance", fontsize=30, labelpad = 15)

# 標示y軸(labelpad代表與圖片的距離)

plt.ylabel("Approximation ratio", fontsize=30, labelpad = 20)

# 顯示出線條標記位置

plt.legend(loc = "best", fontsize=20)

# 畫出圖片

plt.show()