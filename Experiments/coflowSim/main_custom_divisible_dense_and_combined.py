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
rawFDLS = []
rawWeaver = []
FDLS = []
Weaver = []

rseed = 13
turn = 100
listOfTurnsDenseFDLS = []
average_DenseFDLS = 0
listOfTurnsDenseWeaver = []
average_DenseWeaver = 0
    
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
    
    listOfTurnsDenseFDLS.append(value_FDLS / mod.objVal)
    
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
    
    listOfTurnsDenseWeaver.append(value_Weaver / mod.objVal)
    
    rseed += 1
    turn -= 1

for f in listOfTurnsDenseFDLS:
    average_DenseFDLS += f
average_DenseFDLS /= len(listOfTurnsDenseFDLS)
FDLS.append(average_DenseFDLS)

rawFDLS.append(listOfTurnsDenseFDLS)

for w in listOfTurnsDenseWeaver:
    average_DenseWeaver += w
average_DenseWeaver /= len(listOfTurnsDenseWeaver)
Weaver.append(average_DenseWeaver)

rawWeaver.append(listOfTurnsDenseWeaver)
    
rseed = 13
turn = 100
listOfTurnsCombinedFDLS = []
average_CombinedFDLS = 0
listOfTurnsCombinedWeaver = []
average_CombinedWeaver = 0
    
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
    
    listOfTurnsCombinedFDLS.append(value_FDLS / mod.objVal)
    
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
    
    listOfTurnsCombinedWeaver.append(value_Weaver / mod.objVal)
    
    rseed += 1
    turn -= 1

for f in listOfTurnsCombinedFDLS:
    average_CombinedFDLS += f
average_CombinedFDLS /= len(listOfTurnsCombinedFDLS)
FDLS.append(average_CombinedFDLS)

rawFDLS.append(listOfTurnsCombinedFDLS)

for w in listOfTurnsCombinedWeaver:
    average_CombinedWeaver += w
average_CombinedWeaver /= len(listOfTurnsCombinedWeaver)
Weaver.append(average_CombinedWeaver)

rawWeaver.append(listOfTurnsCombinedWeaver)

raw = {'rawFDLS': rawFDLS, 'rawWeaver': rawWeaver}
algo = {'FDLS': FDLS, 'Weaver': Weaver}

file = open('../result/custom_divisible_dense_and_combined/custom_divisible_dense_and_combined.txt','w')

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

# ????????????????????????15??????10

plt.figure(figsize=(15,10),dpi=100,linewidth = 2)

x = np.arange(len(instanceOfCoflows))

width = 0.3

plt.bar(x,FDLS,width,color = 'g', label="FDLS")

plt.bar(x+width,Weaver,width,color = 'b', label="Weaver")


# ????????????????????????????????????????????????x????????????????????????????????????y????????????????????????

plt.title("Divisible coflows from custom", size=40, x=0.5, y=1.03)

# ????????????????????????

plt.xticks(x + width/2,instanceOfCoflows,fontsize=20)

plt.yticks(fontsize=20)

# ??????x???(labelpad????????????????????????)

plt.xlabel("Coflow instance", fontsize=30, labelpad = 15)

# ??????y???(labelpad????????????????????????)

plt.ylabel("Approximation ratio", fontsize=30, labelpad = 20)

# ???????????????????????????

plt.legend(loc = "best", fontsize=20)

# ????????????

plt.show()