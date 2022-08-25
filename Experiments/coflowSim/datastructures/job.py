from datastructures.task import *
from datastructures.machine import *
import random

class Job:
    def __init__(self, jobName, jobID):
        self.jobName = jobName
        self.jobID = jobID
        self.actualStartTime = float("inf")
        self.simulatedStartTime = 0
        self.simulatedFinishTime = 0
        self.tasks = []
        self.tasksInRacks = None
        self.shuffleBytesPerRack = None
        self.numMappersInRacks = None
        self.numMappers = 0
        self.numReducers = 0
        self.totalShuffleBytes = 0
        self.numFlows = 0
        self.weight = random.randint(1, 100)
        #self.weight = 1
    
    def addTask(self, task):
        self.tasks.append(task)
        
        # Determine job arrival times
        if task.actualStartTime < self.actualStartTime:
            self.actualStartTime = task.actualStartTime
        
        # Increase respective task counts
        if task.taskType == TaskType.MAPPER:
            self.numMappers += 1
        elif task.taskType == TaskType.REDUCER:
            self.numReducers += 1
            self.totalShuffleBytes += task.shuffleBytes
            
    def convertMachineToRack(self, machine, machinesPerRack):
        # Subtracting because machine IDs start from 1
        return int((machine - 1) / machinesPerRack)
    
    def addAscending(self, coll, t):
        index = 0
        while index < len(coll):
            if coll[index].shuffleBytesLeft > t.shuffleBytesLeft:
                break
            index += 1
        coll.insert(index, t)
            
    def arrangeTasks(self, numRacks, machinesPerRack):
        if self.numMappersInRacks == None:
            self.numMappersInRacks = []
            for i in range(numRacks):
                self.numMappersInRacks.append(0)
        
        if self.tasksInRacks == None:
            self.tasksInRacks = []
            for i in range(numRacks):
                self.tasksInRacks.append([])
            
            self.shuffleBytesPerRack = []
            for i in range(numRacks):
                self.shuffleBytesPerRack.append(0)
        
        for t in self.tasks:
            if t.taskType == TaskType.MAPPER:
                fromRack = self.convertMachineToRack(t.getPlacement(), machinesPerRack)
                self.numMappersInRacks[fromRack] += 1
            
            if t.taskType == TaskType.REDUCER:
                toRack = self.convertMachineToRack(t.getPlacement(), machinesPerRack)
                self.addAscending(self.tasksInRacks[toRack], t)
                self.shuffleBytesPerRack[toRack] += t.shuffleBytes
        
        self.coalesceMappers(numRacks)
        self.coalesceReducers(numRacks)
    
    def coalesceMappers(self, numRacks):
        newMappers = []
        for t in self.tasks:
            if t.taskType == TaskType.MAPPER:
                newMappers.append(t)
        
        for t in newMappers:
            self.tasks.remove(t)
        newMappers.clear()
        
        # Reset mapper counters in job
        self.numMappers = 0
        
        for i in range(numRacks):
            if self.numMappersInRacks[i] > 0:
                self.numMappers += 1
                
                iThMt = MapTask("JOB-" + str(self.jobID) + "-MAP-" + str(i), i, self, self.actualStartTime, Machine(i+1))
                
                newMappers.append(iThMt)
        
        self.tasks.extend(newMappers)
                
    def coalesceReducers(self, numRacks):
        newReducers = []
        for t in self.tasks:
            if t.taskType == TaskType.REDUCER:
                newReducers.append(t)
        
        for t in newReducers:
            self.tasks.remove(t)
        newReducers.clear()
        
        # Reset shuffle counters in job
        self.numReducers = 0
        self.totalShuffleBytes = 0
        
        for i in range(numRacks):
            if self.tasksInRacks[i] != None and len(self.tasksInRacks[i]) > 0:
                self.numReducers += 1
            
                iThRt = ReduceTask("JOB-" + str(self.jobID) + "-REDUCE-" + str(i), i, self, self.actualStartTime, Machine(i+1), 0)
                
                # Update shuffle counters in task
                for t in self.tasksInRacks[i]:
                    iThRt.shuffleBytes += t.shuffleBytes
                iThRt.shuffleBytesLeft = iThRt.shuffleBytes
                
                
                # Update shuffle counters in job
                self.totalShuffleBytes += iThRt.shuffleBytes;
                
                newReducers.append(iThRt)
                
                self.tasksInRacks[i].clear()
                self.tasksInRacks[i].append(iThRt)
        
        self.tasks.extend(newReducers)
        
        # Fixes for two-sided simulation
        if self.numMappers == 0:
            return
        
        self.totalShuffleBytes = 0
        for t in self.tasks:
            if t.taskType != TaskType.REDUCER:
                continue
            
            # Rounding to numMappers
            t.roundToNearestNMB(self.numMappers)
            
            self.totalShuffleBytes += t.shuffleBytes
            
            # Now create flows
            t.createFlows()
            self.numFlows += t.getNumFlows()
    
    def getNumFlows(self):
        return self.numFlows
        
    def getJobName(self):
        return self.jobName
    
    def getWeight(self):
        return self.weight
    
    def getSimulatedFinishTime(self):
        return self.simulatedFinishTime
    
    def getReleaseTime(self):
        return self.actualStartTime
        
    def initJobFinishedTime(self):
        self.simulatedFinishTime = 0
    
        
