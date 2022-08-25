from enum import Enum
from datastructures.flow import *
import math

class TaskType(Enum):
    MAPPER = "mapper"
    REDUCER = "reducer"    

class Task:
    def __init__(self, taskType, taskName, taskID, parentJob, startTime, assignedMachine):
        self.taskType = taskType
        self.taskName = taskName
        self.taskID = taskID
        self.parentJob = parentJob
        self.actualStartTime = startTime
        self.assignedMachine = assignedMachine
        self.simulatedStartTime = None
        self.simulatedFinishTime = None
        
    def startTask(self, curtime):
        self.simulatedStartTime = curtime
    
    def cleanupTask(self, curtime):
        self.simulatedFinishTime = curtime
    
    def getArrivalTime(self):
        return self.actualStartTime
    
    def getPlacement(self):
        return self.assignedMachine.machineID
    
    def getParentJob(self):
        return self.parentJob
        
    def toString(self):
        return str(self.taskType) + "-" + str(self.taskName)
    
class MapTask(Task):
    def __init__(self, taskName, taskID, parentJob, startTime, assignedMachine):
        super().__init__(TaskType.MAPPER, taskName, taskID, parentJob, startTime, assignedMachine)
        
class ReduceTask(Task):
    def __init__(self, taskName, taskID, parentJob, startTime, assignedMachine, shuffleBytes):
        super().__init__(TaskType.REDUCER, taskName, taskID, parentJob, startTime, assignedMachine)
        self.shuffleBytes = shuffleBytes
        self.shuffleBytesLeft = shuffleBytes
        self.flows = None
        
        # Rounding to the nearest 1MB
        self.roundToNearestNMB(1)
    
    def roundToNearestNMB(self, MB):
        tmp = self.shuffleBytes
        MULT = MB * 1048576
        numMB = math.floor(tmp / MULT)
        
        if tmp % MULT > 0:
            numMB += 1
            
        self.shuffleBytes = MULT * numMB
        self.shuffleBytesLeft = self.shuffleBytes
    
    def createFlows(self):
        self.flows = []
        
        avgFlowSize = self.shuffleBytes / self.parentJob.numMappers
        for t in self.parentJob.tasks:
            if t.taskType != TaskType.MAPPER:
                continue
            
            flowSize = max(avgFlowSize, 1048576)
            self.flows.append(Flow(t, self, flowSize))
        
    def getNumFlows(self):
        return len(self.flows)
    
    