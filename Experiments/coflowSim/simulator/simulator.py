from utils.constants import *

class Simulator:
    def __init__(self, traceProducer):
        self.NUM_RACKS = traceProducer.getNumRacks()
        self.MACHINES_PER_RACK = traceProducer.getMachinesPerRack()
        self.jobs = None
        
        self.initialize(traceProducer)
    
    def initialize(self, traceProducer):
        self.jobs = traceProducer.jobs
        
        self.mergeTasksByRack()
    
    
    '''
        Merges all tasks in the same rack to a single one to form a non-blocking switch.
    '''
    def mergeTasksByRack(self):
        for j in self.jobs.listOfJobs:
            j.arrangeTasks(self.NUM_RACKS, self.MACHINES_PER_RACK)
        
    '''
    Event loop of the simulator that proceed epoch by epoch.
     * Simulate the time steps in each epoch, where each time step (8ms) is as long as it takes to
       transfer 1MB through each link.
    '''
    def simulate(self, flowsInThisCore, EPOCH_IN_MILLIS):
        CURRENT_TIME = 0
        readyFlows = []
        
        # Create the mapper and reducer rack table
        rackMapperInfoTable = []
        rackReducerInfoTable = []
        for i in range(self.NUM_RACKS):
            rackMapperInfoTable.append(False)
            rackReducerInfoTable.append(False)
        
        while(len(flowsInThisCore) > 0 or len(readyFlows) > 0):
            
            newReadyFlows = []
            for flow in flowsInThisCore:
                flowArrivalTime = min(flow[0].getMapper().getArrivalTime(), flow[0].getReducer().getArrivalTime())
                
                # If flow is not arrived, do not add flow to ready flows
                if flowArrivalTime > CURRENT_TIME + EPOCH_IN_MILLIS:
                    continue
                
                # One flow added to ready flows
                newReadyFlows.append(flow)
                
            for flow in newReadyFlows:
                readyFlows.append(flow)
                flowsInThisCore.remove(flow)
            
            readyFlows.sort(key = lambda f: f[1])
            
            finishedFlows = []
            for flow in readyFlows:
                # Convert machine to rack. (Subtracting because machine IDs start from 1)
                i = flow[0].getMapper().getPlacement() - 1
                j = flow[0].getReducer().getPlacement() - 1
                
                # If link (i,j) is not idle, do not assign flow to active flows
                if rackMapperInfoTable[i] == True or rackReducerInfoTable[j] == True:
                    continue
                
                # Update the mapper and reducer rack table
                rackMapperInfoTable[i] = True
                rackReducerInfoTable[j] = True
                
                flow[0].bytesRemaining -= EPOCH_IN_MILLIS * Constants.RACK_BYTES_PER_MILLISEC
                
                # Finished flow
                if flow[0].bytesRemaining <= 0:
                    finishedFlows.append(flow)
                
                    # Update the finished time of job
                    if CURRENT_TIME + EPOCH_IN_MILLIS > flow[0].getReducer().getParentJob().simulatedFinishTime:
                        flow[0].getReducer().getParentJob().simulatedFinishTime = CURRENT_TIME + EPOCH_IN_MILLIS
                        
            for flow in finishedFlows:
                readyFlows.remove(flow)
            
            for i in range(self.NUM_RACKS):
                # Reset the mapper and reducer rack table
                rackMapperInfoTable[i] = False
                rackReducerInfoTable[i] = False
            

            CURRENT_TIME += EPOCH_IN_MILLIS
        