'''
Constructor for Flow.
 * mapper: flow source.
 * reducer: flow destination.
 * totalBytes: size in bytes.
'''

class Flow:
    def __init__(self, mapper, reducer, totalBytes):
        self.mapper = mapper
        self.reducer = reducer
        self.totalBytes = totalBytes
        self.bytesRemaining = totalBytes
    
    def getMapper(self):
        return self.mapper
    
    def getReducer(self):
        return self.reducer
    
    def getFlowSize(self):
        return self.totalBytes
    
    def initFlow(self):
        self.bytesRemaining = self.totalBytes
    
    def toString(self):
        return "FLOW-" + str(self.mapper) + "-->" + str(self.reducer) + " | " + str(self.bytesRemaining)