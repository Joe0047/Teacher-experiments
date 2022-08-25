class Machine:
    def __init__(self, machineID):
        self.machineID = machineID
    
    def toString(self):
        return "Machine-" + str(self.machineID)
    
    