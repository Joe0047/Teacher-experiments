class Utils:
    
    @staticmethod
    def sumArray(array):
        total = 0
        for i in range(len(array)):
            total += array[i]
        return total
    
    @staticmethod
    def sumFlowSetSquare(flowlist):
        total = 0
        for f in flowlist:
            total += f[0]
        return total * total
    
    @staticmethod
    def sumFlowSquareSet(flowlist):
        total = 0
        for f in flowlist:
            total += f[0] * f[0]
        return total