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
    
    @staticmethod
    def sumCoflowSetSquare(coflowlist, port, n):
        total = 0
        if port == "i":
            for k in coflowlist:
                total += k[0][n]
        
        elif port == "j":
            for k in coflowlist:
                total += k[1][n]
        
        return total * total
    
    @staticmethod
    def sumCoflowSquareSet(coflowlist, port, n):
        total = 0
        if port == "i":
            for k in coflowlist:
                total += k[0][n] * k[0][n]
                
        elif port == "j":
            for k in coflowlist:
                total += k[1][n] * k[1][n]
        
        return total