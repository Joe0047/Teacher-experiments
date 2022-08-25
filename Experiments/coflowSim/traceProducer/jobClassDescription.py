'''
Constructor for JobClassDescription.
 * minW: Minimum number of tasks.
 * maxW: Maximum number of tasks.
 * minL: Minimum shuffle size in MB for each reducer.
 * maxL: Maximum shuffle size in MB for each reducer.   
'''

class JobClassDescription:
    def __init__(self, minW, maxW, minL, maxL):
        self.minWidth = max(minW, 1)
        self.maxWidth = max(maxW, self.minWidth)
        self.minLength = max(minL, 1)
        self.maxLength = max(maxL, self.minLength)