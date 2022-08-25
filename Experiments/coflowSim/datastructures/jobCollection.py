from datastructures.job import *

class JobCollection:
    def __init__(self):
        self.hashOfJobs = {}
        self.listOfJobs = []
    
    def getOrAddJob(self, jobName):
        if jobName in self.hashOfJobs:
            return self.hashOfJobs.get(jobName)
        else:
            job = Job(jobName, len(self.listOfJobs))
            
            self.hashOfJobs[jobName] = job
            self.listOfJobs.append(job)
            
            return job
        
    def size(self):
        return len(self.listOfJobs)
    
    def elementAt(self, index):
        return self.listOfJobs[index]
    
    def removeJob(self, job):
        self.listOfJobs.remove(job)
        del self.hashOfJobs[job.getJobName()]