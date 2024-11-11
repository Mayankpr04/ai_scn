import time

class TimeMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time=time.time()
        self.total_time=0
        self.count0

    def update(self,n=1):
        end_time=time.time()
        self.total_time+=(end_time-self.start_time)
        self.count+=n
        self.start_time=end_time

    @property
    def avg(self):
        return self.total_time/self.count if self.count > 0 else 0
