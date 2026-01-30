class SchedulerBase:
    def __init__(self, worker_addrs):
        self.worker_addrs = worker_addrs

    def select_worker(self, model):
        raise NotImplementedError()

# (1) Round-Robin
class RoundRobinScheduler(SchedulerBase):
    def __init__(self, worker_addrs):
        super().__init__(worker_addrs)
        self.index = 0

    def select_worker(self, model):
        worker = self.worker_addrs[self.index]
        self.index = (self.index + 1) % len(self.worker_addrs)
        return worker

# (2) Performance-Only (NPU > TPU > GPU)
class PerformancePriorityScheduler(SchedulerBase):
    def __init__(self, node_info):
        # node_info: [(address, rank), ...]
        self.node_info = sorted(node_info, key=lambda x: x[1])

    def select_worker(self, model):
        return self.node_info[0][0]

    
