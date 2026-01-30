class SchedulerBase:
    def __init__(self, worker_addrs):
        self.worker_addrs = worker_addrs

    def select_worker(self, model):
        raise NotImplementedError()

# (1) 라운드 로빈 스케줄러
class RoundRobinScheduler(SchedulerBase):
    def __init__(self, worker_addrs):
        super().__init__(worker_addrs)
        self.index = 0

    def select_worker(self, model):
        worker = self.worker_addrs[self.index]
        self.index = (self.index + 1) % len(self.worker_addrs)
        return worker

# (2) 성능 우선 스케줄러 (NPU > TPU > GPU)
class PerformancePriorityScheduler(SchedulerBase):
    def __init__(self, node_info):
        # node_info: [(주소, 랭크), ...]
        self.node_info = sorted(node_info, key=lambda x: x[1])

    def select_worker(self, model):
        # 현재는 항상 가장 높은 성능 노드 선택 (추후 부하, 가용성 반영 가능)
        return self.node_info[0][0]

