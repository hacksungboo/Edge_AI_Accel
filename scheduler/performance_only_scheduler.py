# scheduler/performance_only_scheduler.py
from .base_scheduler import BaseScheduler

class PerformanceOnlyScheduler(BaseScheduler):
    def __init__(self, performance_profiler):
        super().__init__()
        self.profiler = performance_profiler

    def schedule(self, task, nodes, mobility_info=None):
        # 각 노드의 예상 처리시간이 가장 짧은 노드 선택
        best = min(nodes,
                   key=lambda n: self.profiler.get_estimated_time(n.name, task.model_name))
        return best