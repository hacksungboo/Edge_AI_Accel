# scheduler/power_only_scheduler.py
from .base_scheduler import BaseScheduler

class PowerOnlyScheduler(BaseScheduler):
    def __init__(self, prometheus_collector):
        super().__init__()
        self.collector = prometheus_collector

    def schedule(self, task, nodes, mobility_info=None):
        # 각 노드의 실시간 전력소모를 비교
        best = min(nodes, key=lambda n: self.collector.get_node_power_consumption(n.name))
        return best
