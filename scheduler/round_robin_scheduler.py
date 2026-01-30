# scheduler/round_robin_scheduler.py
from .base_scheduler import BaseScheduler

class RoundRobinScheduler(BaseScheduler):
    def __init__(self):
        super().__init__()
        self._idx = 0

    def schedule(self, task, nodes, mobility_info=None):
        if not nodes:
            return None
        # 순환 인덱스
        node = nodes[self._idx % len(nodes)]
        self._idx += 1
        return node
