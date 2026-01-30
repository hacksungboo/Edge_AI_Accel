# scheduler/random_scheduler.py
import random
from .base_scheduler import BaseScheduler

class RandomScheduler(BaseScheduler):
    def __init__(self):
        super().__init__()

    def schedule(self, task, nodes, mobility_info=None):
        return random.choice(nodes)
