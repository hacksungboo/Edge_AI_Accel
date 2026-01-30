# scheduler/shortest_queue_scheduler.py
from .base_scheduler import BaseScheduler


class ShortestQueueScheduler(BaseScheduler):
    """
    Shortest Queue First (SQF) Scheduler
    각 태스크를 현재 큐 길이가 가장 짧은 노드에 할당
    """
    
    def __init__(self, task_queues):
        """
        Args:
            task_queues (dict): 노드별 태스크 큐 딕셔너리 {node_name: Queue}
        """
        super().__init__()
        self.task_queues = task_queues
    
    def schedule(self, task, nodes, mobility_info=None):
        """
        큐 길이가 가장 짧은 노드를 선택
        
        Args:
            task: 할당할 태스크
            nodes: 후보 노드 리스트
            mobility_info: 모빌리티 정보 (사용하지 않음)
            
        Returns:
            선택된 노드 객체
        """
        # 각 노드의 현재 큐 길이를 확인하여 가장 짧은 노드 선택
        best_node = min(
            nodes,
            key=lambda n: self.task_queues[n.name].qsize()
        )
        
        return best_node
