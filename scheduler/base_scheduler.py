import time
from abc import ABC, abstractmethod


class BaseScheduler(ABC):
    @abstractmethod
    def schedule(self, task, nodes, mobility_info=None):
        pass


class Task:
    def __init__(self, task_id, model_name, device_id, slo_type="normal", base_deadline=5.0):
        self.task_id = task_id
        self.model_name = model_name
        self.device_id = device_id
        self.slo_type = slo_type  # "soft", "normal", "hard"
        self.base_deadline = base_deadline
        self.adapted_deadline = base_deadline
        self.submit_time = time.time()
        
        # SLO 추적용 필드
        self.assigned_node = None
        self.start_time = None
        self.completion_time = None
        self.is_slo_violated = False
    
    def get_penalty_weight(self):
        # SLO 위반 시 페널티 가중치
        if self.slo_type == "hard":
            return 10.0
        elif self.slo_type == "normal": 
            return 3.0
        else: 
            return 1.0


class Node:
    def __init__(self, name, url, prometheus_collector=None, current_utilization=0.0):
        self.name = name
        self.url = url
        self.prometheus_collector = prometheus_collector
        self.current_utilization = current_utilization
        self.network_latency = 0.1  # 기본값
        self.last_power_update = 0
        self.cached_power = 15.0  # 기본값
        
    @property
    def power_consumption(self):
        """실시간 전력 소모량 (1초 캐시)"""
        current_time = time.time()
        
        # 1초 이내면 캐시된 값 사용
        if current_time - self.last_power_update < 1.0:
            return self.cached_power
            
        # 프로메테우스에서 최신 전력 데이터 가져오기
        if self.prometheus_collector:
            self.cached_power = self.prometheus_collector.get_node_power_consumption(self.name)
            self.last_power_update = current_time
            
        return self.cached_power
