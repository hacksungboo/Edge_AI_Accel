import numpy as np
from .topsis_scheduler import TOPSISScheduler


class PowerAwareTOPSISScheduler(TOPSISScheduler):
    def __init__(self, prometheus_collector, task_queues, weights=None, power_threshold=100.0):
        super().__init__(weights or [0.3, 0.5, 0.1, 0.1])
        self.prometheus_collector = prometheus_collector
        self.task_queues = task_queues
        self.power_threshold = power_threshold
        self.slo_violations = []
        self.slo_weight_dict = {"hard": 10.0, "normal": 5.0, "soft": 2.0}

    def schedule(self, task, nodes, mobility_info=None):
        cluster_power = self.prometheus_collector.get_active_cluster_power(
            [node.name for node in nodes]
        )
        
        if cluster_power > self.power_threshold:
            print(f"Cluster power ({cluster_power:.1f}W) exceeds threshold ({self.power_threshold:.1f}W)")
            nodes = self._filter_low_power_nodes(nodes)
        
        # super().schedule() 대신 직접 TOPSIS 로직 구현
        if len(nodes) == 1:
            return nodes[0]
            
        matrix = self.create_decision_matrix(task, nodes)  # 자체 구현된 매트릭스 사용
        normalized = self.normalize_matrix(matrix)
        weighted = normalized * self.weights
        
        ideal_positive = np.max(weighted, axis=0)
        ideal_negative = np.min(weighted, axis=0)
        
        distances_pos = np.sqrt(np.sum((weighted - ideal_positive)**2, axis=1))
        distances_neg = np.sqrt(np.sum((weighted - ideal_negative)**2, axis=1))
        
        closeness = distances_neg / (distances_pos + distances_neg + 1e-10)
        
        best_node_idx = np.argmax(closeness)
        return nodes[best_node_idx]
    
    def _filter_low_power_nodes(self, nodes):
        """현재 전력 소모가 낮은 노드들만 필터링"""
        sorted_nodes = sorted(nodes, key=lambda n: n.power_consumption)
        num_nodes = max(1, len(sorted_nodes) // 2)
        return sorted_nodes[:num_nodes]

    def create_decision_matrix(self, task, nodes):
        """개선된 decision matrix 생성"""
        matrix = []
        
        for node in nodes:
            # 1. 전력 효율성
            current_power = node.power_consumption
            power_efficiency = 1.0 / (current_power + 0.1)
            
            # 2. SLO 긴급도 (타입별 가중치 적용)
            slo_weight = self._get_slo_weight(task.slo_type)
            slo_urgency = slo_weight / (task.adapted_deadline + 0.1)
            
            # 3. 노드 가용성 (큐 길이 기반)
            queue_size = self.task_queues[node.name].qsize()
            availability = 1.0 / (queue_size + 1)
            
            # 4. 처리 성능
            performance_score = self._get_processing_capability(node, task.model_name)
            
            row = [power_efficiency, slo_urgency, availability, performance_score]
            matrix.append(row)
            
        return np.array(matrix)

    def normalize_matrix(self, matrix):
        """벡터 정규화 (부모에서 복사)"""
        norm = np.sqrt(np.sum(matrix**2, axis=0))
        return matrix / (norm + 1e-10)

    def _get_slo_weight(self, slo_type):
        return self.slo_weight_dict.get(slo_type, 5.0)
    
    def _get_processing_capability(self, node, model_name):
        """노드별 모델 처리 능력 점수"""
        capability_matrix = {
            "coral": {"mobilenet": 0.9, "resnet50": 0.7},
            "hailo": {"mobilenet": 0.8, "resnet50": 0.8},
            "jetson": {"mobilenet": 0.7, "resnet50": 0.9}
        }
        
        for node_type in capability_matrix:
            if node_type in node.name.lower():
                return capability_matrix[node_type].get(model_name, 0.5)
        return 0.5
