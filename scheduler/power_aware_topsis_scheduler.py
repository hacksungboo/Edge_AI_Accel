import numpy as np
from .topsis_scheduler import TOPSISScheduler

class PowerAwareTOPSISScheduler(TOPSISScheduler):
    def __init__(self, prometheus_collector, task_queues, weights=None, power_threshold=100.0):
        super().__init__(weights or [0.3, 0.5, 0.1, 0.1])
        self.prometheus_collector = prometheus_collector
        self.task_queues = task_queues
        self.power_threshold = power_threshold
        self.slo_violations = []
        self.slo_weight_dict = {"hard": 7.0, "normal": 5.0, "soft": 2.0}

        # 마진 전력 프로파일 (실제 실험 데이터)
        self.idle_power = {
            "coral1": 4.56, "coral2": 4.60,
            "jetson1": 2.38, "jetson2": 2.29,
            "hailo1": 5.04, "hailo2": 4.51
        }
        self.busy_power = {
            "coral1": 5.35, "coral2": 5.28,
            "jetson1": 6.47, "jetson2": 6.24,
            "hailo1": 5.33, "hailo2": 5.01
        }
        # 마진 전력 계산 (busy - idle)
        self.delta_power = {}
        for node in self.idle_power:
            self.delta_power[node] = max(
                self.busy_power.get(node, 4.0) - self.idle_power.get(node, 4.0), 
                0.5  # 최소 0.5W 보장
            )

    def schedule(self, task, nodes, mobility_info=None):
        cluster_power = self.prometheus_collector.get_active_cluster_power(
            [node.name for node in nodes]
        )
        
        #if cluster_power > self.power_threshold*0.8:
            #print(f"Cluster power ({cluster_power:.1f}W) exceeds threshold ({self.power_threshold:.1f}W)")
        #    nodes = self._filter_low_power_nodes(nodes)
        
        # super().schedule() 대신 직접 TOPSIS 로직 구현
        if len(nodes) == 1:
            return nodes[0]
            
        matrix = self.create_decision_matrix(task, nodes)  # 자체 구현된 매트릭스 사용
        normalized = self.normalize_matrix(matrix)
        weighted = normalized * self.weights
        
        # 첫 번째 기준(energy cost)는 minimize, 나머지는 maximize
        ideal_positive = np.array([
            np.min(weighted[:, 0]),  # energy cost: 최소값이 ideal
            np.max(weighted[:, 1]),  # slo urgency: 최대값이 ideal
            np.max(weighted[:, 2]),  # availability: 최대값이 ideal
            np.max(weighted[:, 3])   # performance: 최대값이 ideal
        ])
        
        ideal_negative = np.array([
            np.max(weighted[:, 0]),  # energy cost: 최대값이 anti-ideal
            np.min(weighted[:, 1]),  # slo urgency: 최소값이 anti-ideal
            np.min(weighted[:, 2]),  # availability: 최소값이 anti-ideal
            np.min(weighted[:, 3])   # performance: 최소값이 anti-ideal
        ])
        
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
        matrix = []
        
        for node in nodes:
            name = node.name

            # 1. marginal power efficiency (tasks/s per marginal Watt)
#            processing_time = self._get_processing_time(node, task.model_name)
#            throughput = 1.0 / (processing_time + 1e-6)  # tasks/s
#            delta_p = self.delta_power.get(name, 2.0)  # 기본값 2W
#            marginal_efficiency = throughput / (delta_p + 0.1)

            # 1. Energy cost per task (Wh/task) - 낮을수록 좋음
            processing_time = self._get_processing_time(node, task.model_name)
            throughput = 1.0 / (processing_time + 1e-6)  # tasks/s
            delta_p = self.delta_power.get(name, 2.0)  # 기본값 2W
            energy_cost_per_task = (delta_p + 0.1) / throughput  # Wh/task


            # 2. SLO 긴급도 (타입별 가중치)
            slo_weight = self._get_slo_weight(task.slo_type)
            slo_urgency = slo_weight / (task.adapted_deadline + 0.1)
            
            # 3. node availability
            queue_size = self.task_queues[node.name].qsize()
            availability = 1.0 / (queue_size**2.0 + 1)
            
            # 4. 처리 성능
            performance_score = self._get_processing_capability(node, task.model_name)
            
            row = [energy_cost_per_task , slo_urgency, availability, performance_score]
            matrix.append(row)
            
        return np.array(matrix)

    def _get_processing_time(self, node, model_name):
        """노드의 모델별 처리 시간 반환"""
        processing_times = {
            "coral1": {"mobilenet": 0.01389, "resnet50": 0.07767, "efficientnet": 0.02509, "inception": 0.02122},
            "coral2": {"mobilenet": 0.01373, "resnet50": 0.07777, "efficientnet": 0.02601, "inception": 0.02143},
            "jetson1": {"mobilenet": 0.0158, "resnet50": 0.06002, "efficientnet": 0.04279, "inception": 0.02226},
            "jetson2": {"mobilenet": 0.01556, "resnet50": 0.06037, "efficientnet": 0.04269, "inception": 0.02188},
            "hailo1": {"mobilenet": 0.01256, "resnet50": 0.04571, "efficientnet": 0.00837, "inception": 0.00844},
            "hailo2": {"mobilenet": 0.00761, "resnet50": 0.02416, "efficientnet": 0.01299, "inception": 0.0105},
        }
        
        name = node.name.lower()
        if name.startswith("coral1"):
            return processing_times["coral1"][model_name]
        elif name.startswith("coral2"):
            return processing_times["coral2"][model_name]
        elif name.startswith("jetson1"):
            return processing_times["jetson1"][model_name]
        elif name.startswith("jetson2"):
            return processing_times["jetson2"][model_name]
        elif name.startswith("hailo1"):
            return processing_times["hailo1"][model_name]
        elif name.startswith("hailo2"):
            return processing_times["hailo2"][model_name]

        return 0.1  # 기본값

    def normalize_matrix(self, matrix):
        """벡터 정규화 (부모에서 복사)"""
        norm = np.sqrt(np.sum(matrix**2, axis=0))
        return matrix / (norm + 1e-10)

    def _get_slo_weight(self, slo_type):
        return self.slo_weight_dict.get(slo_type, 5.0)
    
    def _get_processing_capability(self, node, model_name):
        """처리 성능 점수 (역 처리시간)"""
        processing_time = self._get_processing_time(node, model_name)
        return 1.0 / (processing_time + 1e-6)
