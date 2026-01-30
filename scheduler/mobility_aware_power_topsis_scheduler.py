import numpy as np
import time
import threading
from .topsis_scheduler import TOPSISScheduler

class MobilityAwarePowerTOPSISScheduler(TOPSISScheduler):
    """
    모빌리티를 올바르게 반영한 전력 인식 TOPSIS 스케줄러
    
    4가지 기준:
    1. Deadline Safety Margin (40%) - 모빌리티 핵심!
    2. Recent Load (30%) - 최근 처리량 기반 로드밸런싱 (NEW!)
    3. Processing Capability (20%) - 처리 성능
    4. Energy Efficiency (10%) - 에너지 효율
    """
    
    def __init__(self, prometheus_collector, task_queues, mobility_manager=None, 
                 weights=None, power_threshold=100.0):
        # 4개 기준에 대한 가중치
        default_weights = [0.40, 0.30, 0.20, 0.10]
        super().__init__(weights or default_weights)
        
        self.prometheus_collector = prometheus_collector
        self.task_queues = task_queues
        self.mobility_manager = mobility_manager
        self.power_threshold = power_threshold
        self.slo_violations = []
        self.slo_weight_dict = {"hard": 7.0, "normal": 5.0, "soft": 2.0}
        

        self.node_processed_count = {}      # 각 노드의 최근 처리 count
        self.node_last_reset_time = {}      # 마지막 reset 시간
        self.count_lock = threading.Lock()  # Thread-safe
        self.processing_times = {
        "coral1": {"mobilenet": 0.01389, "resnet50": 0.07767, "efficientnet": 0.02509, "inception": 0.02122},
        "coral2": {"mobilenet": 0.01373, "resnet50": 0.07777, "efficientnet": 0.02601, "inception": 0.02143},
        "jetson1": {"mobilenet": 0.0158, "resnet50": 0.06002, "efficientnet": 0.04279, "inception": 0.02226},
        "jetson2": {"mobilenet": 0.01556, "resnet50": 0.06037, "efficientnet": 0.04269, "inception": 0.02188},
        "hailo1": {"mobilenet": 0.01256, "resnet50": 0.04571, "efficientnet": 0.00837, "inception": 0.00844},
        "hailo2": {"mobilenet": 0.00761, "resnet50": 0.02416, "efficientnet": 0.01299, "inception": 0.0105},
    }



        self.avg_processing_times = {}
        for node_name, times_dict in self.processing_times.items():
            times_list = [times_dict[model] for model in ['mobilenet', 'resnet50', 'efficientnet', 'inception']]
            self.avg_processing_times[node_name] = np.mean(times_list)


        # 모든 노드 초기화
        for node_name in ['coral1', 'coral2', 'jetson1', 'jetson2', 'hailo1', 'hailo2']:
            self.node_processed_count[node_name] = 0
            self.node_last_reset_time[node_name] = time.time()
        
        # 실제 전력 프로파일 (실험 데이터)
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
                0.5
            )



    def schedule(self, task, nodes, mobility_info=None):
        """
        4가지 기준을 고려한 TOPSIS 기반 스케줄링
        
        Args:
            task: Task 객체
            nodes: 선택 가능한 노드들
            mobility_info: 모빌리티 정보 (device_id 등)
        
        Returns:
            선택된 노드
        """
        
        # 단일 노드 처리
        if len(nodes) == 1:
            return nodes[0]
        
        # 4개 기준을 포함한 의사결정 행렬 생성
        matrix = self.create_decision_matrix(task, nodes, mobility_info)
        normalized = self.normalize_matrix(matrix)
        weighted = normalized * self.weights
        
        # Ideal positive & negative 설정 (모두 maximize)
        ideal_positive = np.array([np.max(weighted[:, i]) for i in range(4)])
        ideal_negative = np.array([np.min(weighted[:, i]) for i in range(4)])
        
        # 거리 계산
        distances_pos = np.sqrt(np.sum((weighted - ideal_positive)**2, axis=1))
        distances_neg = np.sqrt(np.sum((weighted - ideal_negative)**2, axis=1))
        
        # 근접도 계산
        closeness = distances_neg / (distances_pos + distances_neg + 1e-10)
        
        # 최고 점수 노드 선택
        best_node_idx = np.argmax(closeness)

        return nodes[best_node_idx]



    def create_decision_matrix(self, task, nodes, mobility_info=None):
        """
        4가지 기준을 포함한 의사결정 행렬 생성
        
        기준:
        1. Deadline Safety Margin (모빌리티 핵심!)
           = remaining_deadline - expected_completion_time
           
        2. Recent Load (로드밸런싱!) ← NEW!
           = 1 / (최근 60초 처리량 + 1)
           높을수록: 덜 바쁨
           
        3. Processing Capability (처리 성능)
           = 1 / processing_time
           
        4. Energy Efficiency (에너지 효율)
           = throughput / power_consumption
        """
        matrix = []
        
        for node in nodes:
            name = node.name
            
            # 기준 1: Deadline Safety Margin
            remaining_deadline = self._calculate_remaining_deadline(task, mobility_info)
            expected_completion_time = self._get_expected_completion_time(node, task)
            safety_margin = remaining_deadline - expected_completion_time
            
            if safety_margin < 0:
                criterion_1 = -1.0  # 페널티
            elif safety_margin == 0:
                criterion_1 = 0.001
            else:
                criterion_1 = safety_margin
            
        
            # 기준 2: Recent Load (최근 처리량 기반) ← NEW!
            processed_recent = self.get_recent_processed_count(node, time_window=5)
            criterion_2 = 1.0 / (processed_recent + 1)
                      
            # 기준 3: Processing Capability
            processing_time = self._get_processing_time(node, task.model_name)
            throughput = 1.0 / (processing_time + 1e-6)
            criterion_3 = throughput
            
            # 기준 4: Energy Efficiency
            delta_power = self.delta_power.get(name, 2.0)
            energy_cost_per_task = (delta_power + 0.1) / throughput
            criterion_4 = 1.0 / (energy_cost_per_task + 0.01)
            
            row = [
                criterion_1,  # Safety Margin
                criterion_2,  # Recent Load
                criterion_3,  # Processing capability
                criterion_4   # Energy efficiency
            ]
            matrix.append(row)
        
        return np.array(matrix)



    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NEW! 최근 처리량 관리 함수들
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def update_processed_count(self, node_name):
        """
        노드의 처리 완료 시 호출 (worker thread에서)
        
        사용 예:
            scheduler.update_processed_count(selected_node.name)
        """
        with self.count_lock:
            if node_name in self.node_processed_count:
                self.node_processed_count[node_name] += 1
    
    
    def get_recent_processed_count(self, node, time_window=5):
        """
        최근 time_window(초) 동안의 처리량 반환
        
        동작:
        - 60초 이상 지났으면 자동으로 카운터를 0으로 reset
        - 계속 자동 로드밸런싱
        
        Args:
            node: Node 객체
            time_window: 시간 윈도우 (초)
        
        Returns:
            최근 처리량
        """
        #with self.count_lock:
        #    current_time = time.time()
        #    elapsed = current_time - self.node_last_reset_time[node.name]
            
            # 시간 윈도우 초과하면 reset
        #    if elapsed > time_window:
        #        self.node_processed_count[node.name] = 0
        #        self.node_last_reset_time[node.name] = current_time
        #        return 0
            
        #    return self.node_processed_count[node.name]

        current_time = time.time()
        node_name=node.name
        last_reset = self.node_last_reset_time.get(node_name, current_time)
        elapsed = current_time - last_reset
        
        if elapsed > time_window:
            self.node_processed_count[node_name] = 0
            self.node_last_reset_time[node_name] = current_time
            return 0
        
        return self.node_processed_count.get(node_name, 0)

    def _calculate_remaining_deadline(self, task, mobility_info):
        return task.adapted_deadline
        """
        모빌리티를 고려한 남은 deadline 계산
        """
        if mobility_info is None or self.mobility_manager is None:
            return task.adapted_deadline
        
        device_id = task.device_id
        device_info = self.mobility_manager.devices.get(device_id)
        
        if device_info is None:
            return task.adapted_deadline
        
        try:
            current_pos = device_info.current_position
            if not isinstance(current_pos, tuple) or len(current_pos) != 2:
                return task.adapted_deadline
            
            coverage_center = (250, 250)
            coverage_range = 150
            
            current_distance = np.sqrt(
                (current_pos[0] - coverage_center[0])**2 + 
                (current_pos[1] - coverage_center[1])**2
            )
            
            distance_to_boundary = coverage_range - current_distance
            
            if distance_to_boundary <= 0:
                return max(task.adapted_deadline * 0.1, 0.01)
            
            device_speed = getattr(device_info, 'speed', 1.5)
            if device_speed <= 0:
                device_speed = 1.5
            
            time_to_exit = distance_to_boundary / device_speed
            remaining_deadline = min(task.adapted_deadline, time_to_exit)
            
            return max(remaining_deadline, 0.01)
        
        except Exception as e:
            print(f"[WARNING] Error calculating remaining deadline: {e}")
            return task.adapted_deadline


    def _get_expected_completion_time(self, node, task):
        queue_size = self.task_queues[node.name].qsize()
        processing_time = self._get_processing_time(node, task.model_name)
        
        avg_processing_time = self.avg_processing_times[node.name]
        
        queue_wait_time = queue_size * avg_processing_time
        total_time = queue_wait_time + processing_time
        
        return total_time



    def _get_processing_time(self, node, model_name):
        # ✅ 인스턴스 변수 사용!
        name = node.name.lower()
        
        if name.startswith("coral1"):
            return self.processing_times["coral1"].get(model_name, self.avg_processing_times["coral1"])
        elif name.startswith("coral2"):
            return self.processing_times["coral2"].get(model_name, self.avg_processing_times["coral2"])
        elif name.startswith("jetson1"):
            return self.processing_times["jetson1"].get(model_name, self.avg_processing_times["jetson1"])
        elif name.startswith("jetson2"):
            return self.processing_times["jetson2"].get(model_name, self.avg_processing_times["jetson2"])
        elif name.startswith("hailo1"):
            return self.processing_times["hailo1"].get(model_name, self.avg_processing_times["hailo1"])
        elif name.startswith("hailo2"):
            return self.processing_times["hailo2"].get(model_name, self.avg_processing_times["hailo2"])
        
        return 0.05



    def normalize_matrix(self, matrix):
        """벡터 정규화 (Euclidean normalization)"""
        norm = np.sqrt(np.sum(matrix**2, axis=0))
        return matrix / (norm + 1e-10)



    def _get_slo_weight(self, slo_type):
        """SLO 타입별 가중치"""
        return self.slo_weight_dict.get(slo_type, 5.0)

