class PerformanceProfiler:
    def __init__(self):
        # 실측 데이터 (10000회 평균)
        self.processing_times = {
            "coral1": {"mobilenet": 0.034817, "resnet50": 0.077973},
            "coral2": {"mobilenet": 0.034817, "resnet50": 0.077973},
            "jetson1": {"mobilenet": 0.01666, "resnet50": 0.034158},
            "jetson2": {"mobilenet": 0.01666, "resnet50": 0.034158},
            "hailo1": {"mobilenet": 0.01294, "resnet50": 0.070979},
            "hailo2": {"mobilenet": 0.01294, "resnet50": 0.070979}
        }
    
    def get_estimated_time(self, node_name, model_name):
        """예상 처리 시간 반환"""
        return self.processing_times.get(node_name, {}).get(model_name, 1.0)
    
    def can_meet_deadline(self, node_name, model_name, deadline):
        """데드라인 준수 가능 여부"""
        estimated_time = self.get_estimated_time(node_name, model_name)
        # 안전 마진 20% 추가
        return estimated_time * 1.2 < deadline
