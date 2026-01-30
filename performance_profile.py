class PerformanceProfiler:
    def __init__(self):
        # 실측 데이터 (10000회 평균)
        self.processing_times = {
            "coral1": {"mobilenet": 0.01389, "resnet50": 0.07767, "efficientnet": 0.02509, "inception": 0.02122},
            "coral2": {"mobilenet": 0.01373, "resnet50": 0.07777, "efficientnet": 0.02601, "inception": 0.02143},
            "jetson1": {"mobilenet": 0.0158, "resnet50": 0.06002, "efficientnet": 0.04279, "inception": 0.02226},
            "jetson2": {"mobilenet": 0.01556, "resnet50": 0.06037, "efficientnet": 0.04269, "inception": 0.02188},
            "hailo1": {"mobilenet": 0.01256, "resnet50": 0.04571, "efficientnet": 0.00837, "inception": 0.00844},
            "hailo2": {"mobilenet": 0.00761, "resnet50": 0.02416, "efficientnet": 0.01299, "inception": 0.0105},
        }
    

    def get_estimated_time(self, node_name, model_name):
        """예상 처리 시간 반환"""
        return self.processing_times.get(node_name, {}).get(model_name, 1.0)
    
    def can_meet_deadline(self, node_name, model_name, deadline):
        """데드라인 준수 가능 여부"""
        estimated_time = self.get_estimated_time(node_name, model_name)
        # 안전 마진 20% 추가
        return estimated_time * 1.2 < deadline
