from mobility.mobility_manager import MobilityManager

class DeadlineAdapter:
    def __init__(self, mobility_manager: MobilityManager):
        self.mobility_manager = mobility_manager

    def adapt_deadline(self, task):
        """Mobility 정보 기반 deadline 적응"""
        device_info = self.mobility_manager.devices.get(task.device_id)
        
        if not device_info or device_info.device_type == 'fixed':
            task.adapted_deadline = task.base_deadline
            return task

        # 커버리지 이탈 예상 시간
        exit_time = self.mobility_manager.predict_coverage_exit_time(task.device_id, coverage_range=150)
        
        # exit_time이 None이면 이미 밖 (예외)
        if exit_time is None:
            task.adapted_deadline = task.base_deadline
            return task
        
        # 속도 기반 adapt factor
        if device_info.speed > 20:      # 고속 (차량)
            adapt_factor = 0.3
        elif device_info.speed > 5:     # 중속 (드론)
            adapt_factor = 0.6
        else:                           # 저속 (보행자)
            adapt_factor = 1.0

        # Deadline 조정
        if exit_time < task.base_deadline:
            task.adapted_deadline = min(exit_time * 0.8, task.base_deadline * adapt_factor)
        else:
            task.adapted_deadline = task.base_deadline * adapt_factor

        # 최소값 보장
        task.adapted_deadline = max(0.1, task.adapted_deadline)
        
        return task