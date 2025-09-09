import random
import time
from mobility.mobility_manager import MobilityManager
from scheduler.base_scheduler import Task


class TaskGenerator:
    def __init__(self, mobility_manager: MobilityManager):
        self.mobility_manager = mobility_manager
        self.device_types = ['fixed', 'pedestrian', 'vehicle', 'drone']
        self.device_ratios = [0.35, 0.25, 0.25, 0.15]
        self.slo_types = ['soft', 'normal', 'hard']
        self.slo_ratios = [0.5, 0.35, 0.15]

    def generate_devices(self, num_devices=100):
        """가상 디바이스 생성"""
        for i in range(num_devices):
            device_id = f"device_{i:03d}"
            device_type = random.choices(self.device_types, weights=self.device_ratios)[0]
            self.mobility_manager.register_device(device_id, device_type)        
    
    def generate_poisson_tasks(self, lambda_rate=5.0, duration=3600):
        tasks = []
        current_time = 0
        task_id = 0
        
        while current_time < duration:
            interval = random.expovariate(lambda_rate)
            current_time += interval
            
            device_id = f"device_{random.randint(0, 999):03d}"  # 디바이스 개수와 맞도록 조정 필요
            model_name = random.choice(["mobilenet", "resnet50"])
            
            # SLO 타입별 deadline 설정 (수정된 부분)
            slo_type = random.choices(self.slo_types, weights=self.slo_ratios, k=1)[0]
            
            if slo_type == "hard":
                base_deadline = random.uniform(0.05, 0.15) # 50 ~ 150ms
            elif slo_type == "normal":
                base_deadline = random.uniform(0.1, 0.3) # 100 ~ 300ms
            else:
                base_deadline = random.uniform(0.2, 0.5) # 200 ~ 500ms
            
            task = Task(
                task_id=f"task_{task_id:06d}",
                model_name=model_name,
                device_id=device_id,
                slo_type=slo_type,
                base_deadline=base_deadline
            )
            tasks.append((current_time, task))
            task_id += 1
            
        return tasks
