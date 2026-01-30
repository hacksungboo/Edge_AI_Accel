import random
import time
from mobility.mobility_manager import MobilityManager
from scheduler.base_scheduler import Task


class TaskGenerator:
    def __init__(self, mobility_manager: MobilityManager):
        self.mobility_manager = mobility_manager
        self.device_types = ['fixed', 'pedestrian', 'vehicle', 'drone']
        self.device_ratios = [0.4, 0.3, 0.2, 0.1]
        self.slo_types = ['soft', 'normal', 'hard']
        self.slo_ratios = [0.5, 0.35, 0.15]

        self.device_lambdas={
            'fixed': 2.0,       # task/sec per device
            'pedestrian': 0.2,
            'vehicle': 1.0,
            'drone': 0.5
        }
        

    def generate_devices(self, num_devices=125,coverage_range=150):
        """가상 디바이스 생성"""
        for i in range(num_devices):
            device_id = f"device_{i:03d}"
            device_type = random.choices(self.device_types, weights=self.device_ratios)[0]
            self.mobility_manager.register_device(device_id, device_type,coverage_range)        
            
            device = self.mobility_manager.devices[device_id]
            device.lambda_rate = self.device_lambdas[device_type]


    
    def generate_poisson_tasks(self, lambda_rate=10.0, duration=600, coverage_range=150):
        """포아송 분포 기반 태스크 생성 (커버리지 내 디바이스만)"""
        tasks = []
        current_time = 0
        task_id = 0
        
        # 등록된 디바이스 ID 목록
        device_ids = list(self.mobility_manager.devices.keys())
        
        while current_time < duration:
            interval = random.expovariate(lambda_rate)
            current_time += interval
            
            # 현재 커버리지 내에 있는 디바이스만 필터링
            active_devices = [
                dev_id for dev_id in device_ids 
                if self.mobility_manager.is_in_coverage(dev_id, coverage_range)
            ]
            
            # 커버리지 내 디바이스가 없으면 이번 태스크는 생성 안 함
            if not active_devices:
                continue
            
            # 커버리지 내 디바이스 중 랜덤 선택
            device_id = random.choice(active_devices)
            model_name = random.choice(["mobilenet", "resnet50", "efficientnet", "inception"])
            
            # SLO 타입별 deadline 설정
            slo_type = random.choices(self.slo_types, weights=self.slo_ratios, k=1)[0]
            
            if slo_type == "hard":
                base_deadline = random.uniform(0.05, 0.15)
            elif slo_type == "normal":
                base_deadline = random.uniform(0.1, 0.3)
            else:
                base_deadline = random.uniform(0.2, 0.5)
            
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



    def generate_poisson_tasks_dynamic(self, lambda_rate=10.0, duration=600, coverage_range=150):
        """태스크 타임라인만 생성 (디바이스는 나중에 선택)"""
        tasks = []
        current_time = 0
        task_id = 0
        
        while current_time < duration:
            interval = random.expovariate(lambda_rate)
            current_time += interval
            
            model_name = random.choice(["mobilenet", "resnet50", "efficientnet", "inception"])
            slo_type = random.choices(self.slo_types, weights=self.slo_ratios, k=1)[0]
            
            if slo_type == "hard":
                base_deadline = random.uniform(0.05, 0.15)
            elif slo_type == "normal":
                base_deadline = random.uniform(0.1, 0.3)
            else:
                base_deadline = random.uniform(0.2, 0.5)
            
            # device_id는 None으로 설정 (나중에 선택)
            task = Task(
                task_id=f"task_{task_id:06d}",
                model_name=model_name,
                device_id=None,  # ← 나중에 설정
                slo_type=slo_type,
                base_deadline=base_deadline
            )
            tasks.append((current_time, task))
            task_id += 1
            
        return tasks




        