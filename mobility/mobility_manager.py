import random
import numpy as np
import time

class MobilityInfo:
    def __init__(self, device_type, current_position, speed=0, direction=0):
        self.device_type = device_type
        self.current_position = current_position  # (x, y)
        self.speed = speed                        # m/s, 초기 속도 고정
        self.direction = direction                # radians

class MobilityManager:
    def __init__(self, area_size=(1000, 1000)):
        self.devices = {}  # device_id -> MobilityInfo
        self.area_size = area_size

    def register_device(self, device_id, device_type):
        """디바이스 등록"""
        initial_position = (
            random.uniform(0, self.area_size[0]),
            random.uniform(0, self.area_size[1])
        )
        if device_type == 'fixed':
            speed = 0
            direction = 0
        elif device_type == 'pedestrian':
            speed = random.uniform(1, 3)
            direction = random.uniform(0, 2 * np.pi)
        elif device_type == 'vehicle':
            speed = random.uniform(10, 25)
            direction = random.uniform(0, 2 * np.pi)
        elif device_type == 'drone':
            speed = random.uniform(5, 15)
            direction = random.uniform(0, 2 * np.pi)
        else:
            speed = 0
            direction = 0

        self.devices[device_id] = MobilityInfo(
            device_type, initial_position, speed, direction
        )

    def update_mobility(self, dt=1.0):
        """모든 디바이스의 위치 업데이트 (속도 고정, 방향만 확률적 변경)"""
        for device_id, info in self.devices.items():
            x, y = info.current_position
            speed = info.speed
            direction = info.direction

            if info.device_type == 'fixed':
                continue

            change_probability = 0.1  # 기본 10%
            if info.device_type == 'vehicle':
                change_probability = 0.05
            elif info.device_type == 'drone':
                change_probability = 0.08

            if random.random() < change_probability:
                direction = random.uniform(0, 2 * np.pi)

            vx = speed * np.cos(direction)
            vy = speed * np.sin(direction)

            x += vx * dt
            y += vy * dt

            # 경계 처리 (영역 범위 내 유지)
            x = max(0, min(self.area_size[0], x))
            y = max(0, min(self.area_size[1], y))

            # 상태 업데이트
            info.current_position = (x, y)
            info.direction = direction

    def predict_coverage_exit_time(self, device_id: str, coverage_range=100):
        """디바이스가 커버리지 벗어날 예상 시간 계산"""
        if device_id not in self.devices:
            return float('inf')
        info = self.devices[device_id]
        if info.device_type == 'fixed':
            return float('inf')
        speed = info.speed
        if speed == 0:
            return float('inf')

        center = (self.area_size[0] / 2, self.area_size[1] / 2)
        distance_to_center = np.sqrt(
            (info.current_position[0] - center[0])**2 +
            (info.current_position[1] - center[1])**2
        )

        if distance_to_center > coverage_range:
            return 0.1  # 이미 벗어남

        distance_to_edge = coverage_range - distance_to_center
        time_to_exit = distance_to_edge / (speed + 0.1)

        return max(0.1, time_to_exit)
