import random
import numpy as np
import time

class MobilityInfo:
    def __init__(self, device_type, current_position, speed=0, direction=0):
        self.device_type = device_type
        self.current_position = current_position  # (x, y)
        self.speed = speed                        # m/s, 초기 속도 고정
        self.direction = direction                # radians
        self.lambda_rate=0.0

class MobilityManager:
    def __init__(self, area_size=(500, 500)):
        self.devices = {}  # device_id -> MobilityInfo
        self.area_size = area_size

    def register_device(self, device_id, device_type,coverage_range=150):
        
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


    def is_in_coverage(self, device_id: str, coverage_range=150):
        """디바이스가 현재 커버리지 내에 있는지 확인"""
        if device_id not in self.devices:
            return False
        
        info = self.devices[device_id]
        center = np.array([self.area_size[0] / 2, self.area_size[1] / 2])
        position = np.array(info.current_position)
        
        distance_to_center = np.linalg.norm(position - center)
        return distance_to_center <= coverage_range    

    def _sample_position_in_coverage(self, coverage_range):
        """edge 클라우드 central coverage 내에서만 position 샘플링"""
        center_x, center_y = self.area_size[0] / 2, self.area_size[1] / 2
        while True:
            # Polar sampling (균일 분포)
            r = coverage_range * np.sqrt(random.uniform(0, 1))
            theta = random.uniform(0, 2 * np.pi)
            x = center_x + r * np.cos(theta)
            y = center_y + r * np.sin(theta)

            # 혹시라도 영역 밖으로 갈 경우 제외
            if (0 <= x <= self.area_size[0]) and (0 <= y <= self.area_size[1]):
                return (x, y)


    def update_mobility(self, dt=2.0):
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

            # 경계 처리
            x = max(0, min(self.area_size[0], x))
            y = max(0, min(self.area_size[1], y))

            # 상태 업데이트
            info.current_position = (x, y)
            info.direction = direction


    def predict_coverage_exit_time(self, device_id: str, coverage_range=150):
        """디바이스가 커버리지 벗어날 예상 시간 계산 (방향 고려)"""
        if device_id not in self.devices:
            return float('inf')
        
        info = self.devices[device_id]
        if info.device_type == 'fixed':
            return float('inf')
        
        speed = info.speed
        if speed == 0:
            return float('inf')

        # 현재 위치와 중심
        center = np.array([self.area_size[0] / 2, self.area_size[1] / 2])
        position = np.array(info.current_position)
        
        # 이동 방향 벡터 (단위: m/s)
        velocity = np.array([
            speed * np.cos(info.direction),
            speed * np.sin(info.direction)
        ])
        
        # 중심으로부터의 상대 위치
        relative_pos = position - center
        distance_to_center = np.linalg.norm(relative_pos)
        
        # 이미 커버리지 밖
        if distance_to_center > coverage_range:
            return None  # 스케줄링 불가
        
        # 원과의 교차 시간 계산 (quadratic formula)
        # ||p + vt||² = R²
        # ||p||² + 2(p·v)t + ||v||²t² = R²
        
        a = np.dot(velocity, velocity)  # ||v||²
        b = 2 * np.dot(relative_pos, velocity)  # 2(p·v)
        c = np.dot(relative_pos, relative_pos) - coverage_range**2  # ||p||² - R²
        
        discriminant = b**2 - 4*a*c
        
        # 교차점 없음 (안쪽으로 이동 중이거나 접선 이동)
        if discriminant < 0:
            return float('inf')
        
        # 두 교차점 중 미래의 양수 시간만 선택
        t1 = (-b + np.sqrt(discriminant)) / (2*a)
        t2 = (-b - np.sqrt(discriminant)) / (2*a)
        
        # 양수 시간 중 가장 빠른 시간
        valid_times = [t for t in [t1, t2] if t > 0]
        
        if not valid_times:
            return float('inf')  # 안쪽으로 이동 중
        
        time_to_exit = min(valid_times)
        return max(0.01, time_to_exit)  # 최소 0.5초 보장