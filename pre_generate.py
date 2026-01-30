"""
사전 생성 모듈
- 모빌리티 시뮬레이션
- Task 생성
- 데이터 저장/로드
"""

import copy
import random
import time
import pickle
from scheduler.base_scheduler import Task


def pre_simulate_mobility(mobility_manager, duration=600, interval=2.0, coverage_range=150):
    """
    모빌리티를 사전 시뮬레이션하여 모든 시점의 디바이스 상태 기록
    
    Args:
        mobility_manager: MobilityManager 인스턴스
        duration: 시뮬레이션 시간 (초)
        interval: 업데이트 간격 (초)
        coverage_range: 커버리지 반경
    
    Returns:
        mobility_snapshots: {
            0.0: {'device_000': {'position': (x,y), 'in_coverage': bool, ...}, ...},
            2.0: {...},
            ...
        }
    """
    mobility_snapshots = {}
    current_time = 0.0
    
    # 원본 보존을 위한 깊은 복사
    sim_mobility = copy.deepcopy(mobility_manager)
    
    print(f"\n{'='*80}")
    print(f"[PRE-SIM] Starting mobility pre-simulation")
    print(f"{'='*80}")
    print(f"Duration: {duration}s")
    print(f"Interval: {interval}s")
    print(f"Coverage range: {coverage_range}m")
    print(f"Expected snapshots: {int(duration/interval) + 1}")
    
    snapshot_count = 0
    
    while current_time <= duration:
        # 현재 시점의 모든 디바이스 상태 저장
        snapshot = {}
        
        for device_id, device in sim_mobility.devices.items():
            # 커버리지 확인
            in_coverage = sim_mobility.is_in_coverage(device_id, coverage_range=coverage_range)
            
            snapshot[device_id] = {
                'position': device.current_position,
                'speed': device.speed,
                'direction': device.direction,
                'device_type': device.device_type,
                'lambda_rate': device.lambda_rate,
                'in_coverage': in_coverage
            }
        
        mobility_snapshots[current_time] = snapshot
        snapshot_count += 1
        
        # 진행 상황 출력 (60초마다)
        if int(current_time) % 60 == 0:
            in_coverage_count = sum(1 for s in snapshot.values() if s['in_coverage'])
            print(f"[PRE-SIM] Progress: {current_time:>6.1f}s / {duration}s "
                  f"({snapshot_count:>4} snapshots, {in_coverage_count:>3}/100 devices in coverage)")
        
        # 다음 시점으로 이동
        if current_time < duration:
            sim_mobility.update_mobility(dt=interval)
        current_time += interval
    
    print(f"\n[PRE-SIM] ✓ Completed: {len(mobility_snapshots)} snapshots (0.0~{duration}s)")
    
    # 통계 출력
    total_coverage = sum(
        sum(1 for d in snapshot.values() if d['in_coverage'])
        for snapshot in mobility_snapshots.values()
    )
    avg_coverage = total_coverage / len(mobility_snapshots)
    print(f"[PRE-SIM] Average devices in coverage: {avg_coverage:.1f} / 100")
    
    return mobility_snapshots


def pre_generate_tasks(mobility_snapshots, slo_types, slo_ratios, duration=600, interval=2.0):
    """
    모빌리티 스냅샷을 참조하여 모든 task 사전 생성
    
    핵심:
    1. 각 디바이스의 lambda_rate 사용
    2. Poisson 분포로 도착 시간 결정
    3. 해당 시점의 커버리지 체크
    4. 커버리지 내부만 task 생성
    
    Args:
        mobility_snapshots: 사전 시뮬레이션한 모빌리티 스냅샷
        slo_types: ['soft', 'normal', 'hard']
        slo_ratios: [0.5, 0.35, 0.15]
        duration: 시뮬레이션 시간 (초)
        interval: 스냅샷 간격 (초)
    
    Returns:
        all_tasks: [(arrival_time, task, device_snapshot), ...]
    """
    all_tasks = []
    model_choices = ["mobilenet", "resnet50", "efficientnet", "inception"]
    
    print(f"\n{'='*80}")
    print(f"[PRE-GEN] Starting task pre-generation")
    print(f"{'='*80}")
    
    # 모든 디바이스 목록
    device_ids = list(mobility_snapshots[0.0].keys())
    print(f"Total devices: {len(device_ids)}")
    
    # 각 디바이스별로 독립적으로 task 생성
    device_task_counts = {}
    
    for idx, device_id in enumerate(device_ids, 1):
        # 디바이스 정보 (t=0 기준)
        device_info = mobility_snapshots[0.0][device_id]
        lambda_rate = device_info['lambda_rate']
        device_type = device_info['device_type']
        
        # Poisson 프로세스로 도착 시간 생성
        current_time = 0.0
        task_count = 0
        
        while current_time < duration:
            # 다음 task까지 간격
            inter_arrival_time = random.expovariate(lambda_rate)
            current_time += inter_arrival_time
            
            if current_time >= duration:
                break
            
            # 가장 가까운 스냅샷 시간 찾기
            snapshot_time = round(current_time / interval) * interval
            snapshot_time = min(snapshot_time, duration)
            
            # 해당 시점의 디바이스 상태
            if snapshot_time not in mobility_snapshots:
                continue
            
            device_snapshot = mobility_snapshots[snapshot_time].get(device_id)
            
            if device_snapshot is None:
                continue
            
            # 커버리지 체크 (핵심!)
            if not device_snapshot['in_coverage']:
                continue  # 커버리지 밖 → task 생성 안함
            
            # Task 생성
            task_id = f"{device_id}_{task_count:06d}"
            model_name = random.choice(model_choices)
            slo_type = random.choices(slo_types, weights=slo_ratios, k=1)[0]
            
            # Deadline 설정
            if slo_type == "hard":
                base_deadline = random.uniform(0.05, 0.15)
            elif slo_type == "normal":
                base_deadline = random.uniform(0.1, 0.3)
            else:  # soft
                base_deadline = random.uniform(0.3, 0.6)
            
            # Task 객체 생성
            task = Task(
                task_id=task_id,
                model_name=model_name,
                device_id=device_id,
                slo_type=slo_type,
                base_deadline=base_deadline
            )
            
            # Task + 해당 시점의 디바이스 스냅샷 저장
            all_tasks.append((current_time, task, device_snapshot))
            task_count += 1
        
        device_task_counts[device_id] = task_count
        
        # 진행 상황 출력 (20개마다)
        if idx % 20 == 0:
            print(f"[PRE-GEN] Progress: {idx:>3}/100 devices processed")
    
    # 시간 순 정렬
    all_tasks.sort(key=lambda x: x[0])
    
    print(f"\n[PRE-GEN] ✓ Completed: {len(all_tasks)} tasks generated")
    
    # 통계 출력
    print(f"\n[PRE-GEN] Task Statistics:")
    
    # SLO 분포
    slo_dist = {}
    for _, task, _ in all_tasks:
        slo_dist[task.slo_type] = slo_dist.get(task.slo_type, 0) + 1
    
    print(f"  SLO distribution:")
    for slo_type in ['soft', 'normal', 'hard']:
        count = slo_dist.get(slo_type, 0)
        percentage = count / len(all_tasks) * 100 if all_tasks else 0
        print(f"    {slo_type:8s}: {count:>5} tasks ({percentage:>5.1f}%)")
    
    # 모델 분포
    model_dist = {}
    for _, task, _ in all_tasks:
        model_dist[task.model_name] = model_dist.get(task.model_name, 0) + 1
    
    print(f"  Model distribution:")
    for model_name in sorted(model_dist.keys()):
        count = model_dist[model_name]
        percentage = count / len(all_tasks) * 100 if all_tasks else 0
        print(f"    {model_name:12s}: {count:>5} tasks ({percentage:>5.1f}%)")
    
    # 디바이스 타입별 통계
    device_type_tasks = {}
    for _, task, snapshot in all_tasks:
        device_type = snapshot['device_type']
        device_type_tasks[device_type] = device_type_tasks.get(device_type, 0) + 1
    
    print(f"  Device type distribution:")
    for device_type in ['fixed', 'pedestrian', 'vehicle', 'drone']:
        count = device_type_tasks.get(device_type, 0)
        percentage = count / len(all_tasks) * 100 if all_tasks else 0
        print(f"    {device_type:12s}: {count:>5} tasks ({percentage:>5.1f}%)")
    
    # 시간 분포
    time_bins = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
    time_dist = {f"{time_bins[i]}-{time_bins[i+1]}s": 0 for i in range(len(time_bins)-1)}
    
    for arrival_time, _, _ in all_tasks:
        for i in range(len(time_bins)-1):
            if time_bins[i] <= arrival_time < time_bins[i+1]:
                key = f"{time_bins[i]}-{time_bins[i+1]}s"
                time_dist[key] += 1
                break
    
    print(f"  Time distribution:")
    for time_range, count in time_dist.items():
        print(f"    {time_range:12s}: {count:>5} tasks")
    
    return all_tasks


def save_pre_generated_data(mobility_snapshots, all_tasks, experiment_id, results_dir='./results'):
    """
    사전 생성된 데이터를 파일로 저장
    
    Args:
        mobility_snapshots: 모빌리티 스냅샷
        all_tasks: 생성된 task 리스트
        experiment_id: 실험 ID
        results_dir: 저장 디렉토리
    """
    import os
    
    os.makedirs(results_dir, exist_ok=True)
    
    filename = f'{results_dir}/pre_generated_data_{experiment_id}.pkl'
    
    data = {
        'mobility_snapshots': mobility_snapshots,
        'all_tasks': all_tasks,
        'metadata': {
            'experiment_id': experiment_id,
            'generation_time': time.time(),
            'num_snapshots': len(mobility_snapshots),
            'num_tasks': len(all_tasks),
            'duration': max(mobility_snapshots.keys()) if mobility_snapshots else 0
        }
    }
    
    print(f"\n{'='*80}")
    print(f"[SAVE] Saving pre-generated data...")
    print(f"{'='*80}")
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    
    print(f"[SAVE] ✓ Saved to: {filename}")
    print(f"[SAVE] File size: {file_size_mb:.2f} MB")
    print(f"[SAVE] Snapshots: {len(mobility_snapshots)}")
    print(f"[SAVE] Tasks: {len(all_tasks)}")


def load_pre_generated_data(experiment_id, results_dir='./results'):
    """
    저장된 사전 생성 데이터 로드
    
    Args:
        experiment_id: 실험 ID
        results_dir: 저장 디렉토리
    
    Returns:
        mobility_snapshots, all_tasks
    """
    filename = f'{results_dir}/pre_generated_data_{experiment_id}.pkl'
    
    print(f"\n{'='*80}")
    print(f"[LOAD] Loading pre-generated data...")
    print(f"{'='*80}")
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    mobility_snapshots = data['mobility_snapshots']
    all_tasks = data['all_tasks']
    metadata = data.get('metadata', {})
    
    print(f"[LOAD] ✓ Loaded from: {filename}")
    print(f"[LOAD] Snapshots: {len(mobility_snapshots)}")
    print(f"[LOAD] Tasks: {len(all_tasks)}")
    print(f"[LOAD] Original experiment ID: {metadata.get('experiment_id', 'unknown')}")
    
    return mobility_snapshots, all_tasks


def replay_mobility(mobility_manager, mobility_snapshots, duration=600, interval=2.0):
    """
    사전 실행한 모빌리티를 실시간으로 재생
    
    핵심:
    - interval(2초)마다 스냅샷에서 디바이스 상태 복원
    - deadline_adapter, scheduler가 이 상태를 참조
    
    Args:
        mobility_manager: 실제 MobilityManager 인스턴스
        mobility_snapshots: 사전 실행한 스냅샷
        duration: 재생 시간 (초)
        interval: 업데이트 간격 (초)
    """
    start_time = time.time()
    
    print(f"\n[REPLAY] Starting mobility replay...")
    print(f"[REPLAY] Duration: {duration}s")
    print(f"[REPLAY] Interval: {interval}s")
    
    update_count = 0
    
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        
        # 현재 시점의 스냅샷 찾기
        snapshot_time = round(elapsed / interval) * interval
        
        if snapshot_time in mobility_snapshots:
            # 모든 디바이스 상태 복원
            snapshot = mobility_snapshots[snapshot_time]
            
            for device_id, state in snapshot.items():
                if device_id in mobility_manager.devices:
                    device = mobility_manager.devices[device_id]
                    device.current_position = state['position']
                    device.speed = state['speed']
                    device.direction = state['direction']
            
            update_count += 1
            
            # 진행 상황 (60초마다)
            if int(elapsed) % 60 == 0 and update_count > 1:
                in_coverage = sum(1 for s in snapshot.values() if s['in_coverage'])
                print(f"[REPLAY] Progress: {elapsed:>6.1f}s / {duration}s "
                      f"({update_count:>3} updates, {in_coverage:>3}/100 in coverage)")
        
        # interval만큼 대기
        time.sleep(interval)
    
    print(f"[REPLAY] ✓ Mobility replay completed ({update_count} updates)")


def inject_pre_generated_tasks(all_tasks, incoming_tasks_queue):
    """
    사전 생성된 task를 정확한 시간에 주입
    
    핵심:
    - 도착 시간까지 정확히 대기
    - incoming_tasks_queue에 넣기
    
    Args:
        all_tasks: [(arrival_time, task, device_snapshot), ...]
        incoming_tasks_queue: Queue
    """
    start_time = time.time()
    injected_count = 0
    
    print(f"\n[INJECTION] Starting task injection...")
    print(f"[INJECTION] Total tasks: {len(all_tasks)}")
    
    for arrival_time, task, device_snapshot in all_tasks:
        # 정확한 시간까지 대기
        target_time = start_time + arrival_time
        
        while time.time() < target_time:
            time.sleep(0.0001)  # 100 microseconds
        
        # Task를 queue에 넣기
        actual_timestamp = time.time()
        incoming_tasks_queue.put((actual_timestamp, task))
        
        injected_count += 1
        
        # 진행 상황 (1000개마다)
        if injected_count % 1000 == 0:
            elapsed = time.time() - start_time
            remaining = len(all_tasks) - injected_count
            print(f"[INJECTION] Progress: {injected_count:>5}/{len(all_tasks)} "
                  f"({elapsed:>6.1f}s elapsed, {remaining:>5} remaining)")
    
    print(f"[INJECTION] ✓ Task injection completed ({injected_count} tasks)")


def inject_pre_generated_tasks_throttled(all_tasks, incoming_tasks_queue, target_rate=20):
    """
    사전 생성된 task를 정확한 시간에 주입 (속도 제어!)
    
    핵심: Processing rate에 맞춰서 injection rate 제한
    """
    import time
    
    start_time = time.time()
    injected_count = 0
    
    print(f"\n[INJECTION] Starting task injection with throttling...")
    print(f"[INJECTION] Total tasks: {len(all_tasks)}")
    print(f"[INJECTION] Target rate: {target_rate} tasks/sec")
    
    for arrival_time, task, device_snapshot in all_tasks:
        # 목표 도착 시간
        target_time = start_time + arrival_time
        
        # 처리율 제한 (핵심!)
        if injected_count > 0:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # 주입되어야 할 task 수
            expected_injected = elapsed * target_rate
            
            # 너무 빠르면 대기
            if injected_count > expected_injected:
                wait_time = (injected_count - expected_injected) / target_rate
                time.sleep(wait_time)
        
        # 도착 시간까지 대기
        while time.time() < target_time:
            time.sleep(0.001)
        
        # Task를 queue에 넣기
        incoming_tasks_queue.put((time.time(), task))
        injected_count += 1
        
        if injected_count % 1000 == 0:
            elapsed = time.time() - start_time
            current_rate = injected_count / elapsed if elapsed > 0 else 0
            print(f"[INJECTION] {injected_count:>5}/{len(all_tasks)} "
                  f"({current_rate:>5.1f} tasks/sec)")
    
    print(f"[INJECTION] ✓ Completed ({injected_count} tasks)")



def inject_pre_generated_tasks_by_arrival_time(all_tasks, incoming_tasks_queue):
    """arrival_time에 맞춰 정확히 주입 → Poisson 유지!"""
    
    start_time = time.time()
    
    for arrival_time, task, device_snapshot in all_tasks:
        elapsed = time.time() - start_time
        wait_time = arrival_time - elapsed  # ← 핵심!
        
        if wait_time > 0:
            time.sleep(wait_time)  # Poisson 간격 유지
        
        incoming_tasks_queue.put((time.time(), task))


def pre_populate_incoming_queue(all_tasks, incoming_tasks_queue):
    """
    모든 task를 arrival_time과 함께 Queue에 미리 넣기 (NEW!)

    text
    핵심:
    - 실험 시작 전에 모든 task를 Queue에 미리 넣음
    - arrival_time을 함께 저장
    - Main loop에서 arrival_time 보고 대기

    Args:
        all_tasks: [(arrival_time, task, device_snapshot), ...]
        incoming_tasks_queue: Queue
    """

    print(f"\n[PRE-POPULATE] Populating incoming queue...")
    print(f"[PRE-POPULATE] Total tasks: {len(all_tasks)}")

    for arrival_time, task, device_snapshot in all_tasks:
        # arrival_time을 함께 저장 (핵심!)
        incoming_tasks_queue.put((arrival_time, task))

    memory_mb = len(all_tasks) * 500 / 1024 / 1024
    print(f"[PRE-POPULATE] ✓ Queue populated with {len(all_tasks)} tasks")
    print(f"[PRE-POPULATE] ✓ Estimated memory: ~{memory_mb:.1f} MB")