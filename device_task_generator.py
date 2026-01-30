import threading
import queue
import itertools
import random
import time
import csv
import os
from scheduler.base_scheduler import Task

# 글로벌 변수들
#incoming_tasks_queue = queue.PriorityQueue()
incoming_tasks_queue = queue.Queue()  # FIFO
task_counter = itertools.count()
simulation_running = True
task_generation_log = []
task_log_lock = threading.Lock()


# 우선순위 큐 -> 삭제
def get_task_priority(task):
    """
    SLO type에 따른 우선순위 반환
    
    Args:
        task: Task 객체
    
    Returns:
        priority: int (1-5 범위)
    """
    # 기본 우선순위
    base_priority = {"hard": 1, "normal": 2, "soft": 3}.get(task.slo_type, 2)
    
    # Deadline 긴급도에 따른 보너스
    if task.base_deadline <= 0.15:
        urgency_bonus = 0  # 최고 우선순위
    elif task.base_deadline <= 0.3:
        urgency_bonus = 1
    else:
        urgency_bonus = 2
    
    return base_priority + urgency_bonus


def device_task_generator(device_id, mobility_manager, slo_types, slo_ratios):
    """
    각 device에서 독립적으로 task를 생성하는 스레드 함수
    
    Args:
        device_id: device 식별자
        mobility_manager: MobilityManager 인스턴스
        slo_types: ['soft', 'normal', 'hard']
        slo_ratios: [0.5, 0.35, 0.15]
    """
    global incoming_tasks_queue, task_counter, simulation_running, task_generation_log
    
    device = mobility_manager.devices[device_id]
    model_choices = ["mobilenet", "resnet50", "efficientnet", "inception"]
    
    task_count = 0
    generated_count = 0
    transmitted_count = 0
    skipped_count = 0
    
    #print(f"[TaskGen] Started task generator for {device_id} (type={device.device_type}, lambda={device.lambda_rate})")
    
    while simulation_running:
        try:
            # Poisson 분포에 따른 다음 task 생성까지의 시간 간격
            interval = random.expovariate(device.lambda_rate)
            time.sleep(interval)
            
            generated_count += 1
            
            # 현재 device 정보 가져오기
            device_info = mobility_manager.devices[device_id]
            current_position = device_info.current_position
            
            # ← 중요! position 타입 체크
            if not isinstance(current_position, tuple) or len(current_position) != 2:
                print(f"[WARNING] Device {device_id} has invalid position: {current_position}")
                continue
            
            # 커버리지 확인
            in_coverage = mobility_manager.is_in_coverage(device_id, coverage_range=150)
            
            if in_coverage:
                # Task 생성
                task_id = f"{device_id}_{task_count:06d}"
                model_name = random.choice(model_choices)
                
                # ← 중요! slo_type이 문자열이어야 함 (list 아님!)
                slo_type_list = random.choices(slo_types, weights=slo_ratios, k=1)
                slo_type = slo_type_list[0]  # list에서 첫 원소 추출
                
                # Deadline 설정 (SLO type별)
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
                
                timestamp = time.time()
                
                # 우선순위 큐 삭제
                #priority = get_task_priority(task)
                #sequence = next(task_counter)  # int
                #incoming_tasks_queue.put((priority, sequence, timestamp, task))

                incoming_tasks_queue.put((timestamp, task))


                transmitted_count += 1
                task_count += 1
                status = "transmitted"
            else:
                skipped_count += 1
                status = "skipped"
                task_id = ""
            
            # ← 중요! 타입 안전 로깅
            with task_log_lock:
                # 위치 값 정규화 (float로 변환)
                try:
                    pos_x = float(current_position[0])
                    pos_y = float(current_position[1])
                except (TypeError, IndexError, ValueError):
                    pos_x = 0.0
                    pos_y = 0.0
                
                # 속도 값 정규화
                try:
                    speed = float(device_info.speed)
                except (TypeError, AttributeError, ValueError):
                    speed = 0.0
                
                # ← 중요! 방향 값 정규화 (tuple 처리!)
                try:
                    if isinstance(device_info.direction, (int, float)):
                        direction = float(device_info.direction)
                    else:
                        # tuple, list, None 등은 0.0으로 기본값
                        direction = 0.0
                except (TypeError, AttributeError, ValueError):
                    direction = 0.0
                
                # 로그 엔트리 생성
                log_entry = {
                    'timestamp': time.time(),
                    'device_id': device_id,
                    'device_type': device_info.device_type,
                    'position_x': round(pos_x, 2),
                    'position_y': round(pos_y, 2),
                    'speed': round(speed, 2),
                    'direction': round(direction, 3),
                    'in_coverage': in_coverage,
                    'status': status,
                    'task_id': task_id,
                    'generated_count': generated_count,
                    'transmitted_count': transmitted_count,
                    'skipped_count': skipped_count
                }
                
                task_generation_log.append(log_entry)
                
        except Exception as e:
            print(f"[ERROR] Device {device_id} task generator error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()  # 에러 상세 출력
            time.sleep(1)


def start_device_task_generators(mobility_manager, slo_types, slo_ratios):
    """
    모든 device의 task 생성 스레드를 시작
    
    Args:
        mobility_manager: MobilityManager 인스턴스
        slo_types: SLO 타입 리스트
        slo_ratios: SLO 타입별 비율
    
    Returns:
        threads: 시작된 스레드 리스트
    """
    threads = []
    
    for device_id in mobility_manager.devices.keys():
        thread = threading.Thread(
            target=device_task_generator,
            args=(device_id, mobility_manager, slo_types, slo_ratios),
            daemon=True,
            name=f"TaskGen-{device_id}"
        )
        thread.start()
        threads.append(thread)
    
    print(f"[DEVICE TASK GEN] Started {len(threads)} device task generator threads")
    
    return threads


def stop_device_task_generators():
    """
    모든 device task generator 스레드 종료
    """
    global simulation_running
    simulation_running = False
    print("[DEVICE TASK GEN] Stopping all device task generators...")


def save_task_generation_log(experiment_id):
    """
    Task 생성 로그를 CSV 파일로 저장
    
    Args:
        experiment_id: 실험 ID
    """
    global task_generation_log
    
    os.makedirs('results', exist_ok=True)
    
    filename = f'results/task_generation_log_{experiment_id}.csv'
    
    # CSV 작성
    if len(task_generation_log) > 0:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = list(task_generation_log[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(task_generation_log)
        
        print(f"[LOG] Task generation log saved: {filename} ({len(task_generation_log)} entries)")
    else:
        print(f"[LOG] No task generation log to save (log is empty)")
    
    # 통계 저장
    save_task_generation_stats(experiment_id)


def save_task_generation_stats(experiment_id):
    """
    Device별 task 생성 통계 요약 저장
    
    Args:
        experiment_id: 실험 ID
    """
    global task_generation_log
    
    from collections import defaultdict
    
    # Device별 통계 수집
    stats = defaultdict(lambda: {
        'device_type': '',
        'total_generated': 0,
        'total_transmitted': 0,
        'total_skipped': 0
    })
    
    for entry in task_generation_log:
        device_id = entry['device_id']
        stats[device_id]['device_type'] = entry['device_type']
        stats[device_id]['total_generated'] = entry['generated_count']
        stats[device_id]['total_transmitted'] = entry['transmitted_count']
        stats[device_id]['total_skipped'] = entry['skipped_count']
    
    filename = f'results/task_generation_stats_{experiment_id}.csv'
    
    # CSV 작성
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'device_id', 'device_type',
            'total_generated', 'total_transmitted', 'total_skipped',
            'transmission_rate (%)'
        ])
        
        for device_id, stat in sorted(stats.items()):
            total_gen = stat['total_generated']
            total_trans = stat['total_transmitted']
            transmission_rate = (total_trans / total_gen * 100) if total_gen > 0 else 0.0
            
            writer.writerow([
                device_id,
                stat['device_type'],
                total_gen,
                total_trans,
                stat['total_skipped'],
                f"{transmission_rate:.1f}%"
            ])
    
    print(f"[LOG] Task generation stats saved: {filename}")
    
    # 요약 통계 출력
    total_generated = sum(s['total_generated'] for s in stats.values())
    total_transmitted = sum(s['total_transmitted'] for s in stats.values())
    total_skipped = sum(s['total_skipped'] for s in stats.values())
    overall_rate = (total_transmitted / total_generated * 100) if total_generated > 0 else 0.0
    
    print(f"\n[SUMMARY] Task Generation Statistics:")
    print(f"  Total generated: {total_generated}")
    print(f"  Total transmitted: {total_transmitted}")
    print(f"  Total skipped: {total_skipped}")
    print(f"  Overall transmission rate: {overall_rate:.1f}%")


def get_incoming_queue_status():
    """
    incoming_tasks_queue 상태 반환 (디버깅용)
    
    Returns:
        dict: queue_size, is_empty
    """
    return {
        'queue_size': incoming_tasks_queue.qsize(),
        'is_empty': incoming_tasks_queue.empty()
    }


def reset_device_generator():
    """
    Device task generator 완전 초기화
    각 실험 사이에 호출하여 전역 상태를 리셋
    """
    global incoming_tasks_queue, task_counter, simulation_running
    global task_generation_log, task_log_lock
    
    import time
    import queue
    import itertools
    import threading
    
    print("[DEVICE GEN] Starting reset...")
    
    # 1. simulation 플래그 종료
    simulation_running = False
    print("[DEVICE GEN] Set simulation_running = False")
    
    # 2. 모든 device generator 스레드 종료 대기
    time.sleep(2)
    print("[DEVICE GEN] Waited for threads to terminate")
    
    # 3. 새로운 Queue 생성
    incoming_tasks_queue = queue.PriorityQueue()
    print("[DEVICE GEN] New priority queue created")
    
    # 4. Task counter 리셋
    task_counter = itertools.count()
    print("[DEVICE GEN] Task counter reset")
    
    # 5. 로그 초기화
    task_generation_log = []
    print("[DEVICE GEN] Task generation log cleared")
    
    # 6. Lock 재생성
    task_log_lock = threading.Lock()
    print("[DEVICE GEN] Lock recreated")
    
    print("[DEVICE GEN] Reset completed successfully")


def set_simulation_running(flag):
    """
    simulation_running 플래그 제어
    실험 시작/종료 시 호출
    """
    global simulation_running
    simulation_running = flag
    import time
    time.sleep(0.1)  # 플래그 업데이트 반영 시간
    print(f"[DEVICE GEN] simulation_running flag set to: {flag}")


def get_simulation_running():
    """
    현재 simulation_running 상태 조회
    """
    return simulation_running


def get_incoming_tasks_queue_size():
    """
    Incoming queue의 현재 크기 조회
    """
    return incoming_tasks_queue.qsize()