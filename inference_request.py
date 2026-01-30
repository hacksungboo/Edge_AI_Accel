import queue
import threading
import time
import random
import numpy as np
import base64
import requests
import csv
from datetime import datetime
import itertools

WORKERS = {
    "coral1": "http://10.42.7.50:8080/infer",
    "coral2": "http://10.42.3.139:8080/infer",
    "hailo1": "http://10.42.4.56:8080/infer",
    "hailo2": "http://10.42.5.202:8080/infer",
    "jetson1": "http://10.42.2.147:8080/infer",
    "jetson2": "http://10.42.1.189:8080/infer",
    
}

log_file = "./results/master_inference_results.csv"
log_lock = threading.Lock()


# 마이그레이션 로그 파일 및 락
migration_log_file = "./results/migration_log.csv"
migration_log_lock = threading.Lock()

# 전역 변수 - main에서 설정
slo_monitor = None
performance_profiler = None
mobility_manager = None


# 전역 카운터 (고유 순서 보장)
task_counter = itertools.count()



# 완료된 태스크 결과 전달용 큐
completed_tasks_queue = queue.Queue()

def init_log():
    with log_lock:
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "task_id", "device_id","model_name", "slo_type", "base_deadline", "adapted_deadline",
                "assigned_node", "request_time", "inference_start", "inference_end", 
                "waiting_time", "pure_inference_time", "total_response_time", "slo_violated",
                "device_type", "device_speed", "device_direction", "device_x", "device_y"  # 추가
            ])

def format_timestamp(ts: float) -> str:
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def log_result(row):
    with log_lock:
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


def init_migration_log():
    with migration_log_lock, open(migration_log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "from_node",
            "to_node",
            "task_id",
            "model_name",
            "slo_type",
            "base_deadline",
            "adapted_deadline"
        ])

def log_migration(from_node, to_node, task):
    with migration_log_lock, open(migration_log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            format_timestamp(time.time()),
            from_node,
            to_node,
            task.task_id,
            task.model_name,           # 모델 이름 추가
            task.slo_type,
            task.base_deadline,
            task.adapted_deadline
        ])




# 큐, 우선순위 큐 -> main 코드에서도 변경 필요, 해당 코드 하단에서도 설정 필요 (worker_Thread)
task_queues = {name: queue.Queue() for name in WORKERS.keys()}

# 우선순위 큐로 변경 (낮은 숫자 = 높은 우선순위)
#task_queues = {name: queue.PriorityQueue() for name in WORKERS.keys()}


def put_task_to_queue(queue_name, task, request_timestamp):
    """우선순위와 함께 태스크를 큐에 넣기"""
    priority = get_task_priority(task)
    count = next(task_counter)
    task_queues[queue_name].put((priority, count, (task, request_timestamp)))

def get_task_priority(task):
    """ 우선순위 계산"""
    base_priority = {"hard": 1, "normal": 2, "soft": 3}[task.slo_type]
    
    # 데드라인 긴박 -> 우선순위 높음
    if task.adapted_deadline <= 0.15:
        urgency_bonus = 0  # 최고 우선순위
    elif task.adapted_deadline <= 0.3:
        urgency_bonus = 1
    else:
        urgency_bonus = 2
        
    return base_priority + urgency_bonus

def generate_random_image():
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return base64.b64encode(img.tobytes()).decode("utf-8")

def worker_thread(worker_name, worker_url):
    while True:
        task_info = task_queues[worker_name].get()
        if task_info is None:  # 종료 신호
            task_queues[worker_name].task_done()
            break
        
        # task_info = (task, request_timestamp)
        task, request_timestamp = task_info
        

        # mobility info
        device_info = mobility_manager.devices.get(task.device_id)
        if device_info:
            device_type = device_info.device_type
            device_speed = device_info.speed
            device_direction = device_info.direction
            device_x, device_y = device_info.current_position
        else:
            device_type = device_speed = device_direction = device_x = device_y = None

        # 대기 시작 시간
        queue_start_time = time.time()
        waiting_time = queue_start_time - request_timestamp
        
        img_b64 = generate_random_image()
        req = {"model": task.model_name, "input": img_b64}
        
        try:
            # 추론 시작
            inference_start = time.time()
            resp = requests.post(worker_url, json=req, timeout=15)
            inference_end = time.time()
            
            resp_json = resp.json()
            success = (resp.status_code == 200 and resp_json.get("result") == "done")
            
            # 서버에서 반환한 순수 추론 시간
            pure_inference_time = resp_json.get("inference_time", inference_end - inference_start)
            total_response_time = inference_end - request_timestamp
            
            # SLO 위반 체크
            slo_violated = slo_monitor.check_and_record_violation(task, total_response_time)
            
            log_row = [
                task.task_id,
                task.device_id,
                task.model_name,
                task.slo_type,
                task.base_deadline,
                task.adapted_deadline,
                task.assigned_node,
                format_timestamp(request_timestamp),
                format_timestamp(inference_start),
                format_timestamp(inference_end),
                f"{waiting_time:.4f}",
                f"{pure_inference_time:.4f}",
                f"{total_response_time:.4f}",
                int(slo_violated),
                device_type,                    # 디바이스 타입
                f"{device_speed:.2f}",         # 속도
                f"{device_direction:.4f}",     # 방향(라디안)
                f"{device_x:.2f}",            # X 좌표
                f"{device_y:.2f}"             # Y 좌표
            ]
            log_result(log_row)
            
            # 완료 결과를 main으로 전달
            completed_tasks_queue.put({
                "task_id" : task.task_id,
                "response_time": total_response_time, # task 도착 -> 큐 대기 -> 추론 -> 추론 결과 수신
                "pure_inference_time": pure_inference_time, # 추론 노드에서 추론 시간
                "slo_violated": slo_violated,
                "waiting_time":waiting_time # 큐 대기 시간
            })

        except Exception as e:
            failure_time = time.time()
            total_response_time = failure_time - request_timestamp
            
            log_row = [
                task.task_id,
                task.device_id, 
                task.slo_type,
                task.base_deadline,
                task.adapted_deadline,
                task.assigned_node,
                format_timestamp(request_timestamp),
                None, None,
                f"{waiting_time:.4f}",
                None,
                f"{total_response_time:.4f}",
                1  # 실패는 위반으로 간주
            ]
            print(f"[ERROR] Request failed on {worker_name}: {e}")
            log_result(log_row)
            
        task_queues[worker_name].task_done()

# 우선순위 큐
def worker_thread2(worker_name, worker_url):
    while True:
        item = task_queues[worker_name].get()
        if item is None:  # 종료 신호
            task_queues[worker_name].task_done()
            break
        
        # 우선순위 큐에서 받은 아이템: (priority, count, (task, request_timestamp))
        try:
            priority, count, (task, request_timestamp) = item
        except (TypeError, ValueError) as e:
            print(f"[ERROR] Invalid task format in queue: {item}, error: {e}")
            task_queues[worker_name].task_done()
            continue
        
        # mobility info
        device_info = mobility_manager.devices.get(task.device_id)
        if device_info:
            device_type = device_info.device_type
            device_speed = device_info.speed
            device_direction = device_info.direction
            device_x, device_y = device_info.current_position
        else:
            device_type = device_speed = device_direction = device_x = device_y = None


        # ===== 타이밍 정의 =====
        # request_timestamp: main에서 put_task_to_queue() 호출 시점
        
        # 대기 시작 시간
        queue_start_time = time.time()
        waiting_time = queue_start_time - request_timestamp
        
        img_b64 = generate_random_image()
        req = {"model": task.model_name, "input": img_b64}
        
        try:
            # 추론 시작
            inference_start = time.time()
            resp = requests.post(worker_url, json=req, timeout=15)
            inference_end = time.time()
            
            resp_json = resp.json()
            success = (resp.status_code == 200 and resp_json.get("result") == "done")
            
            # 서버에서 반환한 순수 추론 시간
            pure_inference_time = resp_json.get("inference_time", inference_end - inference_start)
            total_response_time = inference_end - request_timestamp
            
            # SLO 위반 체크
            slo_violated = slo_monitor.check_and_record_violation(task, total_response_time)
            
            log_row = [
                task.task_id,
                task.device_id,
                task.model_name,
                task.slo_type,
                task.base_deadline,
                task.adapted_deadline,
                task.assigned_node,
                format_timestamp(request_timestamp),
                format_timestamp(inference_start),
                format_timestamp(inference_end),
                f"{waiting_time:.4f}",
                f"{pure_inference_time:.4f}",
                f"{total_response_time:.4f}",
                int(slo_violated),
                device_type,                    # 디바이스 타입
                f"{device_speed:.2f}" if device_speed is not None else "None",         # 속도
                f"{device_direction:.4f}" if device_direction is not None else "None", # 방향(라디안)
                f"{device_x:.2f}" if device_x is not None else "None",            # X 좌표
                f"{device_y:.2f}" if device_y is not None else "None"             # Y 좌표
            ]
            log_result(log_row)
            
            # 완료 결과를 main으로 전달
            completed_tasks_queue.put({
                "task_id": task.task_id,
                "response_time": total_response_time,
                "pure_inference_time": pure_inference_time,
                "slo_violated": slo_violated,
                "waiting_time": waiting_time
            })

        except Exception as e:
            failure_time = time.time()
            total_response_time = failure_time - request_timestamp
            
            log_row = [
                task.task_id,
                task.device_id, 
                task.model_name,
                task.slo_type,
                task.base_deadline,
                task.adapted_deadline,
                task.assigned_node,
                format_timestamp(request_timestamp),
                None, None,
                f"{waiting_time:.4f}",
                None,
                f"{total_response_time:.4f}",
                1,  # 실패는 위반으로 간주
                device_type,
                f"{device_speed:.2f}" if device_speed is not None else "None",
                f"{device_direction:.4f}" if device_direction is not None else "None",
                f"{device_x:.2f}" if device_x is not None else "None",
                f"{device_y:.2f}" if device_y is not None else "None"
            ]
            print(f"[ERROR] Request failed on {worker_name}: {e}")
            log_result(log_row)
            
            # 실패한 경우에도 completed_tasks_queue에 추가
            completed_tasks_queue.put({
                "task_id": task.task_id,
                "response_time": total_response_time,
                "pure_inference_time": None,
                "slo_violated": True,
                "waiting_time": waiting_time
            })
            
        task_queues[worker_name].task_done()



def start_worker_threads(slo_mon, perf_prof,mob_mgr):
    global slo_monitor, performance_profiler, mobility_manager
    slo_monitor = slo_mon
    performance_profiler = perf_prof
    mobility_manager=mob_mgr
    
    init_log()
    init_migration_log()
    threads = []
    for name, url in WORKERS.items():
        t = threading.Thread(target=worker_thread, args=(name, url), daemon=True) # 설정 필요 (우선순위 큐)
        t.start()
        threads.append(t)
    return threads

def stop_worker_threads(threads):
    for name in WORKERS.keys():
        task_queues[name].put(None)
    for t in threads:
        t.join()
