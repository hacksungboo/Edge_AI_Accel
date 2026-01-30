import queue
import threading
import time
import random
import numpy as np
import base64
import requests
import csv
from datetime import datetime


WORKERS = {
#    "coral1": "http://10.42.7.232:8080/infer",
#    "coral2": "http://10.42.3.112:8080/infer",
    "hailo1": "http://10.42.4.38:8080/infer",
    "hailo2": "http://10.42.5.188:8080/infer",
#    "jetson1": "http://10.42.2.59:8080/infer",
#    "jetson2": "http://10.42.1.149:8080/infer",
#    
     
}

log_file = "master_inference_results.csv"
log_lock = threading.Lock()


# 로그 CSV 헤더 작성 (실패 시간 추가)
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "task_request_time", "worker", "model",
        "inference_start_time", "inference_end_time",
        "inference_time", "success", "error_msg", "failure_time"
    ])


def format_timestamp(ts: float) -> str:
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def log_result(row):
    with log_lock:
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


# 각 워커별 task 큐 생성
task_queues = {name: queue.Queue() for name in WORKERS.keys()}


def generate_random_image():
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return base64.b64encode(img.tobytes()).decode("utf-8")


def worker_thread(worker_name, worker_url):
    while True:
        model_name = task_queues[worker_name].get()
        if model_name is None:  # 종료 신호
            task_queues[worker_name].task_done()
            break
        img_b64 = generate_random_image()
        req = {"model": model_name, "input": img_b64}
        task_request_time = time.time()
        try:
            resp = requests.post(worker_url, json=req, timeout=15)
            resp_json = resp.json()
            success = (resp.status_code == 200 and resp_json.get("result") == "done")
            inference_start_time = resp_json.get("start_time")
            inference_end_time = resp_json.get("end_time")
            inference_time = resp_json.get("inference_time")

            log_row = [
                format_timestamp(task_request_time),
                worker_name,
                model_name,
                inference_start_time,
                inference_end_time,
                inference_time,
                success,
                None,
                None  # 실패 시간 없음
            ]
            print(log_row)
            log_result(log_row)
        except Exception as e:
            failure_time = time.time()
            log_row = [
                format_timestamp(task_request_time),
                worker_name,
                model_name,
                None,
                None,
                None,
                False,
                str(e),
                format_timestamp(failure_time)  # 실패 발생 시간 기록
            ]
            print(f"[ERROR] Request failed on {worker_name}: {e}")
            log_result(log_row)
        task_queues[worker_name].task_done()


def start_worker_threads():
    threads = []
    for name, url in WORKERS.items():
        t = threading.Thread(target=worker_thread, args=(name, url), daemon=True)
        t.start()
        threads.append(t)
    return threads


def stop_worker_threads(threads):
    for name in WORKERS.keys():
        task_queues[name].put(None)  # 종료 신호 넣기
    for t in threads:
        t.join()


def schedule_tasks(model_workload_list):
    """
    모델별 작업 개수를 받아, task 큐에 순차적으로 (라운드로빈 스타일) 분배.
    """
    worker_names = list(WORKERS.keys())
    idx = 0
    tasks = []
    for model_name, count in model_workload_list:
        tasks.extend([model_name] * count)
    random.shuffle(tasks)
    for model_name in tasks:
        target_worker = worker_names[idx % len(worker_names)]
        task_queues[target_worker].put(model_name)
        idx += 1


if __name__ == "__main__":
    threads = start_worker_threads()

    start_time = time.time()

    # 예시1: mobilenet 10000개 task 분배
    #schedule_tasks([("mobilenet", 10000)])
    # 예시2: resnet50 10000개 task 분배
    schedule_tasks([("resnet50", 5000)])
    # 예시3: mobilenet 5000 + resnet50 5000 혼합 task 분배
    #schedule_tasks([("efficientnet", 5000), ("inception", 5000)])

    # 모든 작업 완료 대기
    for q in task_queues.values():
        q.join()

    end_time = time.time()
    stop_worker_threads(threads)

    print(f"All tasks completed in {end_time - start_time:.2f} seconds.")

