# 과거 개발중 사용하던 코드. (최종 실험 코드 아님)
# 최종 학위논문 실험 코드는 main_pregenerated.py
#
#


import time
import threading
import json
import random
import csv
from datetime import datetime
import queue
from scheduler.power_aware_topsis_scheduler import PowerAwareTOPSISScheduler
from scheduler.power_only_scheduler import PowerOnlyScheduler
from scheduler.performance_only_scheduler import PerformanceOnlyScheduler
from scheduler.round_robin_scheduler import RoundRobinScheduler
from scheduler.shortest_queue_scheduler import ShortestQueueScheduler
from scheduler.random_scheduler import RandomScheduler
from scheduler.mobility_aware_power_topsis_scheduler import MobilityAwarePowerTOPSISScheduler
from scheduler.base_scheduler import Node
from mobility.mobility_manager import MobilityManager
from task_generator import TaskGenerator
from deadline_adapter import DeadlineAdapter
from prometheus_collector import PrometheusCollector
from inference_request import start_worker_threads, stop_worker_threads, task_queues, WORKERS, format_timestamp, completed_tasks_queue, put_task_to_queue
from slo_monitor import SLOMonitor
from performance_profile import PerformanceProfiler
from utils import *

# ← 추가! Device task generator import
from device_task_generator import (
    start_device_task_generators, 
    stop_device_task_generators, 
    incoming_tasks_queue,
    get_incoming_queue_status,
    save_task_generation_log
)

# task 생성 후 디바이스 할당 (기존)
# 각 디바이스에서 task 생성 (변경 완료)


def main():
    random.seed(42)

    """메인 실험 실행 함수 - Distributed Task Generation 방식"""
    print("Starting Power-aware Mobility-based SLO Scheduling Experiment...")
    print("Using Distributed Task Generation Model")
    
    # 실험 ID 생성
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Experiment ID: {experiment_id}")
    
    # 1. 초기화
    prometheus_collector = PrometheusCollector("http://localhost:30090")
    slo_monitor = SLOMonitor(experiment_id)
    performance_profiler = PerformanceProfiler()
    
    mobility_manager = MobilityManager(area_size=(500, 500))  # ← 변경! 공간 축소
    task_generator = TaskGenerator(mobility_manager)
    deadline_adapter = DeadlineAdapter(mobility_manager)

    # 2. 동적 전력 임계값 계산
    power_threshold = prometheus_collector.calculate_dynamic_power_threshold(list(WORKERS.keys()))
    print(f"Dynamic power threshold: {power_threshold:.1f}W")
    
    # 3. 스케줄러 초기화
    #scheduler = PowerAwareTOPSISScheduler(
    #    prometheus_collector=prometheus_collector,
    #    task_queues=task_queues, 
    #    power_threshold=power_threshold,
    #    weights=[0.6, 0.3, 0.05, 0.05]
    #)
    #scheduler=PowerOnlyScheduler(prometheus_collector=prometheus_collector)
    #scheduler=PerformanceOnlyScheduler(performance_profiler=performance_profiler)
    #scheduler=RoundRobinScheduler()
    #scheduler=RandomScheduler()
    #scheduler = ShortestQueueScheduler(task_queues=task_queues)

    scheduler = MobilityAwarePowerTOPSISScheduler(
        prometheus_collector=prometheus_collector,
        task_queues=task_queues,
        mobility_manager=mobility_manager,  # ← 중요! 모빌리티 정보 전달
        weights=[0.20, 0.50, 0.15, 0.15],
        power_threshold=power_threshold
    )








    # 4. 워커 노드 객체 생성
    nodes = []
    for name in WORKERS.keys():
        node = Node(name, WORKERS[name], prometheus_collector)
        nodes.append(node)
        print(f"Node registered: {name} -> {WORKERS[name]}")
    
    print(f"Total {len(nodes)} nodes registered for scheduling")
    
    # 5. 가상 IoT 디바이스 생성 (lambda 자동 할당됨)
    print("Generating virtual IoT devices...")
    task_generator.generate_devices(125, coverage_range=150)  # ← 변경! coverage_range 명시
    
    print(f"Generated 100 devices with lambda rates:")
    lambda_summary = {}
    for device_id, device_info in mobility_manager.devices.items():
        device_type = device_info.device_type
        lambda_summary[device_type] = lambda_summary.get(device_type, 0) + 1
    
    for dtype, count in lambda_summary.items():
        lambda_rate = task_generator.device_lambdas[dtype]
        print(f"  {dtype:12s}: {count:2d} devices × λ={lambda_rate:.2f} = {count*lambda_rate:.2f} task/sec")
    
    
    # 6. 초기 전력 상태 확인
    initial_power = prometheus_collector.get_active_cluster_power(list(WORKERS.keys()))
    print(f"Initial cluster power: {initial_power:.1f}W")
    print(f"Power threshold: {power_threshold:.1f}W")
    
    # 7. 실험 시작
    start_time = time.time()
    print(f"Experiment started at: {datetime.fromtimestamp(start_time)}")
    
    print("Starting worker threads...")
    worker_threads = start_worker_threads(slo_monitor, performance_profiler, mobility_manager)
    
    # 평가지표 수집용 변수
    power_samples = [] 

    print("Starting mobility update thread...")
    mobility_thread = threading.Thread(
        target=mobility_update_loop,
        args=(mobility_manager,), 
        daemon=True
    )
    mobility_thread.start()
    
    print("Starting power monitoring thread...")
    power_monitor_thread = threading.Thread(
        target=power_monitor_loop,
        args=(prometheus_collector, list(WORKERS.keys()), power_samples),
        daemon=True
    )
    power_monitor_thread.start()
    
    # Device task generator 스레드 시작
    print("Starting device task generator threads...")
    device_threads = start_device_task_generators(
        mobility_manager, 
        task_generator.slo_types, 
        task_generator.slo_ratios
    )
    
    # 변수 초기화
    total_tasks_processed = 0
    total_tasks_skipped = 0
    power_exceeded_count = 0
    no_eligible_node_count = 0
    scheduling_decisions = []
    
    print("="*80)
    print("EXPERIMENT STARTED - Distributed Task Generation Model")
    print("="*80)
    
    # 8. 메인 루프 - incoming_tasks_queue에서 task 처리
    # 기존 for문 대신 queue 기반으로 변경
    
    simulation_duration = 600  # 600초 시뮬레이션
    target_end_time = start_time + simulation_duration
    
    print(f"Target simulation time: {simulation_duration} seconds")
    print(f"Expected end time: {datetime.fromtimestamp(target_end_time)}")
    
    while time.time() < target_end_time:
        try:
            # incoming_tasks_queue에서 task 꺼내기 (1초 timeout)
            #priority, sequence, request_timestamp, task = incoming_tasks_queue.get(timeout=1) #우선순위 큐
            request_timestamp, task = incoming_tasks_queue.get(timeout=1)
            



            # IoT 디바이스 mobility 기반 deadline 동적 적응
            task = deadline_adapter.adapt_deadline(task)
            
            # 현재 클러스터 전력 상태 확인
            current_cluster_power = prometheus_collector.get_active_cluster_power(list(WORKERS.keys()))
            
            # 전력 임계값 초과 체크
        #    if current_cluster_power > scheduler.power_threshold:
        #        power_exceeded_count += 1
            


            # 데드라인 준수 가능한 노드 필터링
            #eligible_nodes = [
            #    node for node in nodes
            #    if performance_profiler.can_meet_deadline(
            #        node.name, task.model_name, task.adapted_deadline
            #    )
            #]
            # 적합한 노드가 없으면 모든 노드 중 선택
            #if not eligible_nodes:
            #    eligible_nodes = nodes
            #    no_eligible_node_count += 1
            #    print(f"[WARNING] No node can meet deadline for {task.task_id}")


            #eligible_nodes=nodes # 노드 필터링 x(2025.11.11)
        



            # 큐 길이 기반 필터링
            queue_sizes = {name: task_queues[name].qsize() for name, url in WORKERS.items()}

            max_queue_size = max(queue_sizes.values()) if queue_sizes else 0
            threshold = max_queue_size * 0.7
            
            eligible_nodes = [node for node in nodes if queue_sizes[node.name] <= threshold]
            if not eligible_nodes:
                eligible_nodes = nodes


###            
            # TOPSIS 기반 전력 및 SLO 인식 스케줄링
            #selected_node = scheduler.schedule(task, eligible_nodes)


            # mobility_aware_schedule
            mobility_info = {'device_id': task.device_id}
            selected_node = scheduler.schedule(task, eligible_nodes, mobility_info=mobility_info)


            task.assigned_node = selected_node.name
            
            # 스케줄링 결정 기록
            scheduling_decision = {
                'timestamp': request_timestamp,
                'request_timestamp': format_timestamp(request_timestamp),
                'task_id': task.task_id,
                'device_id': task.device_id,
                'slo_type': task.slo_type,
                'base_deadline': task.base_deadline,
                'adapted_deadline': task.adapted_deadline,
                'selected_node': selected_node.name,
                'node_power': selected_node.power_consumption,
                'cluster_power': current_cluster_power,
            }
            scheduling_decisions.append(scheduling_decision)

        
            # 기본 큐---------------
            task_queues[selected_node.name].put((task, request_timestamp))
            
            # 우선순위 큐
            #put_task_to_queue(selected_node.name, task, request_timestamp) 



            
            # 진행 상황 로그 출력
            total_tasks_processed += 1
            
            if total_tasks_processed % 100 == 0:
                queue_status = get_incoming_queue_status()
                print(f"[Progress] Processed: {total_tasks_processed}, "
                      f"Incoming queue: {queue_status['queue_size']}, "
                      f"Cluster power: {current_cluster_power:.1f}W")
        
        except queue.Empty:
            # Timeout: incoming queue가 비어있음
            # 계속 대기
            pass
    
    print("="*80)
    print("SIMULATION TIME COMPLETED - Waiting for processing to finish...")
    print("="*80)
    
    # ← 추가! Device task generator 스레드 종료
    print("Stopping device task generators...")
    stop_device_task_generators()
    
    # Incoming queue에 남아있는 task 처리
    print("Processing remaining tasks in incoming queue...")
    remaining_count = 0
    while not incoming_tasks_queue.empty():
        try:

            request_timestamp, task = incoming_tasks_queue.get(timeout=0.5) 
            task = deadline_adapter.adapt_deadline(task)
            current_cluster_power = prometheus_collector.get_active_cluster_power(list(WORKERS.keys()))
            
            #eligible_nodes = [
            #    node for node in nodes
            #    if performance_profiler.can_meet_deadline(
            #        node.name, task.model_name, task.adapted_deadline
            #    )
            #]
            
            #if not eligible_nodes:
            #    eligible_nodes = nodes

            
            # 큐 길이 기반 필터링
            queue_sizes = {name: task_queues[name].qsize() for name, url in WORKERS.items()}
            max_queue_size = max(queue_sizes.values()) if queue_sizes else 0
            threshold = max_queue_size * 0.7
            
            eligible_nodes = [node for node in nodes if queue_sizes[node.name] <= threshold]
            if not eligible_nodes:
                eligible_nodes = nodes
            
            if total_tasks_processed % 100 == 0:
                print(f"[FILTER] Q: {queue_sizes} | T: {threshold:.1f} | E: {[n.name for n in eligible_nodes]}")

            selected_node = scheduler.schedule(task, eligible_nodes)
            task.assigned_node = selected_node.name
            
            scheduling_decision = {
                'timestamp': request_timestamp,
                'request_timestamp': format_timestamp(request_timestamp),
                'task_id': task.task_id,
                'device_id': task.device_id,
                'slo_type': task.slo_type,
                'base_deadline': task.base_deadline,
                'adapted_deadline': task.adapted_deadline,
                'selected_node': selected_node.name,
                'node_power': selected_node.power_consumption,
                'cluster_power': current_cluster_power,
            }
            scheduling_decisions.append(scheduling_decision)


            # 기본 큐---------------
            task_queues[selected_node.name].put((task, request_timestamp))
            
            # 우선순위 큐
            #put_task_to_queue(selected_node.name, task, request_timestamp) 


            
            remaining_count += 1
            total_tasks_processed += 1
            
        except queue.Empty:
            break
    
    print(f"Processed {remaining_count} remaining tasks from incoming queue")
    
    # 모든 작업 완료 대기
    for queue_name, q in task_queues.items():
        remaining = q.qsize()
        print(f"Waiting for queue {queue_name} to finish ({remaining} tasks remaining)...")
        q.join()
    
    # 실험 종료 및 정리
    end_time = time.time()
    total_duration = end_time - start_time
    
    save_task_generation_log(experiment_id)


    # 워커 스레드 종료
    print("Stopping worker threads...")
    stop_worker_threads(worker_threads)
    
    # 평가지표 계산
    violation_stats = slo_monitor.get_violation_stats()
    violations = violation_stats["total"]
    
    total_response_time = 0.0
    total_waiting_time  = 0.0
    completed_count     = 0

    # completed_tasks_queue에서 결과 읽기
    while not completed_tasks_queue.empty():
        result = completed_tasks_queue.get()
        total_response_time += result["response_time"]
        total_waiting_time  += result["waiting_time"]
        completed_count    += 1

    if len(power_samples) > 1:
        sample_interval = power_samples[1][0] - power_samples[0][0]
    else:
        sample_interval = 1.0

    # 평가 메트릭
    slo_compliance_rate = (completed_count - violations) / completed_count * 100 if completed_count else 0
    throughput = completed_count / total_duration if total_duration > 0 else 0
    avg_response_time = total_response_time / completed_count if completed_count else 0
    avg_waiting_time  = total_waiting_time  / completed_count if completed_count else 0
    total_energy_Wh = sum(power for _, power in power_samples) * sample_interval / 3600
    avg_power_W = sum(power for _, power in power_samples) / len(power_samples) if power_samples else 0
    power_efficiency = completed_count / total_energy_Wh if total_energy_Wh > 0 else 0
    slo_per_energy = slo_compliance_rate / total_energy_Wh if total_energy_Wh > 0 else 0

    # 실험 결과 저장
    save_experiment_results(
        experiment_id, scheduling_decisions, 
        mobility_manager, slo_monitor, 
        start_time, end_time
    )
    
    # 최종 통계 출력
    final_cluster_power = prometheus_collector.get_active_cluster_power(list(WORKERS.keys()))
    
    print("="*80)
    print("EXPERIMENT COMPLETED - FINAL RESULTS")
    print("="*80)
    print(f"Experiment ID: {experiment_id}")
    print(f"Total tasks processed: {total_tasks_processed}")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Average task arrival rate: {total_tasks_processed/total_duration:.2f} tasks/second")
    print()
    
    print("=== POWER CONSUMPTION ===")
    print(f"Initial cluster power: {initial_power:.1f}W")
    print(f"Final cluster power: {final_cluster_power:.1f}W")
    print(f"Power threshold: {power_threshold:.1f}W")
    print(f"Threshold exceeded: {power_exceeded_count} times")
    print()
    
    print("=== SLO PERFORMANCE ===")
    print(f"Total SLO violations: {violation_stats['total']}")
    if violation_stats['total'] > 0:
        print(f"SLO violation rate: {violation_stats['total']/completed_count*100:.2f}%")
        print(f"Violations by type: {violation_stats['by_type']}")
        print(f"Average violation time: {violation_stats['avg_violation']:.3f}s")
    else:
        print("🎉 No SLO violations detected!")
    print()
    
    print("=== SCHEDULING PERFORMANCE ===")
    print(f"Tasks with no eligible nodes: {no_eligible_node_count}")
    
    # 노드별 작업 분배 통계
    node_task_count = {}
    for decision in scheduling_decisions:
        node = decision['selected_node']
        node_task_count[node] = node_task_count.get(node, 0) + 1
    
    print("Node task distribution:")
    for node, count in sorted(node_task_count.items()):
        percentage = (count / total_tasks_processed) * 100 if total_tasks_processed else 0
        print(f"  {node}: {count} tasks ({percentage:.1f}%)")
    print()
    
    print("=== EVALUATION METRICS ===")
    print(f"SLO Compliance Rate: {slo_compliance_rate:.2f}%")
    print(f"System Throughput: {throughput:.2f} tasks/second")
    print(f"Average Response Time: {avg_response_time:.4f}s")
    print(f"Average Waiting Time: {avg_waiting_time:.4f}s")
    print(f"Completed Count: {completed_count}")
    print(f"Total Energy Consumption: {total_energy_Wh:.3f} Wh")
    print(f"Average Power Consumption: {avg_power_W:.3f} W")
    print(f"Power Efficiency: {power_efficiency:.2f} tasks/Wh")
    print(f"SLO per Energy: {slo_per_energy:.2f} %/Wh")
    print()
    
    print("=== FILES GENERATED ===")
    print(f"Master inference results: master_inference_results.csv")
    print(f"SLO violations: slo_violations_{experiment_id}.csv")
    print(f"Scheduling decisions: scheduling_decisions_{experiment_id}.csv")
    print(f"Device information: devices_{experiment_id}.csv")
    print(f"Experiment summary: experiment_summary_{experiment_id}.json")
    print("="*80)


if __name__ == "__main__":
    main()