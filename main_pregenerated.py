"""
수정된 Main Experiment - 사전 생성 방식
"""

import time
import threading
import json
import random
import csv
from datetime import datetime
import queue
import os
# 스케줄러 import
from scheduler.mobility_aware_power_topsis_scheduler import MobilityAwarePowerTOPSISScheduler
from scheduler.base_scheduler import Node
from scheduler.power_aware_topsis_scheduler import PowerAwareTOPSISScheduler
from scheduler.power_only_scheduler import PowerOnlyScheduler
from scheduler.performance_only_scheduler import PerformanceOnlyScheduler
from scheduler.round_robin_scheduler import RoundRobinScheduler
from scheduler.shortest_queue_scheduler import ShortestQueueScheduler
from scheduler.random_scheduler import RandomScheduler

# 기본 모듈 import
from mobility.mobility_manager import MobilityManager
from task_generator import TaskGenerator
from deadline_adapter import DeadlineAdapter
from prometheus_collector import PrometheusCollector
from inference_request import start_worker_threads, stop_worker_threads, task_queues, WORKERS, format_timestamp, completed_tasks_queue
from slo_monitor import SLOMonitor
from performance_profile import PerformanceProfiler
from utils import power_monitor_loop, save_experiment_results

# 사전 생성 모듈 import
from pre_generate import (
    pre_simulate_mobility,
    pre_generate_tasks,
    save_pre_generated_data,
    load_pre_generated_data,
    replay_mobility,
    inject_pre_generated_tasks_by_arrival_time,
    inject_pre_generated_tasks_throttled,
    pre_populate_incoming_queue
)

# Incoming queue
incoming_tasks_queue = queue.Queue()


def main():
    """메인 실험 실행 함수 - 사전 생성 방식"""
    random.seed(42)
    
    print("\n" + "="*80)
    print("Power-aware Mobility-based SLO Scheduling Experiment")
    print("Using PRE-GENERATION Model")
    print("="*80)
    
    # 실험 ID 생성
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nExperiment ID: {experiment_id}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. 초기화
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print(f"\n{'='*80}")
    print("STEP 1: Initialization")
    print("="*80)
    
    prometheus_collector = PrometheusCollector("http://localhost:30090")
    slo_monitor = SLOMonitor(experiment_id)
    performance_profiler = PerformanceProfiler()
    
    mobility_manager = MobilityManager(area_size=(500, 500))
    task_generator = TaskGenerator(mobility_manager)
    deadline_adapter = DeadlineAdapter(mobility_manager)
    
    # 동적 전력 임계값 계산
    power_threshold = prometheus_collector.calculate_dynamic_power_threshold(list(WORKERS.keys()))
    print(f"Dynamic power threshold: {power_threshold:.1f}W")
    
    # 스케줄러 초기화
    #scheduler=PowerOnlyScheduler(prometheus_collector=prometheus_collector)
    #scheduler=PerformanceOnlyScheduler(performance_profiler=performance_profiler)
    #scheduler = ShortestQueueScheduler(task_queues=task_queues)
    #scheduler=RoundRobinScheduler()
    #scheduler=RandomScheduler()
    #scheduler = PowerAwareTOPSISScheduler(prometheus_collector=prometheus_collector,task_queues=task_queues, power_threshold=power_threshold,weights=[0.25, 0.25, 0.25, 0.25]) #[0.6, 0.3, 0.05, 0.05]
    
    
    


    scheduler = MobilityAwarePowerTOPSISScheduler(
        prometheus_collector=prometheus_collector,
        task_queues=task_queues,
       mobility_manager=mobility_manager,
       weights=[0.30, 0.40, 0.20, 0.10], # [Deadline Safety, Recent Load, Processing, Energy]
       #weights=[0.25, 0.20, 0.45, 0.10],
       #weights=[0.25, 0.50, 0.15, 0.10],
        power_threshold=power_threshold
    )
    print("Scheduler: MobilityAwarePowerTOPSISScheduler")
    
    # 워커 노드 객체 생성
    nodes = []
    for name in WORKERS.keys():
        node = Node(name, WORKERS[name], prometheus_collector)
        nodes.append(node)
        print(f"Node registered: {name} -> {WORKERS[name]}")
    
    print(f"Total {len(nodes)} nodes registered")
    
    # 가상 IoT 디바이스 생성
    print(f"\nGenerating virtual IoT devices...")
    task_generator.generate_devices(125, coverage_range=150)
    
    # Lambda rate 통계
    lambda_summary = {}
    for device_id, device_info in mobility_manager.devices.items():
        device_type = device_info.device_type
        lambda_summary[device_type] = lambda_summary.get(device_type, 0) + 1
    
    print(f"\nDevice distribution:")
    total_lambda = 0
    for dtype, count in lambda_summary.items():
        lambda_rate = task_generator.device_lambdas[dtype]
        type_total = count * lambda_rate
        total_lambda += type_total
        print(f"  {dtype:12s}: {count:2d} devices × λ={lambda_rate:.2f} = {type_total:>6.2f} task/sec")
    print(f"  {'Total':12s}: {sum(lambda_summary.values()):2d} devices × λ=avg    = {total_lambda:>6.2f} task/sec")
    
    # 초기 전력 상태 확인
    initial_power = prometheus_collector.get_active_cluster_power(list(WORKERS.keys()))
    print(f"\nInitial cluster power: {initial_power:.1f}W")
    print(f"Power threshold: {power_threshold:.1f}W")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. 사전 생성 (실험 전)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    simulation_duration = 600
    
    # 모빌리티 사전 시뮬레이션
    mobility_snapshots = pre_simulate_mobility(
        mobility_manager, 
        duration=simulation_duration,
        interval=2.0,
        coverage_range=150
    )
    
    # Task 사전 생성
    all_tasks = pre_generate_tasks(
        mobility_snapshots,
        task_generator.slo_types,
        task_generator.slo_ratios,
        duration=simulation_duration,
        interval=2.0
    )
    
    # 저장 (재현 가능)
    save_pre_generated_data(mobility_snapshots, all_tasks, experiment_id)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. 실험 시작
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print(f"\n{'='*80}")
    print("STEP 2: Starting Experiment")
    print("="*80)
    
    start_time = time.time()
    print(f"Experiment started at: {datetime.fromtimestamp(start_time)}")
    
    # Worker threads 시작
    print(f"\nStarting worker threads...")
    worker_threads = start_worker_threads(slo_monitor, performance_profiler, mobility_manager)
    
    # 평가지표 수집용 변수
    power_samples = []
    
    # 모빌리티 재생 스레드 시작
    print(f"Starting mobility replay thread...")
    mobility_thread = threading.Thread(
        target=replay_mobility,
        args=(mobility_manager, mobility_snapshots, simulation_duration, 2.0),
        daemon=True
    )
    mobility_thread.start()
    
#    # Task injection 스레드 시작
#    print(f"Starting task injection thread...")
#    injection_thread = threading.Thread(
#        target=inject_pre_generated_tasks_throttled,
#        args=(all_tasks, incoming_tasks_queue,20),
#        daemon=True
#    )
#    injection_thread.start()
    
    pre_populate_incoming_queue(all_tasks, incoming_tasks_queue)

    # Power monitoring 스레드 시작
    print(f"Starting power monitoring thread...")
    power_monitor_thread = threading.Thread(
        target=power_monitor_loop,
        args=(prometheus_collector, list(WORKERS.keys()), power_samples),
        daemon=True
    )
    power_monitor_thread.start()
    
    # 변수 초기화
    total_tasks_processed = 0
    power_exceeded_count = 0
    scheduling_decisions = []
    
    print(f"\n{'='*80}")
    print("EXPERIMENT RUNNING - Pre-generated Task Injection Model")
    print("="*80)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. Main loop (기존과 거의 동일!)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    target_end_time = start_time + simulation_duration
    
    print(f"Target simulation time: {simulation_duration} seconds")
    print(f"Expected end time: {datetime.fromtimestamp(target_end_time)}\n")
    
    while time.time() < target_end_time:
        try:
            # Incoming queue에서 task 꺼내기
           # request_timestamp, task = incoming_tasks_queue.get(timeout=1)
            
            arrival_time, task = incoming_tasks_queue.get(timeout=1)

            current_time = time.time() - start_time
            wait_time = arrival_time - current_time
            if wait_time > 0:
                time.sleep(wait_time)

            request_timestamp = time.time()
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 중요! deadline_adapter는 재생 중인 모빌리티 참조
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            task = deadline_adapter.adapt_deadline(task)
            
            # 현재 클러스터 전력 상태 확인
            current_cluster_power = prometheus_collector.get_active_cluster_power(list(WORKERS.keys()))
            
            # 큐 길이 기반 필터링
            queue_sizes = {name: task_queues[name].qsize() for name in WORKERS.keys()}
            max_queue_size = max(queue_sizes.values()) if queue_sizes else 0
            threshold = max_queue_size * 0.7
            
            eligible_nodes = [node for node in nodes if queue_sizes[node.name] <= threshold]
            if not eligible_nodes:
                eligible_nodes = nodes
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # mobility_aware_schedule인 경우 주석 변경
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
##          # baseline  스케줄링
            #selected_node = scheduler.schedule(task, eligible_nodes)

##          # mobility_aware_schedule
            mobility_info = {'device_id': task.device_id}
            selected_node = scheduler.schedule(task, eligible_nodes, mobility_info=mobility_info)


            
            task.assigned_node = selected_node.name
# m-meslo일때만 아래 주석 해제               
            scheduler.update_processed_count(selected_node.name)

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
            
            # Worker queue에 넣기
            task_queues[selected_node.name].put((task, request_timestamp))
            
            total_tasks_processed += 1
            
            # 진행 상황 로그 출력
            if total_tasks_processed % 100 == 0:
                incoming_queue_size = incoming_tasks_queue.qsize()
                elapsed = time.time() - start_time
                print(f"[Progress] Processed: {total_tasks_processed:>5}, "
                      f"Incoming: {incoming_queue_size:>4}, "
                      f"Elapsed: {elapsed:>6.1f}s, "
                      f"Power: {current_cluster_power:>5.1f}W")
        
        except queue.Empty:
            # Timeout: incoming queue가 비어있음
            pass
    
    print(f"\n{'='*80}")
    print("SIMULATION TIME COMPLETED - Waiting for processing to finish...")
    print("="*80)
    
    # Incoming queue에 남아있는 task 처리
    print(f"\nProcessing remaining tasks in incoming queue...")
    remaining_count = 0
    
    while not incoming_tasks_queue.empty():
        try:
            request_timestamp, task = incoming_tasks_queue.get(timeout=0.5)
            task = deadline_adapter.adapt_deadline(task)
            current_cluster_power = prometheus_collector.get_active_cluster_power(list(WORKERS.keys()))
            
            queue_sizes = {name: task_queues[name].qsize() for name in WORKERS.keys()}
            max_queue_size = max(queue_sizes.values()) if queue_sizes else 0
            threshold = max_queue_size * 0.7
            
            eligible_nodes = [node for node in nodes if queue_sizes[node.name] <= threshold]
            if not eligible_nodes:
                eligible_nodes = nodes
            

##          # baseline 스케줄링
            #selected_node = scheduler.schedule(task, eligible_nodes)

##          # mobility_aware_schedule
            mobility_info = {'device_id': task.device_id}
            selected_node = scheduler.schedule(task, eligible_nodes, mobility_info=mobility_info)

            task.assigned_node = selected_node.name
#m-meslo일때만 아래 주석 해제           
            scheduler.update_processed_count(selected_node.name)

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
            
            task_queues[selected_node.name].put((task, request_timestamp))
            
            remaining_count += 1
            total_tasks_processed += 1
            
        except queue.Empty:
            break
    
    print(f"Processed {remaining_count} remaining tasks from incoming queue")
    
    # 모든 작업 완료 대기
    for queue_name, q in task_queues.items():
        remaining = q.qsize()
        if remaining > 0:
            print(f"Waiting for queue {queue_name} to finish ({remaining} tasks remaining)...")
        q.join()
    
    # 실험 종료
    end_time = time.time()
    total_duration = end_time - start_time
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. 결과 처리 및 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # 워커 스레드 종료
    print(f"\nStopping worker threads...")
    stop_worker_threads(worker_threads)
    
    # 평가지표 계산
    violation_stats = slo_monitor.get_violation_stats()
    violations = violation_stats["total"]
    
    total_response_time = 0.0
    total_waiting_time = 0.0
    completed_count = 0
    
    # completed_tasks_queue에서 결과 읽기
    while not completed_tasks_queue.empty():
        result = completed_tasks_queue.get()
        total_response_time += result["response_time"]
        total_waiting_time += result["waiting_time"]
        completed_count += 1
    
    # 샘플 간격 계산
    if len(power_samples) > 1:
        sample_interval = power_samples[1][0] - power_samples[0][0]
    else:
        sample_interval = 1.0
    
    # 평가 메트릭
    slo_compliance_rate = (completed_count - violations) / completed_count * 100 if completed_count else 0
    throughput = completed_count / total_duration if total_duration > 0 else 0
    avg_response_time = total_response_time / completed_count if completed_count else 0
    avg_waiting_time = total_waiting_time / completed_count if completed_count else 0
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
    
    print(f"\n{'='*80}")
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
    print(f"Pre-generated data: pre_generated_data_{experiment_id}.pkl")
    print(f"Master inference results: master_inference_results.csv")
    print(f"SLO violations: slo_violations_{experiment_id}.csv")
    print(f"Scheduling decisions: scheduling_decisions_{experiment_id}.csv")
    print(f"Device information: devices_{experiment_id}.csv")
    print(f"Experiment summary: experiment_summary_{experiment_id}.json")
    print("="*80)


if __name__ == "__main__":
    main()
