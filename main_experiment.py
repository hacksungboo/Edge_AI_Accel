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
from scheduler.power_aware_topsis_scheduler import PowerAwareTOPSISScheduler
from scheduler.base_scheduler import Node
from scheduler.power_only_scheduler import PowerOnlyScheduler
from scheduler.performance_only_scheduler import PerformanceOnlyScheduler
from scheduler.round_robin_scheduler import RoundRobinScheduler
from scheduler.shortest_queue_scheduler import ShortestQueueScheduler
from scheduler.random_scheduler import RandomScheduler
from mobility.mobility_manager import MobilityManager
from task_generator import TaskGenerator
from deadline_adapter import DeadlineAdapter
from prometheus_collector import PrometheusCollector
from inference_request import start_worker_threads, stop_worker_threads, task_queues, WORKERS, format_timestamp, completed_tasks_queue, get_task_priority, put_task_to_queue
from slo_monitor import SLOMonitor
from performance_profile import PerformanceProfiler
from utils import *




def main():
    """메인 실험 실행 함수"""
    print("Starting Power-aware Mobility-based SLO Scheduling Experiment...")
    
    # 실험 ID 생성
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Experiment ID: {experiment_id}")
    
    # 1. 초기화
    prometheus_collector = PrometheusCollector("http://localhost:30090")
    slo_monitor = SLOMonitor(experiment_id)
    performance_profiler = PerformanceProfiler()
    
    mobility_manager = MobilityManager()
    task_generator = TaskGenerator(mobility_manager)
    deadline_adapter = DeadlineAdapter(mobility_manager)
 


    # 2. 동적 전력 임계값 계산
    power_threshold = prometheus_collector.calculate_dynamic_power_threshold(list(WORKERS.keys()))
    print(f"Dynamic power threshold: {power_threshold:.1f}W")
    
    # 3. 전력 및 SLO 인식 스케줄러 초기화 (task_queues 전달)
    scheduler = PowerAwareTOPSISScheduler(
        prometheus_collector=prometheus_collector,
        task_queues=task_queues, 
        power_threshold=power_threshold,
        #weights=[0.2, 0.3, 0.1, 0.4]
        weights=[0.6, 0.3, 0.05, 0.05]
    )

    #scheduler=PowerOnlyScheduler(prometheus_collector=prometheus_collector)
    #scheduler=PerformanceOnlyScheduler(performance_profiler=performance_profiler)
    #scheduler=RoundRobinScheduler()
    #scheduler=RandomScheduler()
    #scheduler = ShortestQueueScheduler(task_queues=task_queues)


    # 4. 워커 노드 객체 생성
    nodes = []
    for name in WORKERS.keys():
        node = Node(name, WORKERS[name], prometheus_collector)
        nodes.append(node)
        print(f"Node registered: {name} -> {WORKERS[name]}")
    
    print(f"Total {len(nodes)} nodes registered for scheduling")
    
    # 5. 가상 IoT 디바이스 생성
    print("Generating virtual IoT devices...")
    task_generator.generate_devices(100)
    
    # 6. 태스크 시나리오 생성
    print("Generating high-load task timeline...")
    tasks_timeline = task_generator.generate_poisson_tasks_dynamic(
        lambda_rate=20.0,
        duration=600
    )
    
    print(f"Generated {len(tasks_timeline)} tasks for 600-second simulation")
    print(f"Expected average task rate: {len(tasks_timeline)/1800:.2f} tasks/second")
    
    # 7. 초기 전력 상태 확인
    initial_power = prometheus_collector.get_active_cluster_power(list(WORKERS.keys()))
    print(f"Initial cluster power: {initial_power:.1f}W")
    print(f"Power threshold: {power_threshold:.1f}W")
    
    # 8. 실험 시작
    start_time = time.time()
    print(f"Experiment started at: {datetime.fromtimestamp(start_time)}")
    
    print("Starting worker threads...")
    # 워커 스레드 시작
    threads = start_worker_threads(slo_monitor, performance_profiler, mobility_manager)
    


    # 부하 분산 스레드 시작
#    lb_thread = threading.Thread(
#        target=load_balancer_thread,
#        args=(task_queues, prometheus_collector),
#        daemon=True
#    )
#    lb_thread.start()



    # === 평가지표 수집용 변수 추가 ===
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
        args=(prometheus_collector, list(WORKERS.keys()), power_samples),# prometheus_collector.get_active_cluster_power(active_nodes) 클러스터 전력을 ~초마다 저장
        daemon=True
    )
    power_monitor_thread.start()
    
    # 변수 초기화
    total_tasks = len(tasks_timeline)
    completed_tasks = 0
    skipped_tasks = 0
    power_exceeded_count = 0
    no_eligible_node_count = 0
    scheduling_decisions = []
    
    print("="*80)
    print("EXPERIMENT STARTED - Real-time Power-aware SLO Scheduling")
    print("="*80)
    
    # 9. 태스크 실행 및 스케줄링 메인 루프
    for schedule_time, task in tasks_timeline:
        # === 실제 시간 동기화 ===
        current_elapsed = time.time() - start_time
        if schedule_time > current_elapsed:
            sleep_time = schedule_time - current_elapsed
            time.sleep(sleep_time)





        # ★ 현재 시점에 커버리지 내 디바이스 선택
        device_ids = list(mobility_manager.devices.keys())
        active_devices = [
            dev_id for dev_id in device_ids 
            if mobility_manager.is_in_coverage(dev_id, coverage_range=100)
        ]
        print(len(active_devices))

        # 커버리지 내 디바이스가 없으면 스킵
        if not active_devices:
            print(f"[SKIP] No device in coverage at t={current_elapsed:.1f}s")
            skipped_tasks += 1
            continue
                    
        # 랜덤 디바이스 선택
        task.device_id = random.choice(active_devices)
    



        # IoT 디바이스 mobility 기반 deadline 동적 적응
        task = deadline_adapter.adapt_deadline(task)
        
        # 현재 클러스터 전력 상태 확인
        current_cluster_power = prometheus_collector.get_active_cluster_power(list(WORKERS.keys()))
        
        # 전력 임계값 초과 체크----------------------
        if current_cluster_power > scheduler.power_threshold:
            power_exceeded_count += 1
        
        # 데드라인 준수 가능한 노드 필터링
        eligible_nodes = [
            node for node in nodes
            if performance_profiler.can_meet_deadline(
                node.name, task.model_name, task.adapted_deadline
            )
        ]
        
        # 적합한 노드가 없으면 모든 노드 중 선택
        if not eligible_nodes:
            eligible_nodes = nodes
            no_eligible_node_count += 1
            print(f"[WARNING] No node can meet deadline for {task.task_id}")
        
        # TOPSIS 기반 전력 및 SLO 인식 스케줄링
        selected_node = scheduler.schedule(task, eligible_nodes)
        task.assigned_node = selected_node.name
        
        # 요청 시각 기록
        request_timestamp = time.time()
        
        # 스케줄링 결정 기록
        scheduling_decision = {
            'timestamp': request_timestamp,
            'request_timestamp': format_timestamp(request_timestamp),  # 요청 시각
            'task_id': task.task_id,
            'device_id': task.device_id,
            'slo_type': task.slo_type,
            'base_deadline': task.base_deadline,
            'adapted_deadline': task.adapted_deadline,
            'selected_node': selected_node.name,
            'node_power': selected_node.power_consumption,
            'cluster_power': current_cluster_power,
            'eligible_nodes_count': len(eligible_nodes)
        }
        scheduling_decisions.append(scheduling_decision)
        
        # 기본 큐---------------
        #task_queues[selected_node.name].put((task, request_timestamp))
        
        # 우선순위 큐
        put_task_to_queue(selected_node.name, task, request_timestamp) 






        # 진행 상황 로그 출력
        completed_tasks += 1
        progress = (completed_tasks / total_tasks) * 100
        
        print(f"[{progress:5.1f}%] [{current_cluster_power:5.1f}W] "
            f"Task {task.task_id} ({task.slo_type}) -> {selected_node.name} "
            f"(Node: {selected_node.power_consumption:4.1f}W) "
            f"Base Deadline: {task.base_deadline:.2f}s "
            f"Adapted Deadline: {task.adapted_deadline:.2f}s")
        
        """# 100개 태스크마다 중간 통계 출력
        if completed_tasks % 100 == 0:
            current_violations = len(slo_monitor.violations)
            print(f"\n--- Progress Report: {completed_tasks}/{total_tasks} tasks ---")
            print(f"Current cluster power: {current_cluster_power:.1f}W / {power_threshold:.1f}W")
            print(f"Power threshold exceeded: {power_exceeded_count} times")
            print(f"No eligible nodes: {no_eligible_node_count} times")
            print(f"SLO violations so far: {current_violations}")
            print(f"Elapsed time: {time.time() - start_time:.1f}s")
            print()"""
                
    print("="*80)
    print("TASK GENERATION COMPLETED - Waiting for processing to finish...")
    print("="*80)
    
    # 모든 작업 완료 대기
    for queue_name, q in task_queues.items():
        remaining = q.qsize()
        print(f"Waiting for queue {queue_name} to finish ({remaining} tasks remaining)...")
        q.join()
    #lb_thread.join()
    
    # 실험 종료 및 정리
    end_time = time.time()
    total_duration = end_time - start_time
    
    # 워커 스레드 종료
    print("Stopping worker threads...")
    stop_worker_threads(threads)
    
    # === 평가지표 계산 추가 ===
    violation_stats = slo_monitor.get_violation_stats()
    violations = violation_stats["total"]
    


    total_response_time = 0.0
    total_waiting_time  = 0.0
    completed_count     = 0


    # completed_tasks_queue에서 결과 읽기
    while completed_count < total_tasks:
        result = completed_tasks_queue.get()
        # result는 dict로 put된 값
        total_response_time += result["response_time"]
        total_waiting_time  += result["waiting_time"]
        completed_count    += 1    


    if len(power_samples) > 1:
        sample_interval = power_samples[1][0] - power_samples[0][0]
    else:
        sample_interval = 1.0


    # 1. SLO 준수율
    slo_compliance_rate = (total_tasks - violations) / total_tasks * 100 if total_tasks else 0


    # 2. 시스템 처리량
    throughput = completed_tasks / total_duration if total_duration > 0 else 0
    
    # 3. 평균 응답시간 (total_response_time : task 생성 -> 큐 대기 -> 추론 서버 -> 응답 반환)
    avg_response_time = total_response_time / total_tasks


    # 4. 평균 대기시간
    avg_waiting_time  = total_waiting_time  / total_tasks


    # 5. 총 에너지 소모량 (추론 노드 전체 전력 * 측정 간격 / 3600초)
    total_energy_Wh = sum(power for _, power in power_samples) * sample_interval / 3600


    # 6. 평균 소비 전력 (모든 전력 값 단순 평균)
    avg_power_W = sum(power for _, power in power_samples) / len(power_samples) if power_samples else 0


    # 7. 전력 효율성 (와트시(Wh) 당 처리 태스크 수)
    power_efficiency = completed_tasks / total_energy_Wh if total_energy_Wh > 0 else 0 


    # 8. 에너지당 SLO 준수율 (와트시(Wh)당 준수율 비율)
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
    print(f"Total tasks processed: {total_tasks}")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Average task rate: {total_tasks/total_duration:.2f} tasks/second")
    print(f"Skipped tasks (no coverage): {skipped_tasks}")
    print()
    
    print("=== POWER CONSUMPTION ===")
    print(f"Initial cluster power: {initial_power:.1f}W")
    print(f"Final cluster power: {final_cluster_power:.1f}W")
    print(f"Power threshold: {power_threshold:.1f}W")
    print(f"Threshold exceeded: {power_exceeded_count} times ({power_exceeded_count/total_tasks*100:.1f}%)")
    print()
    
    print("=== SLO PERFORMANCE ===")
    print(f"Total SLO violations: {violation_stats['total']}")
    if violation_stats['total'] > 0:
        print(f"SLO violation rate: {violation_stats['total']/total_tasks*100:.2f}%")
        print(f"Violations by type: {violation_stats['by_type']}")
        print(f"Average violation time: {violation_stats['avg_violation']:.3f}s")
    else:
        print("🎉 No SLO violations detected!")
    print()
    
    print("=== SCHEDULING PERFORMANCE ===")
    print(f"Tasks with no eligible nodes: {no_eligible_node_count} ({no_eligible_node_count/total_tasks*100:.1f}%)")
    
    # 노드별 작업 분배 통계
    node_task_count = {}
    for decision in scheduling_decisions:
        node = decision['selected_node']
        node_task_count[node] = node_task_count.get(node, 0) + 1
    
    print("Node task distribution:")
    for node, count in sorted(node_task_count.items()):
        percentage = (count / total_tasks) * 100
        print(f"  {node}: {count} tasks ({percentage:.1f}%)")
    print()
    
    # === 새로운 평가지표 출력 ===
    print("=== EVALUATION METRICS ===")
    print(f"SLO Compliance Rate: {slo_compliance_rate:.2f}%")
    print(f"System Throughput: {throughput:.2f} tasks/second")
    print(f"Average Response Time : {avg_response_time:.4f}s")
    print(f"Average Waiting Time  : {avg_waiting_time:.4f}s")
    print(f"Completed Count : {completed_count}")
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
