import time
import threading
import json
import csv
from datetime import datetime
from scheduler.power_aware_topsis_scheduler import PowerAwareTOPSISScheduler
from scheduler.base_scheduler import Node
from mobility.mobility_manager import MobilityManager
from task_generator import TaskGenerator
from deadline_adapter import DeadlineAdapter
from prometheus_collector import PrometheusCollector
from inference_request import start_worker_threads, stop_worker_threads, task_queues, WORKERS
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
        weights=[0.3, 0.5, 0.1, 0.1]
    )
    
    # 4. 워커 노드 객체 생성
    nodes = []
    for name in WORKERS.keys():
        node = Node(name, WORKERS[name], prometheus_collector)
        nodes.append(node)
        print(f"Node registered: {name} -> {WORKERS[name]}")
    
    print(f"Total {len(nodes)} nodes registered for scheduling")
    
    # 5. 가상 IoT 디바이스 생성
    print("Generating virtual IoT devices...")
    task_generator.generate_devices(1000)
    
    # 6. 태스크 시나리오 생성
    print("Generating high-load task timeline...")
    tasks_timeline = task_generator.generate_poisson_tasks(
        lambda_rate=40.0,
        duration=30
    )
    
    print(f"Generated {len(tasks_timeline)} tasks for 1800-second simulation")
    print(f"Expected average task rate: {len(tasks_timeline)/1800:.2f} tasks/second")
    
    # 7. 초기 전력 상태 확인
    initial_power = prometheus_collector.get_active_cluster_power(list(WORKERS.keys()))
    print(f"Initial cluster power: {initial_power:.1f}W")
    print(f"Power threshold: {power_threshold:.1f}W")
    
    # 8. 실험 시작
    start_time = time.time()
    print(f"Experiment started at: {datetime.fromtimestamp(start_time)}")
    
    print("Starting worker threads...")
    threads = start_worker_threads()
    
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
        args=(prometheus_collector, list(WORKERS.keys()), power_samples),  # power_samples 추가
        daemon=True
    )
    power_monitor_thread.start()
    
    # 변수 초기화
    total_tasks = len(tasks_timeline)
    completed_tasks = 0
    power_exceeded_count = 0
    no_eligible_node_count = 0
    scheduling_decisions = []
    
    print("="*80)
    print("EXPERIMENT STARTED - Real-time Power-aware SLO Scheduling")
    print("="*80)
    
    # 9. 태스크 실행 및 스케줄링 메인 루프
    for schedule_time, task in tasks_timeline:
        # === 실제 시간 동기화 (복원) ===
        current_elapsed = time.time() - start_time
        if schedule_time > current_elapsed:
            sleep_time = schedule_time - current_elapsed
            time.sleep(sleep_time)
            
        # IoT 디바이스 mobility 기반 deadline 동적 적응
        task = deadline_adapter.adapt_deadline(task)
        
        # 현재 클러스터 전력 상태 확인
        current_cluster_power = prometheus_collector.get_active_cluster_power(list(WORKERS.keys()))
        
        # 전력 임계값 초과 체크
        if current_cluster_power > scheduler.power_threshold:
            power_exceeded_count += 1
        
        # 데드라인 준수 가능한 노드 필터링
        eligible_nodes = []
        for node in nodes:
            if performance_profiler.can_meet_deadline(
                node.name, task.model_name, task.adapted_deadline
            ):
                eligible_nodes.append(node)
        
        # 적합한 노드가 없으면 모든 노드 중 선택
        if not eligible_nodes:
            eligible_nodes = nodes
            no_eligible_node_count += 1
            print(f"[WARNING] No node can meet deadline for {task.task_id}")
        
        # TOPSIS 기반 전력 및 SLO 인식 스케줄링
        selected_node = scheduler.schedule(task, eligible_nodes)
        task.assigned_node = selected_node.name
        
        # 예상 완료 시간 계산
        estimated_completion = performance_profiler.get_estimated_time(
            selected_node.name, task.model_name
        )
        
        # 스케줄링 결정 기록
        scheduling_decision = {
            'timestamp': time.time(),
            'task_id': task.task_id,
            'device_id': task.device_id,
            'slo_type': task.slo_type,
            'base_deadline': task.base_deadline,
            'adapted_deadline': task.adapted_deadline,
            'selected_node': selected_node.name,
            'estimated_completion': estimated_completion,
            'node_power': selected_node.power_consumption,
            'cluster_power': current_cluster_power,
            'eligible_nodes_count': len(eligible_nodes)
        }
        scheduling_decisions.append(scheduling_decision)
        
        # 선택된 노드의 task queue에 할당
        task_queues[selected_node.name].put(task.model_name)
        
        # 진행 상황 로그 출력
        completed_tasks += 1
        progress = (completed_tasks / total_tasks) * 100
        
        # SLO 위험도 표시
        deadline_risk = "🔴" if estimated_completion > task.adapted_deadline else \
                       "🟡" if estimated_completion > task.adapted_deadline * 0.8 else "🟢"
        
        print(f"[{progress:5.1f}%] [{current_cluster_power:5.1f}W] {deadline_risk} "
              f"Task {task.task_id} ({task.slo_type}) -> {selected_node.name} "
              f"(Node: {selected_node.power_consumption:4.1f}W) "
              f"Est: {estimated_completion:.2f}s / DL: {task.adapted_deadline:.2f}s")
        
        # 100개 태스크마다 중간 통계 출력
        if completed_tasks % 100 == 0:
            current_violations = len(slo_monitor.violations)
            print(f"\n--- Progress Report: {completed_tasks}/{total_tasks} tasks ---")
            print(f"Current cluster power: {current_cluster_power:.1f}W / {power_threshold:.1f}W")
            print(f"Power threshold exceeded: {power_exceeded_count} times")
            print(f"No eligible nodes: {no_eligible_node_count} times")
            print(f"SLO violations so far: {current_violations}")
            print(f"Elapsed time: {time.time() - start_time:.1f}s")
            print()
            
    print("="*80)
    print("TASK GENERATION COMPLETED - Waiting for processing to finish...")
    print("="*80)
    
    # 모든 작업 완료 대기
    for queue_name, q in task_queues.items():
        remaining = q.qsize()
        print(f"Waiting for queue {queue_name} to finish ({remaining} tasks remaining)...")
        q.join()
    
    # 실험 종료 및 정리
    end_time = time.time()
    total_duration = end_time - start_time
    
    # 워커 스레드 종료
    print("Stopping worker threads...")
    stop_worker_threads(threads)
    
    # === 평가지표 계산 추가 ===
    violation_stats = slo_monitor.get_violation_stats()
    violations = violation_stats["total"]
    
    # 1. SLO 준수율
    slo_compliance_rate = (total_tasks - violations) / total_tasks * 100 if total_tasks else 0
    
    # 2. 시스템 처리량
    throughput = completed_tasks / total_duration if total_duration > 0 else 0
    
    # 3. 평균 응답시간 (추정값 사용 - 실제 측정은 worker thread에서 해야 함)
    estimated_total_response_time = sum(d['estimated_completion'] for d in scheduling_decisions)
    avg_estimated_response_time = estimated_total_response_time / completed_tasks if completed_tasks else 0
    
    # 4. 전력 관련 지표
    if len(power_samples) > 1:
        sample_interval = power_samples[1][0] - power_samples[0][0]
    else:
        sample_interval = 1.0
    
    total_energy_Wh = sum(power for _, power in power_samples) * sample_interval / 3600
    avg_power_W = sum(power for _, power in power_samples) / len(power_samples) if power_samples else 0
    power_efficiency = completed_tasks / total_energy_Wh if total_energy_Wh > 0 else 0
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
    print(f"Avg Estimated Response Time: {avg_estimated_response_time:.4f} seconds")
    print(f"Total Energy Consumption: {total_energy_Wh:.3f} Wh")
    print(f"Average Power Consumption: {avg_power_W:.3f} W")
    print(f"Power Efficiency: {power_efficiency:.2f} tasks/Wh")
    print(f"SLO per Energy: {slo_per_energy:.2f} %/Wh")
    print()
    
    # === 평가지표 CSV 저장 ===
    with open(f"./results/metrics_{experiment_id}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows([
            ["slo_compliance_rate_%", f"{slo_compliance_rate:.2f}"],
            ["throughput_tasks_per_sec", f"{throughput:.2f}"],
            ["avg_estimated_response_time_sec", f"{avg_estimated_response_time:.4f}"],
            ["total_energy_Wh", f"{total_energy_Wh:.3f}"],
            ["avg_power_W", f"{avg_power_W:.3f}"],
            ["power_efficiency_tasks_per_Wh", f"{power_efficiency:.2f}"],
            ["slo_per_energy_%_per_Wh", f"{slo_per_energy:.2f}"]
        ])
    
    print("=== FILES GENERATED ===")
    print(f"Master inference results: master_inference_results.csv")
    print(f"SLO violations: slo_violations_{experiment_id}.csv")
    print(f"Scheduling decisions: scheduling_decisions_{experiment_id}.csv")
    print(f"Device information: devices_{experiment_id}.csv")
    print(f"Experiment summary: experiment_summary_{experiment_id}.json")
    print(f"Evaluation metrics: metrics_{experiment_id}.csv")  # 새로 추가
    print("="*80)

if __name__ == "__main__":
    main()
