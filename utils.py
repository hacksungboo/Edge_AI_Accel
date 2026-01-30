import time
import threading

def mobility_update_loop(mobility_manager):
    """IoT 디바이스의 위치를 주기적으로 업데이트하는 함수"""
    print("[MOBILITY] Mobility update loop started")
    update_count = 0
    
    while True:
        try:
            mobility_manager.update_mobility(dt=2.0)
            update_count += 1
            
            #if update_count % 300 == 0:
            #    print(f"[MOBILITY] Updated device positions ({update_count} updates, "
            #          f"{update_count/60:.1f} minutes)")
                
        except Exception as e:
            print(f"[ERROR] Mobility update error: {e}")
            
        time.sleep(2.0)

def power_monitor_loop(prometheus_collector, active_nodes, power_samples):
    """전력 모니터링 및 로깅 함수 - power_samples에 데이터 저장"""
    print(f"[POWER] Power monitoring started for nodes: {active_nodes}")
    monitor_count = 0
    
    while True:
        try:
            cluster_power = prometheus_collector.get_active_cluster_power(active_nodes)
            # === 전력 샘플 저장 추가 ===
            power_samples.append((time.time(), cluster_power))
            
            monitor_count += 1
            
            if monitor_count % 12 == 0:  # 60초마다
                #print(f"\n[POWER MONITOR] Cluster total: {cluster_power:.1f}W")
                
                for node_name in active_nodes:
                    node_power = prometheus_collector.get_node_power_consumption(node_name)
                    node_status = prometheus_collector.get_node_status(node_name)
                    status_str = "🟢 UP" if node_status else "🔴 DOWN"
                    #print(f"[POWER]   {node_name:8s}: {node_power:5.1f}W ({status_str})")
                #print()
                
        except Exception as e:
            print(f"[ERROR] Power monitoring error: {e}")
            
        time.sleep(2.0)

def save_experiment_results(experiment_id, scheduling_decisions, mobility_manager, 
                          slo_monitor, start_time, end_time):
    """실험 결과를 파일로 저장"""
    import csv
    import json
    from datetime import datetime
    
    # 1. 스케줄링 결정 저장
    with open(f"./results/[Task] scheduling_decisions_{experiment_id}.csv", 'w', newline='') as f:
        if scheduling_decisions:
            writer = csv.DictWriter(f, fieldnames=scheduling_decisions[0].keys())
            writer.writeheader()
            writer.writerows(scheduling_decisions)
    
    # 2. 디바이스 정보 저장
    with open(f"./results/[Device] devices_{experiment_id}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['device_id', 'device_type', 'final_x', 'final_y'])
        
        for device_id, info in mobility_manager.devices.items():
            writer.writerow([
                device_id, info.device_type,
                info.current_position[0], info.current_position[1]
            ])
    
    # 3. 실험 요약 저장
    summary = {
        'experiment_id': experiment_id,
        'start_time': datetime.fromtimestamp(start_time).isoformat(),
        'end_time': datetime.fromtimestamp(end_time).isoformat(),
        'duration_seconds': end_time - start_time,
        'total_tasks': len(scheduling_decisions),
        'total_devices': len(mobility_manager.devices),
        'slo_violations': len(slo_monitor.violations)
    }
    
    with open(f"./results/[Summary] experiment_summary_{experiment_id}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[SAVE] Experiment results saved with ID: {experiment_id}")


def load_balancer_thread(task_queues, prometheus_collector):
    """부하 불균형 감지 및 태스크 재분배"""
    while True:
        try:
            # 1. 큐 길이 모니터링
            queue_sizes = {name: q.qsize() for name, q in task_queues.items()}
            if not queue_sizes:
                time.sleep(5)
                continue

            # 2. 가장 바쁜 노드와 한가한 노드 찾기
            busy_node, busy_size = max(queue_sizes.items(), key=lambda x: x[1])
            idle_node, idle_size = min(queue_sizes.items(), key=lambda x: x[1])

            # 3. 임계값 초과 시 태스크 이주
            if busy_size - idle_size > 10:
                migrate_tasks(busy_node, idle_node, task_queues)
                print(f"[MIGRATION] {busy_node}({busy_size}) → {idle_node}({idle_size})")

        except Exception as e:
            print(f"[ERROR] Load balancer: {e}")

        time.sleep(5)

def migrate_tasks(from_node, to_node, task_queues, count=5):
    temp_tasks = []
    for _ in range(count):
        if task_queues[from_node].empty():
            break
        task_item = task_queues[from_node].get_nowait()
        temp_tasks.append(task_item)
        task_queues[from_node].task_done()   # 꺼낸 즉시 완료 표시


    # 마이그레이션 로그 작성
    for priority, order, (task, request_ts) in temp_tasks:
        log_migration(from_node, to_node, task)


    for task_item in temp_tasks:
        task_queues[to_node].put(task_item)