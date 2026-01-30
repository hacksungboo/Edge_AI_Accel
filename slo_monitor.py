import csv
import time
from datetime import datetime


class SLOMonitor:
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        self.violations = []
        self.slo_log_file = f"./results/[SLO] slo_violations_{experiment_id}.csv"
        self._init_slo_log()
    
    def _init_slo_log(self):
        with open(self.slo_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'task_id', 'device_id', 'slo_type', 
                'deadline', 'actual_time', 'violation_amount', 'assigned_node'
            ])
    
    def check_and_record_violation(self, task, actual_completion_time):
        """SLO 위반 확인 및 기록"""
        if actual_completion_time > task.adapted_deadline:
            violation_amount = actual_completion_time - task.adapted_deadline
            
            violation_record = {
                'timestamp': datetime.now().isoformat(),
                'task_id': task.task_id,
                'device_id': task.device_id,
                'slo_type': task.slo_type,
                'deadline': task.adapted_deadline,
                'actual_time': actual_completion_time,
                'violation_amount': violation_amount,
                'assigned_node': task.assigned_node
            }
            
            self.violations.append(violation_record)
            self._log_violation(violation_record)
            
            #print(f"[SLO VIOLATION] {task.task_id} ({task.slo_type}): "
            #      f"{actual_completion_time:.2f}s > {task.adapted_deadline:.2f}s "
            #      f"(+{violation_amount:.2f}s)")
            
            return True
        return False
    
    def _log_violation(self, violation):
        with open(self.slo_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                violation['timestamp'], violation['task_id'], 
                violation['device_id'], violation['slo_type'],
                violation['deadline'], violation['actual_time'],
                violation['violation_amount'], violation['assigned_node']
            ])
    
    def get_violation_stats(self):
        """SLO 위반 통계"""
        if not self.violations:
            return {"total": 0, "by_type": {}, "avg_violation": 0}
        
        by_type = {}
        total_violation_time = 0
        
        for violation in self.violations:
            slo_type = violation['slo_type']
            by_type[slo_type] = by_type.get(slo_type, 0) + 1
            total_violation_time += violation['violation_amount']
        
        return {
            "total": len(self.violations),
            "by_type": by_type,
            "avg_violation": total_violation_time / len(self.violations)
        }
