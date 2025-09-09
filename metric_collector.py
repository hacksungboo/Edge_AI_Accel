import csv
import time
from datetime import datetime

class MetricsCollector:
    def __init__(self, experiment_id, results_dir="./results"):
        self.experiment_id = experiment_id
        self.dir = results_dir
        # 파일 경로
        self.task_log_file = f"{self.dir}/tasks_{experiment_id}.csv"
        self.power_log_file = f"{self.dir}/power_{experiment_id}.csv"
        self.summary_file    = f"{self.dir}/summary_{experiment_id}.csv"
        # 초기화
        self._init_task_log()
        self._init_power_log()
        # 내부 저장소
        self.total_tasks = 0
        self.completed_tasks = 0
        self.total_response_time = 0.0
        self.power_samples = []

    def _init_task_log(self):
        with open(self.task_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "task_id", "slo_type", "assigned_node",
                "submit_time", "start_time", "completion_time",
                "response_time", "slo_violated"
            ])

    def _init_power_log(self):
        with open(self.power_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "cluster_power_W"])

    def log_task(self, task):
        """태스크 완료 직후 호출"""
        self.total_tasks += 1
        rt = task.completion_time - task.submit_time
        self.total_response_time += rt
        self.completed_tasks += 1

        with open(self.task_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                task.task_id, task.slo_type, task.assigned_node,
                f"{task.submit_time:.3f}", f"{task.start_time:.3f}",
                f"{task.completion_time:.3f}", f"{rt:.4f}",
                int(task.is_slo_violated)
            ])

    def log_power(self, timestamp, cluster_power):
        """전력 샘플링 스레드에서 주기 호출"""
        self.power_samples.append((timestamp, cluster_power))
        with open(self.power_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{timestamp:.3f}", f"{cluster_power:.3f}"])

    def write_summary(self, slo_monitor, duration_sec):
        """실험 종료 시 한 번 호출하여 요약 지표 저장"""
        # SLO 준수율
        total = self.total_tasks
        violations = slo_monitor.get_violation_stats()["total"]
        slo_compliance = (total - violations) / total * 100 if total else 0

        # 처리량(tasks/sec), 평균 응답시간(sec)
        throughput = self.completed_tasks / duration_sec if duration_sec > 0 else 0
        avg_response = self.total_response_time / self.completed_tasks if self.completed_tasks else 0

        # 전력 효율성 및 총 에너지 소모량
        # 전력 샘플은 (timestamp, power) 리스트
        # 샘플 간격을 일정하다고 가정 (e.g., 5초마다)
        if len(self.power_samples) > 1:
            interval = self.power_samples[1][0] - self.power_samples[0][0]
        else:
            interval = 1.0
        total_energy_Wh = sum(p for _, p in self.power_samples) * interval / 3600
        avg_power = sum(p for _, p in self.power_samples) / len(self.power_samples) if self.power_samples else 0
        power_efficiency = self.completed_tasks / (total_energy_Wh) if total_energy_Wh > 0 else 0
        slo_per_energy = slo_compliance / total_energy_Wh if total_energy_Wh > 0 else 0

        # 요약 파일 헤더 쓰기 (첫 실행 시)
        header = [
            "experiment_id", "duration_sec", "total_tasks", "violations",
            "slo_compliance_%", "throughput_tps", "avg_response_s",
            "total_energy_Wh", "avg_power_W", "power_efficiency_t_per_Wh",
            "slo_per_energy"
        ]
        row = [
            self.experiment_id, f"{duration_sec:.1f}", total, violations,
            f"{slo_compliance:.2f}", f"{throughput:.2f}", f"{avg_response:.4f}",
            f"{total_energy_Wh:.3f}", f"{avg_power:.3f}", f"{power_efficiency:.2f}",
            f"{slo_per_energy:.2f}"
        ]
        # summary 파일에 한 줄 추가
        write_header = False
        try:
            with open(self.summary_file, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                write_header = True
        except FileExistsError:
            pass
        with open(self.summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
