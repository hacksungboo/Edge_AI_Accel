import requests
import time
from typing import Dict, Optional, List


class PrometheusCollector:
    def __init__(self, prometheus_url="http://localhost:30090"):
        self.prometheus_url = prometheus_url
        self.api_url = f"{prometheus_url}/api/v1/query"
        
    def query_metric(self, query: str) -> Optional[Dict]:
        """프로메테우스에서 특정 쿼리 실행"""
        try:
            response = requests.get(self.api_url, params={'query': query}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    return data
                else:
                    print(f"Prometheus query error: {data.get('error', 'Unknown error')}")
                    return None
            else:
                print(f"Prometheus HTTP error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error querying Prometheus: {e}")
            return None
    
    def get_node_power_consumption(self, node_name: str) -> float:
        """특정 노드의 현재 전력 소모량 가져오기"""
        power_value = 4.0  # 기본값
        
        # Jetson 노드 전력 쿼리
        if "jetson" in node_name.lower():
            if "jetson1" in node_name:
                query = 'jetson_board_power_watt{instance="192.168.0.61:8000"}'
            elif "jetson2" in node_name:
                query = 'jetson_board_power_watt{instance="192.168.0.62:8000"}'
            else:
                query = 'jetson_board_power_watt'
                
        # 라즈베리파이 계열 노드 전력 쿼리
        elif any(keyword in node_name.lower() for keyword in ["coral", "hailo", "raspberry", "pi"]):
            ip_mapping = {
                "coral1": "192.168.0.51",
                "coral2": "192.168.0.52", 
                "hailo1": "192.168.0.71",
                "hailo2": "192.168.0.72",
                "master": "192.168.0.11"
            }
            
            if node_name in ip_mapping:
                ip = ip_mapping[node_name]
                query = f'raspberry_power_watt{{instance="{ip}:8000"}}'
            else:
                query = 'raspberry_power_watt'
        else:
            print(f"Unknown node type for {node_name}, using default power")
            return power_value
            
        result = self.query_metric(query)
        
        if result and result['data']['result']:
            try:
                power_value = float(result['data']['result'][0]['value'][1])
                #print(f"[POWER] {node_name}: {power_value:.2f}W")
            except (IndexError, ValueError, KeyError) as e:
                print(f"Error parsing power data for {node_name}: {e}")
        else:
            print(f"No power data found for {node_name}, using default {power_value}W")
            
        return power_value
    
    def get_active_cluster_power(self, active_nodes: List[str]) -> float:
        """현재 활성화된 노드들만의 전력 소모량 합계"""
        total_power = 0.0
        
        for node_name in active_nodes:
            node_power = self.get_node_power_consumption(node_name)
            total_power += node_power
        
        return total_power
    
    def calculate_dynamic_power_threshold(self, active_nodes):
        """동적 전력 임계값 계산 (실측 데이터 반영)"""
        # 실측 데이터 기반 전력 프로파일
        power_profiles = {
            "coral1": {"idle": 4.56, "peak": 5.35},
            "coral2": {"idle": 4.60, "peak": 5.28},
            "jetson1": {"idle": 2.38, "peak": 6.47},
            "jetson2": {"idle": 2.29, "peak": 6.24},
            "hailo1": {"idle": 5.04, "peak": 5.33},
            "hailo2": {"idle": 4.51, "peak": 5.01}
        }
        
        total_idle = 0
        total_peak = 0
        
        for node_name in active_nodes:
            if node_name in power_profiles:
                total_idle += power_profiles[node_name]["idle"]
                total_peak += power_profiles[node_name]["peak"]
            else:
                # 기본값 (알 수 없는 노드)
                total_idle += 4.0
                total_peak += 6.0
        
        # 임계값 = idle + (peak - idle) * 0.8
        threshold = total_idle + (total_peak - total_idle) * 0.8
        return threshold
    
    def get_node_status(self, node_name: str) -> bool:
        """노드 상태 확인 (up/down)"""
        ip_mapping = {
            "jetson1": "192.168.0.61:9100",
            "jetson2": "192.168.0.62:9100",
            "coral1": "192.168.0.51:9100",
            "coral2": "192.168.0.52:9100",
            "hailo1": "192.168.0.71:9100", 
            "hailo2": "192.168.0.72:9100"
        }
        
        if node_name in ip_mapping:
            instance = ip_mapping[node_name]
            query = f'up{{instance="{instance}"}}'
            result = self.query_metric(query)
            
            if result and result['data']['result']:
                status = int(result['data']['result'][0]['value'][1])
                return status == 1
                
        return False
