import numpy as np
from .base_scheduler import BaseScheduler


class TOPSISScheduler(BaseScheduler):
    def __init__(self, weights=None):
        self.weights = weights or [0.4, 0.3, 0.2, 0.1]
        
    def schedule(self, task, nodes, mobility_info=None):
        if len(nodes) == 1:
            return nodes[0]
            
        matrix = self.create_decision_matrix(task, nodes)
        normalized = self.normalize_matrix(matrix)
        weighted = normalized * self.weights
        
        ideal_positive = np.max(weighted, axis=0)
        ideal_negative = np.min(weighted, axis=0)
        
        distances_pos = np.sqrt(np.sum((weighted - ideal_positive)**2, axis=1))
        distances_neg = np.sqrt(np.sum((weighted - ideal_negative)**2, axis=1))
        
        closeness = distances_neg / (distances_pos + distances_neg + 1e-10)
        
        best_node_idx = np.argmax(closeness)
        return nodes[best_node_idx]
    
    def create_decision_matrix(self, task, nodes):
        matrix = []
        for node in nodes:
            slo_urgency = 1.0 / (task.adapted_deadline + 0.1)
            
            row = [
                1/(node.power_consumption + 0.1),
                slo_urgency,
                1/(node.network_latency + 0.01),
                1-node.current_utilization
            ]
            matrix.append(row)
        return np.array(matrix)
    
    def normalize_matrix(self, matrix):
        """벡터 정규화"""
        norm = np.sqrt(np.sum(matrix**2, axis=0))
        return matrix / (norm + 1e-10)
