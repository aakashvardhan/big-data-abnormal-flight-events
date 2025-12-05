# shared_state.py
"""
Shared state management for Streamlit app
Provides thread-safe queues for communication between components
"""

from queue import Queue
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class ProducerMetrics:
    """Metrics from Kafka producer"""
    batch_num: int
    total_sent: int
    records_in_batch: int
    timestamp: float
    
    def to_dict(self):
        return {
            'batch_num': self.batch_num,
            'total_sent': self.total_sent,
            'records_in_batch': self.records_in_batch,
            'timestamp': self.timestamp
        }


@dataclass
class PredictionResult:
    """Prediction result from consumer"""
    carrier: str
    origin: str
    destination: str
    scheduled_departure: int
    prediction: int
    delay_probability: float
    timestamp: float
    
    def to_dict(self):
        return {
            'carrier': self.carrier,
            'origin': self.origin,
            'destination': self.destination,
            'scheduled_departure': self.scheduled_departure,
            'prediction': self.prediction,
            'delay_probability': self.delay_probability,
            'timestamp': self.timestamp
        }


@dataclass
class ConsumerMetrics:
    """Metrics from Kafka consumer"""
    batch_id: int
    records_processed: int
    min_prob: float
    max_prob: float
    avg_prob: float
    timestamp: float
    
    def to_dict(self):
        return {
            'batch_id': self.batch_id,
            'records_processed': self.records_processed,
            'min_prob': self.min_prob,
            'max_prob': self.max_prob,
            'avg_prob': self.avg_prob,
            'timestamp': self.timestamp
        }


# Global queues for inter-thread communication
producer_metrics_queue = Queue()
consumer_output_queue = Queue()
consumer_metrics_queue = Queue()
