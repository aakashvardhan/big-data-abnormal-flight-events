# producer_manager.py
"""
Producer Manager: Wraps Kafka producer for background execution with metrics
"""

import threading
import time
import logging
from typing import Optional, Callable
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
from shared_state import ProducerMetrics, producer_metrics_queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProducerManager:
    """Manages Kafka producer in background thread with metrics reporting"""
    
    def __init__(self):
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.producer: Optional[KafkaProducer] = None
        self.metrics_queue = producer_metrics_queue
        
    def start(self, bootstrap_servers: str, topic: str, csv_path: str, 
              batch_size: int = 100, delay_ms: int = 100):
        """
        Start producer in background thread
        
        Args:
            bootstrap_servers: Kafka broker address
            topic: Kafka topic name
            csv_path: Path to CSV file with flight data
            batch_size: Records per batch
            delay_ms: Delay between batches in milliseconds
        """
        if self.running:
            logger.warning("Producer already running")
            return False
            
        self.running = True
        self.thread = threading.Thread(
            target=self._run_producer,
            args=(bootstrap_servers, topic, csv_path, batch_size, delay_ms),
            daemon=True
        )
        self.thread.start()
        logger.info("Producer thread started")
        return True
    
    def _run_producer(self, bootstrap_servers: str, topic: str, csv_path: str,
                      batch_size: int, delay_ms: int):
        """Internal method to run producer in thread"""
        try:
            # Initialize Kafka producer
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1,
                compression_type='gzip'
            )
            
            logger.info(f"Producer initialized for topic: {topic}")
            
            # Stream data from CSV
            chunk_iter = pd.read_csv(csv_path, chunksize=batch_size)
            total_sent = 0
            batch_num = 0
            
            for chunk in chunk_iter:
                if not self.running:
                    logger.info("Producer stopped by user")
                    break
                
                batch_num += 1
                records = chunk.to_dict('records')
                records_sent = 0
                
                for record in records:
                    if not self.running:
                        break
                    
                    # Clean NaN values
                    cleaned_record = {
                        k: (None if pd.isna(v) else v) 
                        for k, v in record.items()
                    }
                    
                    # Create message key
                    key = f"{cleaned_record.get('ORIGIN', '')}-{cleaned_record.get('DEST', '')}"
                    
                    try:
                        future = self.producer.send(topic, key=key, value=cleaned_record)
                        future.get(timeout=10)
                        total_sent += 1
                        records_sent += 1
                    except KafkaError as e:
                        logger.error(f"Failed to send message: {e}")
                
                # Send metrics
                metrics = ProducerMetrics(
                    batch_num=batch_num,
                    total_sent=total_sent,
                    records_in_batch=records_sent,
                    timestamp=time.time()
                )
                self.metrics_queue.put(metrics)
                
                logger.info(f"Batch {batch_num}: Sent {records_sent} records (Total: {total_sent})")
                
                # Delay to simulate real-time streaming
                if delay_ms > 0 and self.running:
                    time.sleep(delay_ms / 1000.0)
                
                if self.running:
                    time.sleep(1)
            
            if self.running:
                self.producer.flush()
                logger.info(f"✓ Streaming complete! Total records sent: {total_sent}")
            
        except Exception as e:
            logger.error(f"Producer error: {e}")
        finally:
            self._cleanup()
    
    def stop(self):
        """Stop the producer gracefully"""
        if not self.running:
            return
        
        logger.info("Stopping producer...")
        self.running = False
        
        # Wait for thread to finish (with timeout)
        if self.thread:
            self.thread.join(timeout=5)
        
        self._cleanup()
        logger.info("Producer stopped")
    
    def _cleanup(self):
        """Clean up resources"""
        if self.producer:
            try:
                self.producer.close()
            except Exception as e:
                logger.error(f"Error closing producer: {e}")
            self.producer = None
        self.running = False
    
    def is_running(self) -> bool:
        """Check if producer is running"""
        return self.running and self.thread and self.thread.is_alive()
