# consumer_manager.py
"""
Consumer Manager: Manages Spark consumer subprocess and captures output
"""

import subprocess
import threading
import logging
import json
import re
import time
from typing import Optional
from shared_state import PredictionResult, ConsumerMetrics, consumer_output_queue, consumer_metrics_queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsumerManager:
    """Manages Spark consumer subprocess with output capturing"""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.output_thread: Optional[threading.Thread] = None
        self.running = False
        self.output_queue = consumer_output_queue
        self.metrics_queue = consumer_metrics_queue
        
    def start(self, bootstrap_servers: str, topic: str, model_path: str, 
              threshold: float = 0.5, starting_offset: str = "earliest"):
        """
        Start Spark consumer as subprocess
        
        Args:
            bootstrap_servers: Kafka broker address
            topic: Kafka topic to consume from
            model_path: Path to trained ML model
            threshold: Probability threshold for delay prediction (default: 0.5)
            starting_offset: Where to start reading - 'earliest' or 'latest' (default: 'earliest')
        """
        if self.running:
            logger.warning("Consumer already running")
            return False
        
        # Build spark-submit command
        spark_command = [
            'spark-submit',
            '--driver-memory', '4g',
            '--executor-memory', '4g',
            '--conf', 'spark.driver.maxResultSize=2g',
            '--packages', 'org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1',
            'kafka_consumer_spark.py',
            '--bootstrap-servers', bootstrap_servers,
            '--topic', topic,
            '--model-path', model_path,
            '--threshold', str(threshold),
            '--starting-offset', starting_offset
        ]
        
        try:
            # Start subprocess
            self.process = subprocess.Popen(
                spark_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                cwd='/Users/aakashvardhan/Documents/big-data-project-prediction/Big Data ML Model'
            )
            
            self.running = True
            
            # Start thread to read output
            self.output_thread = threading.Thread(
                target=self._read_output,
                daemon=True
            )
            self.output_thread.start()
            
            logger.info("Consumer subprocess started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start consumer: {e}")
            return False
    
    def _read_output(self):
        """Read and parse output from Spark consumer"""
        try:
            for line in self.process.stdout:
                line = line.strip()
                
                if not line or not self.running:
                    continue
                
                # Parse JSON predictions
                if line.startswith('{') and 'carrier' in line:
                    try:
                        data = json.loads(line)
                        prediction = PredictionResult(
                            carrier=data.get('carrier', ''),
                            origin=data.get('origin', ''),
                            destination=data.get('destination', ''),
                            scheduled_departure=data.get('scheduled_departure', 0),
                            prediction=data.get('prediction', 0),
                            delay_probability=data.get('delay_probability', 0.0),
                            timestamp=time.time()
                        )
                        self.output_queue.put(prediction)
                        logger.debug(f"Parsed prediction: {prediction}")
                    except json.JSONDecodeError:
                        pass
                
                # Parse batch processing messages
                if '📥 Processing batch' in line:
                    match = re.search(r'batch (\d+)', line)
                    if match:
                        batch_id = int(match.group(1))
                        logger.info(f"Processing batch {batch_id}")
                
                # Parse probability statistics
                if 'Records:' in line and 'Delay probability' in line:
                    try:
                        # Example: "  Records: 20, Delay probability - min: 0.123, max: 0.456, avg: 0.234"
                        records_match = re.search(r'Records:\s*(\d+)', line)
                        prob_match = re.search(
                            r'min:\s*([\d.]+),\s*max:\s*([\d.]+),\s*avg:\s*([\d.]+)',
                            line
                        )
                        if records_match and prob_match:
                            metrics = ConsumerMetrics(
                                batch_id=0,  # Will be updated if we track it
                                records_processed=int(records_match.group(1)),
                                min_prob=float(prob_match.group(1)),
                                max_prob=float(prob_match.group(2)),
                                avg_prob=float(prob_match.group(3)),
                                timestamp=time.time()
                            )
                            self.metrics_queue.put(metrics)
                    except Exception as e:
                        logger.error(f"Error parsing metrics: {e}")
                
                # Log important messages
                if any(marker in line for marker in ['✓', '❌', '⚠️', 'ERROR', 'WARN']):
                    logger.info(line)
                    
        except Exception as e:
            logger.error(f"Error reading consumer output: {e}")
        finally:
            logger.info("Consumer output thread ended")
    
    def stop(self):
        """Stop the consumer subprocess"""
        if not self.running:
            return
        
        logger.info("Stopping consumer...")
        self.running = False
        
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Consumer didn't stop gracefully, killing...")
                self.process.kill()
            except Exception as e:
                logger.error(f"Error stopping consumer: {e}")
            
            self.process = None
        
        if self.output_thread:
            self.output_thread.join(timeout=5)
        
        logger.info("Consumer stopped")
    
    def is_running(self) -> bool:
        """Check if consumer is running"""
        return (self.running and 
                self.process and 
                self.process.poll() is None)
