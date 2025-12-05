# kafka_producer.py
"""
Kafka Producer: Sends flight data to Kafka topic for real-time prediction
Usage: python kafka_producer.py --topic flight-data --csv may_2025.csv
"""

import json
import time
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FlightDataProducer:
    """Kafka producer for streaming flight data"""
    
    def __init__(self, bootstrap_servers, topic):
        """
        Initialize Kafka producer
        
        Args:
            bootstrap_servers: Kafka broker addresses (e.g., 'localhost:9092')
            topic: Kafka topic name
        """
        self.topic = topic
        
        # Create Kafka producer with JSON serialization
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',  # Wait for all replicas to acknowledge
            retries=3,
            max_in_flight_requests_per_connection=1,  # Maintain order
            compression_type='gzip'  # Compress data
        )
        
        logger.info(f"Kafka producer initialized for topic: {topic}")
    
    def send_flight_record(self, flight_data, key=None):
        """
        Send a single flight record to Kafka
        
        Args:
            flight_data: Dictionary containing flight information
            key: Optional message key (e.g., flight ID)
        """
        try:
            # Send message
            future = self.producer.send(
                self.topic,
                key=key,
                value=flight_data
            )
            
            # Wait for send to complete
            record_metadata = future.get(timeout=10)
            
            logger.debug(
                f"Sent record to partition {record_metadata.partition} "
                f"at offset {record_metadata.offset}"
            )
            
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def stream_from_csv(self, csv_path, batch_size=100, delay_ms=100):
        """
        Stream flight data from CSV file to Kafka
        
        Args:
            csv_path: Path to CSV file
            batch_size: Number of records to send before delay
            delay_ms: Delay in milliseconds between batches (simulates real-time)
        """
        logger.info(f"Starting to stream data from {csv_path}")
        
        # Read CSV in chunks to avoid memory issues
        chunk_iter = pd.read_csv(csv_path, chunksize=batch_size)
        
        total_sent = 0
        batch_num = 0
        
        try:
            for chunk in chunk_iter:
                batch_num += 1
                
                # Convert chunk to records
                records = chunk.to_dict('records')
                
                # Send each record
                for record in records:
                    # Clean NaN values
                    cleaned_record = {
                        k: (None if pd.isna(v) else v) 
                        for k, v in record.items()
                    }
                    
                    # Use origin-dest-time as message key for partitioning
                    key = f"{cleaned_record.get('ORIGIN', '')}-{cleaned_record.get('DEST', '')}"
                    
                    if self.send_flight_record(cleaned_record, key=key):
                        total_sent += 1
                
                logger.info(f"Batch {batch_num}: Sent {len(records)} records (Total: {total_sent})")
                
                # Simulate real-time streaming with delay
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)
                
                # Additional 1 second delay per batch
                time.sleep(1)
            
            # Flush remaining messages
            self.producer.flush()
            logger.info(f"✓ Streaming complete! Total records sent: {total_sent}")
            
        except Exception as e:
            logger.error(f"Error streaming data: {e}")
            raise
        finally:
            self.close()
    
    def close(self):
        """Close Kafka producer"""
        self.producer.close()
        logger.info("Kafka producer closed")


def main():
    parser = argparse.ArgumentParser(description='Stream flight data to Kafka')
    parser.add_argument(
        '--bootstrap-servers',
        default='localhost:29092',
        help='Kafka bootstrap servers (default: localhost:29092)'
    )
    parser.add_argument(
        '--topic',
        default='flight-data',
        help='Kafka topic name (default: flight-data)'
    )
    parser.add_argument(
        '--csv',
        required=True,
        help='Path to CSV file containing flight data'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of records per batch (default: 100)'
    )
    parser.add_argument(
        '--delay-ms',
        type=int,
        default=100,
        help='Delay between batches in ms (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Create producer and stream data
    producer = FlightDataProducer(args.bootstrap_servers, args.topic)
    producer.stream_from_csv(args.csv, args.batch_size, args.delay_ms)


if __name__ == "__main__":
    main()
