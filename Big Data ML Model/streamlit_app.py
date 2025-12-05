# streamlit_app.py
"""
Streamlit Dashboard for Real-Time Flight Delay Prediction
Monitors Kafka Producer and Consumer with live metrics and predictions
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os
from producer_manager import ProducerManager
from consumer_manager import ConsumerManager
from shared_state import (
    producer_metrics_queue,
    consumer_output_queue,
    consumer_metrics_queue
)

# Page configuration
st.set_page_config(
    page_title="Flight Delay Prediction Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    .status-stopped {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'producer_manager' not in st.session_state:
        st.session_state.producer_manager = None
    
    if 'consumer_manager' not in st.session_state:
        st.session_state.consumer_manager = None
    
    if 'producer_metrics' not in st.session_state:
        st.session_state.producer_metrics = {
            'batch_num': 0,
            'total_sent': 0,
            'records_in_batch': 0,
            'messages_per_sec': 0,
            'last_update': None
        }
    
    if 'consumer_metrics' not in st.session_state:
        st.session_state.consumer_metrics = {
            'batches_processed': 0,
            'total_predictions': 0,
            'sum_probabilities': 0.0,
            'min_prob': 1.0,
            'max_prob': 0.0,
            'avg_prob': 0.0,
            'last_update': None
        }
    
    if 'predictions_history' not in st.session_state:
        st.session_state.predictions_history = []
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True


def update_metrics():
    """Update metrics from queues"""
    # Update producer metrics
    while not producer_metrics_queue.empty():
        metrics = producer_metrics_queue.get()
        st.session_state.producer_metrics.update({
            'batch_num': metrics.batch_num,
            'total_sent': metrics.total_sent,
            'records_in_batch': metrics.records_in_batch,
            'last_update': metrics.timestamp
        })
    
    # Update consumer metrics with running average
    while not consumer_metrics_queue.empty():
        metrics = consumer_metrics_queue.get()
        
        # Update cumulative stats
        current = st.session_state.consumer_metrics
        current['batches_processed'] += 1
        current['total_predictions'] += metrics.records_processed
        current['sum_probabilities'] += metrics.avg_prob * metrics.records_processed
        
        # Update running min/max
        current['min_prob'] = min(current['min_prob'], metrics.min_prob)
        current['max_prob'] = max(current['max_prob'], metrics.max_prob)
        
        # Calculate running average
        if current['total_predictions'] > 0:
            current['avg_prob'] = current['sum_probabilities'] / current['total_predictions']
        
        current['last_update'] = metrics.timestamp
    
    # Update predictions
    while not consumer_output_queue.empty():
        prediction = consumer_output_queue.get()
        st.session_state.predictions_history.append(prediction.to_dict())
        
        # Keep only last 100 predictions
        if len(st.session_state.predictions_history) > 100:
            st.session_state.predictions_history.pop(0)


def main():
    initialize_session_state()
    
    # Header
    st.title("✈️ Real-Time Flight Delay Prediction Dashboard")
    st.markdown("Monitor Kafka producer/consumer with live ML predictions")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Kafka Settings")
        bootstrap_servers = st.text_input(
            "Bootstrap Servers",
            value="localhost:29092"
        )
        topic = st.text_input(
            "Topic",
            value="flight-data"
        )
        
        st.subheader("Producer Settings")
        csv_file = st.text_input(
            "CSV File",
            value="may_2025.csv"
        )
        batch_size = st.number_input(
            "Batch Size",
            min_value=5,
            max_value=1000,
            value=20,
            help="Number of records per batch (lower = slower streaming)"
        )
        delay_ms = st.number_input(
            "Delay (ms)",
            min_value=0,
            max_value=10000,
            value=500,
            help="Delay between batches in milliseconds (higher = slower streaming)"
        )
        
        st.subheader("Consumer Settings")
        model_path = st.text_input(
            "Model Path",
            value="flight_delay_rf_model_new"
        )
        
        # Check if consumer is running for threshold control
        consumer_is_active = bool(
            st.session_state.consumer_manager and 
            st.session_state.consumer_manager.is_running()
        )
        
        # Store threshold in session state if not exists
        if 'threshold' not in st.session_state:
            st.session_state.threshold = 0.5
        
        # Disable threshold slider if consumer is running
        threshold = st.slider(
            "Prediction Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.threshold,
            step=0.01,
            help="Probability threshold for classifying flights as delayed. Lower = more sensitive.",
            disabled=consumer_is_active,
            key="threshold_slider"
        )
        
        # Update session state when slider changes
        if not consumer_is_active:
            st.session_state.threshold = threshold
        
        # Show warning if consumer is running
        if consumer_is_active:
            st.info(f"ℹ️ Current threshold: {st.session_state.threshold:.2f} (Stop consumer to change)")
        
        # Starting offset option
        read_from_beginning = st.checkbox(
            "Read from beginning (first start only)",
            value=False,
            help="On first start: read all messages from beginning. On restart: resumes from last offset.",
            disabled=consumer_is_active
        )
        
        st.markdown("---")
        st.subheader("Dashboard Settings")
        st.session_state.auto_refresh = st.checkbox(
            "Auto-refresh",
            value=True
        )
        refresh_rate = st.slider(
            "Refresh Rate (seconds)",
            min_value=1,
            max_value=10,
            value=2
        )
        
        st.markdown("---")
        if st.button("🔄 Reset Metrics", use_container_width=True):
            st.session_state.producer_metrics = {
                'batch_num': 0,
                'total_sent': 0,
                'records_in_batch': 0,
                'messages_per_sec': 0,
                'last_update': None
            }
            st.session_state.consumer_metrics = {
                'batches_processed': 0,
                'total_predictions': 0,
                'sum_probabilities': 0.0,
                'min_prob': 1.0,
                'max_prob': 0.0,
                'avg_prob': 0.0,
                'last_update': None
            }
            st.session_state.predictions_history = []
            # Clear queues
            while not producer_metrics_queue.empty():
                producer_metrics_queue.get()
            while not consumer_metrics_queue.empty():
                consumer_metrics_queue.get()
            while not consumer_output_queue.empty():
                consumer_output_queue.get()
            st.success("✅ Metrics and history cleared!")
            st.rerun()
    
    # Main content
    col1, col2 = st.columns(2)
    
    # Producer Section
    with col1:
        st.header("📤 Kafka Producer")
        
        producer_running = bool(
            st.session_state.producer_manager and 
            st.session_state.producer_manager.is_running()
        )
        
        status_text = "🟢 Running" if producer_running else "🔴 Stopped"
        st.markdown(f"**Status:** {status_text}")
        
        # Control buttons
        col1a, col1b = st.columns(2)
        
        with col1a:
            if st.button("▶️ Start Producer", disabled=producer_running):
                if not os.path.exists(csv_file):
                    st.error(f"CSV file not found: {csv_file}")
                else:
                    manager = ProducerManager()
                    if manager.start(bootstrap_servers, topic, csv_file, batch_size, delay_ms):
                        st.session_state.producer_manager = manager
                        st.success("Producer started!")
                        st.rerun()
        
        with col1b:
            if st.button("⏹️ Stop Producer", disabled=not producer_running):
                if st.session_state.producer_manager:
                    st.session_state.producer_manager.stop()
                    st.session_state.producer_manager = None
                    st.info("Producer stopped")
                    st.rerun()
        
        # Metrics
        st.subheader("📊 Metrics")
        
        metrics = st.session_state.producer_metrics
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Total Records Sent", metrics['total_sent'])
            st.metric("Current Batch", metrics['batch_num'])
        
        with metric_col2:
            st.metric("Records in Last Batch", metrics['records_in_batch'])
            if metrics['last_update']:
                time_ago = int(time.time() - metrics['last_update'])
                st.metric("Last Update", f"{time_ago}s ago")
            else:
                st.metric("Last Update", "N/A")
    
    # Consumer Section
    with col2:
        st.header("📥 Kafka Consumer")
        
        consumer_running = bool(
            st.session_state.consumer_manager and 
            st.session_state.consumer_manager.is_running()
        )
        
        status_text = "🟢 Running" if consumer_running else "🔴 Stopped"
        st.markdown(f"**Status:** {status_text}")
        
        # Control buttons
        col2a, col2b = st.columns(2)
        
        with col2a:
            if st.button("▶️ Start Consumer", disabled=consumer_running):
                if not os.path.exists(model_path):
                    st.error(f"Model not found: {model_path}")
                else:
                    manager = ConsumerManager()
                    starting_offset = "earliest" if read_from_beginning else "latest"
                    if manager.start(bootstrap_servers, topic, model_path, threshold, starting_offset):
                        st.session_state.consumer_manager = manager
                        offset_text = "from beginning" if read_from_beginning else "latest only"
                        st.success(f"Consumer started with threshold: {threshold} ({offset_text})")
                        st.rerun()
        
        with col2b:
            if st.button("⏹️ Stop Consumer", disabled=not consumer_running):
                if st.session_state.consumer_manager:
                    st.session_state.consumer_manager.stop()
                    st.session_state.consumer_manager = None
                    st.info("Consumer stopped")
                    st.rerun()
        
        # Metrics
        st.subheader("📊 Metrics")
        
        c_metrics = st.session_state.consumer_metrics
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Batches Processed", c_metrics['batches_processed'])
            st.metric("Total Predictions", c_metrics.get('total_predictions', 0))
        
        with metric_col2:
            st.metric("Avg Delay Probability", f"{c_metrics['avg_prob']:.3f}")
            st.metric("Min / Max Prob", f"{c_metrics['min_prob']:.3f} / {c_metrics['max_prob']:.3f}")
    
    # Predictions Section
    st.markdown("---")
    st.header("🎯 Live Predictions")
    
    if st.session_state.predictions_history:
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.predictions_history)
        
        # Display latest predictions
        st.subheader("Latest High-Risk Flights")
        display_df = df.tail(10).sort_values('delay_probability', ascending=False)
        
        # Format the dataframe
        display_df['scheduled_departure'] = display_df['scheduled_departure'].astype(str)
        display_df['delay_probability'] = display_df['delay_probability'].apply(lambda x: f"{x:.3f}")
        display_df['prediction'] = display_df['prediction'].map({0: '✅ On Time', 1: '⚠️ Delayed'})
        
        st.dataframe(
            display_df[['carrier', 'origin', 'destination', 'scheduled_departure', 
                       'prediction', 'delay_probability']],
            use_container_width=True,
            hide_index=True
        )
        
        # Chart
        st.subheader("Delay Probability Trend")
        chart_df = df.tail(50).copy()
        chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'], unit='s')
        
        st.line_chart(
            chart_df.set_index('timestamp')['delay_probability'],
            use_container_width=True
        )
    else:
        st.info("No predictions yet. Start both producer and consumer to see real-time predictions.")
    
    # Auto-refresh
    if st.session_state.auto_refresh:
        update_metrics()
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
