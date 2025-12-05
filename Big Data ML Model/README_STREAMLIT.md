# Flight Delay Prediction Streamlit Dashboard

Real-time monitoring dashboard for Kafka producer and consumer with live ML predictions.

## Features

- **Kafka Producer Control**: Start/stop producer with configurable batch size and delay
- **Kafka Consumer Control**: Start/stop Spark consumer with real-time prediction capture
- **Live Metrics**: Real-time monitoring of producer and consumer performance
- **Prediction Feed**: Display latest high-risk flight predictions
- **Visualization**: Trend charts for delay probability over time
- **Auto-refresh**: Configurable dashboard refresh rate

## Architecture

```
┌─────────────────┐
│  Streamlit UI   │
│   (Dashboard)   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────┐
│Producer │ │Consumer  │
│Manager  │ │Manager   │
│(Thread) │ │(Subprocess)
└────┬────┘ └────┬─────┘
     │           │
     ▼           ▼
  Kafka ←──────→ Spark
  Topic         Consumer
```

## Prerequisites

1. **Kafka Running**: Ensure Kafka is running on `localhost:29092`
   ```bash
   # Start Kafka (if using Docker)
   docker-compose up -d
   ```

2. **Python Dependencies**:
   ```bash
   pip install streamlit kafka-python pandas pyspark
   ```

3. **Required Files**:
   - `may_2025.csv` - Flight data CSV
   - `flight_delay_rf_model_new/` - Trained ML model directory
   - `kafka_consumer_spark.py` - Spark consumer script

## Installation

All required files are already created:
- `streamlit_app.py` - Main dashboard
- `producer_manager.py` - Producer wrapper
- `consumer_manager.py` - Consumer wrapper
- `shared_state.py` - State management

## Usage

### 1. Start the Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard will open at `http://localhost:8501`

### 2. Configure Settings (Sidebar)

**Kafka Settings:**
- Bootstrap Servers: `localhost:29092`
- Topic: `flight-data`

**Producer Settings:**
- CSV File: `may_2025.csv`
- Batch Size: 100 (records per batch)
- Delay: 100ms (between batches)

**Consumer Settings:**
- Model Path: `flight_delay_rf_model_new`

**Dashboard Settings:**
- Auto-refresh: Enable/disable
- Refresh Rate: 1-10 seconds

### 3. Start Producer

1. Click **"▶️ Start Producer"** button
2. Watch metrics update in real-time:
   - Total Records Sent
   - Current Batch
   - Records in Last Batch
   - Last Update timestamp

### 4. Start Consumer

1. Click **"▶️ Start Consumer"** button
2. Wait ~30 seconds for Spark initialization
3. Watch predictions appear in the feed:
   - Batches Processed
   - Delay Probability Statistics
   - High-Risk Flight Predictions

### 5. Monitor Real-Time Predictions

The bottom section shows:
- **Latest High-Risk Flights**: Table with carrier, route, and delay probability
- **Delay Probability Trend**: Line chart of predictions over time

### 6. Stop Components

- Click **"⏹️ Stop Producer"** to stop data streaming
- Click **"⏹️ Stop Consumer"** to stop Spark consumer

## Configuration Options

### Producer Configuration
- **Batch Size**: Number of records sent before delay (10-1000)
- **Delay (ms)**: Milliseconds to wait between batches (0-5000)
- Lower delay = faster streaming, higher resource usage

### Consumer Configuration
- **Model Path**: Directory containing trained Random Forest model
- Consumer runs with 4GB driver and executor memory

### Dashboard Configuration
- **Auto-refresh**: Enable continuous updates
- **Refresh Rate**: How often to poll for new data (1-10 seconds)

## Troubleshooting

### Producer Won't Start
- **Issue**: "CSV file not found"
- **Solution**: Verify `may_2025.csv` exists in current directory

### Consumer Won't Start
- **Issue**: "Model not found"
- **Solution**: Verify model directory exists: `flight_delay_rf_model_new/`

### No Predictions Appearing
- **Check**:
  1. Producer is running and sending data
  2. Consumer is running (wait 30s for initialization)
  3. Kafka is accessible at `localhost:29092`
  4. Check terminal logs for errors

### High Memory Usage
- **Issue**: Spark consuming too much memory
- **Solution**: Already configured with 4GB limits
- If still issues, reduce producer batch size or increase delay

### Dashboard Not Refreshing
- **Check**: "Auto-refresh" is enabled in sidebar
- **Try**: Manually refresh browser or adjust refresh rate

## File Structure

```
Big Data ML Model/
├── streamlit_app.py           # Main dashboard
├── producer_manager.py        # Producer wrapper
├── consumer_manager.py        # Consumer wrapper  
├── shared_state.py           # Shared state/queues
├── kafka_producer.py         # Original producer
├── kafka_consumer_spark.py   # Original consumer
├── may_2025.csv              # Flight data
└── flight_delay_rf_model_new/ # ML model
```

## How It Works

### Producer Manager
- Runs Kafka producer in background thread
- Reads CSV in chunks and streams to Kafka
- Sends metrics to queue for dashboard updates

### Consumer Manager
- Launches Spark consumer as subprocess
- Captures stdout/stderr in real-time
- Parses JSON predictions and metrics
- Pushes to queues for dashboard display

### Shared State
- Thread-safe queues for communication
- Producer metrics: batch info, records sent
- Consumer metrics: probability statistics
- Prediction results: high-risk flights

### Streamlit Dashboard
- Polls queues every N seconds
- Updates metrics and predictions
- Maintains history of last 100 predictions
- Provides controls for start/stop

## Performance Notes

- **Producer**: Lightweight, runs in thread
- **Consumer**: Heavy Spark process, allocates 4GB RAM
- **Dashboard**: Refreshes every 2 seconds by default
- **History**: Keeps last 100 predictions in memory

## Tips

1. **Start Consumer First**: Give Spark time to initialize before streaming data
2. **Monitor Resources**: Spark consumer is memory-intensive
3. **Adjust Refresh Rate**: Lower rate for faster updates, but higher CPU usage
4. **Check Logs**: Terminal shows detailed logs from both components
5. **Clean Shutdown**: Always use Stop buttons before closing dashboard

## Next Steps

- Add error alerting for failed predictions
- Implement prediction accuracy tracking
- Add data export functionality
- Create historical analysis views
- Add Kafka broker health monitoring
