# Abnormal Flight Events Detection Project

## Overview
Machine learning project to detect abnormal flight patterns from flight event data using unsupervised anomaly detection techniques.

## Dataset
- **Training Data**: ~6 million flight events, ~254,767 unique flights (Jan 10-31, 2022)
- **Event Types**: 25+ categories including phase transitions, altitude crossings, ground operations
- **Features**: flight_id, event_type, timestamp, longitude, latitude, altitude, info (JSON)

## Project Structure
```
project/
├── data/
│   ├── raw/              # Original data files from Google Drive
│   ├── processed/        # Cleaned and processed data
│   └── features/         # Engineered features
├── notebooks/
│   ├── phase1_eda.ipynb  # Data exploration and quality assessment
│   ├── phase2_features.ipynb  # Feature engineering
│   └── phase3_modeling.ipynb  # Model development
├── src/
│   ├── data/             # Data loading and processing
│   ├── features/         # Feature engineering modules
│   ├── models/           # Model implementations
│   └── utils/            # Utility functions
├── outputs/
│   ├── figures/          # Visualizations
│   ├── models/           # Saved models
│   └── results/          # Anomaly detection results
├── config/
│   └── config.yaml       # Configuration parameters
└── requirements.txt
```

## Implementation Phases

### Phase 1: Data Preparation & EDA ✓ (Current)
- Data quality assessment
- Exploratory data analysis
- Flight reconstruction pipeline

### Phase 2: Feature Engineering (Upcoming)
- Flight-level aggregate features
- Event sequence features
- Contextual/comparative features

### Phase 3: Model Development (Upcoming)
- Isolation Forest baseline
- Clustering + outlier detection
- Autoencoder for complex patterns
- Ensemble approaches

### Phase 4: Evaluation & Validation (Upcoming)
- Manual inspection of anomalies
- Cluster coherence analysis
- Stability testing

### Phase 5: Interpretation & Insights (Upcoming)
- Anomaly characterization
- Feature attribution (SHAP)
- Temporal/spatial pattern analysis

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download data (see notebooks/phase1_eda.ipynb)

3. Run Phase 1 notebook for data exploration

## Google Drive Links
- Training Data: https://drive.google.com/drive/u/1/folders/1Scvz11KmgIt5dzmSMzS4MCv0IXnte2a-
- Testing Data: https://drive.google.com/drive/u/1/folders/1l3LtAp0u2svJ46lBL80WwwABz7TllSXF
