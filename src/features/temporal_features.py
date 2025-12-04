"""
Temporal feature engineering for flight anomaly detection.
Extracts time-based features from flight event sequences.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def extract_temporal_features(flight_df: pd.DataFrame) -> Dict:
    """
    Extract temporal features from a single flight's event sequence.
    
    Args:
        flight_df: DataFrame for a single flight, sorted by timestamp
        
    Returns:
        Dictionary of temporal features
    """
    if len(flight_df) == 0:
        return {}
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(flight_df['timestamp']):
        flight_df['timestamp'] = pd.to_datetime(flight_df['timestamp'])
    
    flight_df = flight_df.sort_values('timestamp').reset_index(drop=True)
    
    features = {}
    
    # Basic temporal features
    features['total_duration_seconds'] = (
        flight_df['timestamp'].max() - flight_df['timestamp'].min()
    ).total_seconds()
    
    features['n_events'] = len(flight_df)
    features['avg_time_between_events'] = (
        features['total_duration_seconds'] / max(features['n_events'] - 1, 1)
    )
    
    # Time deltas between consecutive events
    time_deltas = flight_df['timestamp'].diff().dt.total_seconds().dropna()
    if len(time_deltas) > 0:
        features['min_time_delta'] = time_deltas.min()
        features['max_time_delta'] = time_deltas.max()
        features['mean_time_delta'] = time_deltas.mean()
        features['std_time_delta'] = time_deltas.std()
        features['median_time_delta'] = time_deltas.median()
    
    # Phase-specific temporal features
    features.update(_extract_phase_times(flight_df))
    
    # Ground operation times
    features.update(_extract_ground_times(flight_df))
    
    # Time-of-day features
    features.update(_extract_time_of_day_features(flight_df))
    
    return features


def _extract_phase_times(flight_df: pd.DataFrame) -> Dict:
    """Extract time spent in different flight phases."""
    features = {}
    
    # Common flight phase event types (adjust based on actual data)
    phase_keywords = {
        'ground': ['parking', 'taxi', 'gate', 'stand', 'ramp'],
        'taxi': ['taxi', 'taxiway'],
        'takeoff': ['take-off', 'takeoff', 'departure'],
        'climb': ['climb', 'climbing'],
        'cruise': ['cruise', 'level'],
        'descent': ['descent', 'descending', 'approach'],
        'landing': ['landing', 'touchdown', 'arrival']
    }
    
    # Calculate time in each phase
    for phase, keywords in phase_keywords.items():
        phase_events = flight_df[
            flight_df['event_type'].str.lower().str.contains('|'.join(keywords), na=False, regex=True)
        ]
        
        if len(phase_events) > 0:
            phase_duration = (
                phase_events['timestamp'].max() - phase_events['timestamp'].min()
            ).total_seconds()
            features[f'{phase}_time_seconds'] = phase_duration
            features[f'{phase}_event_count'] = len(phase_events)
        else:
            features[f'{phase}_time_seconds'] = 0
            features[f'{phase}_event_count'] = 0
    
    # Climb rate (if altitude data available)
    if 'altitude' in flight_df.columns:
        climb_events = flight_df[flight_df['altitude'].notna()].copy()
        if len(climb_events) > 1:
            climb_events = climb_events.sort_values('timestamp')
            altitude_diff = climb_events['altitude'].iloc[-1] - climb_events['altitude'].iloc[0]
            time_diff = (climb_events['timestamp'].iloc[-1] - climb_events['timestamp'].iloc[0]).total_seconds()
            
            if time_diff > 0:
                features['avg_climb_rate_ms'] = altitude_diff / time_diff
            else:
                features['avg_climb_rate_ms'] = 0
        else:
            features['avg_climb_rate_ms'] = 0
    
    return features


def _extract_ground_times(flight_df: pd.DataFrame) -> Dict:
    """Extract ground operation temporal features."""
    features = {}
    
    # Identify ground events (low altitude or ground-specific event types)
    if 'altitude' in flight_df.columns:
        ground_events = flight_df[flight_df['altitude'] <= 50]  # Below 50m considered ground
    else:
        ground_keywords = ['parking', 'gate', 'stand', 'taxi', 'ramp']
        ground_events = flight_df[
            flight_df['event_type'].str.lower().str.contains('|'.join(ground_keywords), na=False, regex=True)
        ]
    
    if len(ground_events) > 0:
        features['ground_time_seconds'] = (
            ground_events['timestamp'].max() - ground_events['timestamp'].min()
        ).total_seconds()
        features['ground_event_count'] = len(ground_events)
    else:
        features['ground_time_seconds'] = 0
        features['ground_event_count'] = 0
    
    # Taxi time (specific ground movement)
    taxi_keywords = ['taxi', 'taxiway']
    taxi_events = flight_df[
        flight_df['event_type'].str.lower().str.contains('|'.join(taxi_keywords), na=False, regex=True)
    ]
    
    if len(taxi_events) > 0:
        features['taxi_time_seconds'] = (
            taxi_events['timestamp'].max() - taxi_events['timestamp'].min()
        ).total_seconds()
        features['taxi_event_count'] = len(taxi_events)
    else:
        features['taxi_time_seconds'] = 0
        features['taxi_event_count'] = 0
    
    # Parking/stand time
    parking_keywords = ['parking', 'gate', 'stand', 'ramp']
    parking_events = flight_df[
        flight_df['event_type'].str.lower().str.contains('|'.join(parking_keywords), na=False, regex=True)
    ]
    
    if len(parking_events) > 0:
        features['parking_time_seconds'] = (
            parking_events['timestamp'].max() - parking_events['timestamp'].min()
        ).total_seconds()
        features['parking_event_count'] = len(parking_events)
    else:
        features['parking_time_seconds'] = 0
        features['parking_event_count'] = 0
    
    return features


def _extract_time_of_day_features(flight_df: pd.DataFrame) -> Dict:
    """Extract time-of-day related features."""
    features = {}
    
    first_timestamp = flight_df['timestamp'].min()
    
    features['hour_of_day'] = first_timestamp.hour
    features['day_of_week'] = first_timestamp.dayofweek  # 0=Monday, 6=Sunday
    features['is_weekend'] = 1 if features['day_of_week'] >= 5 else 0
    
    # Time of day categories
    hour = features['hour_of_day']
    if 5 <= hour < 12:
        features['time_period'] = 'morning'
    elif 12 <= hour < 17:
        features['time_period'] = 'afternoon'
    elif 17 <= hour < 21:
        features['time_period'] = 'evening'
    else:
        features['time_period'] = 'night'
    
    # Encode time period as numeric (for ML)
    period_map = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
    features['time_period_encoded'] = period_map[features['time_period']]
    
    return features


def extract_temporal_features_batch(flight_summary_df: pd.DataFrame, 
                                     events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features for all flights in batch.
    
    Args:
        flight_summary_df: Flight-level summary DataFrame (from Phase 1)
        events_df: Event-level DataFrame (sorted by flight and timestamp)
        
    Returns:
        DataFrame with temporal features added
    """
    # If there is no timestamp column (as in the BTS on‑time dataset that has one row per
    # flight and no event timestamps), we cannot compute temporal dynamics. In that case,
    # simply return the original flight_summary_df unchanged.
    if 'timestamp' not in events_df.columns:
        print("No 'timestamp' column found in events_df; skipping temporal feature extraction.")
        return flight_summary_df

    print("Extracting temporal features for all flights...")
    
    temporal_features_list = []
    
    for flight_id in flight_summary_df['flight_id'].unique():
        flight_events = events_df[events_df['flight_id'] == flight_id].copy()
        
        if len(flight_events) > 0:
            features = extract_temporal_features(flight_events)
            features['flight_id'] = flight_id
            temporal_features_list.append(features)
    
    temporal_df = pd.DataFrame(temporal_features_list)
    
    # Merge with flight summary
    result_df = flight_summary_df.merge(temporal_df, on='flight_id', how='left')
    
    print(f"✓ Extracted temporal features for {len(result_df)} flights")
    
    return result_df



