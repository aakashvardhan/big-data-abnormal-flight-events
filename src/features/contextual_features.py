"""
Contextual feature engineering for flight anomaly detection.
Extracts airport norms, peer comparisons, and contextual deviations.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from scipy import stats


def extract_contextual_features(flight_df: pd.DataFrame,
                                 flight_summary_df: pd.DataFrame,
                                 flight_id: str) -> Dict:
    """
    Extract contextual features by comparing a flight to its peers.
    
    Args:
        flight_df: DataFrame for a single flight
        flight_summary_df: DataFrame with all flights (for comparison)
        flight_id: ID of the current flight
        
    Returns:
        Dictionary of contextual features
    """
    if len(flight_df) == 0:
        return {}
    
    features = {}
    
    # Extract basic flight metrics for comparison
    flight_metrics = _extract_flight_metrics(flight_df)
    
    # Airport-specific features
    features.update(_extract_airport_context(flight_df, flight_summary_df))
    
    # Time-of-day context
    features.update(_extract_time_context(flight_df, flight_summary_df))
    
    # Peer comparison features
    features.update(_extract_peer_comparison(flight_metrics, flight_summary_df, flight_id))
    
    return features


def _extract_flight_metrics(flight_df: pd.DataFrame) -> Dict:
    """Extract key metrics from a flight for comparison."""
    metrics = {}
    
    if 'timestamp' in flight_df.columns:
        metrics['duration'] = (
            flight_df['timestamp'].max() - flight_df['timestamp'].min()
        ).total_seconds()
    else:
        metrics['duration'] = 0
    
    metrics['n_events'] = len(flight_df)
    metrics['n_event_types'] = flight_df['event_type'].nunique()
    
    if 'altitude' in flight_df.columns:
        altitude_data = flight_df['altitude'].dropna()
        if len(altitude_data) > 0:
            metrics['max_altitude'] = altitude_data.max()
            metrics['mean_altitude'] = altitude_data.mean()
        else:
            metrics['max_altitude'] = 0
            metrics['mean_altitude'] = 0
    else:
        metrics['max_altitude'] = 0
        metrics['mean_altitude'] = 0
    
    # Extract airport
    metrics['airport'] = _extract_airport_from_flight(flight_df)
    
    # Extract time of day
    if 'timestamp' in flight_df.columns:
        first_timestamp = flight_df['timestamp'].min()
        metrics['hour'] = first_timestamp.hour if hasattr(first_timestamp, 'hour') else 0
        metrics['day_of_week'] = first_timestamp.dayofweek if hasattr(first_timestamp, 'dayofweek') else 0
    else:
        metrics['hour'] = 0
        metrics['day_of_week'] = 0
    
    return metrics


def _extract_airport_from_flight(flight_df: pd.DataFrame) -> Optional[str]:
    """Extract airport identifier from flight data."""
    import json
    
    # Check if airport column exists
    if 'airport' in flight_df.columns:
        airports = flight_df['airport'].dropna().unique()
        if len(airports) > 0:
            return str(airports[0])
    
    # Check info_parsed
    if 'info_parsed' in flight_df.columns:
        for info in flight_df['info_parsed'].dropna():
            if isinstance(info, dict):
                if 'airport' in info:
                    return str(info['airport'])
            elif isinstance(info, str):
                try:
                    info_dict = json.loads(info)
                    if 'airport' in info_dict:
                        return str(info_dict['airport'])
                except:
                    pass
    
    # Check info column
    if 'info' in flight_df.columns:
        for info in flight_df['info'].dropna():
            try:
                if isinstance(info, str):
                    info_dict = json.loads(info)
                    if 'airport' in info_dict:
                        return str(info_dict['airport'])
            except:
                pass
    
    return None


def _extract_airport_context(flight_df: pd.DataFrame,
                             flight_summary_df: pd.DataFrame) -> Dict:
    """Extract airport-specific contextual features."""
    features = {}
    
    airport = _extract_airport_from_flight(flight_df)
    
    if airport is None:
        features['airport_deviation_duration'] = 0
        features['airport_deviation_events'] = 0
        features['airport_zscore_duration'] = 0
        features['airport_zscore_events'] = 0
        return features
    
    # Get all flights from the same airport
    airport_flights = flight_summary_df[
        flight_summary_df.get('airport', pd.Series([None] * len(flight_summary_df))) == airport
    ]
    
    if len(airport_flights) < 2:
        # Not enough data for comparison
        features['airport_deviation_duration'] = 0
        features['airport_deviation_events'] = 0
        features['airport_zscore_duration'] = 0
        features['airport_zscore_events'] = 0
        return features
    
    # Calculate airport norms
    if 'duration_seconds' in airport_flights.columns:
        airport_durations = airport_flights['duration_seconds'].dropna()
        if len(airport_durations) > 0:
            airport_mean_duration = airport_durations.mean()
            airport_std_duration = airport_durations.std()
            
            # Current flight duration
            current_duration = (
                flight_df['timestamp'].max() - flight_df['timestamp'].min()
            ).total_seconds() if 'timestamp' in flight_df.columns else 0
            
            # Deviation from airport norm
            if airport_std_duration > 0:
                features['airport_zscore_duration'] = (current_duration - airport_mean_duration) / airport_std_duration
            else:
                features['airport_zscore_duration'] = 0
            
            features['airport_deviation_duration'] = abs(current_duration - airport_mean_duration)
        else:
            features['airport_zscore_duration'] = 0
            features['airport_deviation_duration'] = 0
    else:
        features['airport_zscore_duration'] = 0
        features['airport_deviation_duration'] = 0
    
    # Event count comparison
    if 'n_events' in airport_flights.columns:
        airport_events = airport_flights['n_events'].dropna()
        if len(airport_events) > 0:
            airport_mean_events = airport_events.mean()
            airport_std_events = airport_events.std()
            
            current_events = len(flight_df)
            
            if airport_std_events > 0:
                features['airport_zscore_events'] = (current_events - airport_mean_events) / airport_std_events
            else:
                features['airport_zscore_events'] = 0
            
            features['airport_deviation_events'] = abs(current_events - airport_mean_events)
        else:
            features['airport_zscore_events'] = 0
            features['airport_deviation_events'] = 0
    else:
        features['airport_zscore_events'] = 0
        features['airport_deviation_events'] = 0
    
    return features


def _extract_time_context(flight_df: pd.DataFrame,
                         flight_summary_df: pd.DataFrame) -> Dict:
    """Extract time-of-day contextual features."""
    features = {}
    
    if 'timestamp' not in flight_df.columns:
        features['time_deviation_duration'] = 0
        features['time_deviation_events'] = 0
        return features
    
    first_timestamp = flight_df['timestamp'].min()
    hour = first_timestamp.hour if hasattr(first_timestamp, 'hour') else 0
    day_of_week = first_timestamp.dayofweek if hasattr(first_timestamp, 'dayofweek') else 0
    
    # Get flights at similar time (same hour ± 1 hour, same day of week)
    if 'first_seen' in flight_summary_df.columns:
        flight_summary_df['hour'] = pd.to_datetime(flight_summary_df['first_seen']).dt.hour
        flight_summary_df['day_of_week'] = pd.to_datetime(flight_summary_df['first_seen']).dt.dayofweek
        
        similar_time_flights = flight_summary_df[
            (flight_summary_df['hour'] >= hour - 1) & 
            (flight_summary_df['hour'] <= hour + 1) &
            (flight_summary_df['day_of_week'] == day_of_week)
        ]
    else:
        similar_time_flights = pd.DataFrame()
    
    if len(similar_time_flights) < 2:
        features['time_deviation_duration'] = 0
        features['time_deviation_events'] = 0
        return features
    
    # Compare duration
    if 'duration_seconds' in similar_time_flights.columns:
        time_durations = similar_time_flights['duration_seconds'].dropna()
        if len(time_durations) > 0:
            time_mean_duration = time_durations.mean()
            current_duration = (
                flight_df['timestamp'].max() - flight_df['timestamp'].min()
            ).total_seconds()
            features['time_deviation_duration'] = abs(current_duration - time_mean_duration)
        else:
            features['time_deviation_duration'] = 0
    else:
        features['time_deviation_duration'] = 0
    
    # Compare event count
    if 'n_events' in similar_time_flights.columns:
        time_events = similar_time_flights['n_events'].dropna()
        if len(time_events) > 0:
            time_mean_events = time_events.mean()
            current_events = len(flight_df)
            features['time_deviation_events'] = abs(current_events - time_mean_events)
        else:
            features['time_deviation_events'] = 0
    else:
        features['time_deviation_events'] = 0
    
    return features


def _extract_peer_comparison(flight_metrics: Dict,
                            flight_summary_df: pd.DataFrame,
                            flight_id: str) -> Dict:
    """Compare flight to all peers (global comparison)."""
    features = {}
    
    # Duration comparison
    if 'duration_seconds' in flight_summary_df.columns:
        all_durations = flight_summary_df['duration_seconds'].dropna()
        if len(all_durations) > 0:
            mean_duration = all_durations.mean()
            std_duration = all_durations.std()
            median_duration = all_durations.median()
            
            current_duration = flight_metrics.get('duration', 0)
            
            if std_duration > 0:
                features['global_zscore_duration'] = (current_duration - mean_duration) / std_duration
            else:
                features['global_zscore_duration'] = 0
            
            features['global_percentile_duration'] = stats.percentileofscore(all_durations, current_duration) / 100
            features['deviation_from_median_duration'] = abs(current_duration - median_duration)
        else:
            features['global_zscore_duration'] = 0
            features['global_percentile_duration'] = 0.5
            features['deviation_from_median_duration'] = 0
    else:
        features['global_zscore_duration'] = 0
        features['global_percentile_duration'] = 0.5
        features['deviation_from_median_duration'] = 0
    
    # Event count comparison
    if 'n_events' in flight_summary_df.columns:
        all_events = flight_summary_df['n_events'].dropna()
        if len(all_events) > 0:
            mean_events = all_events.mean()
            std_events = all_events.std()
            median_events = all_events.median()
            
            current_events = flight_metrics.get('n_events', 0)
            
            if std_events > 0:
                features['global_zscore_events'] = (current_events - mean_events) / std_events
            else:
                features['global_zscore_events'] = 0
            
            features['global_percentile_events'] = stats.percentileofscore(all_events, current_events) / 100
            features['deviation_from_median_events'] = abs(current_events - median_events)
        else:
            features['global_zscore_events'] = 0
            features['global_percentile_events'] = 0.5
            features['deviation_from_median_events'] = 0
    else:
        features['global_zscore_events'] = 0
        features['global_percentile_events'] = 0.5
        features['deviation_from_median_events'] = 0
    
    # Altitude comparison
    if 'max_altitude' in flight_summary_df.columns:
        all_altitudes = flight_summary_df['max_altitude'].dropna()
        if len(all_altitudes) > 0 and flight_metrics.get('max_altitude', 0) > 0:
            mean_altitude = all_altitudes.mean()
            std_altitude = all_altitudes.std()
            
            current_altitude = flight_metrics.get('max_altitude', 0)
            
            if std_altitude > 0:
                features['global_zscore_altitude'] = (current_altitude - mean_altitude) / std_altitude
            else:
                features['global_zscore_altitude'] = 0
        else:
            features['global_zscore_altitude'] = 0
    else:
        features['global_zscore_altitude'] = 0
    
    return features


def extract_contextual_features_batch(flight_summary_df: pd.DataFrame,
                                      events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract contextual features for all flights in batch.
    
    Args:
        flight_summary_df: Flight-level summary DataFrame
        events_df: Event-level DataFrame (sorted by flight and timestamp)
        
    Returns:
        DataFrame with contextual features added
    """
    print("Extracting contextual features for all flights...")
    
    # First, add airport and time info to flight_summary_df if not present
    if 'airport' not in flight_summary_df.columns:
        print("  Extracting airport information...")
        airports = []
        for flight_id in flight_summary_df['flight_id']:
            flight_events = events_df[events_df['flight_id'] == flight_id]
            airport = _extract_airport_from_flight(flight_events)
            airports.append(airport)
        flight_summary_df['airport'] = airports
    
    contextual_features_list = []
    
    for flight_id in flight_summary_df['flight_id'].unique():
        flight_events = events_df[events_df['flight_id'] == flight_id].copy()
        
        if len(flight_events) > 0:
            features = extract_contextual_features(flight_events, flight_summary_df, flight_id)
            features['flight_id'] = flight_id
            contextual_features_list.append(features)
    
    contextual_df = pd.DataFrame(contextual_features_list)
    
    # Merge with flight summary
    result_df = flight_summary_df.merge(contextual_df, on='flight_id', how='left')
    
    print(f"✓ Extracted contextual features for {len(result_df)} flights")
    
    return result_df



