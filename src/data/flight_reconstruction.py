"""
Flight reconstruction utilities.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm


def reconstruct_flights(df, flight_id_col='flight_id', timestamp_col='timestamp'):
    """
    Group events by flight and sort chronologically.
    
    Args:
        df: DataFrame with flight events
        flight_id_col: Column name for flight ID
        timestamp_col: Column name for timestamp
    
    Returns:
        DataFrame sorted by flight and timestamp
    """
    # Convert timestamp to datetime if needed
    if df[timestamp_col].dtype == 'object':
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Sort by flight_id and timestamp
    df_sorted = df.sort_values([flight_id_col, timestamp_col]).reset_index(drop=True)
    
    return df_sorted


def extract_flight_summary(flight_df):
    """
    Extract summary statistics for a single flight.
    
    Args:
        flight_df: DataFrame for a single flight (sorted by timestamp)
    
    Returns:
        Dictionary with flight summary
    """
    if len(flight_df) == 0:
        return None
    
    summary = {
        'flight_id': flight_df['flight_id'].iloc[0],
        'n_events': len(flight_df),
        'first_seen': flight_df['timestamp'].min(),
        'last_seen': flight_df['timestamp'].max(),
        'duration_seconds': (flight_df['timestamp'].max() - flight_df['timestamp'].min()).total_seconds(),
        'event_types': flight_df['event_type'].unique().tolist(),
        'n_event_types': flight_df['event_type'].nunique(),
    }
    
    # Add altitude statistics if available
    if 'altitude' in flight_df.columns:
        altitude_data = flight_df['altitude'].dropna()
        if len(altitude_data) > 0:
            summary.update({
                'min_altitude': altitude_data.min(),
                'max_altitude': altitude_data.max(),
                'mean_altitude': altitude_data.mean(),
                'std_altitude': altitude_data.std()
            })
    
    # Add coordinate range if available
    if 'latitude' in flight_df.columns and 'longitude' in flight_df.columns:
        lat_data = flight_df['latitude'].dropna()
        lon_data = flight_df['longitude'].dropna()
        if len(lat_data) > 0 and len(lon_data) > 0:
            summary.update({
                'lat_min': lat_data.min(),
                'lat_max': lat_data.max(),
                'lon_min': lon_data.min(),
                'lon_max': lon_data.max()
            })
    
    return summary


def create_flight_summary_dataset(df, flight_id_col='flight_id'):
    """
    Create flight-level summary dataset from event-level data.
    
    Args:
        df: DataFrame with flight events (sorted)
        flight_id_col: Column name for flight ID
    
    Returns:
        DataFrame with one row per flight
    """
    print(f"Creating flight summary for {df[flight_id_col].nunique():,} flights...")
    
    summaries = []
    for flight_id, group in tqdm(df.groupby(flight_id_col)):
        summary = extract_flight_summary(group)
        if summary:
            summaries.append(summary)
    
    summary_df = pd.DataFrame(summaries)
    print(f"✓ Created summary dataset with {len(summary_df):,} flights")
    
    return summary_df


def validate_trajectory_consistency(flight_df, max_speed_mps=300):
    """
    Check for impossible trajectories (teleportation).
    
    Args:
        flight_df: DataFrame for a single flight (sorted by timestamp)
        max_speed_mps: Maximum realistic speed in meters per second (300 m/s ≈ 670 mph)
    
    Returns:
        Dictionary with validation results
    """
    if len(flight_df) < 2:
        return {'valid': True, 'issues': []}
    
    from geopy.distance import geodesic
    
    issues = []
    
    # Check consecutive points
    for i in range(len(flight_df) - 1):
        row1 = flight_df.iloc[i]
        row2 = flight_df.iloc[i + 1]
        
        # Skip if coordinates are missing
        if pd.isna(row1['latitude']) or pd.isna(row2['latitude']):
            continue
        
        # Calculate distance
        coords1 = (row1['latitude'], row1['longitude'])
        coords2 = (row2['latitude'], row2['longitude'])
        distance_m = geodesic(coords1, coords2).meters
        
        # Calculate time difference
        time_diff_s = (row2['timestamp'] - row1['timestamp']).total_seconds()
        
        if time_diff_s > 0:
            speed_mps = distance_m / time_diff_s
            
            if speed_mps > max_speed_mps:
                issues.append({
                    'event_index': i,
                    'speed_mps': speed_mps,
                    'distance_m': distance_m,
                    'time_diff_s': time_diff_s
                })
    
    return {
        'valid': len(issues) == 0,
        'n_issues': len(issues),
        'issues': issues[:5]  # Return first 5 issues
    }


def add_temporal_features(flight_df):
    """
    Add temporal features to flight events.
    
    Args:
        flight_df: DataFrame for a single flight (sorted by timestamp)
    
    Returns:
        DataFrame with additional temporal features
    """
    flight_df = flight_df.copy()
    
    # Time since first event
    flight_df['time_since_start'] = (
        flight_df['timestamp'] - flight_df['timestamp'].iloc[0]
    ).dt.total_seconds()
    
    # Time between consecutive events
    flight_df['time_delta'] = flight_df['timestamp'].diff().dt.total_seconds()
    
    # Event sequence number
    flight_df['event_sequence'] = range(len(flight_df))
    
    return flight_df


def add_spatial_features(flight_df):
    """
    Add spatial features to flight events.
    
    Args:
        flight_df: DataFrame for a single flight (sorted by timestamp)
    
    Returns:
        DataFrame with additional spatial features
    """
    from geopy.distance import geodesic
    
    flight_df = flight_df.copy()
    
    # Distance from previous point
    distances = [0]  # First point has 0 distance
    
    for i in range(1, len(flight_df)):
        if (pd.notna(flight_df.iloc[i]['latitude']) and 
            pd.notna(flight_df.iloc[i-1]['latitude'])):
            coords1 = (flight_df.iloc[i-1]['latitude'], flight_df.iloc[i-1]['longitude'])
            coords2 = (flight_df.iloc[i]['latitude'], flight_df.iloc[i]['longitude'])
            dist = geodesic(coords1, coords2).meters
            distances.append(dist)
        else:
            distances.append(np.nan)
    
    flight_df['distance_from_prev'] = distances
    
    # Cumulative distance
    flight_df['cumulative_distance'] = flight_df['distance_from_prev'].cumsum()
    
    # Altitude change
    if 'altitude' in flight_df.columns:
        flight_df['altitude_change'] = flight_df['altitude'].diff()
    
    return flight_df
