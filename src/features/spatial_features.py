"""
Spatial feature engineering for flight anomaly detection.
Extracts geographic and trajectory-based features.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from geopy.distance import geodesic


def extract_spatial_features(flight_df: pd.DataFrame) -> Dict:
    """
    Extract spatial features from a single flight's trajectory.
    
    Args:
        flight_df: DataFrame for a single flight, sorted by timestamp
        
    Returns:
        Dictionary of spatial features
    """
    if len(flight_df) == 0:
        return {}
    
    features = {}
    
    # Basic spatial statistics
    if 'latitude' in flight_df.columns and 'longitude' in flight_df.columns:
        valid_coords = flight_df[['latitude', 'longitude']].dropna()
        
        if len(valid_coords) > 0:
            features.update(_extract_coordinate_features(valid_coords))
            features.update(_extract_distance_features(flight_df))
            features.update(_extract_trajectory_features(flight_df))
        else:
            # No valid coordinates
            features['has_valid_coordinates'] = 0
    else:
        features['has_valid_coordinates'] = 0
    
    # Altitude features
    if 'altitude' in flight_df.columns:
        features.update(_extract_altitude_features(flight_df))
    
    return features


def _extract_coordinate_features(coords_df: pd.DataFrame) -> Dict:
    """Extract basic coordinate statistics."""
    features = {}
    
    features['has_valid_coordinates'] = 1
    features['lat_min'] = coords_df['latitude'].min()
    features['lat_max'] = coords_df['latitude'].max()
    features['lat_mean'] = coords_df['latitude'].mean()
    features['lat_std'] = coords_df['latitude'].std()
    features['lat_range'] = features['lat_max'] - features['lat_min']
    
    features['lon_min'] = coords_df['longitude'].min()
    features['lon_max'] = coords_df['longitude'].max()
    features['lon_mean'] = coords_df['longitude'].mean()
    features['lon_std'] = coords_df['longitude'].std()
    features['lon_range'] = features['lon_max'] - features['lon_min']
    
    # Geographic center
    features['centroid_lat'] = features['lat_mean']
    features['centroid_lon'] = features['lon_mean']
    
    return features


def _extract_distance_features(flight_df: pd.DataFrame) -> Dict:
    """Extract distance-based features."""
    features = {}
    
    valid_coords = flight_df[['latitude', 'longitude', 'timestamp']].dropna()
    
    if len(valid_coords) < 2:
        features['total_distance_meters'] = 0
        features['straight_line_distance_meters'] = 0
        features['avg_distance_per_event'] = 0
        return features
    
    # Sort by timestamp
    valid_coords = valid_coords.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate distances between consecutive points
    distances = []
    for i in range(len(valid_coords) - 1):
        coord1 = (valid_coords.iloc[i]['latitude'], valid_coords.iloc[i]['longitude'])
        coord2 = (valid_coords.iloc[i+1]['latitude'], valid_coords.iloc[i+1]['longitude'])
        dist = geodesic(coord1, coord2).meters
        distances.append(dist)
    
    if len(distances) > 0:
        features['total_distance_meters'] = sum(distances)
        features['avg_distance_per_event'] = np.mean(distances)
        features['max_distance_per_event'] = np.max(distances)
        features['min_distance_per_event'] = np.min(distances)
        features['std_distance_per_event'] = np.std(distances)
    else:
        features['total_distance_meters'] = 0
        features['avg_distance_per_event'] = 0
        features['max_distance_per_event'] = 0
        features['min_distance_per_event'] = 0
        features['std_distance_per_event'] = 0
    
    # Straight-line distance (first to last point)
    first_coord = (valid_coords.iloc[0]['latitude'], valid_coords.iloc[0]['longitude'])
    last_coord = (valid_coords.iloc[-1]['latitude'], valid_coords.iloc[-1]['longitude'])
    features['straight_line_distance_meters'] = geodesic(first_coord, last_coord).meters
    
    # Path efficiency (straight line / total distance)
    if features['total_distance_meters'] > 0:
        features['path_efficiency'] = features['straight_line_distance_meters'] / features['total_distance_meters']
    else:
        features['path_efficiency'] = 0
    
    return features


def _extract_trajectory_features(flight_df: pd.DataFrame) -> Dict:
    """Extract trajectory complexity features."""
    features = {}
    
    valid_coords = flight_df[['latitude', 'longitude', 'timestamp']].dropna()
    
    if len(valid_coords) < 3:
        features['trajectory_sinuosity'] = 0
        features['num_direction_changes'] = 0
        features['avg_turning_angle'] = 0
        return features
    
    valid_coords = valid_coords.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate bearings (directions) between consecutive points
    bearings = []
    for i in range(len(valid_coords) - 1):
        coord1 = (valid_coords.iloc[i]['latitude'], valid_coords.iloc[i]['longitude'])
        coord2 = (valid_coords.iloc[i+1]['latitude'], valid_coords.iloc[i+1]['longitude'])
        
        # Calculate bearing using geopy
        bearing = geodesic(coord1, coord2).initial_bearing
        bearings.append(bearing)
    
    if len(bearings) < 2:
        features['trajectory_sinuosity'] = 0
        features['num_direction_changes'] = 0
        features['avg_turning_angle'] = 0
        return features
    
    # Calculate turning angles (change in bearing)
    turning_angles = []
    for i in range(len(bearings) - 1):
        angle_diff = abs(bearings[i+1] - bearings[i])
        # Normalize to 0-180 degrees
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        turning_angles.append(angle_diff)
    
    if len(turning_angles) > 0:
        # Count significant direction changes (> 30 degrees)
        features['num_direction_changes'] = sum(1 for angle in turning_angles if angle > 30)
        features['avg_turning_angle'] = np.mean(turning_angles)
        features['max_turning_angle'] = np.max(turning_angles)
        features['std_turning_angle'] = np.std(turning_angles)
    else:
        features['num_direction_changes'] = 0
        features['avg_turning_angle'] = 0
        features['max_turning_angle'] = 0
        features['std_turning_angle'] = 0
    
    # Trajectory sinuosity (ratio of path length to straight-line distance)
    # Already calculated in distance features, but we can refine it
    total_dist = sum([
        geodesic(
            (valid_coords.iloc[i]['latitude'], valid_coords.iloc[i]['longitude']),
            (valid_coords.iloc[i+1]['latitude'], valid_coords.iloc[i+1]['longitude'])
        ).meters
        for i in range(len(valid_coords) - 1)
    ])
    
    straight_dist = geodesic(
        (valid_coords.iloc[0]['latitude'], valid_coords.iloc[0]['longitude']),
        (valid_coords.iloc[-1]['latitude'], valid_coords.iloc[-1]['longitude'])
    ).meters
    
    if straight_dist > 0:
        features['trajectory_sinuosity'] = total_dist / straight_dist
    else:
        features['trajectory_sinuosity'] = 0
    
    return features


def _extract_altitude_features(flight_df: pd.DataFrame) -> Dict:
    """Extract altitude-based features."""
    features = {}
    
    altitude_data = flight_df['altitude'].dropna()
    
    if len(altitude_data) == 0:
        features['has_altitude_data'] = 0
        features['altitude_min'] = 0
        features['altitude_max'] = 0
        features['altitude_mean'] = 0
        features['altitude_std'] = 0
        features['altitude_range'] = 0
        features['num_altitude_changes'] = 0
        features['avg_altitude_change_rate'] = 0
        return features
    
    features['has_altitude_data'] = 1
    features['altitude_min'] = altitude_data.min()
    features['altitude_max'] = altitude_data.max()
    features['altitude_mean'] = altitude_data.mean()
    features['altitude_median'] = altitude_data.median()
    features['altitude_std'] = altitude_data.std()
    features['altitude_range'] = features['altitude_max'] - features['altitude_min']
    
    # Altitude changes (if timestamp available)
    if 'timestamp' in flight_df.columns:
        alt_with_time = flight_df[['altitude', 'timestamp']].dropna().sort_values('timestamp')
        
        if len(alt_with_time) > 1:
            altitude_changes = alt_with_time['altitude'].diff().dropna()
            time_diffs = alt_with_time['timestamp'].diff().dt.total_seconds().dropna()
            
            # Count significant altitude changes (> 100m)
            features['num_altitude_changes'] = sum(1 for change in altitude_changes if abs(change) > 100)
            
            # Calculate altitude change rate
            valid_pairs = pd.DataFrame({
                'alt_change': altitude_changes,
                'time_diff': time_diffs
            }).dropna()
            
            if len(valid_pairs) > 0:
                valid_pairs = valid_pairs[valid_pairs['time_diff'] > 0]
                if len(valid_pairs) > 0:
                    rates = valid_pairs['alt_change'] / valid_pairs['time_diff']
                    features['avg_altitude_change_rate'] = rates.mean()
                    features['max_altitude_change_rate'] = rates.max()
                    features['min_altitude_change_rate'] = rates.min()
                else:
                    features['avg_altitude_change_rate'] = 0
                    features['max_altitude_change_rate'] = 0
                    features['min_altitude_change_rate'] = 0
            else:
                features['avg_altitude_change_rate'] = 0
                features['max_altitude_change_rate'] = 0
                features['min_altitude_change_rate'] = 0
            
            # Count altitude crossings (e.g., flight level crossings)
            # Count transitions across common flight levels (1000, 2000, 5000, 10000, 20000 feet)
            flight_levels = [304.8, 609.6, 1524, 3048, 6096]  # meters
            crossings = 0
            prev_alt = None
            for alt in alt_with_time['altitude']:
                if prev_alt is not None:
                    for level in flight_levels:
                        if (prev_alt < level <= alt) or (prev_alt > level >= alt):
                            crossings += 1
                prev_alt = alt
            
            features['num_altitude_crossings'] = crossings
        else:
            features['num_altitude_changes'] = 0
            features['avg_altitude_change_rate'] = 0
            features['max_altitude_change_rate'] = 0
            features['min_altitude_change_rate'] = 0
            features['num_altitude_crossings'] = 0
    else:
        features['num_altitude_changes'] = 0
        features['avg_altitude_change_rate'] = 0
        features['max_altitude_change_rate'] = 0
        features['min_altitude_change_rate'] = 0
        features['num_altitude_crossings'] = 0
    
    return features


def extract_spatial_features_batch(flight_summary_df: pd.DataFrame,
                                     events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract spatial features for all flights in batch.
    
    Args:
        flight_summary_df: Flight-level summary DataFrame
        events_df: Event-level DataFrame (sorted by flight and timestamp)
        
    Returns:
        DataFrame with spatial features added
    """
    print("Extracting spatial features for all flights...")
    
    spatial_features_list = []
    
    for flight_id in flight_summary_df['flight_id'].unique():
        flight_events = events_df[events_df['flight_id'] == flight_id].copy()
        
        if len(flight_events) > 0:
            features = extract_spatial_features(flight_events)
            features['flight_id'] = flight_id
            spatial_features_list.append(features)
    
    spatial_df = pd.DataFrame(spatial_features_list)
    
    # Merge with flight summary
    result_df = flight_summary_df.merge(spatial_df, on='flight_id', how='left')
    
    print(f"✓ Extracted spatial features for {len(result_df)} flights")
    
    return result_df



