"""
Operational feature engineering for flight anomaly detection.
Extracts runway, taxiway, parking, and ground operation features.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json


def extract_operational_features(flight_df: pd.DataFrame) -> Dict:
    """
    Extract operational features from a single flight's events.
    
    Args:
        flight_df: DataFrame for a single flight, sorted by timestamp
        
    Returns:
        Dictionary of operational features
    """
    if len(flight_df) == 0:
        return {}
    
    features = {}
    
    # Event type counts
    features.update(_extract_event_type_counts(flight_df))
    
    # Runway operations
    features.update(_extract_runway_features(flight_df))
    
    # Taxiway operations
    features.update(_extract_taxiway_features(flight_df))
    
    # Parking/stand operations
    features.update(_extract_parking_features(flight_df))
    
    # Ground complexity
    features.update(_extract_ground_complexity(flight_df))
    
    # Airport operations
    features.update(_extract_airport_features(flight_df))
    
    return features


def _extract_event_type_counts(flight_df: pd.DataFrame) -> Dict:
    """Count occurrences of different event types."""
    features = {}
    
    event_counts = flight_df['event_type'].value_counts().to_dict()
    
    # Common event type categories (adjust based on actual data)
    event_categories = {
        'ground': ['parking', 'gate', 'stand', 'ramp', 'taxi', 'taxiway'],
        'runway': ['runway', 'takeoff', 'take-off', 'landing', 'touchdown'],
        'airborne': ['climb', 'cruise', 'descent', 'level', 'altitude'],
        'transition': ['transition', 'crossing', 'entry', 'exit']
    }
    
    for category, keywords in event_categories.items():
        count = sum(
            event_counts.get(et, 0) 
            for et in event_counts.keys()
            if any(kw in str(et).lower() for kw in keywords)
        )
        features[f'{category}_event_count'] = count
    
    # Total unique event types
    features['unique_event_types'] = flight_df['event_type'].nunique()
    
    # Most common event type
    if len(event_counts) > 0:
        most_common = max(event_counts.items(), key=lambda x: x[1])
        features['most_common_event_type'] = most_common[0]
        features['most_common_event_count'] = most_common[1]
    else:
        features['most_common_event_type'] = None
        features['most_common_event_count'] = 0
    
    return features


def _extract_runway_features(flight_df: pd.DataFrame) -> Dict:
    """Extract runway-related features."""
    features = {}
    
    # Identify runway events
    runway_keywords = ['runway', 'takeoff', 'take-off', 'landing', 'touchdown', 'departure', 'arrival']
    runway_events = flight_df[
        flight_df['event_type'].str.lower().str.contains('|'.join(runway_keywords), na=False, regex=True)
    ]
    
    features['num_runway_events'] = len(runway_events)
    
    # Extract runway information from info field if available
    if 'info' in flight_df.columns or 'info_parsed' in flight_df.columns:
        runways = []
        for idx, row in runway_events.iterrows():
            runway = _extract_runway_from_info(row)
            if runway:
                runways.append(runway)
        
        if runways:
            features['num_unique_runways'] = len(set(runways))
            features['runway_changes'] = len([i for i in range(1, len(runways)) if runways[i] != runways[i-1]])
            features['most_used_runway'] = max(set(runways), key=runways.count) if runways else None
        else:
            features['num_unique_runways'] = 0
            features['runway_changes'] = 0
            features['most_used_runway'] = None
    else:
        features['num_unique_runways'] = 0
        features['runway_changes'] = 0
        features['most_used_runway'] = None
    
    # Takeoff and landing counts
    takeoff_events = flight_df[
        flight_df['event_type'].str.lower().str.contains('takeoff|take-off|departure', na=False, regex=True)
    ]
    landing_events = flight_df[
        flight_df['event_type'].str.lower().str.contains('landing|touchdown|arrival', na=False, regex=True)
    ]
    
    features['num_takeoffs'] = len(takeoff_events)
    features['num_landings'] = len(landing_events)
    features['num_go_arounds'] = max(0, features['num_landings'] - 1)  # Multiple landings = go-arounds
    
    return features


def _extract_taxiway_features(flight_df: pd.DataFrame) -> Dict:
    """Extract taxiway-related features."""
    features = {}
    
    # Identify taxiway events
    taxiway_keywords = ['taxi', 'taxiway']
    taxiway_events = flight_df[
        flight_df['event_type'].str.lower().str.contains('|'.join(taxiway_keywords), na=False, regex=True)
    ]
    
    features['num_taxiway_events'] = len(taxiway_events)
    
    # Extract taxiway information from info field
    if 'info' in flight_df.columns or 'info_parsed' in flight_df.columns:
        taxiways = []
        for idx, row in taxiway_events.iterrows():
            taxiway = _extract_taxiway_from_info(row)
            if taxiway:
                taxiways.append(taxiway)
        
        if taxiways:
            features['num_unique_taxiways'] = len(set(taxiways))
            features['taxiway_changes'] = len([i for i in range(1, len(taxiways)) if taxiways[i] != taxiways[i-1]])
        else:
            features['num_unique_taxiways'] = 0
            features['taxiway_changes'] = 0
    else:
        features['num_unique_taxiways'] = 0
        features['taxiway_changes'] = 0
    
    return features


def _extract_parking_features(flight_df: pd.DataFrame) -> Dict:
    """Extract parking/stand-related features."""
    features = {}
    
    # Identify parking events
    parking_keywords = ['parking', 'gate', 'stand', 'ramp']
    parking_events = flight_df[
        flight_df['event_type'].str.lower().str.contains('|'.join(parking_keywords), na=False, regex=True)
    ]
    
    features['num_parking_events'] = len(parking_events)
    
    # Extract parking information from info field
    if 'info' in flight_df.columns or 'info_parsed' in flight_df.columns:
        parkings = []
        for idx, row in parking_events.iterrows():
            parking = _extract_parking_from_info(row)
            if parking:
                parkings.append(parking)
        
        if parkings:
            features['num_unique_parkings'] = len(set(parkings))
            features['parking_changes'] = len([i for i in range(1, len(parkings)) if parkings[i] != parkings[i-1]])
        else:
            features['num_unique_parkings'] = 0
            features['parking_changes'] = 0
    else:
        features['num_unique_parkings'] = 0
        features['parking_changes'] = 0
    
    return features


def _extract_ground_complexity(flight_df: pd.DataFrame) -> Dict:
    """Calculate ground operation complexity metrics."""
    features = {}
    
    # Identify ground events (low altitude or ground-specific)
    if 'altitude' in flight_df.columns:
        ground_events = flight_df[flight_df['altitude'] <= 50]  # Below 50m
    else:
        ground_keywords = ['parking', 'gate', 'stand', 'taxi', 'ramp', 'taxiway']
        ground_events = flight_df[
            flight_df['event_type'].str.lower().str.contains('|'.join(ground_keywords), na=False, regex=True)
        ]
    
    features['ground_event_count'] = len(ground_events)
    
    # Ground complexity score (combination of factors)
    complexity_score = 0
    
    # Factor 1: Number of different ground locations
    if 'info' in flight_df.columns or 'info_parsed' in flight_df.columns:
        locations = set()
        for idx, row in ground_events.iterrows():
            runway = _extract_runway_from_info(row)
            taxiway = _extract_taxiway_from_info(row)
            parking = _extract_parking_from_info(row)
            if runway:
                locations.add(f"RWY_{runway}")
            if taxiway:
                locations.add(f"TWY_{taxiway}")
            if parking:
                locations.add(f"PARK_{parking}")
        
        complexity_score += len(locations) * 2
    
    # Factor 2: Number of transitions between ground locations
    if len(ground_events) > 1:
        transitions = 0
        prev_location = None
        for idx, row in ground_events.iterrows():
            current_location = _get_ground_location(row)
            if prev_location and current_location and prev_location != current_location:
                transitions += 1
            prev_location = current_location
        
        complexity_score += transitions
    
    # Factor 3: Number of unique event types during ground operations
    unique_ground_events = ground_events['event_type'].nunique()
    complexity_score += unique_ground_events
    
    features['ground_complexity_score'] = complexity_score
    
    # Ground operation duration (if timestamp available)
    if 'timestamp' in flight_df.columns and len(ground_events) > 0:
        ground_duration = (
            ground_events['timestamp'].max() - ground_events['timestamp'].min()
        ).total_seconds()
        features['ground_duration_seconds'] = ground_duration
    else:
        features['ground_duration_seconds'] = 0
    
    return features


def _extract_airport_features(flight_df: pd.DataFrame) -> Dict:
    """Extract airport-related features."""
    features = {}
    
    # Extract airport from info field
    airports = []
    if 'info' in flight_df.columns or 'info_parsed' in flight_df.columns:
        for idx, row in flight_df.iterrows():
            airport = _extract_airport_from_info(row)
            if airport:
                airports.append(airport)
    
    if airports:
        unique_airports = list(set(airports))
        features['num_airports'] = len(unique_airports)
        features['primary_airport'] = max(set(airports), key=airports.count) if airports else None
        features['airport_changes'] = len([i for i in range(1, len(airports)) if airports[i] != airports[i-1]])
    else:
        features['num_airports'] = 0
        features['primary_airport'] = None
        features['airport_changes'] = 0
    
    return features


def _extract_runway_from_info(row: pd.Series) -> Optional[str]:
    """Extract runway identifier from info field."""
    if 'runway' in row and pd.notna(row['runway']):
        return str(row['runway'])
    
    if 'info_parsed' in row and pd.notna(row['info_parsed']):
        if isinstance(row['info_parsed'], dict):
            return row['info_parsed'].get('runway')
        elif isinstance(row['info_parsed'], str):
            try:
                info_dict = json.loads(row['info_parsed'])
                return info_dict.get('runway')
            except:
                pass
    
    if 'info' in row and pd.notna(row['info']):
        try:
            if isinstance(row['info'], str):
                info_dict = json.loads(row['info'])
                return info_dict.get('runway')
        except:
            pass
    
    return None


def _extract_taxiway_from_info(row: pd.Series) -> Optional[str]:
    """Extract taxiway identifier from info field."""
    if 'taxiway' in row and pd.notna(row['taxiway']):
        return str(row['taxiway'])
    
    if 'info_parsed' in row and pd.notna(row['info_parsed']):
        if isinstance(row['info_parsed'], dict):
            return row['info_parsed'].get('taxiway')
        elif isinstance(row['info_parsed'], str):
            try:
                info_dict = json.loads(row['info_parsed'])
                return info_dict.get('taxiway')
            except:
                pass
    
    if 'info' in row and pd.notna(row['info']):
        try:
            if isinstance(row['info'], str):
                info_dict = json.loads(row['info'])
                return info_dict.get('taxiway')
        except:
            pass
    
    return None


def _extract_parking_from_info(row: pd.Series) -> Optional[str]:
    """Extract parking/stand identifier from info field."""
    if 'parking' in row and pd.notna(row['parking']):
        return str(row['parking'])
    
    if 'info_parsed' in row and pd.notna(row['info_parsed']):
        if isinstance(row['info_parsed'], dict):
            return row['info_parsed'].get('parking') or row['info_parsed'].get('gate') or row['info_parsed'].get('stand')
        elif isinstance(row['info_parsed'], str):
            try:
                info_dict = json.loads(row['info_parsed'])
                return info_dict.get('parking') or info_dict.get('gate') or info_dict.get('stand')
            except:
                pass
    
    if 'info' in row and pd.notna(row['info']):
        try:
            if isinstance(row['info'], str):
                info_dict = json.loads(row['info'])
                return info_dict.get('parking') or info_dict.get('gate') or info_dict.get('stand')
        except:
            pass
    
    return None


def _extract_airport_from_info(row: pd.Series) -> Optional[str]:
    """Extract airport identifier from info field."""
    if 'airport' in row and pd.notna(row['airport']):
        return str(row['airport'])
    
    if 'info_parsed' in row and pd.notna(row['info_parsed']):
        if isinstance(row['info_parsed'], dict):
            return row['info_parsed'].get('airport')
        elif isinstance(row['info_parsed'], str):
            try:
                info_dict = json.loads(row['info_parsed'])
                return info_dict.get('airport')
            except:
                pass
    
    if 'info' in row and pd.notna(row['info']):
        try:
            if isinstance(row['info'], str):
                info_dict = json.loads(row['info'])
                return info_dict.get('airport')
        except:
            pass
    
    return None


def _get_ground_location(row: pd.Series) -> Optional[str]:
    """Get a unique identifier for the ground location."""
    runway = _extract_runway_from_info(row)
    taxiway = _extract_taxiway_from_info(row)
    parking = _extract_parking_from_info(row)
    
    if runway:
        return f"RWY_{runway}"
    elif taxiway:
        return f"TWY_{taxiway}"
    elif parking:
        return f"PARK_{parking}"
    else:
        return None


def extract_operational_features_batch(flight_summary_df: pd.DataFrame,
                                        events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract operational features for all flights in batch.
    
    Args:
        flight_summary_df: Flight-level summary DataFrame
        events_df: Event-level DataFrame (sorted by flight and timestamp)
        
    Returns:
        DataFrame with operational features added
    """
    print("Extracting operational features for all flights...")
    
    operational_features_list = []
    
    for flight_id in flight_summary_df['flight_id'].unique():
        flight_events = events_df[events_df['flight_id'] == flight_id].copy()
        
        if len(flight_events) > 0:
            features = extract_operational_features(flight_events)
            features['flight_id'] = flight_id
            operational_features_list.append(features)
    
    operational_df = pd.DataFrame(operational_features_list)
    
    # Merge with flight summary
    result_df = flight_summary_df.merge(operational_df, on='flight_id', how='left')
    
    print(f"✓ Extracted operational features for {len(result_df)} flights")
    
    return result_df



