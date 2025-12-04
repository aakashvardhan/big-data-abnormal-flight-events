"""
Data quality assessment functions.
"""
import pandas as pd
import numpy as np
from datetime import datetime


def check_missing_values(df):
    """
    Check for missing values in the dataset.
    
    Args:
        df: DataFrame to check
    
    Returns:
        DataFrame with missing value statistics
    """
    missing_stats = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_percent': (df.isnull().sum() / len(df) * 100).values,
        'dtype': df.dtypes.values
    })
    
    missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values(
        'missing_percent', ascending=False
    )
    
    return missing_stats


def check_duplicates(df, subset=None):
    """
    Check for duplicate records.
    
    Args:
        df: DataFrame to check
        subset: Columns to check for duplicates (None for all columns)
    
    Returns:
        Dictionary with duplicate statistics
    """
    n_duplicates = df.duplicated(subset=subset).sum()
    duplicate_percent = (n_duplicates / len(df)) * 100
    
    return {
        'n_duplicates': n_duplicates,
        'duplicate_percent': duplicate_percent,
        'subset': subset
    }


def validate_timestamps(df, timestamp_col='timestamp'):
    """
    Validate timestamp data.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
    
    Returns:
        Dictionary with validation results
    """
    # Convert to datetime if not already
    if df[timestamp_col].dtype == 'object':
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    
    invalid_timestamps = df[timestamp_col].isnull().sum()
    
    valid_data = df[df[timestamp_col].notna()]
    
    return {
        'invalid_count': invalid_timestamps,
        'invalid_percent': (invalid_timestamps / len(df)) * 100,
        'min_timestamp': valid_data[timestamp_col].min(),
        'max_timestamp': valid_data[timestamp_col].max(),
        'date_range_days': (valid_data[timestamp_col].max() - valid_data[timestamp_col].min()).days
    }


def validate_coordinates(df, lat_col='latitude', lon_col='longitude'):
    """
    Validate geographic coordinates.
    
    Args:
        df: DataFrame with coordinate columns
        lat_col: Latitude column name
        lon_col: Longitude column name
    
    Returns:
        Dictionary with validation results
    """
    results = {}
    
    # Check latitude range (-90 to 90)
    if lat_col in df.columns:
        invalid_lat = ((df[lat_col] < -90) | (df[lat_col] > 90)).sum()
        results['invalid_latitude'] = invalid_lat
        results['latitude_range'] = (df[lat_col].min(), df[lat_col].max())
    
    # Check longitude range (-180 to 180)
    if lon_col in df.columns:
        invalid_lon = ((df[lon_col] < -180) | (df[lon_col] > 180)).sum()
        results['invalid_longitude'] = invalid_lon
        results['longitude_range'] = (df[lon_col].min(), df[lon_col].max())
    
    # Check for missing coordinates
    if lat_col in df.columns and lon_col in df.columns:
        missing_coords = (df[lat_col].isnull() | df[lon_col].isnull()).sum()
        results['missing_coordinates'] = missing_coords
        results['missing_coords_percent'] = (missing_coords / len(df)) * 100
    
    return results


def validate_altitude(df, alt_col='altitude'):
    """
    Validate altitude data.
    
    Args:
        df: DataFrame with altitude column
        alt_col: Altitude column name
    
    Returns:
        Dictionary with validation results
    """
    if alt_col not in df.columns:
        return {'error': f'Column {alt_col} not found'}
    
    # Check for negative altitudes (suspicious but possible for ground operations)
    negative_alt = (df[alt_col] < 0).sum()
    
    # Check for unrealistic altitudes (> 50,000 feet = ~15,240 meters)
    unrealistic_alt = (df[alt_col] > 15240).sum()
    
    # Missing altitudes
    missing_alt = df[alt_col].isnull().sum()
    
    return {
        'negative_altitude': negative_alt,
        'unrealistic_altitude': unrealistic_alt,
        'missing_altitude': missing_alt,
        'missing_altitude_percent': (missing_alt / len(df)) * 100,
        'altitude_range': (df[alt_col].min(), df[alt_col].max()),
        'altitude_stats': df[alt_col].describe().to_dict()
    }


def check_event_sequence_validity(flight_df):
    """
    Check if event sequence for a single flight is logically ordered.
    
    Args:
        flight_df: DataFrame for a single flight (sorted by timestamp)
    
    Returns:
        Dictionary with validation results
    """
    issues = []
    
    # Check for timestamp ordering
    if not flight_df['timestamp'].is_monotonic_increasing:
        issues.append('timestamps_not_ordered')
    
    # Check for duplicate timestamps
    if flight_df['timestamp'].duplicated().any():
        issues.append('duplicate_timestamps')
    
    # Check for required events (customize based on domain knowledge)
    event_types = set(flight_df['event_type'].unique())
    
    # Example checks (adjust based on actual event types)
    if 'take-off' in event_types and 'landing' not in event_types:
        issues.append('missing_landing')
    
    if 'landing' in event_types and 'take-off' not in event_types:
        issues.append('missing_takeoff')
    
    return {
        'has_issues': len(issues) > 0,
        'issues': issues,
        'n_events': len(flight_df)
    }


def identify_incomplete_flights(df, group_col='flight_id'):
    """
    Identify flights with missing critical information.
    
    Args:
        df: DataFrame with flight data
        group_col: Column to group by (flight_id)
    
    Returns:
        DataFrame with incomplete flight statistics
    """
    flight_stats = df.groupby(group_col).agg({
        'timestamp': ['min', 'max', 'count'],
        'event_type': 'nunique'
    }).reset_index()
    
    flight_stats.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] 
                           for col in flight_stats.columns]
    
    # Identify potentially incomplete flights
    # Customize thresholds based on domain knowledge
    flight_stats['too_few_events'] = flight_stats['event_type_nunique'] < 3
    flight_stats['single_event'] = flight_stats['timestamp_count'] == 1
    
    incomplete = flight_stats[flight_stats['too_few_events'] | flight_stats['single_event']]
    
    return incomplete


def generate_quality_report(df):
    """
    Generate comprehensive data quality report.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_records': len(df),
        'missing_values': check_missing_values(df).to_dict('records'),
        'duplicates': check_duplicates(df),
        'timestamp_validation': validate_timestamps(df) if 'timestamp' in df.columns else None,
        'coordinate_validation': validate_coordinates(df) if 'latitude' in df.columns else None,
        'altitude_validation': validate_altitude(df) if 'altitude' in df.columns else None
    }
    
    return report
