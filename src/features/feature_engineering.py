"""
Main feature engineering orchestrator.
Combines all feature extraction modules into a unified pipeline.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import yaml

from .temporal_features import extract_temporal_features_batch
from .spatial_features import extract_spatial_features_batch
from .operational_features import extract_operational_features_batch
from .sequence_features import extract_sequence_features_batch
from .contextual_features import extract_contextual_features_batch


def load_processed_data(processed_dir: str = 'data/processed') -> tuple:
    """
    Load processed data from Phase 1.
    
    Args:
        processed_dir: Directory containing processed data
        
    Returns:
        Tuple of (events_df, flight_summary_df)
    """
    processed_path = Path(processed_dir)
    
    # Load events
    events_file = processed_path / 'events_sorted.csv.gz'
    if not events_file.exists():
        raise FileNotFoundError(
            f"Processed events file not found: {events_file}\n"
            "Please run Phase 1 notebook first to generate processed data."
        )
    
    print(f"Loading events from {events_file}...")
    events_df = pd.read_csv(events_file, compression='gzip')
    
    # Convert timestamp to datetime
    if 'timestamp' in events_df.columns:
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    
    print(f"✓ Loaded {len(events_df):,} events")
    
    # Load flight summary
    summary_file = processed_path / 'flight_summary.csv.gz'
    if not summary_file.exists():
        raise FileNotFoundError(
            f"Flight summary file not found: {summary_file}\n"
            "Please run Phase 1 notebook first to generate processed data."
        )
    
    print(f"Loading flight summary from {summary_file}...")
    flight_summary_df = pd.read_csv(summary_file, compression='gzip')
    
    # Convert timestamp columns to datetime
    for col in ['first_seen', 'last_seen']:
        if col in flight_summary_df.columns:
            flight_summary_df[col] = pd.to_datetime(flight_summary_df[col])
    
    print(f"✓ Loaded {len(flight_summary_df):,} flights")
    
    return events_df, flight_summary_df


def extract_all_features(events_df: pd.DataFrame,
                        flight_summary_df: pd.DataFrame,
                        feature_types: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extract all features for all flights.
    
    Args:
        events_df: Event-level DataFrame (sorted by flight and timestamp)
        flight_summary_df: Flight-level summary DataFrame
        feature_types: List of feature types to extract. If None, extracts all.
                      Options: 'temporal', 'spatial', 'operational', 'sequence', 'contextual'
        
    Returns:
        DataFrame with all features
    """
    if feature_types is None:
        feature_types = ['temporal', 'spatial', 'operational', 'sequence', 'contextual']
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Start with flight summary
    features_df = flight_summary_df.copy()
    
    # Extract each feature type
    if 'temporal' in feature_types:
        print("\n[1/5] Extracting temporal features...")
        features_df = extract_temporal_features_batch(features_df, events_df)
    
    if 'spatial' in feature_types:
        print("\n[2/5] Extracting spatial features...")
        features_df = extract_spatial_features_batch(features_df, events_df)
    
    if 'operational' in feature_types:
        print("\n[3/5] Extracting operational features...")
        features_df = extract_operational_features_batch(features_df, events_df)
    
    if 'sequence' in feature_types:
        print("\n[4/5] Extracting sequence features...")
        features_df = extract_sequence_features_batch(features_df, events_df)
    
    if 'contextual' in feature_types:
        print("\n[5/5] Extracting contextual features...")
        features_df = extract_contextual_features_batch(features_df, events_df)
    
    print("\n" + "="*60)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*60)
    print(f"Total features: {len(features_df.columns)}")
    print(f"Total flights: {len(features_df)}")
    
    return features_df


def prepare_features_for_ml(features_df: pd.DataFrame,
                            target_col: Optional[str] = None,
                            exclude_cols: Optional[List[str]] = None) -> tuple:
    """
    Prepare features for machine learning (handle missing values, encoding, scaling).
    
    Args:
        features_df: DataFrame with extracted features
        target_col: Name of target column (if any)
        exclude_cols: Columns to exclude from features
        
    Returns:
        Tuple of (X, y, feature_names, feature_info)
    """
    print("\nPreparing features for machine learning...")
    
    # Default columns to exclude
    default_exclude = ['flight_id', 'first_seen', 'last_seen', 'event_types']
    if exclude_cols:
        default_exclude.extend(exclude_cols)
    
    # Identify feature columns
    feature_cols = [col for col in features_df.columns 
                   if col not in default_exclude and col != target_col]
    
    # Separate features and target
    X = features_df[feature_cols].copy()
    
    if target_col and target_col in features_df.columns:
        y = features_df[target_col]
    else:
        y = None
    
    # Handle missing values
    print(f"  Missing values before imputation: {X.isnull().sum().sum()}")
    
    # Fill missing values with median for numeric, mode for categorical
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            X[col].fillna(X[col].median(), inplace=True)
        else:
            X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown', inplace=True)
    
    print(f"  Missing values after imputation: {X.isnull().sum().sum()}")
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    encoding_map = {}
    
    for col in categorical_cols:
        # Use label encoding for now (can be upgraded to one-hot if needed)
        unique_vals = X[col].unique()
        encoding_map[col] = {val: idx for idx, val in enumerate(unique_vals)}
        X[col] = X[col].map(encoding_map[col])
        X[col].fillna(-1, inplace=True)  # For any unmapped values
    
    print(f"  Encoded {len(categorical_cols)} categorical columns")
    
    # Feature information
    feature_info = {
        'feature_names': list(X.columns),
        'n_features': len(X.columns),
        'n_samples': len(X),
        'encoding_map': encoding_map,
        'excluded_cols': default_exclude
    }
    
    print(f"✓ Prepared {len(X.columns)} features for {len(X)} flights")
    
    return X, y, feature_info


def save_features(features_df: pd.DataFrame,
                 output_path: str = 'data/features/flight_features.csv.gz',
                 X: Optional[pd.DataFrame] = None,
                 feature_info: Optional[Dict] = None):
    """
    Save extracted features to disk.
    
    Args:
        features_df: Full features DataFrame
        output_path: Path to save features
        X: Prepared ML-ready features (optional)
        feature_info: Feature information dictionary (optional)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving features to {output_path}...")
    features_df.to_csv(output_path, index=False, compression='gzip')
    print(f"✓ Saved {len(features_df):,} flights with {len(features_df.columns)} features")
    
    # Save ML-ready features if provided
    if X is not None:
        ml_path = output_path.parent / 'flight_features_ml_ready.csv.gz'
        X.to_csv(ml_path, index=False, compression='gzip')
        print(f"✓ Saved ML-ready features to {ml_path}")
    
    # Save feature info if provided
    if feature_info is not None:
        import json
        info_path = output_path.parent / 'feature_info.json'
        # Convert numpy types to native Python types for JSON serialization
        info_serializable = {}
        for key, value in feature_info.items():
            if key == 'encoding_map':
                info_serializable[key] = {
                    k: {str(k2): int(v2) for k2, v2 in v.items()}
                    for k, v in value.items()
                }
            else:
                info_serializable[key] = value
        
        with open(info_path, 'w') as f:
            json.dump(info_serializable, f, indent=2)
        print(f"✓ Saved feature info to {info_path}")


def run_feature_engineering_pipeline(config_path: str = 'config/config.yaml',
                                    processed_dir: str = 'data/processed',
                                    output_dir: str = 'data/features',
                                    feature_types: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.
    
    Args:
        config_path: Path to configuration file
        processed_dir: Directory with processed data from Phase 1
        output_dir: Directory to save features
        feature_types: List of feature types to extract (None = all)
        
    Returns:
        DataFrame with all extracted features
    """
    print("\n" + "="*60)
    print("PHASE 2: FEATURE ENGINEERING")
    print("="*60)
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}, using defaults")
        config = {}
    
    # Load processed data
    events_df, flight_summary_df = load_processed_data(processed_dir)
    
    # Extract all features
    features_df = extract_all_features(events_df, flight_summary_df, feature_types)
    
    # Prepare for ML
    X, y, feature_info = prepare_features_for_ml(features_df)
    
    # Save features
    output_path = Path(output_dir) / 'flight_features.csv.gz'
    save_features(features_df, str(output_path), X, feature_info)
    
    print("\n" + "="*60)
    print("PHASE 2 COMPLETE!")
    print("="*60)
    print(f"\nFeatures saved to: {output_path}")
    print(f"Ready for Phase 3: Model Development")
    
    return features_df



