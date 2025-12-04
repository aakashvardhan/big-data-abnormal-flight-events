"""Data loading utilities for flight event data."""
import pandas as pd
import gdown
import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import json


def extract_zip_files(zip_path, output_dir):
    """
    Extract ZIP files to output directory.
    
    Args:
        zip_path: Path to ZIP file
        output_dir: Directory to extract files to
    
    Returns:
        List of extracted file paths
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_files = []
    print(f"Extracting {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files
        file_list = zip_ref.namelist()
        
        # Extract with progress bar
        for file in tqdm(file_list, desc="Extracting"):
            zip_ref.extract(file, output_dir)
            extracted_path = output_dir / file
            if extracted_path.suffix == '.csv':
                extracted_files.append(extracted_path)
    
    print(f"✓ Extracted {len(extracted_files)} CSV file(s)")
    return extracted_files


def extract_all_zip_files(directory):
    """
    Find and extract all ZIP files in a directory.
    
    Args:
        directory: Directory containing ZIP files
    
    Returns:
        List of all extracted CSV file paths
    """
    directory = Path(directory)
    zip_files = list(directory.glob('*.zip'))
    
    if not zip_files:
        print(f"No ZIP files found in {directory}")
        return []
    
    print(f"Found {len(zip_files)} ZIP file(s)")
    
    all_extracted = []
    for zip_file in zip_files:
        extracted = extract_zip_files(zip_file, directory)
        all_extracted.extend(extracted)
    
    return all_extracted


def download_from_gdrive_folder(folder_id, output_dir):
    """
    Download files from a Google Drive folder.
    
    Args:
        folder_id: Google Drive folder ID
        output_dir: Local directory to save files
    """
    # Note: Direct folder download requires folder to be public
    # For direct file download, use: gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path)
    
    folder_url = f'https://drive.google.com/drive/folders/{folder_id}'
    print(f"Google Drive folder URL: {folder_url}")
    print(f"\nPlease manually download files from the folder and place them in: {output_dir}")
    print("\nSupported formats: .csv, .zip (will be auto-extracted)")
    print("\nAlternatively, if you have individual file IDs, update this function.")
    
    return folder_url


def load_flight_data(file_path, nrows=None, chunksize=None):
    """
    Load flight event data from CSV file.
    
    Args:
        file_path: Path to CSV file
        nrows: Number of rows to load (None for all)
        chunksize: If specified, return iterator for chunked reading
    
    Returns:
        DataFrame or iterator
    """
    print(f"Loading data from {file_path}...")
    
    if chunksize:
        return pd.read_csv(file_path, chunksize=chunksize, nrows=nrows)
    else:
        df = pd.read_csv(file_path, nrows=nrows)
        print(f"Loaded {len(df):,} records")
        return df


def parse_info_field(info_str):
    """
    Parse JSON info field from string.
    
    Args:
        info_str: JSON string from info column
    
    Returns:
        Dictionary or None if parsing fails
    """
    if pd.isna(info_str):
        return None
    
    try:
        return json.loads(info_str)
    except (json.JSONDecodeError, TypeError):
        return None


def extract_info_fields(df):
    """
    Extract useful fields from info column.
    
    Args:
        df: DataFrame with 'info' column
    
    Returns:
        DataFrame with additional columns from info field
    """
    print("Parsing info field...")
    
    # Parse info JSON
    df['info_parsed'] = df['info'].apply(parse_info_field)
    
    # Extract common fields (adjust based on actual data structure)
    df['airport'] = df['info_parsed'].apply(lambda x: x.get('airport') if x else None)
    df['runway'] = df['info_parsed'].apply(lambda x: x.get('runway') if x else None)
    df['taxiway'] = df['info_parsed'].apply(lambda x: x.get('taxiway') if x else None)
    df['parking'] = df['info_parsed'].apply(lambda x: x.get('parking') if x else None)
    
    return df


def load_all_data_files(data_dir, auto_extract_zip=True):
    """
    Load all CSV files from a directory and combine them.
    Optionally extracts ZIP files first.
    
    Args:
        data_dir: Directory containing CSV files (and optionally ZIP files)
        auto_extract_zip: If True, automatically extract ZIP files first
    
    Returns:
        Combined DataFrame
    """
    data_path = Path(data_dir)
    
    # Extract ZIP files if present and auto_extract is enabled
    if auto_extract_zip:
        zip_files = list(data_path.glob('*.zip'))
        if zip_files:
            print(f"Found {len(zip_files)} ZIP file(s), extracting...")
            extract_all_zip_files(data_path)
    
    # Now load CSV files (search recursively because Google Drive ZIPs
    # often contain nested folders)
    csv_files = list(data_path.rglob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        print("If you have ZIP files, they should be automatically extracted.")
        print("If CSVs are inside nested folders, they will be picked up after this change.")
        return None
    
    print(f"Found {len(csv_files)} CSV file(s)")
    
    dfs = []
    for file in tqdm(csv_files, desc="Loading files"):
        df = pd.read_csv(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(combined_df):,}")
    
    return combined_df


def save_processed_data(df, output_path, compression='gzip'):
    """
    Save processed DataFrame to compressed format.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
        compression: Compression method ('gzip', 'bz2', 'zip', 'xz', None)
    """
    print(f"Saving data to {output_path}...")
    
    if compression:
        if not output_path.endswith('.gz'):
            output_path = f"{output_path}.gz"
    
    df.to_csv(output_path, index=False, compression=compression)
    print(f"✓ Saved {len(df):,} records")


def get_data_summary(df):
    """
    Get summary statistics for the dataset.
    
    Args:
        df: DataFrame to summarize
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_records': len(df),
        'unique_flights': df['flight_id'].nunique() if 'flight_id' in df.columns else None,
        'date_range': (df['timestamp'].min(), df['timestamp'].max()) if 'timestamp' in df.columns else None,
        'event_types': df['event_type'].nunique() if 'event_type' in df.columns else None,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict()
    }
    
    return summary
