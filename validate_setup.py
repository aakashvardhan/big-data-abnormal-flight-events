"""
Installation and setup validation script.
Run this to verify all dependencies are installed correctly.
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required packages can be imported."""
    print("="*60)
    print("CHECKING PACKAGE IMPORTS")
    print("="*60)
    
    required_packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('plotly.express', 'px'),
        ('sklearn', None),
        ('yaml', None),
        ('tqdm', None),
        ('geopy', None),
    ]
    
    failed = []
    for package, alias in required_packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package} - {e}")
            failed.append(package)
    
    if failed:
        print(f"\n⚠️  {len(failed)} package(s) failed to import")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True


def check_directories():
    """Check if all required directories exist."""
    print("\n" + "="*60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("="*60)
    
    project_root = Path(__file__).parent
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'data/features',
        'outputs/figures',
        'outputs/models',
        'outputs/results',
        'src/data',
        'src/features',
        'src/models',
        'src/utils',
        'notebooks',
        'config'
    ]
    
    missing = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - Missing")
            missing.append(dir_path)
    
    if missing:
        print(f"\n⚠️  {len(missing)} directory(ies) missing")
        print("Creating missing directories...")
        for dir_path in missing:
            (project_root / dir_path).mkdir(parents=True, exist_ok=True)
        print("✅ Directories created")
    else:
        print("\n✅ All directories exist!")
    
    return True


def check_data_files():
    """Check if data files exist."""
    print("\n" + "="*60)
    print("CHECKING DATA FILES")
    print("="*60)
    
    project_root = Path(__file__).parent
    raw_data_dir = project_root / 'data' / 'raw'
    
    csv_files = list(raw_data_dir.glob('*.csv'))
    zip_files = list(raw_data_dir.glob('*.zip'))
    
    if csv_files:
        print(f"✓ Found {len(csv_files)} CSV file(s) in data/raw/")
        for f in csv_files[:5]:  # Show first 5
            print(f"  - {f.name}")
        if len(csv_files) > 5:
            print(f"  ... and {len(csv_files) - 5} more")
        return True
    elif zip_files:
        print(f"✓ Found {len(zip_files)} ZIP file(s) in data/raw/")
        for f in zip_files:
            print(f"  - {f.name}")
        print("\n💡 ZIP files will be automatically extracted when you run the notebook")
        return True
    else:
        print("⚠️  No data files found in data/raw/")
        print("\nPlease download data from:")
        print("  Training: https://drive.google.com/drive/u/1/folders/1Scvz11KmgIt5dzmSMzS4MCv0IXnte2a-")
        print("  Testing: https://drive.google.com/drive/u/1/folders/1l3LtAp0u2svJ46lBL80WwwABz7TllSXF")
        print("\nSupported formats: .csv or .zip files")
        print("Place files in: data/raw/")
        return False


def check_config():
    """Check if configuration file exists and is valid."""
    print("\n" + "="*60)
    print("CHECKING CONFIGURATION")
    print("="*60)
    
    project_root = Path(__file__).parent
    config_file = project_root / 'config' / 'config.yaml'
    
    if not config_file.exists():
        print("✗ config/config.yaml not found")
        return False
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print("✓ config.yaml exists and is valid")
        print(f"  - Raw data dir: {config['data']['raw_dir']}")
        print(f"  - Processed data dir: {config['data']['processed_dir']}")
        return True
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False


def main():
    """Run all validation checks."""
    print("\n" + "="*60)
    print("FLIGHT ANOMALY DETECTION - SETUP VALIDATION")
    print("="*60)
    
    checks = [
        ("Package Imports", check_imports),
        ("Directory Structure", check_directories),
        ("Configuration", check_config),
        ("Data Files", check_data_files),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            results.append(check_func())
        except Exception as e:
            print(f"\n❌ Error during {name} check: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (name, _) in enumerate(checks):
        status = "✅" if results[i] else "❌"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 Setup is complete! You're ready to run Phase 1.")
        print("\nNext steps:")
        print("1. Open notebooks/phase1_eda.ipynb")
        print("2. Run cells to explore your data")
        print("3. Review outputs in outputs/figures/")
    else:
        print("\n⚠️  Setup incomplete. Please resolve the issues above.")
        print("\nFor help, see QUICKSTART.md")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
