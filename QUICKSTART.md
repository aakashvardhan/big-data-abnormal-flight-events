# Quick Start Guide - Abnormal Flight Events Detection

## Phase 1: Data Preparation & EDA - COMPLETE ✅

### What Has Been Created

#### 1. Project Structure
```
project/
├── config/
│   └── config.yaml           # Configuration parameters
├── data/
│   ├── raw/                  # Place downloaded CSV files here
│   ├── processed/            # Cleaned data output
│   └── features/             # Feature datasets
├── src/
│   ├── data/                 # Data processing modules
│   │   ├── data_loader.py
│   │   ├── data_quality.py
│   │   └── flight_reconstruction.py
│   ├── features/             # Feature engineering (Phase 2)
│   ├── models/               # ML models (Phase 3)
│   └── utils/
│       └── helpers.py
├── notebooks/
│   └── phase1_eda.ipynb     # ⭐ START HERE
├── outputs/
│   ├── figures/             # Visualizations
│   ├── models/              # Saved models
│   └── results/             # Analysis results
├── requirements.txt
└── README.md
```

#### 2. Utilities Created
- **Data Loader**: Load and parse flight event data from CSV
- **Data Quality**: Check missing values, duplicates, validate timestamps/coordinates/altitude
- **Flight Reconstruction**: Group events by flight, create flight-level summaries
- **Helper Functions**: Configuration loading, directory setup

#### 3. Phase 1 Notebook
Comprehensive EDA notebook with:
- Data loading and quality assessment
- Missing values and duplicate analysis
- Event type frequency distribution
- Altitude and temporal pattern analysis
- Flight reconstruction and summary statistics
- Geographic distribution visualization
- Comprehensive summary report

## Getting Started

### Step 1: Install Dependencies
```powershell
# Navigate to project directory
cd "c:\Users\aiish\OneDrive\Desktop\MSDA-SJSU\Fall 2025\Big Data\project"

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

### Step 2: Download Data
1. **Training Data**: https://drive.google.com/drive/u/1/folders/1Scvz11KmgIt5dzmSMzS4MCv0IXnte2a-
2. **Testing Data**: https://drive.google.com/drive/u/1/folders/1l3LtAp0u2svJ46lBL80WwwABz7TllSXF

Download files and place them in:
```
data/raw/
```

**Supported formats:**
- ✅ `.csv` files (used directly)
- ✅ `.zip` files (automatically extracted)

The code will automatically detect and extract ZIP files when you run the notebook!

### Step 3: Run Phase 1 Notebook
```powershell
# Start Jupyter
jupyter notebook

# Open: notebooks/phase1_eda.ipynb
```

**OR** open the notebook in VS Code and run cells interactively.

### Step 4: Review Outputs
After running the notebook, check:
- `data/processed/` - Cleaned data files
- `outputs/figures/` - Generated visualizations
- Notebook output - Summary statistics and findings

## Expected Dataset Structure

Your CSV files should have these columns:
- `flight_id` - Unique flight identifier
- `event_type` - Type of event (level-start, climb, descent, etc.)
- `timestamp` - Event timestamp
- `latitude` - Geographic latitude
- `longitude` - Geographic longitude
- `altitude` - Altitude in meters
- `info` - JSON field with metadata (airport, runway, etc.)

## Phase 1 Outputs

After completing Phase 1, you'll have:
1. ✅ Clean, validated event-level data
2. ✅ Flight-level summary dataset
3. ✅ Quality assessment report
4. ✅ Exploratory visualizations
5. ✅ Identified potential anomalies (preliminary)

## Next Steps: Phase 2 (Coming Soon)

Phase 2 will focus on **Feature Engineering**:
- Temporal features (duration, taxi times, phase times)
- Spatial features (distance, trajectory sinuosity)
- Operational features (runway events, taxiway transitions)
- Event sequence features (n-grams, state transitions)
- Contextual features (airport-specific norms, peer comparison)

## Troubleshooting

### Issue: Module not found
```powershell
# Make sure you're in the project root and environment is activated
pip install -r requirements.txt
```

### Issue: No CSV files found
- Verify files are in `data/raw/` directory
- Check file extensions are `.csv`

### Issue: Memory errors with large dataset
- In the notebook, reduce `nrows` parameter for testing
- Use chunked reading for full dataset processing

### Issue: Import errors in notebook
- Make sure you're running notebook from `notebooks/` directory
- The notebook adds parent directory to path automatically

## Configuration

Edit `config/config.yaml` to adjust:
- Data paths
- Sample sizes for testing
- Model parameters
- Visualization settings

## Key Features of the Implementation

### Data Quality Checks ✅
- Missing value analysis
- Duplicate detection
- Timestamp validation
- Coordinate range validation
- Altitude reasonability checks

### Exploratory Analysis ✅
- Event type distributions
- Altitude patterns and flight levels
- Temporal patterns (hourly, daily)
- Airport activity analysis
- Geographic coverage

### Flight Reconstruction ✅
- Event grouping by flight
- Chronological ordering
- Flight-level summaries
- Duration calculations
- Event counting

### Visualizations ✅
- Distribution plots
- Box plots for outlier detection
- Time series analysis
- Geographic scatter plots
- Summary dashboards

## Support

For questions or issues:
1. Check the notebook comments and markdown cells
2. Review function docstrings in `src/` modules
3. Consult `plan.md` for project strategy

## Project Timeline

- ✅ **Week 1**: Phase 1 - Data Preparation & EDA (COMPLETE)
- 🔄 **Week 2**: Phase 2 - Feature Engineering (NEXT)
- ⏳ **Week 3**: Phase 3 - Model Development
- ⏳ **Week 4**: Phase 4 - Evaluation & Refinement
- ⏳ **Week 5**: Phase 5 - Insights & Production

---

**Ready to begin!** Open `notebooks/phase1_eda.ipynb` and start exploring your flight data.
