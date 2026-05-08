import pandas as pd
import numpy as np
import glob
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Well splits (determined from previous analysis)
TEST_WELLS = ['RAJIAN-03A', 'PINDORI-2', 'TURKWAL DEEP X 2', 'Balkassar POL 01']
VALIDATION_WELLS = ['MINWAL-X-1', 'MINWAL-2', 'MISSA KESWAL-02', 'Balkassar OXY 01']

# Essential features (must exist in all wells)
ESSENTIAL_FEATURES = ['tvd', 'dt', 'dt_nct', 'gr', 'sphi', 'hp', 'ob']

# Optional features (use if available with good coverage)
OPTIONAL_FEATURES = ['rhob_combined', 'res_deep'] # 'temp', 'vp'

# Features to exclude (target leakage or redundant)
EXCLUDE_FEATURES = ['mw', 'pp', 'fp', 'ves', 'dphi', 'nphi', 'phit', 'phie', 'velocity',
                    'velocity_nct', 'ai', 'cgr', 'ac', 'cal', 'cali', 'dst', 'lld', 'lls', 'msfl']

TARGET = 'ppp' # Primary target
MIN_SAMPLES_PER_WELL = 600
MIN_FEATURE_COVERAGE = 0.65  # Feature must have >65% non-NaN to be included

# Physical constraints - CRITICAL FOR AVOIDING OUTLIERS
CONSTRAINTS = {
    'tvd': (0, 6000),              # meters
    'gr': (0, 500),                # API
    'dt': (40, 250),               # us/ft
    'dt_nct': (40, 200),           # us/ft
    'rhob_combined': (1.5, 3.5),   # g/cc
    'res_deep': (0.01, 100000),    # ohmm (Caps ridiculous resistivity values)
    'sphi': (-0.1, 0.6),           # decimal
    'ob': (100, 30000),            # psi
    'hp': (100, 20000),            # psi
    'ppp': (100, 30000),           # psi
    'temp': (30, 500)              # degF
}

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def load_and_standardize_well(filepath):
    """Load well data and standardize column names"""
    df = pd.read_csv(filepath)
    
    # Standardize column names: lowercase, no spaces
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '')
    
    # Replace missing value flags with NaN
    df.replace([-999.25, -999, -999.0], np.nan, inplace=True)
    
    # Add well identifier
    well_name = filepath.split('/')[-1].replace('.csv', '').replace('.CSV', '')
    df['well_id'] = well_name
    
    return df, well_name


def validate_essential_features(df, well_name):
    """Check if well has all essential features"""
    missing = [f for f in ESSENTIAL_FEATURES if f not in df.columns]
    if missing:
        print(f"  {well_name}: Missing essential features {missing} - Skipping")
        return False
    return True


def clean_well_data(df):
    """Remove impossible values and handle NaN using physical constraints"""
    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])
    
    # Drop rows with missing features
    df = df.dropna(subset=ESSENTIAL_FEATURES)
    df = df.dropna(subset=OPTIONAL_FEATURES)
    
    # Apply physical constraints to filter out garbage data
    for feat, (min_val, max_val) in CONSTRAINTS.items():
        if feat in df.columns:
            df = df[(df[feat] >= min_val) & (df[feat] <= max_val)]
            
    # Underpressure is valid, so we don't drop ppp < hp here.
    
    # Exclude features with target leakage
    for feat in EXCLUDE_FEATURES:
        if feat in df.columns:
            df = df.drop(columns=[feat])
    
    return df


def engineer_features(df):
    """Create physics-based features"""
    # Sort by depth for consistency
    df = df.sort_values('tvd').reset_index(drop=True)
    
    # Core physics features
    df['eaton_ratio'] = (df['dt'] / df['dt_nct']) ** 3
    
    # Pressure gradients (psi/ft) - removes depth dependency
    tvd_ft = df['tvd'] * 3.28084
    df['hp_gradient'] = df['hp'] / tvd_ft
    df['ob_gradient'] = df['ob'] / tvd_ft
    
    # Normalized depth
    df['tvd_normalized'] = df['tvd'] / df['tvd'].max()
    
    # Target transformations
    df['pressure_ratio'] = df['ppp'] / df['hp']
    df['overpressure'] = df['ppp'] - df['hp']
    
    # Pressure regime classification
    df['pressure_regime'] = pd.cut(
        df['pressure_ratio'],
        bins=[0, 0.9, 1.05, 1.2, 1.5, np.inf],
        labels=['Underpressured', 'Normal', 'Mild_OP', 'Moderate_OP', 'Severe_OP']
    )
    
    return df


def process_well(filepath):
    """Process a single well"""
    df, well_name = load_and_standardize_well(filepath)
    
    # Validate essential features
    if not validate_essential_features(df, well_name):
        return None

    # Keep only Murree formation for Missa and Qazian 
    if well_name == 'MISSA KESWAL-01':
        df = df[(df['depth']>=1136.8) & (df['depth']<=1802.3)].reset_index(drop=True)
        print(f"  {well_name}: Extracted Murree formation range")
    elif well_name == 'MISSA KESWAL-02':
        df = df[(df['depth']>=1115) & (df['depth']<=1792.9)].reset_index(drop=True)
        print(f"  {well_name}: Extracted Murree formation range")
    elif well_name == 'MISSA KESWAL-03':
        df = df[(df['depth']>=1124.8) & (df['depth']<=1870.9)].reset_index(drop=True)
        print(f"  {well_name}: Extracted Murree formation range")
    elif well_name == 'QAZIAN -1X':
        df = df[(df['depth']>=1198.8) & (df['depth']<=2062.9)].reset_index(drop=True)
        print(f"  {well_name}: Extracted Murree formation range")
    
    # Clean data
    initial_samples = len(df)
    df = clean_well_data(df)
    
    # Check minimum samples
    if len(df) < MIN_SAMPLES_PER_WELL:
        print(f"  {well_name}: Too few samples ({len(df)}) - Skipping")
        return None
    
    # Engineer features
    df = engineer_features(df)
    
    print(f"  {well_name}: {len(df):,} samples retained ({len(df)/initial_samples*100:.1f}%)")
    return df


def select_features(df_all):
    """Select features based on coverage across all data"""
    feature_coverage = {}
    
    # Check essential features (should all be 100%)
    for feat in ESSENTIAL_FEATURES:
        if feat in df_all.columns:
            coverage = df_all[feat].notna().sum() / len(df_all)
            feature_coverage[feat] = coverage
    
    # Check optional features
    for feat in OPTIONAL_FEATURES:
        if feat in df_all.columns:
            coverage = df_all[feat].notna().sum() / len(df_all)
            feature_coverage[feat] = coverage
    
    # Select features with sufficient coverage
    selected_features = []
    for feat, coverage in feature_coverage.items():
        if coverage >= MIN_FEATURE_COVERAGE:
            selected_features.append(feat)
            print(f"  {feat}: {coverage*100:.1f}% coverage - INCLUDED")
        else:
            print(f"  {feat}: {coverage*100:.1f}% coverage - EXCLUDED")
    
    # Add engineered features
    engineered = ['eaton_ratio', 'hp_gradient', 'ob_gradient', 'tvd_normalized']
    selected_features.extend(engineered)
    
    return selected_features

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute preprocessing pipeline"""
    print("="*70)
    print("PORE PRESSURE PREDICTION - PREPROCESSING")
    print("="*70)
    
    # Process all wells
    file_paths = glob.glob('./datasets/*.csv')
    print(f"\nProcessing {len(file_paths)} wells...")
    
    processed_wells = []
    for fp in file_paths:
        df = process_well(fp)
        if df is not None:
            processed_wells.append(df)
    
    if not processed_wells:
        print("ERROR: No wells survived preprocessing!")
        return None, None, None
    
    print(f"\n{len(processed_wells)}/{len(file_paths)} wells successfully processed")
    
    # Combine all wells
    df_all = pd.concat(processed_wells, ignore_index=True)
    print(f"Total samples: {len(df_all):,}")
    
    # Select features based on coverage
    print("\nFeature selection based on coverage:")
    predictor_features = select_features(df_all)
    
    # Split by wells
    df_test = df_all[df_all['well_id'].isin(TEST_WELLS)]
    df_val = df_all[df_all['well_id'].isin(VALIDATION_WELLS)]
    train_wells = df_all[~df_all['well_id'].isin(TEST_WELLS + VALIDATION_WELLS)]['well_id'].unique().tolist()
    df_train = df_all[df_all['well_id'].isin(train_wells)]
    
    # Print split statistics
    print(f"\nData splits:")
    print(f"  Training:   {len(df_train):,} samples ({len(df_train)/len(df_all)*100:.1f}%)")
    print(f"  Validation: {len(df_val):,} samples ({len(df_val)/len(df_all)*100:.1f}%)")
    print(f"  Test:       {len(df_test):,} samples ({len(df_test)/len(df_all)*100:.1f}%)")
    
    # Print pressure distributions
    print(f"\nPressure regime distributions:")
    for name, df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
        if len(df) > 0:
            dist = df['pressure_regime'].value_counts(normalize=True).sort_index()
            under = dist.get('Underpressured', 0)
            normal = dist.get('Normal', 0)
            over = dist[['Mild_OP', 'Moderate_OP', 'Severe_OP']].sum() if len(dist) > 2 else 0
            print(f"  {name}: Under={under*100:.1f}%, Normal={normal*100:.1f}%, Over={over*100:.1f}%")
    
    # Save datasets
    print(f"\nSaving datasets...")
    df_train.to_csv('train_data.csv', index=False)
    df_val.to_csv('val_data.csv', index=False)
    df_test.to_csv('test_data.csv', index=False)
    
    # Save metadata
    metadata = {
        'train_wells': train_wells,
        'validation_wells': VALIDATION_WELLS,
        'test_wells': TEST_WELLS,
        'predictor_features': predictor_features,
        'target_options': ['ppp', 'pressure_ratio', 'overpressure'],
        'train_samples': len(df_train),
        'val_samples': len(df_val),
        'test_samples': len(df_test),
        'feature_coverage_threshold': MIN_FEATURE_COVERAGE
    }
    
    with open('preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Preprocessing complete!")
    print(f"✓ Features selected: {len(predictor_features)}")
    print(f"✓ Files saved: train_data.csv, val_data.csv, test_data.csv, preprocessing_metadata.json")
    
    return df_train, df_val, df_test


if __name__ == "__main__":
    df_train, df_val, df_test = main()