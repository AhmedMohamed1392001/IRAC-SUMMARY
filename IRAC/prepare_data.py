import pandas as pd
import json
import re
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_raw_example(row, idx):
    """
    Create a raw data example from a row of data.
    Handles missing data gracefully without skipping entries.
    Ensures ALL values are strings to prevent type inconsistencies.
    
    Args:
        row: DataFrame row
        idx: Row index (used as ID)
    
    Returns:
        dict: Raw example with id, case_name, case, and IRAC_summ fields
    """
    # Handle Case Name - use placeholder if missing
    case_name = row.get('Case Name', '')
    if pd.isna(case_name) or str(case_name).strip() == '':
        case_name = f"Case #{idx + 1}"
    else:
        case_name = str(case_name)  # Ensure it's a string
    
    # Handle Case (full case text) - use placeholder if missing
    case_text = row.get('Case', '')
    if pd.isna(case_text) or str(case_text).strip() == '':
        case_text = "No case text available."
    else:
        case_text = str(case_text)  # Ensure it's a string
    
    # Handle IRAC Summary - preprocess to start with "Issue"
    irac_summary = row.get('IRAC Summary', '')
    irac_summary = str(irac_summary) if irac_summary else "No IRAC summary available."
    
    # Create raw example (no instruction formatting)
    # Explicitly convert all values to strings to prevent type issues
    raw_example = {
        "id": int(idx + 1),  # ID can be int
        "case_name": str(case_name).strip(),
        "case": str(case_text).strip(),
        "IRAC_summ": str(irac_summary).strip()  # Explicitly ensure string type
    }
    
    return raw_example


def convert_xlsx_to_train_dev_test(input_file, dev_size=0.10, test_size=0.10, random_state=42):
    """
    Convert XLSX legal case data to train, dev, and test JSONL files in RAW format.
    Does NOT skip any entries - handles missing data with placeholders.
    The instruction formatting will be done as a preprocessing step in the notebook.
    
    Args:
        input_file: Path to input XLSX file
        dev_size: Fraction of data to use for dev/validation (default: 0.10)
        test_size: Fraction of data to use for testing (default: 0.10)
        random_state: Random seed for reproducibility
    
    Returns:
        tuple: (train_data, dev_data, test_data)
    """
    
    # Read the XLSX file
    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file)
    
    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    # Process ALL rows - no skipping
    all_data = []
    missing_case_name = 0
    missing_case = 0
    missing_irac = 0
    
    for idx, row in df.iterrows():
        # Track missing data for reporting
        if pd.isna(row.get('Case Name')) or str(row.get('Case Name', '')).strip() == '':
            missing_case_name += 1
        if pd.isna(row.get('Case')) or str(row.get('Case', '')).strip() == '':
            missing_case += 1
        if pd.isna(row.get('IRAC Summary')) or str(row.get('IRAC Summary', '')).strip() == '':
            missing_irac += 1
        
        # Create raw example (handles missing data internally)
        example = create_raw_example(row, idx)
        all_data.append(example)
    
    # Split into train+dev and test first
    print(f"\nSplitting data (dev_size={dev_size}, test_size={test_size}, random_state={random_state})...")
    train_dev_data, test_data = train_test_split(
        all_data, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Then split train+dev into train and dev
    # Calculate dev_size as proportion of train+dev
    dev_size_adjusted = dev_size / (1 - test_size)
    train_data, dev_data = train_test_split(
        train_dev_data,
        test_size=dev_size_adjusted,
        random_state=random_state
    )
    
    print(f"  Train set: {len(train_data)} examples ({len(train_data)/len(all_data)*100:.1f}%)")
    print(f"  Dev set: {len(dev_data)} examples ({len(dev_data)/len(all_data)*100:.1f}%)")
    print(f"  Test set: {len(test_data)} examples ({len(test_data)/len(all_data)*100:.1f}%)")
    
    # Write train file
    train_file = "data/processed/train.jsonl"
    print(f"\nWriting {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ Created {train_file} ({len(train_data)} examples)")
    
    # Write dev file
    dev_file = "data/processed/dev.jsonl"
    print(f"Writing {dev_file}...")
    with open(dev_file, 'w', encoding='utf-8') as f:
        for item in dev_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ Created {dev_file} ({len(dev_data)} examples)")
    
    # Write test file
    test_file = "data/processed/test.jsonl"
    print(f"Writing {test_file}...")
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ Created {test_file} ({len(test_data)} examples)")
    
    # Verify total
    total_output = len(train_data) + len(dev_data) + len(test_data)
    
    return train_data, dev_data, test_data



if __name__ == "__main__":
    input_file = "data/raw/DataCases.xlsx"
    
    # Check if file exists
    if not Path(input_file).exists():
        print(f"\n⚠ Error: File '{input_file}' not found!")
        print("Please update the 'input_file' variable with your file name.")
    else:
        # Convert and split into train/dev/test
        train_data, dev_data, test_data = convert_xlsx_to_train_dev_test(
            input_file,
            dev_size=0.10,   # 10% for dev/validation
            test_size=0.10,  # 10% for testing
            random_state=42  # For reproducibility
        )
        
        print("✓ Conversion complete!")