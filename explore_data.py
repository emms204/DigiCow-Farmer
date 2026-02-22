#!/usr/bin/env python3
"""
Data Exploration Script for DigiCow Farmer Training Adoption Challenge
This script reads and analyzes all CSV files in the dataset.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

def print_separator(title=""):
    """Print a visual separator with optional title"""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)
    else:
        print("="*80 + "\n")

def explore_csv_file(filepath, file_name):
    """Comprehensive exploration of a CSV file"""
    print_separator(f"Exploring: {file_name}")
    
    try:
        # Read the CSV file
        print(f"Reading {file_name}...")
        df = pd.read_csv(filepath)
        
        # Basic Information
        print(f"\n📊 BASIC INFORMATION")
        print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Column Information
        print(f"\n📋 COLUMNS ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            print(f"   {i:2d}. {col:30s} ({dtype})")
        
        # Data Types Summary
        print(f"\n🔢 DATA TYPES SUMMARY:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")
        
        # Missing Values
        print(f"\n❌ MISSING VALUES:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df.to_string())
        else:
            print("   ✓ No missing values found!")
        
        # Duplicate Rows
        duplicates = df.duplicated().sum()
        print(f"\n🔄 DUPLICATE ROWS: {duplicates}")
        if duplicates > 0:
            print(f"   ⚠️  Found {duplicates} duplicate rows")
        
        # Sample Data
        print(f"\n👀 SAMPLE DATA (First 3 rows):")
        print(df.head(3).to_string())
        
        # Numerical Columns Statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            print(f"\n📈 NUMERICAL COLUMNS STATISTICS:")
            print(df[numerical_cols].describe().to_string())
        
        # Categorical Columns Summary
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            print(f"\n📝 CATEGORICAL COLUMNS SUMMARY:")
            for col in categorical_cols[:10]:  # Limit to first 10 to avoid too much output
                unique_count = df[col].nunique()
                print(f"\n   {col}:")
                print(f"      Unique values: {unique_count}")
                if unique_count <= 20:
                    value_counts = df[col].value_counts()
                    print(f"      Value distribution:")
                    for val, count in value_counts.items():
                        pct = (count / len(df) * 100)
                        print(f"         {str(val)[:40]:40s}: {count:6,} ({pct:5.2f}%)")
                else:
                    print(f"      Top 10 values:")
                    top_values = df[col].value_counts().head(10)
                    for val, count in top_values.items():
                        pct = (count / len(df) * 100)
                        print(f"         {str(val)[:40]:40s}: {count:6,} ({pct:5.2f}%)")
        
        # Target Variables Analysis (if present)
        target_cols = [col for col in df.columns if 'adopted_within' in col.lower()]
        if target_cols:
            print(f"\n🎯 TARGET VARIABLES ANALYSIS:")
            for col in target_cols:
                print(f"\n   {col}:")
                value_counts = df[col].value_counts().sort_index()
                for val, count in value_counts.items():
                    pct = (count / len(df) * 100)
                    print(f"      {val}: {count:6,} ({pct:5.2f}%)")
                print(f"      Class imbalance ratio: {value_counts.min() / value_counts.max():.3f}")
        
        # Date Columns Analysis
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'day' in col.lower()]
        if date_cols:
            print(f"\n📅 DATE COLUMNS ANALYSIS:")
            for col in date_cols:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    print(f"\n   {col}:")
                    print(f"      Date range: {df[col].min()} to {df[col].max()}")
                    print(f"      Unique dates: {df[col].nunique()}")
                    print(f"      Missing dates: {df[col].isnull().sum()}")
                except:
                    print(f"   {col}: Could not parse as date")
        
        # Special Columns Analysis
        if 'topics_list' in df.columns:
            print(f"\n📚 TOPICS LIST ANALYSIS:")
            # Try to understand the structure
            sample_topics = df['topics_list'].dropna().head(5)
            print(f"   Sample topics_list values:")
            for idx, val in enumerate(sample_topics, 1):
                print(f"      {idx}. {str(val)[:100]}")
        
        if 'trainer' in df.columns:
            print(f"\n👨‍🏫 TRAINER ANALYSIS:")
            trainer_counts = df['trainer'].value_counts()
            print(f"   Unique trainers: {trainer_counts.nunique()}")
            print(f"   Top 10 trainers:")
            for trainer, count in trainer_counts.head(10).items():
                pct = (count / len(df) * 100)
                print(f"      {str(trainer)[:40]:40s}: {count:6,} ({pct:5.2f}%)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading {file_name}: {str(e)}")
        return None

def main():
    """Main function to explore all CSV files"""
    print_separator("DIGICOW FARMER TRAINING ADOPTION - DATA EXPLORATION")
    
    # Get the directory where the script is located
    script_dir = Path(__file__).parent
    csv_files = {
        'Train.csv': script_dir / 'Train.csv',
        'Test.csv': script_dir / 'Test.csv',
        'Prior.csv': script_dir / 'Prior.csv',
        'SampleSubmission.csv': script_dir / 'SampleSubmission.csv',
        'dataset_data_dictionary.csv': script_dir / 'dataset_data_dictionary.csv'
    }
    
    dataframes = {}
    
    # Explore each CSV file
    for file_name, filepath in csv_files.items():
        if filepath.exists():
            df = explore_csv_file(filepath, file_name)
            if df is not None:
                dataframes[file_name] = df
        else:
            print_separator(f"⚠️  File not found: {file_name}")
    
    # Comparative Analysis
    print_separator("COMPARATIVE ANALYSIS")
    
    if 'Train.csv' in dataframes and 'Test.csv' in dataframes:
        train_df = dataframes['Train.csv']
        test_df = dataframes['Test.csv']
        
        print("\n📊 Train vs Test Comparison:")
        print(f"   Train rows: {len(train_df):,}")
        print(f"   Test rows: {len(test_df):,}")
        print(f"   Ratio: {len(train_df) / len(test_df):.2f}:1")
        
        # Check common columns
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        common_cols = train_cols & test_cols
        train_only = train_cols - test_cols
        test_only = test_cols - train_cols
        
        print(f"\n   Common columns: {len(common_cols)}")
        print(f"   Train-only columns: {train_only}")
        print(f"   Test-only columns: {test_only}")
        
        # Check for ID overlap
        if 'ID' in train_df.columns and 'ID' in test_df.columns:
            train_ids = set(train_df['ID'])
            test_ids = set(test_df['ID'])
            overlap = train_ids & test_ids
            print(f"\n   ID overlap: {len(overlap)} IDs appear in both Train and Test")
            if len(overlap) > 0:
                print(f"   ⚠️  Warning: {len(overlap)} IDs overlap between Train and Test!")
    
    if 'Prior.csv' in dataframes:
        prior_df = dataframes['Prior.csv']
        print(f"\n📊 Prior Dataset:")
        print(f"   Prior rows: {len(prior_df):,}")
        if 'Train.csv' in dataframes:
            train_df = dataframes['Train.csv']
            if 'ID' in prior_df.columns and 'ID' in train_df.columns:
                prior_ids = set(prior_df['ID'])
                train_ids = set(train_df['ID'])
                overlap = prior_ids & train_ids
                print(f"   ID overlap with Train: {len(overlap)} IDs")
    
    print_separator("EXPLORATION COMPLETE")
    print("\n✅ Data exploration finished!")
    print("\nNext steps:")
    print("   1. Review the findings above")
    print("   2. Identify feature engineering opportunities")
    print("   3. Check for data quality issues")
    print("   4. Plan modeling strategy")

if __name__ == "__main__":
    main()

