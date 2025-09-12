import pandas as pd
import numpy as np

def analyze_excel_file(file_path):
    """Analyze Excel file structure and content"""
    print("="*60)
    print("COMPREHENSIVE EXCEL FILE ANALYSIS")
    print("="*60)
    
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    
    print(f"File: {file_path}")
    print(f"Number of sheets: {len(sheet_names)}")
    print(f"Sheet names: {sheet_names}")
    print()
    
    # Analyze each sheet
    for sheet_name in sheet_names:
        print("="*50)
        print(f"SHEET: {sheet_name}")
        print("="*50)
        
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            print(f"Shape: {df.shape} (rows × columns)")
            print(f"Columns: {list(df.columns)}")
            print()
            
            # Show data types
            print("Data Types:")
            for col, dtype in df.dtypes.items():
                non_null_count = df[col].count()
                null_count = df[col].isnull().sum()
                print(f"  {col}: {dtype} ({non_null_count} non-null, {null_count} null)")
            print()
            
            # Show first few rows
            print("First 5 rows:")
            print(df.head().to_string())
            print()
            
            # Show basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print("Numeric Column Statistics:")
                stats = df[numeric_cols].describe()
                print(stats.to_string())
                print()
            
            # Show unique values for non-numeric columns with few unique values
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype == 'string':
                    unique_count = df[col].nunique()
                    if unique_count <= 20:
                        print(f"Unique values in '{col}' ({unique_count} unique):")
                        print(f"  {list(df[col].dropna().unique())}")
                        print()
            
        except Exception as e:
            print(f"Error reading sheet '{sheet_name}': {e}")
            print()

if __name__ == "__main__":
    file_path = "E1_Anand_Code_Q4 Q5 Q6.xlsx"
    analyze_excel_file(file_path)
