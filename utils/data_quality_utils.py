# Data quality calculations
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple


@st.cache_data(max_entries=5)
def fill_missing_contextually(
    df: pd.DataFrame,
    pivot_info: Dict,
    fill_mode: str = "primary",
    similarity_threshold: float = 0.8
) -> pd.DataFrame:
    """
    Context-aware missing value filling using pivot statistics.
    Safe fallback implementation.
    """

    if df is None or pivot_info is None:
        return df

    df_filled = df.copy()

    pivot_df = pivot_info.get("data")
    config = pivot_info.get("config", {})

    if pivot_df is None:
        return df_filled

    row_fields = config.get("rows", [])
    value_fields = config.get("values", [])

    if not row_fields or not value_fields:
        return df_filled

    pivot_reset = pivot_df.reset_index()

    for idx, row in df.iterrows():
        for value_col in value_fields:
            if value_col not in df.columns:
                continue

            if pd.isna(row[value_col]):
                mask = True
                for rf in row_fields:
                    if rf in df.columns and rf in pivot_reset.columns:
                        mask = mask & (pivot_reset[rf] == row[rf])

                matches = pivot_reset[mask]
                if not matches.empty and value_col in matches.columns:
                    fill_val = matches.iloc[0][value_col]
                    if pd.notna(fill_val):
                        df_filled.at[idx, value_col] = fill_val

    return df_filled


@st.cache_data
def calculate_data_quality_score(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive data quality metrics"""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    
    # Completeness (0-40 points)
    completeness = ((total_cells - missing_cells) / total_cells * 40) if total_cells > 0 else 0
    
    # Consistency (0-30 points) - based on data types appropriateness
    consistency = 0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            consistency += 3
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            consistency += 3
        else:
            consistency += 1
    consistency = min(30, consistency / len(df.columns) * 30)
    
    # Uniqueness (0-20 points) - duplicate analysis
    duplicate_rows = df.duplicated().sum()
    uniqueness = max(0, 20 - (duplicate_rows / len(df) * 20)) if len(df) > 0 else 0
    
    # Validity (0-10 points) - basic validity checks
    validity = 10  # Start with full points, deduct for issues
    for col in df.select_dtypes(include=[np.number]).columns:
        if (df[col] < 0).any():
            validity -= 1
    validity = max(0, validity)
    
    total_score = completeness + consistency + uniqueness + validity
    
    return {
        'total_score': round(total_score, 2),
        'completeness': round(completeness, 2),
        'consistency': round(consistency, 2),
        'uniqueness': round(uniqueness, 2),
        'validity': round(validity, 2),
        'missing_percentage': round((missing_cells / total_cells * 100) if total_cells > 0 else 0, 2),
        'duplicate_rows': duplicate_rows,
        'duplicate_percentage': round((duplicate_rows / len(df) * 100) if len(df) > 0 else 0, 2)
    }

@st.cache_data
def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Detect duplicate rows"""
    duplicates = df[df.duplicated(keep=False)]
    return duplicates

def ensure_nullable_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame uses consistent nullable types"""
    df_compat = df.copy()
    
    for col in df_compat.columns:
        # Convert object columns with NaN to string type
        if df_compat[col].dtype == 'object':
            # Check if column contains mixed types or NaN
            if df_compat[col].isna().any():
                try:
                    df_compat[col] = df_compat[col].astype('string')
                except:
                    pass  # Keep as object if conversion fails
        
        # Convert float columns that should be integers to Int64
        elif pd.api.types.is_float_dtype(df_compat[col]):
            # Check if all non-null values are integers
            non_null = df_compat[col].dropna()
            if len(non_null) > 0 and (non_null % 1 == 0).all():
                try:
                    df_compat[col] = df_compat[col].astype('Int64')
                except:
                    pass  # Keep as float if conversion fails
    
    return df_compat

def convert_column_type(df: pd.DataFrame, col_name: str, new_type: str) -> Tuple[bool, str]:
    """Enhanced column type conversion with nullable type support"""
    try:
        original_dtype = str(df[col_name].dtype)
        
        # Handle pandas nullable types
        if new_type == 'Int64':
            temp = pd.to_numeric(df[col_name], errors='coerce')
            temp = temp.round(0)
            df[col_name] = temp.astype('Int64')
            
        elif new_type == 'Float64':
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype('Float64')
            
        elif new_type == 'boolean':
            df[col_name] = df[col_name].replace({
                "True": True, "False": False, "true": True, "false": False,
                "1": True, "0": False, 1: True, 0: False,
                "yes": True, "no": False, "Yes": True, "No": False,
                "Y": True, "N": False, "y": True, "n": False
            }).astype('boolean')
            
        elif new_type == 'string':
            df[col_name] = df[col_name].astype('string')
            
        elif new_type == 'datetime':
            df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
            
        elif new_type == 'category':
            df[col_name] = df[col_name].astype('category')
            
        elif new_type == 'int':
            temp = pd.to_numeric(df[col_name], errors='coerce')
            temp = temp.round(0)
            df[col_name] = temp.astype('Int64')
            new_type = 'Int64'  # Update for message
            
        elif new_type == 'float':
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype('Float64')
            new_type = 'Float64'  # Update for message
            
        elif new_type == 'bool':
            df[col_name] = df[col_name].replace({
                "True": True, "False": False, "true": True, "false": False,
                "1": True, "0": False, 1: True, 0: False
            }).astype('boolean')
            new_type = 'boolean'  # Update for message
            
        else:
            # Try custom type
            try:
                df[col_name] = df[col_name].astype(new_type)
            except:
                return False, f"Unknown type '{new_type}'"
        
        # Ensure nullable compatibility
        df = ensure_nullable_compatibility(df)
        
        return True, f"Successfully converted from {original_dtype} to {new_type}"
        
    except Exception as e:
        error_msg = f"Conversion failed: {str(e)}"
        # Try to restore original type if possible
        try:
            df[col_name] = df[col_name].astype(original_dtype)
        except:
            pass
        return False, error_msg

@st.cache_data
def detect_data_types(df: pd.DataFrame, sample_size: int = 500) -> pd.DataFrame:
    """Enhanced intelligent data type detection with better sampling"""
    suggestions = []
    
    # Use representative sampling
    if len(df) > sample_size:
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    else:
        sample_df = df
    
    for col in df.columns:
        current_type = str(df[col].dtype)
        suggested_type = current_type
        confidence = 'Low'
        reason = ''
        sample_values = []
        
        # Skip if already optimal type
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        
        col_data = sample_df[col].dropna()
        if len(col_data) == 0:
            continue
        
        # Convert to string for analysis
        str_data = col_data.astype(str)
        
        # Try to detect datetime with multiple formats
        datetime_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y.%m.%d',
            '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M', '%m/%d/%Y %I:%M %p'
        ]
        
        datetime_matches = 0
        for fmt in datetime_formats:
            try:
                parsed = pd.to_datetime(str_data, format=fmt, errors='coerce')
                matches = parsed.notna().sum()
                if matches > datetime_matches:
                    datetime_matches = matches
                    best_format = fmt
            except:
                continue
        
        if datetime_matches / len(col_data) > 0.8:
            suggested_type = 'datetime'
            confidence = 'High'
            reason = f'{datetime_matches/len(col_data)*100:.0f}% parseable as dates (format: {best_format})'
            sample_values = str_data.head(3).tolist()
        
        # Try to detect numeric (if not datetime)
        if suggested_type == current_type:
            try:
                numeric = pd.to_numeric(str_data, errors='coerce')
                valid_pct = numeric.notna().sum() / len(col_data)
                
                if valid_pct > 0.9:
                    # Check if integer or float
                    numeric_full = pd.to_numeric(df[col], errors='coerce')
                    if (numeric_full.dropna() % 1 == 0).all():
                        suggested_type = 'Int64'
                    else:
                        suggested_type = 'float64'
                    
                    confidence = 'High' if valid_pct > 0.95 else 'Medium'
                    reason = f'{valid_pct*100:.0f}% numeric values'
                    sample_values = col_data.head(3).astype(str).tolist()
            except:
                pass
        
        # Detect category (if not datetime or numeric)
        if suggested_type == current_type:
            unique_ratio = df[col].nunique() / len(df[col])
            if unique_ratio < 0.05:
                suggested_type = 'category'
                confidence = 'High'
                reason = f'Only {df[col].nunique()} unique values ({unique_ratio*100:.1f}%)'
                sample_values = df[col].dropna().unique()[:3].tolist()
        
        # Detect mixed types
        if suggested_type == current_type and df[col].dtype == 'object':
            # Check for mixed content
            has_numbers = any(str(v).replace('.', '').replace('-', '').isdigit() for v in col_data.head(10))
            has_letters = any(any(c.isalpha() for c in str(v)) for v in col_data.head(10))
            
            if has_numbers and has_letters:
                suggested_type = 'string'
                confidence = 'Medium'
                reason = 'Mixed content (numbers and text)'
                sample_values = col_data.head(3).tolist()
        
        if suggested_type != current_type:
            suggestions.append({
                'Column': col,
                'Current Type': current_type,
                'Suggested Type': suggested_type,
                'Confidence': confidence,
                'Reason': reason,
                'Sample Values': str(sample_values)[:100] + '...' if len(str(sample_values)) > 100 else str(sample_values)
            })
    
    return pd.DataFrame(suggestions)