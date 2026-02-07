# Memory optimization functions
import pandas as pd
import numpy as np
import streamlit as st
import gc
from typing import Dict, Tuple

def optimize_dataframe_memory(df: pd.DataFrame, categorical_threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict]:
    """Optimize DataFrame memory usage by downcasting and converting types"""
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    df_optimized = df.copy()
    changes = {}
    
    # Downcast numeric columns
    numeric_cols = df_optimized.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_min = df_optimized[col].min()
        col_max = df_optimized[col].max()
        
        # Integer columns
        if pd.api.types.is_integer_dtype(df_optimized[col]):
            if col_min >= 0:
                if col_max < 256:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='unsigned')
                    changes[col] = 'uint8'
                elif col_max < 65536:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='unsigned')
                    changes[col] = 'uint16'
                elif col_max < 4294967296:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='unsigned')
                    changes[col] = 'uint32'
            else:
                if col_min > -128 and col_max < 128:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
                    changes[col] = 'int8'
                elif col_min > -32768 and col_max < 32768:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
                    changes[col] = 'int16'
                elif col_min > -2147483648 and col_max < 2147483648:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
                    changes[col] = 'int32'
        # Float columns
        elif pd.api.types.is_float_dtype(df_optimized[col]):
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
            changes[col] = 'float32'
    
    # Convert object columns to category if low cardinality
    object_cols = df_optimized.select_dtypes(include=['object']).columns
    for col in object_cols:
        unique_ratio = df_optimized[col].nunique() / len(df_optimized)
        if unique_ratio < categorical_threshold:
            df_optimized[col] = df_optimized[col].astype('category')
            changes[col] = 'category'
    
    optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
    savings = original_memory - optimized_memory
    savings_pct = (savings / original_memory) * 100 if original_memory > 0 else 0
    
    return df_optimized, {
        'original_memory_mb': round(original_memory, 2),
        'optimized_memory_mb': round(optimized_memory, 2),
        'savings_mb': round(savings, 2),
        'savings_pct': round(savings_pct, 2),
        'changes': changes
    }

def clear_model_cache() -> None:
    """Clear trained models from memory"""
    if 'trained_models' in st.session_state:
        st.session_state.trained_models = {}
    if 'model_results' in st.session_state:
        st.session_state.model_results = []
    gc.collect()
    st.success("✅ Model cache cleared!")

def reset_feature_engineering_state() -> None:
    """Reset feature engineering state when dataset changes"""
    st.session_state.feature_engineering_columns = {'selected': [], 'excluded': []}
    st.session_state.feature_pipelines = {}
    # Note: We'll need to import log_action from history_utils
    # log_action("Feature engineering state reset", snapshot=False)

def sync_feature_engineering_state(df: pd.DataFrame) -> None:
    """Synchronize feature engineering state with current DataFrame"""
    current_columns = set(df.columns)
    
    # Remove references to columns that no longer exist
    st.session_state.feature_engineering_columns['selected'] = [
        col for col in st.session_state.feature_engineering_columns['selected'] 
        if col in current_columns
    ]
    
    # Update excluded columns
    new_excluded = {}
    for col, reason in st.session_state.feature_engineering_columns.get('excluded', {}).items():
        if col in current_columns:
            new_excluded[col] = reason
    st.session_state.feature_engineering_columns['excluded'] = new_excluded
    
    # Auto-select if no columns selected
    if not st.session_state.feature_engineering_columns['selected']:
        essential, excluded = auto_select_essential_columns(df)
        st.session_state.feature_engineering_columns['selected'] = essential
        st.session_state.feature_engineering_columns['excluded'] = excluded

def auto_select_essential_columns(df: pd.DataFrame) -> Tuple[list, dict]:
    """Automatically select essential columns and provide reasons for exclusions"""
    essential = []
    excluded = {}
    
    for col in df.columns:
        # Check if constant
        if df[col].nunique() <= 1:
            excluded[col] = "Constant column (only 1 unique value)"
            continue
        
        # Check if ID column (high cardinality)
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.95 and df[col].dtype in ['object', 'int64', 'int32']:
            excluded[col] = f"Likely ID column ({unique_ratio*100:.1f}% unique)"
            continue
        
        # Check if high missing
        missing_pct = df[col].isnull().sum() / len(df)
        if missing_pct > 0.8:
            excluded[col] = f"High missing values ({missing_pct*100:.1f}%)"
            continue
        
        # Check if text/object without encoding
        if df[col].dtype == 'object' and df[col].nunique() > 20:
            excluded[col] = "Text column (needs encoding first)"
            continue
        
        # Otherwise, it's essential
        essential.append(col)
    
    # Check for high correlation among essential numeric columns
    numeric_essential = [col for col in essential if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_essential) >= 2:
        corr_matrix = df[numeric_essential].corr().abs()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    col_to_drop = corr_matrix.columns[j]
                    col_to_keep = corr_matrix.columns[i]
                    if col_to_drop in essential:
                        essential.remove(col_to_drop)
                        excluded[col_to_drop] = f"High correlation with {col_to_keep} (r={corr_matrix.iloc[i,j]:.3f})"
    
    return essential, excluded