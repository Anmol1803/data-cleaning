
import warnings
warnings.filterwarnings('ignore')

import io
import json
import ast
import gc
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any, Union
from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Optional imports with fallbacks
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    from sklearn.ensemble import (IsolationForest, RandomForestClassifier, RandomForestRegressor, 
                                   GradientBoostingClassifier, GradientBoostingRegressor, 
                                   AdaBoostClassifier, AdaBoostRegressor, VotingClassifier, VotingRegressor,
                                   StackingClassifier, StackingRegressor)
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, PolynomialFeatures
    from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, KNeighborsRegressor
    from sklearn.linear_model import (LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet)
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                                 confusion_matrix, classification_report, roc_auc_score, roc_curve,
                                 mean_squared_error, mean_absolute_error, r2_score)
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

# ---------------------------------------------------------------------
# NEW: Dependency checking system
# ---------------------------------------------------------------------
def check_dependencies() -> Dict[str, bool]:
    """Check availability of all optional packages"""
    dependencies = {
        'scikit-learn': SKLEARN_AVAILABLE,
        'rapidfuzz': RAPIDFUZZ_AVAILABLE,
        'ydata_profiling': PROFILING_AVAILABLE,
        'xgboost': XGBOOST_AVAILABLE,
        'lightgbm': LIGHTGBM_AVAILABLE,
        'joblib': 'joblib' in globals() or 'joblib' in locals()
    }
    return dependencies

def graceful_fallback(package_name: str, feature_name: str) -> None:
    """Show user-friendly message for missing dependencies"""
    st.warning(f"⚠️ **{feature_name}** requires `{package_name}`")
    st.info(f"Install with: `pip install {package_name}`")
    if package_name == 'scikit-learn':
        st.info("Alternative: Use manual outlier detection methods")
    elif package_name == 'rapidfuzz':
        st.info("Alternative: Use exact text matching instead of fuzzy")
    elif package_name == 'ydata_profiling':
        st.info("Alternative: Use the built-in data overview features")
    elif package_name in ['xgboost', 'lightgbm']:
        st.info("Alternative: Use Random Forest or Gradient Boosting from scikit-learn")

# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Data Cleaning Pipeline",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------
# Enhanced Session state initialization
# ---------------------------------------------------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'action_log' not in st.session_state:
    st.session_state.action_log = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'future' not in st.session_state:
    st.session_state.future = []
if 'action_seq' not in st.session_state:
    st.session_state.action_seq = 0
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'pivot_tables' not in st.session_state:
    st.session_state.pivot_tables = []
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}
if 'data_quality_score' not in st.session_state:
    st.session_state.data_quality_score = None
if 'feature_pipelines' not in st.session_state:
    st.session_state.feature_pipelines = {}
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = []
if 'feature_engineering_columns' not in st.session_state:
    st.session_state.feature_engineering_columns = {'selected': [], 'excluded': []}
if 'detailed_changes' not in st.session_state:
    st.session_state.detailed_changes = []
if 'memory_optimized' not in st.session_state:
    st.session_state.memory_optimized = False
if 'active_expanders' not in st.session_state:
    st.session_state.active_expanders = {}
if 'dependencies_checked' not in st.session_state:
    st.session_state.dependencies_checked = False

# ---------------------------------------------------------------------
# NEW: Memory management functions
# ---------------------------------------------------------------------
@st.cache_data
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

# ---------------------------------------------------------------------
# NEW: Feature engineering state management
# ---------------------------------------------------------------------
def reset_feature_engineering_state() -> None:
    """Reset feature engineering state when dataset changes"""
    st.session_state.feature_engineering_columns = {'selected': [], 'excluded': []}
    st.session_state.feature_pipelines = {}
    log_action("Feature engineering state reset", snapshot=False)

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

# ---------------------------------------------------------------------
# Enhanced History utilities with Power BI-style tracking
# ---------------------------------------------------------------------
def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _now_ts() -> float:
    return datetime.now().timestamp()

def _snapshot(label: str, detailed_changes: Optional[List[Dict]] = None) -> None:
    """Enhanced snapshot with detailed change tracking"""
    if st.session_state.df is not None:
        # Increment action sequence
        st.session_state.action_seq += 1
        
        # Create history entry with enhanced details
        history_entry = {
            'id': st.session_state.action_seq,
            'label': label,
            'time': _now_str(),
            'timestamp': _now_ts(),
            'df': st.session_state.df.copy(deep=True),
            'detailed_changes': detailed_changes or [],
            'step': st.session_state.current_step
        }
        
        # Add to history and clear redo stack
        st.session_state.history.append(history_entry)
        st.session_state.future = []  # Clear redo stack on new action

def init_history_on_upload(label: str) -> None:
    """Initialize history only when a new dataset is uploaded."""
    if 'df' not in st.session_state or st.session_state.get('dataset_label') != label:
        # Clear all states on new dataset
        st.session_state.history = []
        st.session_state.future = []
        st.session_state.action_seq = 0
        st.session_state.action_log = []
        st.session_state.detailed_changes = []
        st.session_state.pivot_tables = []
        st.session_state.trained_models = {}
        st.session_state.model_results = []
        st.session_state.dataset_label = label
        
        # Reset feature engineering
        reset_feature_engineering_state()
        
        # Take initial snapshot
        if st.session_state.df is not None:
            _snapshot(label, [{'type': 'dataset_upload', 'details': f'Uploaded: {label}'}])

def log_bulk_action(base_message: str, changes: List[Dict]) -> None:
    """Log bulk operations with individual change tracking"""
    timestamp = _now_str()
    
    # Add main action
    st.session_state.action_log.append(f"[{timestamp}] {base_message}")
    
    # Add individual changes
    for change in changes:
        col = change.get('column', 'Unknown')
        operation = change.get('operation', 'Unknown')
        details = change.get('details', '')
        st.session_state.action_log.append(f"  └─ {col}: {operation} {details}")
    
    # Store detailed changes for history
    st.session_state.detailed_changes = changes

def log_action(message: str, snapshot: bool = False, detailed_changes: Optional[List[Dict]] = None) -> None:
    """Enhanced log action with detailed change tracking"""
    timestamp = _now_str()
    st.session_state.action_log.append(f"[{timestamp}] {message}")
    
    if snapshot:
        _snapshot(message, detailed_changes or st.session_state.detailed_changes)
        # Clear detailed changes after snapshot
        st.session_state.detailed_changes = []

def undo_last() -> None:
    """Enhanced undo with detailed tracking"""
    if len(st.session_state.history) <= 1:
        st.warning("No more steps to undo.")
        return
    
    last = st.session_state.history.pop()
    st.session_state.future.append(last)
    prev = st.session_state.history[-1]
    st.session_state.df = prev['df'].copy(deep=True)
    
    # Log undo action
    change_count = len(last.get('detailed_changes', []))
    log_message = f"Undone: {last['label']} ({change_count} changes)"
    st.session_state.action_log.append(f"[{_now_str()}] ↩️ {log_message}")
    st.success(log_message)
    
    # Sync feature engineering state
    if st.session_state.df is not None:
        sync_feature_engineering_state(st.session_state.df)

def redo_next() -> None:
    """Enhanced redo with detailed tracking"""
    if not st.session_state.future:
        st.warning("Nothing to redo.")
        return
    
    nxt = st.session_state.future.pop()
    st.session_state.history.append({
        'id': nxt['id'],
        'label': nxt['label'],
        'time': _now_str(),
        'timestamp': _now_ts(),
        'df': nxt['df'].copy(deep=True),
        'detailed_changes': nxt.get('detailed_changes', []),
        'step': nxt.get('step', st.session_state.current_step)
    })
    st.session_state.df = nxt['df'].copy(deep=True)
    
    # Log redo action
    change_count = len(nxt.get('detailed_changes', []))
    log_message = f"Redone: {nxt['label']} ({change_count} changes)"
    st.session_state.action_log.append(f"[{_now_str()}] ↪️ {log_message}")
    st.success(log_message)
    
    # Sync feature engineering state
    if st.session_state.df is not None:
        sync_feature_engineering_state(st.session_state.df)

def revert_to_action(action_id: int) -> None:
    """Revert to specific action with detailed tracking"""
    ids = [h['id'] for h in st.session_state.history]
    if action_id not in ids:
        st.error("Selected action not found in history.")
        return
    
    idx = ids.index(action_id)
    trimmed = st.session_state.history[:idx+1]
    removed = st.session_state.history[idx+1:]
    
    # Calculate total changes being reverted
    total_changes = sum(len(h.get('detailed_changes', [])) for h in removed)
    
    st.session_state.history = trimmed
    st.session_state.future = removed
    st.session_state.df = st.session_state.history[-1]['df'].copy(deep=True)
    
    # Log revert action
    log_message = f"Reverted to step #{action_id}: {st.session_state.history[-1]['label']} ({len(removed)} steps, {total_changes} changes)"
    st.session_state.action_log.append(f"[{_now_str()}] ⏪ {log_message}")
    st.success(log_message)
    
    # Sync feature engineering state
    if st.session_state.df is not None:
        sync_feature_engineering_state(st.session_state.df)

def render_enhanced_action_log_ui():
    """Enhanced action log UI with Power BI-style features"""
    st.markdown("#### 📝 Applied Steps Timeline")
    
    if not st.session_state.history:
        st.info("No steps yet.")
        return
    
    # Timeline view
    hist_rows = []
    for h in st.session_state.history:
        change_count = len(h.get('detailed_changes', []))
        hist_rows.append({
            "Step": f"#{h['id']}",
            "Action": h['label'],
            "Time": h['time'],
            "Changes": f"{change_count}",
            "Status": "✅"
        })
    
    hist_df = pd.DataFrame(hist_rows)
    st.dataframe(hist_df, use_container_width=True, height=240)
    
    # Step selection with details
    st.markdown("---")
    st.markdown("#### 🔍 Step Details")
    
    step_options = [f"#{h['id']} - {h['label']}" for h in st.session_state.history]
    
    # FIXED: Use unique key based on current step
    selected_step = st.selectbox(
        "Select step to view details:", 
        step_options, 
        key=f"step_select_pipeline_{st.session_state.current_step}"
    )
    
    if selected_step:
        step_id = int(selected_step.split('#')[1].split(' ')[0])
        selected_history = next((h for h in st.session_state.history if h['id'] == step_id), None)
        
        if selected_history:
            st.write(f"**Action:** {selected_history['label']}")
            st.write(f"**Time:** {selected_history['time']}")
            
            # Show detailed changes
            detailed_changes = selected_history.get('detailed_changes', [])
            if detailed_changes:
                st.markdown("**Individual Changes:**")
                for change in detailed_changes:
                    col = change.get('column', 'Unknown')
                    operation = change.get('operation', 'Unknown')
                    details = change.get('details', '')
                    st.write(f"  • **{col}**: {operation} {details}")
            else:
                st.info("No detailed change tracking for this step")
    
    # Enhanced controls - USING UNIQUE KEYS BASED ON CURRENT STEP
    st.markdown("---")
    st.markdown("#### ⚡ Controls")
    
    # FIXED: Use current_step in key instead of time.time()
    if st.button("↩️ Undo Last Step", use_container_width=True, key=f"undo_{st.session_state.current_step}"):
        undo_last()
        st.rerun()
    
    if st.button("↪️ Redo Next", use_container_width=True, key=f"redo_{st.session_state.current_step}"):
        redo_next()
        st.rerun()
    
    if st.session_state.history:
        revert_options = [f"#{h['id']} - {h['label'][:30]}..." for h in st.session_state.history]
        
        # FIXED: Unique key for revert selectbox
        revert_to = st.selectbox(
            "Revert to:", 
            revert_options, 
            key=f"revert_select_pipeline_{st.session_state.current_step}"
        )
        
        # FIXED: Unique key for revert button
        if st.button("⏪ Revert to Selected Step", use_container_width=True, key=f"revert_{st.session_state.current_step}"):
            step_id = int(revert_to.split('#')[1].split(' ')[0])
            revert_to_action(step_id)
            st.rerun()
    
    # FIXED: Unique key for clear button
    if st.button("🗑️ Clear Entire History", use_container_width=True, type="secondary", key=f"clear_{st.session_state.current_step}"):
        st.session_state.action_log = []
        st.session_state.detailed_changes = []
        st.rerun()
    
    # Export action log
    st.markdown("---")
    
    # FIXED: Unique key for export button
    if st.button("📥 Export Action Log", use_container_width=True, key=f"export_{st.session_state.current_step}"):
        log_text = "\n".join(st.session_state.action_log[-100:])  # Last 100 entries
        st.download_button(
            label="Download Log",
            data=log_text,
            file_name=f"action_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
# ---------------------------------------------------------------------
# Enhanced Smart Feature Selection for Engineering
# ---------------------------------------------------------------------
def auto_select_essential_columns(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
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

# ---------------------------------------------------------------------
# Enhanced Data Type Detection
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Enhanced Column Type Conversion with Nullable Type Support
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Enhanced ML Model Training & Evaluation with Validation
# ---------------------------------------------------------------------
def validate_classification_data(X: pd.DataFrame, y: pd.Series) -> Tuple[bool, List[str]]:
    """Validate data for classification tasks"""
    warnings = []
    
    # Check class distribution
    class_counts = y.value_counts()
    total_samples = len(y)
    
    if len(class_counts) < 2:
        return False, ["Need at least 2 classes for classification"]
    
    # Check for severe class imbalance
    majority_pct = (class_counts.iloc[0] / total_samples) * 100
    if majority_pct > 90:
        warnings.append(f"Severe class imbalance: {majority_pct:.1f}% in majority class")
    
    # Check minimum samples per class
    min_samples = class_counts.min()
    if min_samples < 10:
        warnings.append(f"Very small class: only {min_samples} samples")
    
    # Check feature variance
    constant_features = X.columns[X.nunique() <= 1]
    if len(constant_features) > 0:
        warnings.append(f"Constant features found: {', '.join(constant_features[:3])}")
    
    return True, warnings

def validate_regression_data(X: pd.DataFrame, y: pd.Series) -> Tuple[bool, List[str]]:
    """Validate data for regression tasks"""
    warnings = []
    
    # Check target variance
    target_var = y.var()
    if target_var < 1e-10:
        return False, ["Target variable is constant"]
    
    # Check target distribution
    y_skew = y.skew()
    if abs(y_skew) > 3:
        warnings.append(f"Highly skewed target (skewness: {y_skew:.2f})")
    
    # Check for outliers in target
    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr = q3 - q1
    outliers = ((y < (q1 - 1.5 * iqr)) | (y > (q3 + 1.5 * iqr))).sum()
    if outliers / len(y) > 0.05:
        warnings.append(f"{outliers} outliers in target ({outliers/len(y)*100:.1f}%)")
    
    # Check feature variance
    constant_features = X.columns[X.nunique() <= 1]
    if len(constant_features) > 0:
        warnings.append(f"Constant features found: {', '.join(constant_features[:3])}")
    
    return True, warnings

def validate_ml_parameters(problem_type: str, test_size: float, cv_folds: int, 
                          selected_models: List[str]) -> Tuple[bool, List[str]]:
    """Validate ML parameters before training"""
    warnings = []
    
    # Validate test size
    if test_size < 0.1 or test_size > 0.5:
        warnings.append(f"Test size {test_size:.1%} is unusual. Recommended: 20-30%")
    
    # Validate CV folds
    if cv_folds < 3 or cv_folds > 10:
        warnings.append(f"CV folds {cv_folds} is unusual. Recommended: 5-10")
    
    # Check model availability
    missing_models = []
    if problem_type == 'classification':
        available_models = list(get_classification_models().keys())
    else:
        available_models = list(get_regression_models().keys())
    
    for model in selected_models:
        if model not in available_models:
            missing_models.append(model)
    
    if missing_models:
        warnings.append(f"Some models not available: {', '.join(missing_models)}")
    
    return len(warnings) == 0, warnings

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, problem_type, cv_folds=5):
    """Train model and return comprehensive evaluation metrics"""
    try:
        # Train model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Predictions
        y_pred = model.predict(X_test)
        
        results = {
            'model_name': model_name,
            'model_object': model,
            'training_time': training_time,
            'predictions': y_pred
        }
        
        if problem_type == 'classification':
            # Classification metrics
            results['accuracy'] = accuracy_score(y_test, y_pred)
            results['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            results['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            results['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            
            # ROC AUC (for binary classification)
            if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                results['y_pred_proba'] = y_pred_proba
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            results['cv_mean'] = cv_scores.mean()
            results['cv_std'] = cv_scores.std()
            
        else:  # regression
            # Regression metrics
            results['r2'] = r2_score(y_test, y_pred)
            results['mse'] = mean_squared_error(y_test, y_pred)
            results['rmse'] = np.sqrt(results['mse'])
            results['mae'] = mean_absolute_error(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
            results['cv_mean'] = cv_scores.mean()
            results['cv_std'] = cv_scores.std()
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            results['feature_importance'] = np.abs(model.coef_).flatten()
        
        return results
        
    except Exception as e:
        return {'model_name': model_name, 'error': str(e)}

def generate_model_insights(results, problem_type, X_train, y_train):
    """Generate AI-powered insights about model performance"""
    insights = []
    warnings = []
    recommendations = []
    
    if 'error' in results:
        return {
            'insights': [],
            'warnings': [f"⚠️ Model training failed: {results['error']}"],
            'recommendations': ["Try scaling your features or checking for invalid values"]
        }
    
    model_name = results['model_name']
    
    # Performance insights
    if problem_type == 'classification':
        acc = results['accuracy']
        if acc >= 0.9:
            insights.append(f"🌟 Excellent performance! {model_name} achieves {acc*100:.1f}% accuracy")
        elif acc >= 0.75:
            insights.append(f"👍 Good performance. {model_name} achieves {acc*100:.1f}% accuracy")
        elif acc >= 0.6:
            insights.append(f"⚠️ Moderate performance. {model_name} achieves {acc*100:.1f}% accuracy")
        else:
            warnings.append(f"❌ Poor performance. {model_name} only achieves {acc*100:.1f}% accuracy")
        
        # Check for overfitting
        train_acc = results['cv_mean']
        test_acc = acc
        if train_acc - test_acc > 0.1:
            warnings.append(f"⚠️ Possible overfitting detected (CV: {train_acc*100:.1f}%, Test: {test_acc*100:.1f}%)")
            recommendations.append("Consider: reducing model complexity, adding regularization, or collecting more data")
    
    else:  # regression
        r2 = results['r2']
        if r2 >= 0.9:
            insights.append(f"🌟 Excellent fit! {model_name} explains {r2*100:.1f}% of variance")
        elif r2 >= 0.7:
            insights.append(f"👍 Good fit. {model_name} explains {r2*100:.1f}% of variance")
        elif r2 >= 0.5:
            insights.append(f"⚠️ Moderate fit. {model_name} explains {r2*100:.1f}% of variance")
        else:
            warnings.append(f"❌ Poor fit. {model_name} only explains {r2*100:.1f}% of variance")
        
        # Check for overfitting
        train_r2 = results['cv_mean']
        test_r2 = r2
        if train_r2 - test_r2 > 0.15:
            warnings.append(f"⚠️ Possible overfitting detected (CV R²: {train_r2:.3f}, Test R²: {test_r2:.3f})")
            recommendations.append("Consider: regularization (Ridge/Lasso) or simpler model")
    
    # Feature importance insights
    if 'feature_importance' in results:
        top_3_idx = np.argsort(results['feature_importance'])[-3:][::-1]
        insights.append(f"📊 Top 3 most important features drive {results['feature_importance'][top_3_idx].sum()*100:.1f}% of predictions")
    
    # Training time insights
    if results['training_time'] > 10:
        warnings.append(f"⏱️ Slow training ({results['training_time']:.1f}s). Consider feature selection or simpler model for production")
    elif results['training_time'] < 0.5:
        insights.append(f"⚡ Fast training ({results['training_time']:.2f}s) - great for real-time applications!")
    
    # Model-specific recommendations
    if 'Linear' in model_name or 'Logistic' in model_name:
        recommendations.append("💡 Linear models assume linear relationships. Try polynomial features for non-linear patterns")
    elif 'Random Forest' in model_name:
        recommendations.append("💡 Random Forest is robust and interpretable. Good for production deployment")
    elif 'XGBoost' in model_name or 'LightGBM' in model_name:
        recommendations.append("💡 Gradient boosting models often win competitions. Consider hyperparameter tuning for peak performance")
    
    return {
        'insights': insights,
        'warnings': warnings,
        'recommendations': recommendations
    }

# ---------------------------------------------------------------------
# Enhanced Data Quality Calculations
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Enhanced Smart Auto-Clean Suggestions
# ---------------------------------------------------------------------
@st.cache_data
def generate_auto_clean_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    """Generate smart cleaning suggestions based on data characteristics"""
    suggestions = []
    
    for col in df.columns:
        col_data = df[col]
        missing_pct = (col_data.isnull().sum() / len(df)) * 100 if len(df) > 0 else 0
        
        if missing_pct == 0:
            continue
            
        suggestion = {
            'Column': col,
            'Missing %': round(missing_pct, 2),
            'Type': str(df[col].dtype),
            'Recommended Action': 'None',
            'Reason': ''
        }
        
        # High missing rate - suggest drop
        if missing_pct > 80:
            suggestion['Recommended Action'] = 'Drop Column'
            suggestion['Reason'] = 'Too many missing values (>80%)'
        
        # Numeric columns
        elif pd.api.types.is_numeric_dtype(col_data):
            skewness = col_data.skew()
            
            if abs(skewness) < 0.5:
                suggestion['Recommended Action'] = 'Fill with Mean'
                suggestion['Reason'] = 'Nearly symmetric distribution'
            elif abs(skewness) < 1.5:
                suggestion['Recommended Action'] = 'Fill with Median'
                suggestion['Reason'] = 'Moderately skewed distribution'
            else:
                if skewness > 0:
                    suggestion['Recommended Action'] = 'Fill with 10th Percentile'
                    suggestion['Reason'] = 'Highly right-skewed'
                else:
                    suggestion['Recommended Action'] = 'Fill with 90th Percentile'
                    suggestion['Reason'] = 'Highly left-skewed'
        
        # Categorical columns
        elif col_data.dtype == 'object' or col_data.dtype.name == 'category':
            mode_series = col_data.mode()
            if not mode_series.empty:
                mode_freq = (col_data == mode_series.iloc[0]).sum() / len(col_data)
                if mode_freq > 0.5:
                    suggestion['Recommended Action'] = 'Fill with Mode'
                    suggestion['Reason'] = f'Clear mode ({mode_freq*100:.1f}% frequency)'
                else:
                    suggestion['Recommended Action'] = 'Fill with "Unknown"'
                    suggestion['Reason'] = 'No dominant category'
            else:
                suggestion['Recommended Action'] = 'Fill with "Unknown"'
                suggestion['Reason'] = 'No mode available'
        
        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            suggestion['Recommended Action'] = 'Forward Fill'
            suggestion['Reason'] = 'Temporal data - use chronological fill'
        
        suggestions.append(suggestion)
    
    return pd.DataFrame(suggestions)

def apply_auto_clean_suggestions(df: pd.DataFrame, suggestions_df: pd.DataFrame, selected_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Apply selected auto-clean suggestions with bulk logging"""
    df_cleaned = df.copy()
    applied = []
    detailed_changes = []
    
    for _, row in suggestions_df.iterrows():
        col = row['Column']
        if col not in selected_cols:
            continue
            
        action = row['Recommended Action']
        
        try:
            if action == 'Drop Column':
                df_cleaned.drop(columns=[col], inplace=True)
                applied.append(f"Dropped {col}")
                detailed_changes.append({
                    'column': col,
                    'operation': 'drop_column',
                    'details': f'Dropped column with {row["Missing %"]}% missing'
                })
            
            elif action == 'Fill with Mean':
                mean_val = df_cleaned[col].mean()
                df_cleaned[col].fillna(mean_val, inplace=True)
                applied.append(f"Filled {col} with mean")
                detailed_changes.append({
                    'column': col,
                    'operation': 'fill_missing',
                    'details': f'Filled with mean: {mean_val:.4f}'
                })
            
            elif action == 'Fill with Median':
                median_val = df_cleaned[col].median()
                df_cleaned[col].fillna(median_val, inplace=True)
                applied.append(f"Filled {col} with median")
                detailed_changes.append({
                    'column': col,
                    'operation': 'fill_missing',
                    'details': f'Filled with median: {median_val:.4f}'
                })
            
            elif action == 'Fill with 10th Percentile':
                p10_val = df_cleaned[col].quantile(0.1)
                df_cleaned[col].fillna(p10_val, inplace=True)
                applied.append(f"Filled {col} with 10th percentile")
                detailed_changes.append({
                    'column': col,
                    'operation': 'fill_missing',
                    'details': f'Filled with 10th percentile: {p10_val:.4f}'
                })
            
            elif action == 'Fill with 90th Percentile':
                p90_val = df_cleaned[col].quantile(0.9)
                df_cleaned[col].fillna(p90_val, inplace=True)
                applied.append(f"Filled {col} with 90th percentile")
                detailed_changes.append({
                    'column': col,
                    'operation': 'fill_missing',
                    'details': f'Filled with 90th percentile: {p90_val:.4f}'
                })
            
            elif action == 'Fill with Mode':
                mode_val = df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else 'Unknown'
                df_cleaned[col].fillna(mode_val, inplace=True)
                applied.append(f"Filled {col} with mode")
                detailed_changes.append({
                    'column': col,
                    'operation': 'fill_missing',
                    'details': f'Filled with mode: {mode_val}'
                })
            
            elif action == 'Fill with "Unknown"':
                df_cleaned[col].fillna('Unknown', inplace=True)
                applied.append(f"Filled {col} with 'Unknown'")
                detailed_changes.append({
                    'column': col,
                    'operation': 'fill_missing',
                    'details': 'Filled with "Unknown"'
                })
            
            elif action == 'Forward Fill':
                df_cleaned[col] = df_cleaned[col].ffill()
                applied.append(f"Forward filled {col}")
                detailed_changes.append({
                    'column': col,
                    'operation': 'fill_missing',
                    'details': 'Forward filled'
                })
                
        except Exception as e:
            st.warning(f"Could not apply action for {col}: {str(e)}")
    
    return df_cleaned, applied

# ---------------------------------------------------------------------
# Performance Optimized Contextual Filling
# ---------------------------------------------------------------------

@st.cache_data(max_entries=5)
def fill_missing_contextually(
    df: pd.DataFrame,
    pivot_info: Dict,  # Your app's pivot table structure
    fill_mode: str = 'primary',  # 'primary', 'all', or 'best'
    similarity_threshold: float = 0.8
) -> pd.DataFrame:
    """
    CONTEXT-AWARE FILLING SPECIFICALLY FOR YOUR APP'S PIVOT TABLES
    
    Your app creates pivot tables with this structure:
    {
        'name': 'Pivot_Table_1',
        'data': pivot_df,  # The actual pivot DataFrame
        'config': {
            'rows': ['CityTier', 'Occupation'],  # Index columns
            'columns': ['Department'],  # Column headers (optional)
            'values': ['income', 'rent'],  # Value columns
            'agg_functions': ['mean', 'median'],  # Aggregation functions
            'normalized': False
        }
    }
    
    This function intelligently fills missing values by:
    1. Using row categories (CityTier, Occupation) to find matching groups
    2. Filling missing values from aggregated statistics (mean, median, etc.)
    3. Handling both single and multiple aggregation functions
    """
    
    df_filled = df.copy()
    pivot_df = pivot_info['data']
    config = pivot_info['config']
    
    # Get configuration
    row_fields = config.get('rows', [])
    value_fields = config.get('values', [])
    agg_functions = config.get('agg_functions', ['mean'])
    has_columns = 'columns' in config and config['columns']
    
    # Track filling statistics
    filled_count = 0
    attempted_fills = 0
    
    # --- SINGLE AGGREGATION FUNCTION (SIMPLE CASE) ---
    if len(agg_functions) == 1:
        agg_func = agg_functions[0]
        
        # Reset pivot to get row fields as columns
        pivot_reset = pivot_df.reset_index()
        
        # For each row with missing values
        for idx, row in df.iterrows():
            missing_cols = [col for col in value_fields if pd.isna(row.get(col))]
            if not missing_cols:
                continue
            
            # Try to match based on row fields
            match_mask = pd.Series(True, index=pivot_reset.index)
            match_found = True
            
            for row_field in row_fields:
                if row_field in df.columns and pd.notna(row.get(row_field)):
                    if row_field in pivot_reset.columns:
                        # Exact match for categorical
                        match_mask = match_mask & (pivot_reset[row_field] == row[row_field])
                    else:
                        # Field not in pivot (might be aggregated differently)
                        match_found = False
                        break
                else:
                    # Missing this row field, can't use for matching
                    match_found = False
                    break
            
            if not match_found:
                continue
            
            matching_rows = pivot_reset[match_mask]
            
            if len(matching_rows) > 0:
                match_row = matching_rows.iloc[0]  # Take first match
                
                # Fill missing values
                for missing_col in missing_cols:
                    if missing_col in pivot_reset.columns:
                        fill_value = match_row[missing_col]
                        if pd.notna(fill_value):
                            df_filled.at[idx, missing_col] = fill_value
                            filled_count += 1
                    else:
                        # Check if column exists with aggregation suffix
                        col_with_suffix = f"{missing_col}_{agg_func}"
                        if col_with_suffix in pivot_reset.columns:
                            fill_value = match_row[col_with_suffix]
                            if pd.notna(fill_value):
                                df_filled.at[idx, missing_col] = fill_value
                                filled_count += 1
    
    # --- MULTIPLE AGGREGATION FUNCTIONS (COMPLEX CASE) ---
    else:
        # Multiple agg functions create MultiIndex columns
        # Format depends on whether pivot has column grouping
        
        pivot_reset = pivot_df.reset_index()
        
        # Determine available aggregation values for each column
        available_aggs = {}
        for value_col in value_fields:
            available_aggs[value_col] = []
            for agg_func in agg_functions:
                # Check different possible column formats
                possible_cols = [
                    value_col,  # Simple column name
                    f"{value_col}_{agg_func}",  # With suffix
                    (value_col, agg_func),  # MultiIndex format
                    f"({value_col}, {agg_func})"  # String representation
                ]
                
                for col_name in possible_cols:
                    if col_name in pivot_reset.columns:
                        available_aggs[value_col].append((agg_func, col_name))
                        break
        
        # Fill mode selection
        if fill_mode == 'primary':
            # Use first aggregation function only
            primary_agg = agg_functions[0]
        elif fill_mode == 'best':
            # Choose based on data characteristics
            primary_agg = 'median' if 'median' in agg_functions else agg_functions[0]
        else:  # 'all' - will try all aggregation functions
            
            # For each row with missing values
            for idx, row in df.iterrows():
                missing_cols = [col for col in value_fields if pd.isna(row.get(col))]
                if not missing_cols:
                    continue
                
                # Try to match based on row fields
                match_mask = pd.Series(True, index=pivot_reset.index)
                match_found = True
                
                for row_field in row_fields:
                    if row_field in df.columns and pd.notna(row.get(row_field)):
                        if row_field in pivot_reset.columns:
                            match_mask = match_mask & (pivot_reset[row_field] == row[row_field])
                        else:
                            match_found = False
                            break
                    else:
                        match_found = False
                        break
                
                if not match_found:
                    continue
                
                matching_rows = pivot_reset[match_mask]
                
                if len(matching_rows) > 0:
                    match_row = matching_rows.iloc[0]
                    
                    # Try to fill each missing column
                    for missing_col in missing_cols:
                        if missing_col in available_aggs:
                            # Try all available aggregation functions
                            for agg_func, col_name in available_aggs[missing_col]:
                                if col_name in match_row:
                                    fill_value = match_row[col_name]
                                    if pd.notna(fill_value):
                                        df_filled.at[idx, missing_col] = fill_value
                                        filled_count += 1
                                        break
    
    # Report statistics
    if filled_count > 0:
        print(f"✅ Context-aware filling completed!")
        print(f"   Rows processed: {len(df)}")
        print(f"   Values filled: {filled_count}")
        print(f"   Pivot structure: {row_fields} → {value_fields}")
        print(f"   Aggregations: {agg_functions}")
    
    return df_filled

# ---------------------------------------------------------------------
# Enhanced Fuzzy Duplicate Detection with Validation
# ---------------------------------------------------------------------
def find_fuzzy_duplicates(df: pd.DataFrame, column: str, threshold: int = 80) -> pd.DataFrame:
    """Find fuzzy duplicate values in a column with validation"""
    if not RAPIDFUZZ_AVAILABLE:
        graceful_fallback('rapidfuzz', 'Fuzzy Duplicate Detection')
        return pd.DataFrame()
    
    if column not in df.columns:
        st.error(f"Column '{column}' not found in dataset")
        return pd.DataFrame()
    
    if df[column].isnull().all():
        st.warning(f"Column '{column}' is all null values")
        return pd.DataFrame()
    
    values = df[column].dropna().astype(str).unique()
    
    if len(values) < 2:
        st.info(f"Column '{column}' has less than 2 unique non-null values")
        return pd.DataFrame()
    
    duplicates = []
    
    # Process in batches for performance
    batch_size = 100
    for i in range(0, len(values), batch_size):
        batch = values[i:i + batch_size]
        for j, val1 in enumerate(batch):
            for val2 in values[i + j + 1:]:
                score = fuzz.ratio(val1, val2)
                if score >= threshold:
                    duplicates.append({
                        'Value 1': val1,
                        'Value 2': val2,
                        'Similarity': score,
                        'Count 1': (df[column] == val1).sum(),
                        'Count 2': (df[column] == val2).sum()
                    })
    
    if duplicates:
        return pd.DataFrame(duplicates).sort_values('Similarity', ascending=False)
    else:
        return pd.DataFrame()

# ---------------------------------------------------------------------
# Enhanced ML Anomaly Detection with Parameter Validation
# ---------------------------------------------------------------------
def validate_anomaly_detection_params(df: pd.DataFrame, column: str, method: str, 
                                     contamination: float) -> Tuple[bool, List[str]]:
    """Validate parameters for ML anomaly detection"""
    warnings = []
    
    # Check column exists
    if column not in df.columns:
        return False, [f"Column '{column}' not found"]
    
    # Check data type
    if not pd.api.types.is_numeric_dtype(df[column]):
        return False, [f"Column '{column}' must be numeric for ML anomaly detection"]
    
    # Check sample size
    data = df[[column]].dropna()
    if len(data) < 10:
        return False, ["Need at least 10 non-null values for ML detection"]
    elif len(data) < 50:
        warnings.append(f"Small dataset ({len(data)} samples). Results may not be reliable.")
    
    # Validate contamination parameter
    if contamination < 0.01 or contamination > 0.5:
        return False, ["Contamination must be between 0.01 and 0.5"]
    elif contamination > 0.3:
        warnings.append(f"High contamination ({contamination:.2f}). May flag too many points as anomalies.")
    
    # Check column variance
    if data[column].std() < 1e-10:
        warnings.append(f"Column '{column}' has very low variance. Consider manual outlier detection.")
    
    return True, warnings

def detect_anomalies_ml(df: pd.DataFrame, column: str, method: str = 'isolation_forest', contamination: float = 0.1):
    """Detect anomalies using machine learning with validation"""
    if not SKLEARN_AVAILABLE:
        graceful_fallback('scikit-learn', 'ML Anomaly Detection')
        return None
    
    # Validate parameters
    is_valid, warnings = validate_anomaly_detection_params(df, column, method, contamination)
    if not is_valid:
        st.error("Parameter validation failed")
        return None
    
    for warning in warnings:
        st.warning(warning)
    
    data = df[[column]].dropna()
    
    try:
        if method == 'isolation_forest':
            model = IsolationForest(contamination=contamination, random_state=42)
            predictions = model.fit_predict(data)
        else:  # LOF
            model = LocalOutlierFactor(contamination=contamination)
            predictions = model.fit_predict(data)
        
        anomaly_indices = data.index[predictions == -1]
        return anomaly_indices
    except Exception as e:
        st.error(f"Error in ML detection: {str(e)}")
        return None

# ---------------------------------------------------------------------
# Enhanced Quick Insights Generator
# ---------------------------------------------------------------------
def generate_quick_insights(df: pd.DataFrame) -> Dict:
    """Generate comprehensive quick insights"""
    insights = {
        'narrative': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Missing data insights
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        total_missing = df.isnull().sum().sum()
        insights['narrative'].append(
            f"📊 Found {len(missing_cols)} columns with missing values ({total_missing:,} cells total)"
        )
        high_missing = [col for col in missing_cols if (df[col].isnull().sum() / len(df)) > 0.5]
        if high_missing:
            insights['warnings'].append(
                f"⚠️ {len(high_missing)} columns have >50% missing data: {', '.join(high_missing[:3])}"
            )
    
    # Duplicate insights
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        insights['narrative'].append(f"🔄 Found {dup_count} duplicate rows ({dup_count/len(df)*100:.1f}%)")
        if dup_count / len(df) > 0.1:
            insights['recommendations'].append("Consider removing duplicate rows")
    
    # Skewness insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        highly_skewed = []
        for col in numeric_cols:
            skew = df[col].skew()
            if abs(skew) > 2:
                highly_skewed.append((col, skew))
        if highly_skewed:
            insights['narrative'].append(
                f"📈 {len(highly_skewed)} columns are highly skewed (|skew| > 2)"
            )
            insights['recommendations'].append("Consider log transformation for skewed features")
    
    # Correlation insights
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.9:
                    high_corr_pairs.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                    )
        if high_corr_pairs:
            insights['warnings'].append(
                f"⚠️ Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.9)"
            )
            insights['recommendations'].append("Consider removing redundant features")
    
    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        insights['warnings'].append(
            f"⚠️ {len(constant_cols)} columns have only 1 unique value"
        )
        insights['recommendations'].append(f"Remove constant columns: {', '.join(constant_cols)}")
    
    # Class imbalance (for categorical with few classes)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if 2 <= df[col].nunique() <= 10:
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.iloc[0] > 0.9:
                insights['warnings'].append(
                    f"⚠️ '{col}' is highly imbalanced ({value_counts.iloc[0]*100:.1f}% in one class)"
                )
    
    return insights

# ---------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------
@st.cache_data
def compute_correlations(df: pd.DataFrame):
    pearson_corr = df.corr(numeric_only=True, method='pearson')
    spearman_corr = df.corr(numeric_only=True, method='spearman')
    return pearson_corr, spearman_corr

def encode_categorical_columns(df: pd.DataFrame):
    df_encoded = df.copy()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    encoding_info = {}
    for col in categorical_cols:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 20:
            cat = pd.Categorical(df[col])
            df_encoded[col] = cat.codes
            df_encoded[col] = df_encoded[col].replace(-1, np.nan)
            encoding_info[col] = dict(zip(cat.categories, range(len(cat.categories))))
    return df_encoded, encoding_info, categorical_cols

# ---------------------------------------------------------------------
# Enhanced UI helpers
# ---------------------------------------------------------------------
def show_overview_metrics(df: pd.DataFrame):
    st.header("📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with c2:
        st.metric("Columns", f"{df.shape[1]:,}")
    with c3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory (MB)", f"{memory_mb:.2f}")
    with c4:
        total_missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        st.metric("Missing (%)", f"{missing_pct:.1f}%")
    
    a, b = st.columns(2)
    with a:
        st.subheader("Data Types Distribution")
        dtype_counts = df.dtypes.value_counts()
        fig = px.pie(values=dtype_counts.values.tolist(),
                     names=[str(x) for x in dtype_counts.index],
                     title="Column Types")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.subheader("Missing Values by Column")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
        if len(missing_data) > 0:
            fig = px.bar(x=missing_data.values.tolist(),
                         y=[str(x) for x in missing_data.index],
                         orientation='h', title="Missing Values Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 No missing values found!")

def show_column_stats_card(df: pd.DataFrame, col: str):
    col_data = df[col]
    st.subheader(f"📋 Column: {col}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Count", f"{col_data.count():,}")
    with c2:
        st.metric("Missing", f"{col_data.isnull().sum():,}")
    with c3:
        st.metric("Unique", f"{col_data.nunique():,}")
    with c4:
        missing_pct = (col_data.isnull().sum() / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Missing %", f"{missing_pct:.1f}%")
    
    if pd.api.types.is_numeric_dtype(col_data):
        st.subheader("📈 Numerical Statistics")
        a, b = st.columns(2)
        with a:
            stats = col_data.describe()
            median = stats.get('50%') if '50%' in stats.index else col_data.median()
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std', 'Min', 'Max'],
                'Value': [stats.get('mean', np.nan), median, stats.get('std', np.nan),
                          stats.get('min', np.nan), stats.get('max', np.nan)]
            })
            st.dataframe(stats_df, use_container_width=True)
        with b:
            clean_data = col_data.dropna().tolist()
            temp_df = pd.DataFrame({col: clean_data})
            fig = px.histogram(temp_df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader("📝 Categorical Statistics")
        value_counts = col_data.value_counts().head(10)
        a, b = st.columns(2)
        with a:
            st.dataframe(value_counts, use_container_width=True)
        with b:
            if len(value_counts) > 1:
                fig = px.bar(
                    x=[str(x) for x in value_counts.index],
                    y=value_counts.values.tolist(),
                    title=f"Top Values in {col}"
                )
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Enhanced Steps with Visibility Fixes
# ---------------------------------------------------------------------
def data_type_management_step(df: pd.DataFrame):
    st.header("🔧 Data Type Management")
    
    # Smart Type Detection - EXPANDED BY DEFAULT
    with st.expander("🤖 Smart Type Detection", expanded=True):  # CHANGED: expanded=True
        st.markdown("**Automatically detect optimal data types for your columns**")
        if st.button("🔍 Analyze Data Types", type="primary"):
            with st.spinner("Analyzing data types..."):
                suggestions_df = detect_data_types(df)
                if not suggestions_df.empty:
                    st.success(f"Found {len(suggestions_df)} improvement suggestions!")
                    st.dataframe(suggestions_df, use_container_width=True)
                    
                    st.subheader("Apply Suggestions")
                    cols_to_convert = st.multiselect(
                        "Select columns to convert:",
                        suggestions_df['Column'].tolist()
                    )
                    
                    if cols_to_convert and st.button("✅ Apply Selected Conversions"):
                        detailed_changes = []
                        for col in cols_to_convert:
                            row = suggestions_df[suggestions_df['Column'] == col].iloc[0]
                            new_type = row['Suggested Type']
                            success, msg = convert_column_type(df, col, new_type)
                            if success:
                                st.success(f"✅ {col}: {msg}")
                                detailed_changes.append({
                                    'column': col,
                                    'operation': 'convert_type',
                                    'details': f'{row["Current Type"]} → {new_type} ({row["Confidence"]})'
                                })
                            else:
                                st.error(f"❌ {col}: {msg}")
                        
                        if detailed_changes:
                            log_bulk_action(f"Smart type conversion", detailed_changes)
                            log_action(f"Smart type conversion: {', '.join(cols_to_convert)}", snapshot=True)
                        st.rerun()
                else:
                    st.info("✨ All data types look optimal!")
    
    st.divider()
    
    # Existing functionality
    dtype_df = pd.DataFrame({
        'Column': df.columns,
        'Current Type': df.dtypes.astype(str),
        'Sample Values': [str(df[col].dropna().head(3).tolist()) for col in df.columns]
    })
    st.subheader("Current Data Types")
    st.dataframe(dtype_df, use_container_width=True)
    
    col_to_change = st.selectbox("Select column to modify:", [''] + list(df.columns))
    if col_to_change:
        show_column_stats_card(df, col_to_change)
        st.subheader("Change Data Type")
        new_type = st.selectbox("Select new data type:",
                                ['Int64', 'Float64', 'string', 'datetime', 'category', 'boolean', 'custom'])
        if new_type == 'custom':
            custom_type = st.text_input("Enter pandas dtype:")
            new_type = custom_type
        if st.button(f"Convert {col_to_change} to {new_type}"):
            success, message = convert_column_type(df, col_to_change, new_type)
            if success:
                st.success(f"✅ {message}")
                log_action(f"Changed {col_to_change} to {new_type}", snapshot=True)
                st.rerun()
            else:
                st.error(f"❌ {message}")

def text_cleaning_step(df: pd.DataFrame):
    st.header("📝 Text Cleaning Tools")
    text_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    if not text_cols:
        st.info("No text/categorical columns found in dataset.")
        return
    
    col_choice = st.selectbox("Select column to clean:", text_cols)
    
    # Fuzzy Duplicate Detection - EXPANDED BY DEFAULT
    with st.expander("🔍 Fuzzy Duplicate Detection", expanded=True):  # CHANGED: expanded=True
        if RAPIDFUZZ_AVAILABLE:
            st.markdown("**Find similar values that might be duplicates**")
            threshold = st.slider("Similarity Threshold (%):", 70, 95, 80, 5)
            
            if st.button("🔎 Find Fuzzy Duplicates"):
                with st.spinner("Analyzing for fuzzy duplicates..."):
                    fuzzy_dupes = find_fuzzy_duplicates(df, col_choice, threshold)
                    if not fuzzy_dupes.empty:
                        st.warning(f"Found {len(fuzzy_dupes)} potential duplicate pairs!")
                        st.dataframe(fuzzy_dupes, use_container_width=True)
                        
                        st.subheader("Merge Similar Values")
                        if len(fuzzy_dupes) > 0:
                            selected_pair = st.selectbox(
                                "Select pair to merge:",
                                range(len(fuzzy_dupes)),
                                format_func=lambda x: f"{fuzzy_dupes.iloc[x]['Value 1']} ↔ {fuzzy_dupes.iloc[x]['Value 2']} ({fuzzy_dupes.iloc[x]['Similarity']}%)"
                            )
                            pair = fuzzy_dupes.iloc[selected_pair]
                            keep_value = st.radio(
                                "Keep which value?",
                                [pair['Value 1'], pair['Value 2']]
                            )
                            replace_value = pair['Value 2'] if keep_value == pair['Value 1'] else pair['Value 1']
                            
                            if st.button("🔄 Merge These Values"):
                                df[col_choice] = df[col_choice].replace(replace_value, keep_value)
                                st.success(f"Merged '{replace_value}' into '{keep_value}'")
                                log_action(f"Fuzzy merge in {col_choice}: '{replace_value}' → '{keep_value}'", snapshot=True)
                                st.rerun()
                    else:
                        st.success("✨ No fuzzy duplicates found!")
        else:
            graceful_fallback('rapidfuzz', 'Fuzzy Duplicate Detection')
    
    # Text Profile - EXPANDED BY DEFAULT
    with st.expander("📊 Text Profile", expanded=True):  # CHANGED: expanded=True
        text_data = df[col_choice].dropna().astype(str)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Avg Length", f"{text_data.str.len().mean():.1f}")
        with c2:
            st.metric("Max Length", f"{text_data.str.len().max()}")
        with c3:
            st.metric("Min Length", f"{text_data.str.len().min()}")
        with c4:
            st.metric("Unique %", f"{(df[col_choice].nunique()/len(df)*100):.1f}%")
        
        # Length distribution
        lengths = text_data.str.len()
        fig = px.histogram(x=lengths, nbins=30, title="Text Length Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Existing text cleaning
    remove_spaces = st.checkbox("Remove extra spaces")
    remove_specials = st.checkbox("Remove special characters (keep letters/numbers/spaces)")
    case_action = st.radio("Case Conversion:", ["None", "Lowercase", "Uppercase", "Title Case"])
    
    if st.button("Apply Text Cleaning"):
        before = df[col_choice].copy()
        col_series = df[col_choice].astype(str)
        steps_applied = []
        if remove_spaces:
            col_series = col_series.str.strip().str.replace(r"\s+", " ", regex=True)
            steps_applied.append("trim-spaces")
        if remove_specials:
            col_series = col_series.str.replace(r"[^A-Za-z0-9\s]+", "", regex=True)
            steps_applied.append("remove-specials")
        if case_action == "Lowercase":
            col_series = col_series.str.lower()
            steps_applied.append("lowercase")
        elif case_action == "Uppercase":
            col_series = col_series.str.upper()
            steps_applied.append("uppercase")
        elif case_action == "Title Case":
            col_series = col_series.str.title()
            steps_applied.append("titlecase")
        df[col_choice] = col_series
        if steps_applied:
            log_action(f"Text clean {col_choice}: {', '.join(steps_applied)}", snapshot=True)
        else:
            log_action(f"Text clean {col_choice}: no-op", snapshot=False)
        st.success("Applied text cleaning")
        st.rerun()
    
    st.subheader("🔄 Replace Text")
    find_text = st.text_input("Find text:", key=f"find_{col_choice}")
    replace_text = st.text_input("Replace with:", key=f"replace_{col_choice}")
    if st.button("Replace", key=f"replace_btn_{col_choice}"):
        if find_text != "":
            df[col_choice] = df[col_choice].astype(str).str.replace(find_text, replace_text, regex=False)
            st.success(f"Replaced '{find_text}' with '{replace_text}' in '{col_choice}'")
            log_action(f"Replace in {col_choice}: '{find_text}' → '{replace_text}'", snapshot=True)
            st.rerun()
        else:
            st.warning("Please enter text to find.")
    
    st.subheader("🔎 Preview After Cleaning")
    st.write(df[[col_choice]].head(10))

def missing_values_treatment_step(df: pd.DataFrame):
    st.header("Step 4 · Missing Values")
    st.markdown(f"**Step 4 of 12**")
    
    # Create two columns layout for main content and pipeline panel
    col_main, col_pipeline = st.columns([3, 1])
    
    with col_main:
        # 1) OVERVIEW & QUICK FIXES (TOP, ALWAYS VISIBLE)
        st.markdown("## 1️⃣ Overview & Quick Fixes")
        
        # Show missing values summary
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if not missing_cols:
            st.success("🎉 No missing values found!")
            return df
        
        missing_summary = pd.DataFrame({
            'Column': missing_cols,
            'Missing Count': [df[col].isnull().sum() for col in missing_cols],
            'Missing %': [df[col].isnull().sum() / len(df) * 100 for col in missing_cols]
        }).sort_values('Missing Count', ascending=False)
        
        st.subheader("📊 Missing Values Summary")
        st.dataframe(missing_summary, use_container_width=True)
        
        # Missing values chart
        if len(missing_cols) > 0:
            fig = px.bar(
                missing_summary,
                x='Column',
                y='Missing %',
                title="Missing Values Percentage by Column",
                color='Missing %',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Column selection for quick fixes
        st.subheader("🎯 Quick Fixes for Selected Column")
        col_to_clean = st.selectbox("Select column to work on:", [''] + missing_cols)
        
        if col_to_clean:
            show_column_stats_card(df, col_to_clean)
            
            # Determine column type and show appropriate quick fixes
            is_numeric = pd.api.types.is_numeric_dtype(df[col_to_clean])
            
            # Auto recommendation
            if is_numeric:
                skewness = df[col_to_clean].skew()
                if abs(skewness) < 0.5:
                    recommendation = "Fill with Mean (nearly symmetric distribution)"
                elif abs(skewness) < 1.5:
                    recommendation = "Fill with Median (moderately skewed)"
                else:
                    if skewness > 0:
                        recommendation = "Fill with 10th Percentile (highly right-skewed)"
                    else:
                        recommendation = "Fill with 90th Percentile (highly left-skewed)"
            else:
                recommendation = "Fill with Most Frequent (categorical data)"
            
            st.info(f"🤖 **Recommended:** {recommendation}")
            
            # Quick fill options grouped by type
            st.subheader("🔧 Quick Fill Options")
            
            if is_numeric:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Central Tendency**")
                    if st.button("Fill with Mean", key="mean_btn"):
                        mean_val = df[col_to_clean].mean()
                        df[col_to_clean].fillna(mean_val, inplace=True)
                        st.success(f"Filled with mean ({mean_val:.2f})")
                        log_action(f"Filled {col_to_clean} with mean", snapshot=True)
                        st.rerun()
                    if st.button("Fill with Median", key="median_btn"):
                        median_val = df[col_to_clean].median()
                        df[col_to_clean].fillna(median_val, inplace=True)
                        st.success(f"Filled with median ({median_val:.2f})")
                        log_action(f"Filled {col_to_clean} with median", snapshot=True)
                        st.rerun()
                    if st.button("Fill with Mode", key="mode_btn"):
                        mode_series = df[col_to_clean].mode()
                        most_frequent = mode_series.iloc[0] if not mode_series.empty else 'Unknown'
                        df[col_to_clean].fillna(most_frequent, inplace=True)
                        st.success(f"Filled with most frequent value ({most_frequent})")
                        log_action(f"Filled {col_to_clean} with most frequent", snapshot=True)
                        st.rerun()
                
                with col2:
                    st.markdown("**Extremes & Quantiles**")
                    if st.button("Fill with Min", key="min_btn"):
                        min_val = df[col_to_clean].min()
                        df[col_to_clean].fillna(min_val, inplace=True)
                        st.success(f"Filled with minimum ({min_val:.2f})")
                        log_action(f"Filled {col_to_clean} with min", snapshot=True)
                        st.rerun()
                    if st.button("Fill with Max", key="max_btn"):
                        max_val = df[col_to_clean].max()
                        df[col_to_clean].fillna(max_val, inplace=True)
                        st.success(f"Filled with maximum ({max_val:.2f})")
                        log_action(f"Filled {col_to_clean} with max", snapshot=True)
                        st.rerun()
                
                with col3:
                    st.markdown("**Special Actions**")
                    if st.button("Forward Fill", key="ffill_btn"):
                        df[col_to_clean] = df[col_to_clean].ffill()
                        st.success("Forward filled")
                        log_action(f"Forward filled {col_to_clean}", snapshot=True)
                        st.rerun()
                    if st.button("Backward Fill", key="bfill_btn"):
                        df[col_to_clean] = df[col_to_clean].bfill()
                        st.success("Backward filled")
                        log_action(f"Backward filled {col_to_clean}", snapshot=True)
                        st.rerun()
                    if st.button("Drop Rows", key="drop_rows_btn"):
                        original_len = len(df)
                        df.dropna(subset=[col_to_clean], inplace=True)
                        dropped = original_len - len(df)
                        st.success(f"Dropped {dropped} rows")
                        log_action(f"Dropped {dropped} rows from {col_to_clean}", snapshot=True)
                        st.rerun()
                    
                    # Custom fill
                    custom_value = st.text_input("Custom fill value:", key=f"custom_{col_to_clean}")
                    if st.button("Fill with Custom", key="custom_btn") and custom_value != "":
                        try:
                            if pd.api.types.is_numeric_dtype(df[col_to_clean]):
                                cast_val = float(custom_value)
                            else:
                                cast_val = custom_value
                            df[col_to_clean].fillna(cast_val, inplace=True)
                            st.success(f"Filled with '{custom_value}'")
                            log_action(f"Filled {col_to_clean} with custom value: {custom_value}", snapshot=True)
                            st.rerun()
                        except Exception:
                            st.error("Could not cast custom value to column dtype. Filled as string instead.")
                            df[col_to_clean].fillna(custom_value, inplace=True)
                            log_action(f"Filled {col_to_clean} with custom value as string: {custom_value}", snapshot=True)
                            st.rerun()
            
            else:  # Categorical columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Categorical Fills**")
                    if st.button("Fill with Most Frequent", key="cat_mode_btn"):
                        mode_series = df[col_to_clean].mode()
                        most_frequent = mode_series.iloc[0] if not mode_series.empty else 'Unknown'
                        df[col_to_clean].fillna(most_frequent, inplace=True)
                        st.success(f"Filled with most frequent value ({most_frequent})")
                        log_action(f"Filled {col_to_clean} with most frequent", snapshot=True)
                        st.rerun()
                    
                    if st.button('Fill with "Unknown"', key="unknown_btn"):
                        df[col_to_clean].fillna('Unknown', inplace=True)
                        st.success('Filled with "Unknown"')
                        log_action(f'Filled {col_to_clean} with "Unknown"', snapshot=True)
                        st.rerun()
                
                with col2:
                    st.markdown("**Special Actions**")
                    if st.button("Show Unique Values", key="show_values_btn"):
                        st.subheader("Value Frequency")
                        value_counts = df[col_to_clean].value_counts(dropna=False)
                        a, b = st.columns(2)
                        with a:
                            st.dataframe(value_counts.head(20))
                        with b:
                            fig = px.bar(
                                x=[str(x) for x in value_counts.head(10).index],
                                y=value_counts.head(10).values.tolist(),
                                title="Top 10 Values"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if st.button("Drop Rows", key="cat_drop_btn"):
                        original_len = len(df)
                        df.dropna(subset=[col_to_clean], inplace=True)
                        dropped = original_len - len(df)
                        st.success(f"Dropped {dropped} rows")
                        log_action(f"Dropped {dropped} rows from {col_to_clean}", snapshot=True)
                        st.rerun()
        
        # 2) SMART AUTO-CLEAN (COLLAPSED EXPANDER)
        with st.expander("🤖 Smart Auto-Clean (Multi-column Suggestions)", expanded=False):
            st.markdown("**AI-powered missing value treatment based on data characteristics**")
            st.info("💡 This analyzes skewness, data types, and patterns to suggest optimal filling strategies")
            
            if st.button("🔍 Generate Suggestions", type="primary"):
                with st.spinner("Analyzing data and generating suggestions..."):
                    suggestions_df = generate_auto_clean_suggestions(df)
                    if not suggestions_df.empty:
                        st.session_state['auto_clean_suggestions'] = suggestions_df
                        st.success(f"✅ Generated {len(suggestions_df)} cleaning suggestions!")
                    else:
                        st.info("✨ No missing values to clean!")
            
            if 'auto_clean_suggestions' in st.session_state and st.session_state['auto_clean_suggestions'] is not None:
                suggestions_df = st.session_state['auto_clean_suggestions']
                if not suggestions_df.empty:
                    st.subheader("📋 Suggested Actions")
                    st.dataframe(suggestions_df, use_container_width=True)
                    
                    st.subheader("✅ Select Actions to Apply")
                    selected_cols = st.multiselect(
                        "Choose columns to apply suggestions:",
                        suggestions_df['Column'].tolist(),
                        default=suggestions_df['Column'].tolist()
                    )
                    
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        if st.button("🚀 Apply Selected Auto-Clean", type="primary", use_container_width=True):
                            with st.spinner("Applying auto-clean..."):
                                df_cleaned, applied = apply_auto_clean_suggestions(df, suggestions_df, selected_cols)
                                st.session_state.df = df_cleaned
                                
                                # Use bulk logging for multiple changes
                                if applied:
                                    detailed_changes = []
                                    for action in applied:
                                        # Extract column name from action string
                                        if "Filled" in action:
                                            col = action.split("Filled ")[1].split(" with")[0]
                                            op = "fill_missing"
                                            details = action.split("with ")[1] if "with " in action else ""
                                        elif "Dropped" in action:
                                            col = action.split("Dropped ")[1]
                                            op = "drop_column"
                                            details = ""
                                        elif "Forward filled" in action:
                                            col = action.split("Forward filled ")[1]
                                            op = "forward_fill"
                                            details = ""
                                        else:
                                            col = "unknown"
                                            op = "unknown"
                                            details = ""
                                        
                                        detailed_changes.append({
                                            'column': col,
                                            'operation': op,
                                            'details': details
                                        })
                                    
                                    log_bulk_action("Auto-clean applied to multiple columns", detailed_changes)
                                    log_action(f"Auto-clean: {len(applied)} actions", snapshot=True)
                                
                                st.success(f"✅ Applied {len(applied)} cleaning actions!")
                                for action in applied:
                                    st.write(f"  • {action}")
                                st.session_state['auto_clean_suggestions'] = None
                                st.rerun()
                    with c2:
                        if st.button("❌ Clear Suggestions"):
                            st.session_state['auto_clean_suggestions'] = None
                            st.rerun()
        
        # 3) PIVOT-BASED CONTEXT FILL (ADVANCED, COLLAPSED)
        with st.expander("🧠 Pivot-Based Context Fill (Advanced)", expanded=False):
            st.markdown("""
            ### **INTELLIGENT FILLING USING YOUR PIVOT TABLES**
            
            **How it works with YOUR app's pivot tables:**
            1. **Match Categories**: Uses your pivot's row fields (CityTier, Occupation) to find similar groups
            2. **Use Aggregated Stats**: Fills missing values with group statistics (mean, median, etc.)
            3. **Smart Selection**: Chooses the best aggregation method based on your pivot configuration
            
            **Example from YOUR app:**
            - Pivot: `rows=['CityTier', 'Occupation'], values=['income', 'rent'], agg=['mean', 'median']`
            - Missing income for Engineer in CityTier 1
            - Looks up: `pivot[CityTier=1, Occupation='Engineer']` → `income_mean=85000`
            - Fills missing income with 85000
            """)
            
            if st.session_state.pivot_tables:
                # Display available pivot tables
                pivot_options = []
                for i, pt in enumerate(st.session_state.pivot_tables):
                    config = pt['config']
                    desc = f"{pt['name']}: {config.get('rows', [])} → {config.get('values', [])} ({', '.join(config.get('agg_functions', []))})"
                    pivot_options.append((i, desc))
                
                chosen_idx = st.selectbox(
                    "Select pivot table:",
                    range(len(pivot_options)),
                    format_func=lambda x: pivot_options[x][1],
                    key="pivot_select_fill"
                )
                
                pivot_info = st.session_state.pivot_tables[chosen_idx]
                config = pivot_info['config']
                
                # Show pivot configuration
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Row Categories:** {', '.join(config.get('rows', []))}")
                    st.info(f"**Value Columns:** {', '.join(config.get('values', []))}")
                with col2:
                    st.info(f"**Aggregations:** {', '.join(config.get('agg_functions', []))}")
                    has_cols = 'columns' in config and config['columns']
                    st.info(f"**Column Groups:** {'Yes' if has_cols else 'No'}")
                
                # Configuration options
                st.markdown("---")
                st.markdown("### ⚙️ Filling Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(config.get('agg_functions', [])) > 1:
                        fill_mode = st.selectbox(
                            "Aggregation to use:",
                            ['primary', 'best', 'all'],
                            format_func=lambda x: {
                                'primary': f"Use {config['agg_functions'][0]} only",
                                'best': 'Use median if available',
                                'all': 'Try all aggregations'
                            }[x],
                            help="Which aggregation function to use for filling"
                        )
                    else:
                        fill_mode = 'primary'
                        st.info(f"Using: {config['agg_functions'][0]}")
                
                with col2:
                    similarity = st.slider(
                        "Matching strictness:",
                        0.5, 1.0, 0.8, 0.05,
                        help="Higher = stricter matching on row categories"
                    )
                
                # Preview what will be filled
                st.markdown("---")
                st.markdown("### 🔍 Preview Potential Fills")
                
                if st.button("🔍 Analyze Missing Values", key="analyze_fills"):
                    # Find rows with missing values in pivot value columns
                    missing_in_values = df[config.get('values', [])].isnull().any(axis=1)
                    missing_rows = df[missing_in_values]
                    
                    if len(missing_rows) > 0:
                        st.success(f"Found {len(missing_rows)} rows with missing values in pivot columns")
                        
                        # Show sample of what could be filled
                        sample = missing_rows.head(5).copy()
                        
                        # Add what would be used for matching
                        for row_field in config.get('rows', []):
                            if row_field in df.columns:
                                sample[f"Match_{row_field}"] = sample[row_field]
                        
                        st.dataframe(sample, use_container_width=True)
                        
                        # Estimate fill rate
                        total_missing = df[config.get('values', [])].isnull().sum().sum()
                        st.info(f"**Total missing values in target columns:** {total_missing}")
                    else:
                        st.success("🎉 No missing values in the pivot's value columns!")
                
                # Execute filling
                st.markdown("---")
                if st.button("🚀 Fill Missing Values Using Pivot", type="primary", use_container_width=True):
                    with st.spinner(f"Filling missing values using {pivot_info['name']}..."):
                        df_new = fill_missing_contextually(
                            df,
                            pivot_info,
                            fill_mode=fill_mode,
                            similarity_threshold=similarity
                        )
                        
                        # Calculate statistics
                        original_missing = df.isnull().sum().sum()
                        new_missing = df_new.isnull().sum().sum()
                        filled = original_missing - new_missing
                        
                        # Only columns in pivot values
                        pivot_value_missing_before = df[config.get('values', [])].isnull().sum().sum()
                        pivot_value_missing_after = df_new[config.get('values', [])].isnull().sum().sum()
                        pivot_value_filled = pivot_value_missing_before - pivot_value_missing_after
                        
                        st.session_state.df = df_new
                        
                        # Log action
                        if filled > 0:
                            log_action(
                                f"Context-aware fill using pivot '{pivot_info['name']}': Filled {filled} values ({pivot_value_filled} in target columns)",
                                snapshot=True
                            )
                            
                            st.success(f"✅ Successfully filled {filled} missing values!")
                            
                            # Detailed report
                            with st.expander("📊 Filling Report", expanded=True):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Values Filled", filled)
                                with col2:
                                    st.metric("Target Columns Filled", pivot_value_filled)
                                with col3:
                                    st.metric("Remaining Missing", new_missing)
                                
                                # Show what was filled
                                changed_cols = []
                                for col in df.columns:
                                    if not df[col].equals(df_new[col]):
                                        changed_cols.append(col)
                                
                                if changed_cols:
                                    st.info(f"**Columns modified:** {', '.join(changed_cols)}")
                        else:
                            st.warning("⚠️ No values could be filled. Check that your data has matching categories.")
                        
                        st.rerun()
            else:
                st.info("📊 No pivot tables found. Please create one first in the Pivot Table section.")
                
                st.markdown("""
                **How to create an effective pivot table for filling:**
                
                1. **Go to Pivot Tables section**
                2. **Choose Row Fields**: Select categorical columns for grouping (e.g., CityTier, Occupation)
                3. **Choose Value Fields**: Select numeric columns with missing values (e.g., income, rent)
                4. **Choose Aggregation**: Select mean, median, or both
                5. **Create Pivot Table**
                
                **Example configuration:**
                - Rows: CityTier, Occupation
                - Values: income, rent, expenses
                - Aggregation: mean, median
                
                This creates a knowledge base that can intelligently fill missing values!
                """)
        
        # 4) ADVANCED RULES & CALCULATED COLUMNS (BOTTOM, COLLAPSED WITH TABS)
        with st.expander("🧮 Advanced Rules & Calculated Columns", expanded=False):
            tab1, tab2, tab3 = st.tabs([
                "Conditional Fill (Missing Only)",
                "Full Column Formula",
                "Conditional Formula (If-Then-Else)",
            ])
            
            with tab1:
                st.markdown("**Fill missing values based on conditions in other columns**")
                st.info("💡 Example: If Gender='Male', fill Age with 25 (only where Age is missing)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    condition_col = st.selectbox("If column:", df.columns.tolist(), key="cond_col")
                    condition_operator = st.selectbox("Operator:", ["equals (=)", "not equals (≠)", "greater than (>)", "less than (<)", "contains text"], key="cond_op")
                    condition_value = st.text_input("Value:", key="cond_val")
                
                with col2:
                    target_col = st.selectbox("Then fill column:", df.columns.tolist(), key="target_col")
                    fill_value = st.text_input("With value:", key="fill_val")
                
                if st.button("🔍 Preview Conditional Fill", key="preview_cond"):
                    try:
                        # Create condition mask
                        if condition_operator == "equals (=)":
                            mask = df[condition_col].astype(str) == condition_value
                        elif condition_operator == "not equals (≠)":
                            mask = df[condition_col].astype(str) != condition_value
                        elif condition_operator == "greater than (>)":
                            mask = pd.to_numeric(df[condition_col], errors='coerce') > float(condition_value)
                        elif condition_operator == "less than (<)":
                            mask = pd.to_numeric(df[condition_col], errors='coerce') < float(condition_value)
                        else:  # contains
                            mask = df[condition_col].astype(str).str.contains(condition_value, case=False, na=False)
                        
                        # Additional filter for missing values
                        mask = mask & df[target_col].isnull()
                        
                        affected_rows = mask.sum()
                        st.metric("Rows that will be affected:", affected_rows)
                        
                        if affected_rows > 0:
                            st.dataframe(df[mask][[condition_col, target_col]].head(10), use_container_width=True)
                        else:
                            st.info("No rows match the condition")
                        
                    except Exception as e:
                        st.error(f"Error in preview: {str(e)}")
                
                if st.button("✅ Apply Conditional Fill", type="primary", key="apply_cond"):
                    try:
                        # Create condition mask
                        if condition_operator == "equals (=)":
                            mask = df[condition_col].astype(str) == condition_value
                        elif condition_operator == "not equals (≠)":
                            mask = df[condition_col].astype(str) != condition_value
                        elif condition_operator == "greater than (>)":
                            mask = pd.to_numeric(df[condition_col], errors='coerce') > float(condition_value)
                        elif condition_operator == "less than (<)":
                            mask = pd.to_numeric(df[condition_col], errors='coerce') < float(condition_value)
                        else:  # contains
                            mask = df[condition_col].astype(str).str.contains(condition_value, case=False, na=False)
                        
                        # Additional filter for missing values
                        mask = mask & df[target_col].isnull()
                        
                        affected_rows = mask.sum()
                        
                        # Try to convert fill_value to appropriate type
                        if pd.api.types.is_numeric_dtype(df[target_col]):
                            try:
                                fill_value_converted = float(fill_value)
                            except:
                                fill_value_converted = fill_value
                        else:
                            fill_value_converted = fill_value
                        
                        # Apply fill
                        df.loc[mask, target_col] = fill_value_converted
                        
                        st.success(f"✅ Filled {affected_rows} rows!")
                        log_action(f"Conditional fill: If {condition_col} {condition_operator} '{condition_value}', fill {target_col} with '{fill_value}'", snapshot=True)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error applying fill: {str(e)}")
            
            with tab2:
                st.markdown("**Fill any column using formulas from other columns**")
                st.info("💡 Perfect for filling blank columns you just created!")
                
                target_calc_col = st.selectbox("Select column to fill:", df.columns.tolist(), key="target_calc_col")
                
                calc_method = st.radio(
                    "Calculation method:",
                    ["🔢 Simple Formula (A ± B)", "🧮 Advanced Formula (Custom)", "📋 Copy from Column"],
                    key="calc_method"
                )
                
                if calc_method == "🔢 Simple Formula (A ± B)":
                    st.markdown("**Create formula using two columns**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        formula_col1 = st.selectbox("First column:", 
                            [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
                            key="formula_col1"
                        )
                        
                    with col2:
                        formula_operator = st.selectbox("Operator:", 
                            ["Add (+)", "Subtract (-)", "Multiply (×)", "Divide (÷)", "Power (^)", "Modulo (%)"],
                            key="formula_op"
                        )
                    
                    formula_col2 = st.selectbox("Second column:", 
                        [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
                        key="formula_col2"
                    )
                    
                    # Preview
                    st.markdown("**Preview Formula:**")
                    st.code(f"{target_calc_col} = {formula_col1} {formula_operator.split()[1]} {formula_col2}")
                    
                    if st.button("🔍 Preview Result", key="preview_simple"):
                        try:
                            if formula_operator == "Add (+)":
                                result = df[formula_col1] + df[formula_col2]
                            elif formula_operator == "Subtract (-)":
                                result = df[formula_col1] - df[formula_col2]
                            elif formula_operator == "Multiply (×)":
                                result = df[formula_col1] * df[formula_col2]
                            elif formula_operator == "Divide (÷)":
                                result = df[formula_col1] / df[formula_col2].replace(0, np.nan)
                            elif formula_operator == "Power (^)":
                                result = df[formula_col1] ** df[formula_col2]
                            else:  # Modulo
                                result = df[formula_col1] % df[formula_col2]
                            
                            preview_df = pd.DataFrame({
                                formula_col1: df[formula_col1],
                                formula_col2: df[formula_col2],
                                'Result': result
                            })
                            st.dataframe(preview_df.head(10), use_container_width=True)
                        except Exception as e:
                            st.error(f"Error in calculation: {str(e)}")
                    
                    if st.button("✅ Apply Simple Formula", type="primary", key="apply_simple"):
                        try:
                            if formula_operator == "Add (+)":
                                df[target_calc_col] = df[formula_col1] + df[formula_col2]
                            elif formula_operator == "Subtract (-)":
                                df[target_calc_col] = df[formula_col1] - df[formula_col2]
                            elif formula_operator == "Multiply (×)":
                                df[target_calc_col] = df[formula_col1] * df[formula_col2]
                            elif formula_operator == "Divide (÷)":
                                df[target_calc_col] = df[formula_col1] / df[formula_col2].replace(0, np.nan)
                            elif formula_operator == "Power (^)":
                                df[target_calc_col] = df[formula_col1] ** df[formula_col2]
                            else:  # Modulo
                                df[target_calc_col] = df[formula_col1] % df[formula_col2]
                            
                            st.session_state.df = df
                            st.success(f"✅ Applied formula to {target_calc_col}!")
                            log_action(f"Formula: {target_calc_col} = {formula_col1} {formula_operator.split()[1]} {formula_col2}", snapshot=True)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                elif calc_method == "🧮 Advanced Formula (Custom)":
                    st.markdown("**Write custom Python expression**")
                    st.info("💡 Use column names in expression. Example: `df['Age'] * 2 + df['Score']`")
                    
                    custom_formula = st.text_area(
                        "Enter formula (use df['column_name'] syntax):",
                        value=f"df[''] * 2",
                        height=100,
                        key="custom_formula"
                    )
                    
                    st.markdown("**Available functions:** `np.sqrt()`, `np.log()`, `np.abs()`, `np.round()`, `np.max()`, `np.min()`")
                    
                    # Safe evaluation using ast.literal_eval for security
                    def safe_eval_formula(formula: str, df: pd.DataFrame) -> Any:
                        """Safely evaluate formula with restricted access"""
                        try:
                            # Extract column names from formula
                            column_pattern = r"df\['([^']+)'\]"
                            import re
                            columns = re.findall(column_pattern, formula)
                            
                            # Check all columns exist
                            for col in columns:
                                if col not in df.columns:
                                    raise ValueError(f"Column '{col}' not found")
                            
                            # Replace df['col'] with actual values
                            for col in columns:
                                formula = formula.replace(f"df['{col}']", f"df['{col}'].values")
                            
                            # Evaluate with restricted globals
                            allowed_globals = {
                                'df': df,
                                'np': np,
                                'pd': pd,
                                'math': __import__('math')
                            }
                            
                            return eval(formula, {"__builtins__": {}}, allowed_globals)
                        except Exception as e:
                            raise ValueError(f"Formula evaluation error: {str(e)}")
                    
                    if st.button("🔍 Preview Custom Formula", key="preview_custom"):
                        try:
                            result = safe_eval_formula(custom_formula, df)
                            preview_df = pd.DataFrame({
                                'Result': result
                            })
                            st.dataframe(preview_df.head(10), use_container_width=True)
                            st.success("✅ Formula is valid!")
                        except Exception as e:
                            st.error(f"Error in formula: {str(e)}")
                    
                    if st.button("✅ Apply Custom Formula", type="primary", key="apply_custom"):
                        try:
                            df[target_calc_col] = safe_eval_formula(custom_formula, df)
                            st.session_state.df = df
                            st.success(f"✅ Applied custom formula to {target_calc_col}!")
                            log_action(f"Custom formula: {target_calc_col} = {custom_formula}", snapshot=True)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                else:  # Copy from Column
                    st.markdown("**Copy values from another column**")
                    
                    source_col = st.selectbox("Copy from:", 
                        [col for col in df.columns if col != target_calc_col],
                        key="copy_source"
                    )
                    
                    apply_transform = st.checkbox("Apply transformation", key="copy_transform")
                    
                    if apply_transform:
                        transform_type = st.selectbox(
                            "Transformation:",
                            ["Multiply by constant", "Add constant", "Convert to uppercase", "Convert to lowercase", 
                             "Remove spaces", "Round to decimals"],
                            key="transform_type"
                        )
                        
                        if "constant" in transform_type or "decimals" in transform_type:
                            constant_val = st.number_input("Value:", value=1.0, key="constant_val")
                    
                    if st.button("✅ Copy Column", type="primary", key="apply_copy"):
                        try:
                            if not apply_transform:
                                df[target_calc_col] = df[source_col]
                                action_msg = f"Copied {source_col} to {target_calc_col}"
                            else:
                                if transform_type == "Multiply by constant":
                                    df[target_calc_col] = df[source_col] * constant_val
                                    action_msg = f"Copied {source_col} × {constant_val} to {target_calc_col}"
                                elif transform_type == "Add constant":
                                    df[target_calc_col] = df[source_col] + constant_val
                                    action_msg = f"Copied {source_col} + {constant_val} to {target_calc_col}"
                                elif transform_type == "Convert to uppercase":
                                    df[target_calc_col] = df[source_col].astype(str).str.upper()
                                    action_msg = f"Copied {source_col} (uppercase) to {target_calc_col}"
                                elif transform_type == "Convert to lowercase":
                                    df[target_calc_col] = df[source_col].astype(str).str.lower()
                                    action_msg = f"Copied {source_col} (lowercase) to {target_calc_col}"
                                elif transform_type == "Remove spaces":
                                    df[target_calc_col] = df[source_col].astype(str).str.replace(" ", "")
                                    action_msg = f"Copied {source_col} (no spaces) to {target_calc_col}"
                                else:  # Round
                                    df[target_calc_col] = df[source_col].round(int(constant_val))
                                    action_msg = f"Copied {source_col} (rounded) to {target_calc_col}"
                            
                            st.session_state.df = df
                            st.success(f"✅ {action_msg}")
                            log_action(action_msg, snapshot=True)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            with tab3:
                st.markdown("**Fill based on condition (If-Then)**")
                st.info("💡 Example: If Age > 18, fill Status with 'Adult'")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    cond_source_col = st.selectbox("If column:", df.columns.tolist(), key="cond_source")
                    cond_op = st.selectbox("Condition:", 
                        ["equals (=)", "not equals (≠)", "greater than (>)", "less than (<)", 
                         "greater or equal (≥)", "less or equal (≤)", "contains text"],
                        key="cond_operator"
                    )
                    cond_val = st.text_input("Value:", key="cond_value")
                
                with col2:
                    then_value = st.text_input("Then fill with:", key="then_val")
                    else_value = st.text_input("Else fill with (optional):", key="else_val")
                
                if st.button("🔍 Preview Conditional", key="preview_cond_calc"):
                    try:
                        if cond_op == "equals (=)":
                            mask = df[cond_source_col].astype(str) == cond_val
                        elif cond_op == "not equals (≠)":
                            mask = df[cond_source_col].astype(str) != cond_val
                        elif cond_op == "greater than (>)":
                            mask = pd.to_numeric(df[cond_source_col], errors='coerce') > float(cond_val)
                        elif cond_op == "less than (<)":
                            mask = pd.to_numeric(df[cond_source_col], errors='coerce') < float(cond_val)
                        elif cond_op == "greater or equal (≥)":
                            mask = pd.to_numeric(df[cond_source_col], errors='coerce') >= float(cond_val)
                        elif cond_op == "less or equal (≤)":
                            mask = pd.to_numeric(df[cond_source_col], errors='coerce') <= float(cond_val)
                        else:  # contains
                            mask = df[cond_source_col].astype(str).str.contains(cond_val, case=False, na=False)
                        
                        st.metric("Rows matching condition:", mask.sum())
                        st.metric("Rows not matching:", (~mask).sum())
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                
                if st.button("✅ Apply Conditional Fill", type="primary", key="apply_cond_calc"):
                    try:
                        if cond_op == "equals (=)":
                            mask = df[cond_source_col].astype(str) == cond_val
                        elif cond_op == "not equals (≠)":
                            mask = df[cond_source_col].astype(str) != cond_val
                        elif cond_op == "greater than (>)":
                            mask = pd.to_numeric(df[cond_source_col], errors='coerce') > float(cond_val)
                        elif cond_op == "less than (<)":
                            mask = pd.to_numeric(df[cond_source_col], errors='coerce') < float(cond_val)
                        elif cond_op == "greater or equal (≥)":
                            mask = pd.to_numeric(df[cond_source_col], errors='coerce') >= float(cond_val)
                        elif cond_op == "less or equal (≤)":
                            mask = pd.to_numeric(df[cond_source_col], errors='coerce') <= float(cond_val)
                        else:  # contains
                            mask = df[cond_source_col].astype(str).str.contains(cond_val, case=False, na=False)
                        
                        df.loc[mask, target_calc_col] = then_value
                        if else_value:
                            df.loc[~mask, target_calc_col] = else_value
                        
                        st.session_state.df = df
                        st.success(f"✅ Applied conditional fill!")
                        log_action(f"Conditional: If {cond_source_col} {cond_op} '{cond_val}', {target_calc_col} = '{then_value}'", snapshot=True)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # "Go to Next Step" button
        st.markdown("---")
        if st.button("Go to Next Step", type="primary", use_container_width=True):
            st.session_state.current_step = min(12, st.session_state.current_step + 1)
            st.rerun()
    

    
    return df

def outlier_cleaning_step(df: pd.DataFrame):
    st.header("Step 5 · Outliers")
    st.markdown(f"**Step 5 of 12**")
    
    # Create two columns layout for main content and pipeline panel
    col_main, col_pipeline = st.columns([3, 1])
    
    with col_main:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.info("No numeric columns found for outlier detection.")
            return
        
        selected_col = st.selectbox("Select column for outlier analysis:", [''] + numeric_cols)
        if not selected_col:
            return
        
        col_data = df[selected_col].dropna()
        
        # ML Anomaly Detection - EXPANDED BY DEFAULT
        with st.expander("🤖 ML Anomaly Detection", expanded=True):
            if SKLEARN_AVAILABLE:
                st.markdown("**Use machine learning to detect complex anomalies**")
                ml_method = st.radio(
                    "Detection Method:",
                    ['isolation_forest', 'local_outlier_factor'],
                    format_func=lambda x: "Isolation Forest" if x == 'isolation_forest' else "Local Outlier Factor"
                )
                contamination = st.slider("Expected outlier proportion:", 0.01, 0.3, 0.1, 0.01)
                
                if st.button("🔍 Detect Anomalies (ML)"):
                    with st.spinner("Running ML anomaly detection..."):
                        anomaly_indices = detect_anomalies_ml(df, selected_col, ml_method, contamination)
                        if anomaly_indices is not None and len(anomaly_indices) > 0:
                            st.warning(f"Found {len(anomaly_indices)} anomalies using {ml_method}")
                            st.session_state['ml_anomalies'] = anomaly_indices
                            
                            # Show anomalies
                            anomaly_values = df.loc[anomaly_indices, selected_col]
                            st.subheader("Detected Anomalies")
                            st.write(anomaly_values.describe())
                            
                            # Visualization
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=df[selected_col],
                                mode='markers',
                                name='Normal',
                                marker=dict(color='blue', size=5)
                            ))
                            fig.add_trace(go.Scatter(
                                y=df.loc[anomaly_indices, selected_col],
                                mode='markers',
                                name='Anomaly',
                                marker=dict(color='red', size=8, symbol='x')
                            ))
                            fig.update_layout(title=f"Anomaly Detection: {selected_col}")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Treatment options
                            st.subheader("Treatment Options")
                            action = st.radio(
                                "Choose action:",
                                ['Flag as outliers (add column)', 'Remove anomalies', 'Keep all']
                            )
                            
                            if st.button("Apply ML Treatment"):
                                if action == 'Flag as outliers (add column)':
                                    df['is_outlier'] = False
                                    df.loc[anomaly_indices, 'is_outlier'] = True
                                    st.success(f"✅ Added 'is_outlier' column")
                                    log_action(f"ML anomaly flagging in {selected_col}", snapshot=True)
                                elif action == 'Remove anomalies':
                                    df.drop(index=anomaly_indices, inplace=True)
                                    st.success(f"✅ Removed {len(anomaly_indices)} anomalies")
                                    log_action(f"Removed {len(anomaly_indices)} ML anomalies from {selected_col}", snapshot=True)
                                st.rerun()
                        else:
                            st.success("✨ No anomalies detected!")
            else:
                graceful_fallback('scikit-learn', 'ML Anomaly Detection')
        
        st.divider()
        
        # Existing outlier detection
        method = st.radio("Outlier Detection Method:", ['IQR Method', 'Z-Score Method', 'Custom Range'])
        lower_bound = upper_bound = None
        min_val = max_val = None
        if method == 'IQR Method':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
        elif method == 'Z-Score Method':
            z_threshold = st.slider("Z-Score Threshold:", 1.0, 4.0, 3.0, 0.1)
            mean = col_data.mean()
            std = col_data.std()
            z_scores = (df[selected_col] - mean).abs() / std
            outliers = df[z_scores > z_threshold]
        else:
            min_val = st.number_input("Minimum value:", value=float(col_data.min()))
            max_val = st.number_input("Maximum value:", value=float(col_data.max()))
            outliers = df[(df[selected_col] < min_val) | (df[selected_col] > max_val)]
        
        a, b = st.columns(2)
        with a:
            st.metric("Outliers Found", len(outliers))
            st.metric("Outlier %", f"{(len(outliers)/len(df)*100) if len(df)>0 else 0:.1f}%")
        with b:
            clean_data = df[selected_col].dropna().tolist()
            temp_df = pd.DataFrame({selected_col: clean_data})
            fig = px.box(temp_df, y=selected_col, title=f"Box Plot of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        if len(outliers) > 0:
            st.subheader("Outlier Treatment")
            treatment = st.radio("Treatment method:", ['Remove Outliers', 'Cap to Bounds', 'Flag Outliers', 'Keep Outliers'])
            if st.button("Apply Treatment"):
                if treatment == 'Remove Outliers':
                    original_len = len(df)
                    df.drop(outliers.index, inplace=True)
                    removed = original_len - len(df)
                    st.success(f"Removed {removed} outliers")
                    log_action(f"Removed {removed} outliers from {selected_col}", snapshot=True)
                    st.rerun()
                elif treatment == 'Cap to Bounds':
                    if method == 'IQR Method' and lower_bound is not None and upper_bound is not None:
                        df[selected_col] = df[selected_col].clip(lower_bound, upper_bound)
                    elif method == 'Custom Range' and min_val is not None and max_val is not None:
                        df[selected_col] = df[selected_col].clip(min_val, max_val)
                    st.success("Outliers capped to bounds")
                    log_action(f"Capped outliers in {selected_col}", snapshot=True)
                    st.rerun()
                elif treatment == 'Flag Outliers':
                    df['is_outlier'] = False
                    df.loc[outliers.index, 'is_outlier'] = True
                    st.success(f"✅ Flagged {len(outliers)} outliers in new column")
                    log_action(f"Flagged {len(outliers)} outliers in {selected_col}", snapshot=True)
                    st.rerun()
        
        # "Go to Next Step" button
        st.markdown("---")
        if st.button("Go to Next Step", type="primary", use_container_width=True):
            st.session_state.current_step = min(12, st.session_state.current_step + 1)
            st.rerun()
    
    

def drop_columns_step(df: pd.DataFrame):
    st.header("Step 6 · Columns (Add/Drop)")
    st.markdown(f"**Step 6 of 12**")
    
    # Create two columns layout for main content and pipeline panel
    col_main, col_pipeline = st.columns([3, 1])
    
    with col_main:
        # Add New Columns - EXPANDED BY DEFAULT
        with st.expander("➕ Add New Columns", expanded=True):
            st.markdown("**Create new blank columns to fill later**")
            st.info("💡 Create columns here, then fill them with formulas in the Missing Values step!")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                new_col_name = st.text_input("New column name:", key="new_col_name_drop_step")
            
            with col2:
                col_type = st.selectbox(
                    "Column type:",
                    ["Text (string)", "Number (Float64)", "Integer (Int64)", "Empty (null)"],
                    key="new_col_type"
                )
            
            default_value = st.text_input("Default value (optional, leave blank for empty):", key="default_val_drop_step")
            
            if st.button("➕ Add Column", type="primary"):
                if new_col_name:
                    if new_col_name in df.columns:
                        st.error(f"❌ Column '{new_col_name}' already exists!")
                    else:
                        # Determine value based on type and default
                        if default_value:
                            if col_type == "Number (Float64)":
                                try:
                                    df[new_col_name] = float(default_value)
                                except:
                                    st.error("Invalid number format")
                                    return
                            elif col_type == "Integer (Int64)":
                                try:
                                    df[new_col_name] = int(default_value)
                                except:
                                    st.error("Invalid integer format")
                                    return
                            else:  # Text
                                df[new_col_name] = default_value
                        else:
                            # Empty column based on type
                            if col_type == "Number (Float64)":
                                df[new_col_name] = np.nan
                            elif col_type == "Integer (Int64)":
                                df[new_col_name] = pd.NA
                            else:  # Text or Empty
                                df[new_col_name] = pd.NA
                        
                        st.session_state.df = df
                        st.success(f"✅ Created column '{new_col_name}'")
                        log_action(f"Added new column: {new_col_name}", snapshot=True)
                        
                        # Sync feature engineering state
                        sync_feature_engineering_state(df)
                        st.rerun()
                else:
                    st.warning("⚠️ Please enter a column name")
            
            # Quick add multiple columns
            st.markdown("---")
            st.markdown("**Quick Add Multiple Columns**")
            multi_cols = st.text_area(
                "Enter column names (one per line):",
                placeholder="Column1\nColumn2\nColumn3",
                key="multi_cols"
            )
            
            if st.button("➕ Add Multiple Columns"):
                if multi_cols:
                    col_names = [name.strip() for name in multi_cols.split('\n') if name.strip()]
                    added = []
                    skipped = []
                    detailed_changes = []
                    
                    for col_name in col_names:
                        if col_name in df.columns:
                            skipped.append(col_name)
                        else:
                            df[col_name] = pd.NA
                            added.append(col_name)
                            detailed_changes.append({
                                'column': col_name,
                                'operation': 'add_column',
                                'details': 'Added new column'
                            })
                    
                    if added:
                        st.session_state.df = df
                        
                        # Bulk logging for multiple columns
                        log_bulk_action("Added multiple columns", detailed_changes)
                        log_action(f"Added {len(added)} columns", snapshot=True)
                        
                        st.success(f"✅ Added {len(added)} columns: {', '.join(added)}")
                    
                    if skipped:
                        st.warning(f"⚠️ Skipped {len(skipped)} existing columns: {', '.join(skipped)}")
                    
                    if added:
                        # Sync feature engineering state
                        sync_feature_engineering_state(df)
                        st.rerun()
                else:
                    st.warning("⚠️ Please enter at least one column name")
        
        st.divider()
        
        # Smart Drop Suggestions - EXPANDED BY DEFAULT
        with st.expander("🤖 Smart Drop Suggestions", expanded=True):
            st.markdown("**Automatically identify columns that might be safe to drop**")
            
            if st.button("🔍 Analyze Columns"):
                with st.spinner("Analyzing columns..."):
                    suggestions = []
                    
                    # Constant columns
                    for col in df.columns:
                        if df[col].nunique() <= 1:
                            suggestions.append({
                                'Column': col,
                                'Reason': 'Constant (only 1 unique value)',
                                'Type': 'Constant',
                                'Priority': 'High'
                            })
                    
                    # High missing
                    for col in df.columns:
                        missing_pct = (df[col].isnull().sum() / len(df)) * 100
                        if missing_pct > 90:
                            suggestions.append({
                                'Column': col,
                                'Reason': f'{missing_pct:.1f}% missing values',
                                'Type': 'High Missing',
                                'Priority': 'High'
                            })
                    
                    # High correlation (numeric only)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) >= 2:
                        corr_matrix = df[numeric_cols].corr().abs()
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                if corr_matrix.iloc[i, j] > 0.95:
                                    suggestions.append({
                                        'Column': corr_matrix.columns[j],
                                        'Reason': f'Highly correlated with {corr_matrix.columns[i]} (r={corr_matrix.iloc[i,j]:.3f})',
                                        'Type': 'High Correlation',
                                        'Priority': 'Medium'
                                    })
                    
                    if suggestions:
                        st.warning(f"⚠️ Found {len(suggestions)} columns that might be safe to drop")
                        suggestions_df = pd.DataFrame(suggestions)
                        st.dataframe(suggestions_df, use_container_width=True)
                        
                        cols_to_drop_smart = st.multiselect(
                            "Select columns to drop:",
                            suggestions_df['Column'].unique().tolist()
                        )
                        
                        if cols_to_drop_smart and st.button("🗑️ Drop Selected (Smart)", type="primary"):
                            # Create detailed changes for bulk logging
                            detailed_changes = []
                            for col in cols_to_drop_smart:
                                reason = suggestions_df[suggestions_df['Column'] == col]['Reason'].iloc[0]
                                detailed_changes.append({
                                    'column': col,
                                    'operation': 'drop_column',
                                    'details': f'Smart drop: {reason}'
                                })
                            
                            df.drop(columns=cols_to_drop_smart, inplace=True)
                            
                            # Bulk logging
                            log_bulk_action(f"Smart drop: {len(cols_to_drop_smart)} columns", detailed_changes)
                            log_action(f"Dropped {len(cols_to_drop_smart)} columns via smart suggestions", snapshot=True)
                            
                            st.success(f"✅ Dropped {len(cols_to_drop_smart)} columns")
                            
                            # Sync feature engineering state
                            sync_feature_engineering_state(df)
                            st.rerun()
                    else:
                        st.success("✨ No obvious columns to drop!")
        
        st.divider()
        
        # Existing manual drop
        if len(df.columns) == 0:
            st.warning("No columns to drop!")
            return
        
        st.subheader("Current Columns")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Missing': [df[col].isnull().sum() for col in df.columns],
            'Unique': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
        
        cols_to_drop = st.multiselect("Select columns to drop:", df.columns.tolist())
        if cols_to_drop:
            st.warning(f"You are about to drop {len(cols_to_drop)} columns: {cols_to_drop}")
            
            if st.button("Confirm Drop Columns"):
                # Create detailed changes for bulk logging
                detailed_changes = []
                for col in cols_to_drop:
                    col_type = str(df[col].dtype)
                    missing_count = df[col].isnull().sum()
                    detailed_changes.append({
                        'column': col,
                        'operation': 'drop_column',
                        'details': f'Type: {col_type}, Missing: {missing_count}'
                    })
                
                df.drop(columns=cols_to_drop, inplace=True)
                
                # Bulk logging
                log_bulk_action(f"Manual drop: {len(cols_to_drop)} columns", detailed_changes)
                log_action(f"Dropped columns: {cols_to_drop}", snapshot=True)
                
                st.success(f"✅ Dropped {len(cols_to_drop)} columns")
                
                # Sync feature engineering state
                sync_feature_engineering_state(df)
                st.rerun()
        
        # "Go to Next Step" button
        st.markdown("---")
        if st.button("Go to Next Step", type="primary", use_container_width=True):
            st.session_state.current_step = min(12, st.session_state.current_step + 1)
            st.rerun()
    
   

# Enhanced Feature Engineering Step with ALL TABS VISIBLE
def feature_engineering_step(df: pd.DataFrame):
    st.header("Step 8 · Feature Engineering")
    st.markdown(f"**Step 8 of 12**")
    
    # Create two columns layout for main content and pipeline panel
    col_main, col_pipeline = st.columns([3, 1])
    
    with col_main:
        st.markdown("**Smart column selection for modeling** - Export keeps ALL columns, engineering uses only essential ones")
        
        # Auto-select essential columns
        if not st.session_state.feature_engineering_columns['selected']:
            essential, excluded = auto_select_essential_columns(df)
            st.session_state.feature_engineering_columns['selected'] = essential
            st.session_state.feature_engineering_columns['excluded'] = excluded
        
        essential = st.session_state.feature_engineering_columns['selected']
        excluded = st.session_state.feature_engineering_columns['excluded']
        
        # Show summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Columns", len(df.columns))
        with col2:
            st.metric("✅ Selected for Engineering", len(essential))
        with col3:
            st.metric("➖ Auto-Excluded", len(excluded))
        
        st.info("💡 Export will include ALL cleaned columns. Feature engineering uses only selected columns for modeling.")
        
        # Show selected columns
        with st.expander("✅ Selected Columns for Feature Engineering", expanded=True):
            if essential:
                selected_df = pd.DataFrame({
                    'Column': essential,
                    'Type': [str(df[col].dtype) for col in essential],
                    'Unique': [df[col].nunique() for col in essential],
                    'Missing %': [f"{df[col].isnull().sum()/len(df)*100:.1f}%" for col in essential]
                })
                st.dataframe(selected_df, use_container_width=True)
            else:
                st.warning("No columns selected! Add columns from excluded list below.")
        
        # Show excluded columns with reasons
        with st.expander("➖ Auto-Excluded Columns (Click to Add Back)", expanded=False):
            if excluded:
                for col, reason in excluded.items():
                    col_container = st.container()
                    with col_container:
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.write(f"**{col}**: {reason}")
                        with c2:
                            if st.button("➕ Add", key=f"add_{col}"):
                                st.session_state.feature_engineering_columns['selected'].append(col)
                                del st.session_state.feature_engineering_columns['excluded'][col]
                                st.rerun()
            else:
                st.success("All columns are selected!")
        
        st.divider()
        
        # Feature Engineering Operations - ALL TABS VISIBLE
        tab1, tab2, tab3, tab4 = st.tabs(["🔢 Encoding", "📏 Scaling", "📅 Datetime", "🔗 Interactions"])
        
        with tab1:
            st.subheader("Categorical Encoding")
            cat_cols = [col for col in essential if df[col].dtype in ['object', 'category', 'string']]
            
            if cat_cols:
                col_to_encode = st.selectbox("Select column to encode:", cat_cols)
                encoding_method = st.radio(
                    "Encoding method:",
                    ['One-Hot Encoding', 'Label Encoding']
                )
                
                if st.button("Apply Encoding"):
                    try:
                        if encoding_method == 'One-Hot Encoding':
                            dummies = pd.get_dummies(df[col_to_encode], prefix=col_to_encode)
                            
                            # Bulk logging for one-hot encoding
                            detailed_changes = []
                            for dummy_col in dummies.columns:
                                detailed_changes.append({
                                    'column': dummy_col,
                                    'operation': 'one_hot_encode',
                                    'details': f'Created from {col_to_encode}'
                                })
                            
                            df = pd.concat([df, dummies], axis=1)
                            
                            # Update selected columns
                            if col_to_encode in st.session_state.feature_engineering_columns['selected']:
                                st.session_state.feature_engineering_columns['selected'].remove(col_to_encode)
                            st.session_state.feature_engineering_columns['selected'].extend(dummies.columns.tolist())
                            
                            df.drop(columns=[col_to_encode], inplace=True)
                            st.session_state.df = df
                            
                            # Bulk logging
                            log_bulk_action(f"One-hot encoded {col_to_encode}", detailed_changes)
                            log_action(f"One-hot encoded {col_to_encode} into {len(dummies.columns)} columns", snapshot=True)
                            
                            st.success(f"✅ One-hot encoded {col_to_encode} into {len(dummies.columns)} columns")
                            
                        else:  # Label Encoding
                            if SKLEARN_AVAILABLE:
                                le = LabelEncoder()
                                df[f'{col_to_encode}_encoded'] = le.fit_transform(df[col_to_encode].astype(str))
                                st.session_state.feature_engineering_columns['selected'].append(f'{col_to_encode}_encoded')
                                st.session_state.df = df
                                
                                log_action(f"Label encoded {col_to_encode}", snapshot=True)
                                st.success(f"✅ Label encoded {col_to_encode}")
                            else:
                                graceful_fallback('scikit-learn', 'Label Encoding')
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.info("No categorical columns in selected features")
        
        with tab2:
            st.subheader("Feature Scaling")
            if SKLEARN_AVAILABLE:
                numeric_cols = [col for col in essential if pd.api.types.is_numeric_dtype(df[col])]
                if numeric_cols:
                    cols_to_scale = st.multiselect("Select columns to scale:", numeric_cols)
                    scaler_type = st.selectbox(
                        "Scaling method:",
                        ['Standard (Z-score)', 'Min-Max (0-1)', 'Robust (median/IQR)']
                    )
                    
                    if cols_to_scale and st.button("Apply Scaling"):
                        try:
                            if scaler_type == 'Standard (Z-score)':
                                scaler = StandardScaler()
                            elif scaler_type == 'Min-Max (0-1)':
                                scaler = MinMaxScaler()
                            else:
                                scaler = RobustScaler()
                            
                            # Bulk logging for scaling
                            detailed_changes = []
                            for col in cols_to_scale:
                                detailed_changes.append({
                                    'column': col,
                                    'operation': 'scale',
                                    'details': f'{scaler_type} scaling applied'
                                })
                            
                            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                            st.session_state.df = df
                            
                            # Bulk logging
                            log_bulk_action(f"Scaled {len(cols_to_scale)} columns", detailed_changes)
                            log_action(f"Scaled: {', '.join(cols_to_scale)} ({scaler_type})", snapshot=True)
                            
                            st.success(f"✅ Scaled {len(cols_to_scale)} columns using {scaler_type}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.info("No numeric columns in selected features")
            else:
                graceful_fallback('scikit-learn', 'Feature Scaling')
        
        with tab3:
            st.subheader("Datetime Feature Extraction")
            datetime_cols = [col for col in essential if pd.api.types.is_datetime64_any_dtype(df[col])]
            if datetime_cols:
                col_to_extract = st.selectbox("Select datetime column:", datetime_cols)
                features = st.multiselect(
                    "Select features to extract:",
                    ['Year', 'Month', 'Day', 'DayOfWeek', 'Hour', 'Quarter', 'WeekOfYear'],
                    default=['Year', 'Month', 'Day']
                )
                
                if features and st.button("Extract Features"):
                    try:
                        new_cols = []
                        detailed_changes = []
                        
                        for feature in features:
                            new_col_name = f'{col_to_extract}_{feature.lower()}'
                            if feature == 'Year':
                                df[new_col_name] = df[col_to_extract].dt.year
                            elif feature == 'Month':
                                df[new_col_name] = df[col_to_extract].dt.month
                            elif feature == 'Day':
                                df[new_col_name] = df[col_to_extract].dt.day
                            elif feature == 'DayOfWeek':
                                df[new_col_name] = df[col_to_extract].dt.dayofweek
                            elif feature == 'Hour':
                                df[new_col_name] = df[col_to_extract].dt.hour
                            elif feature == 'Quarter':
                                df[new_col_name] = df[col_to_extract].dt.quarter
                            elif feature == 'WeekOfYear':
                                df[new_col_name] = df[col_to_extract].dt.isocalendar().week
                            
                            new_cols.append(new_col_name)
                            detailed_changes.append({
                                'column': new_col_name,
                                'operation': 'extract_datetime',
                                'details': f'Extracted {feature} from {col_to_extract}'
                            })
                        
                        st.session_state.feature_engineering_columns['selected'].extend(new_cols)
                        st.session_state.df = df
                        
                        # Bulk logging
                        log_bulk_action(f"Extracted {len(features)} datetime features", detailed_changes)
                        log_action(f"Datetime extraction from {col_to_extract}: {', '.join(features)}", snapshot=True)
                        
                        st.success(f"✅ Extracted {len(features)} datetime features")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.info("No datetime columns in selected features")
        
        with tab4:
            st.subheader("Interaction & Polynomial Features")
            numeric_cols = [col for col in essential if pd.api.types.is_numeric_dtype(df[col])]
            
            if len(numeric_cols) >= 2:
                st.markdown("**Create Interaction Features**")
                col1 = st.selectbox("Select first column:", numeric_cols, key='int_col1')
                col2 = st.selectbox("Select second column:", [c for c in numeric_cols if c != col1], key='int_col2')
                operation = st.selectbox("Operation:", ['Multiply', 'Divide', 'Add', 'Subtract'])
                new_col_name = st.text_input("New column name:", value=f"{col1}_{operation.lower()}_{col2}")
                
                if st.button("Create Interaction"):
                    try:
                        if operation == 'Multiply':
                            df[new_col_name] = df[col1] * df[col2]
                        elif operation == 'Divide':
                            df[new_col_name] = df[col1] / df[col2].replace(0, np.nan)
                        elif operation == 'Add':
                            df[new_col_name] = df[col1] + df[col2]
                        else:  # Subtract
                            df[new_col_name] = df[col1] - df[col2]
                        
                        st.session_state.feature_engineering_columns['selected'].append(new_col_name)
                        st.session_state.df = df
                        
                        log_action(f"Created interaction: {new_col_name} = {col1} {operation} {col2}", snapshot=True)
                        st.success(f"✅ Created interaction feature: {new_col_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                
                st.divider()
                st.markdown("**Create Polynomial Features**")
                st.info("Generate polynomial and interaction features for selected columns")
                
                poly_cols = st.multiselect("Select columns for polynomial features:", numeric_cols)
                degree = st.slider("Polynomial degree:", 2, 3, 2)
                
                if poly_cols and st.button("Generate Polynomial Features"):
                    try:
                        if SKLEARN_AVAILABLE:
                            poly = PolynomialFeatures(degree=degree, include_bias=False)
                            poly_features = poly.fit_transform(df[poly_cols])
                            feature_names = poly.get_feature_names_out(poly_cols)
                            
                            # Add new polynomial features with bulk logging
                            detailed_changes = []
                            new_cols_added = 0
                            
                            for i, name in enumerate(feature_names):
                                if name not in poly_cols:  # Skip original features
                                    df[name] = poly_features[:, i]
                                    st.session_state.feature_engineering_columns['selected'].append(name)
                                    detailed_changes.append({
                                        'column': name,
                                        'operation': 'polynomial',
                                        'details': f'Degree {degree} polynomial feature'
                                    })
                                    new_cols_added += 1
                            
                            st.session_state.df = df
                            
                            # Bulk logging
                            if new_cols_added > 0:
                                log_bulk_action(f"Created {new_cols_added} polynomial features", detailed_changes)
                                log_action(f"Polynomial features (degree={degree}): {', '.join(poly_cols)}", snapshot=True)
                            
                            st.success(f"✅ Created {new_cols_added} polynomial features")
                            st.rerun()
                        else:
                            graceful_fallback('scikit-learn', 'Polynomial Features')
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.info("Need at least 2 numeric columns for interactions")
        
        # "Go to Next Step" button
        st.markdown("---")
        if st.button("Go to Next Step", type="primary", use_container_width=True):
            st.session_state.current_step = min(12, st.session_state.current_step + 1)
            st.rerun()
   

def get_classification_models():
    """Return classification models dictionary"""
    models = {}
    
    if SKLEARN_AVAILABLE:
        models.update({
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Naive Bayes': GaussianNB(),
            'K-Neighbors': KNeighborsClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
        })
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(random_state=42, use_label_encoder=False)
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(random_state=42)
    
    return models

def get_regression_models():
    """Return regression models dictionary"""
    models = {}
    
    if SKLEARN_AVAILABLE:
        models.update({
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'SVR': SVR(),
            'K-Neighbors': KNeighborsRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'AdaBoost': AdaBoostRegressor(random_state=42),
            'Neural Network': MLPRegressor(max_iter=1000, random_state=42)
        })
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(random_state=42)
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMRegressor(random_state=42)
    
    return models

# Enhanced ML Modeling & AutoML Step with Validation
def modeling_and_automl_step(df: pd.DataFrame):
    st.header("Step 9 · ML & AutoML")
    st.markdown(f"**Step 9 of 12**")
    
    # Create two columns layout for main content and pipeline panel
    col_main, col_pipeline = st.columns([3, 1])
    
    with col_main:
        st.markdown("**Train multiple models, compare performance, and get AI-powered insights**")
        
        if not SKLEARN_AVAILABLE:
            graceful_fallback('scikit-learn', 'Machine Learning')
            return
        
        # Get feature engineering columns (or all if not set)
        if st.session_state.feature_engineering_columns['selected']:
            available_features = [col for col in st.session_state.feature_engineering_columns['selected'] if col in df.columns]
            st.info(f"💡 Using {len(available_features)} features from Feature Engineering step")
        else:
            available_features = df.columns.tolist()
            st.warning("⚠️ No features selected in Feature Engineering. Using all columns.")
        
        # Problem Setup
        st.subheader("🎯 Problem Setup")
        col1, col2 = st.columns(2)
        
        with col1:
            problem_type = st.radio(
                "Problem Type:",
                ['Classification', 'Regression'],
                help="Classification for categorical targets, Regression for continuous targets"
            )
        
        with col2:
            target_col = st.selectbox(
                "Select Target Column:",
                [col for col in df.columns if col in available_features or col not in available_features],
                help="The column you want to predict"
            )
        
        if not target_col:
            st.warning("Please select a target column")
            return
        
        # Feature selection
        feature_cols = st.multiselect(
            "Select Feature Columns (X):",
            [col for col in available_features if col != target_col],
            default=[col for col in available_features if col != target_col][:min(10, len(available_features)-1)],
            help="Columns to use for prediction"
        )
        
        if not feature_cols:
            st.warning("Please select at least one feature column")
            return
        
        # Check for missing values
        X = df[feature_cols]
        y = df[target_col]
        
        if X.isnull().any().any() or y.isnull().any():
            st.error("❌ Missing values detected! Please clean your data first in the Missing Values step.")
            return
        
        # Check for non-numeric features
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            st.error(f"❌ Non-numeric features detected: {non_numeric}. Please encode them in Feature Engineering step.")
            return
        
        # Data Validation
        st.subheader("🔍 Data Validation")
        
        if problem_type.lower() == 'classification':
            is_valid, warnings = validate_classification_data(X, y)
            validation_type = "Classification"
        else:
            is_valid, warnings = validate_regression_data(X, y)
            validation_type = "Regression"
        
        if not is_valid:
            st.error(f"❌ {validation_type} data validation failed")
            for warning in warnings:
                st.error(warning)
            return
        
        for warning in warnings:
            st.warning(warning)
        
        # AI Data Analysis
        with st.expander("🔍 AI Data Analysis", expanded=True):
            st.markdown("### Dataset Characteristics")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Samples", f"{len(X):,}")
            with c2:
                st.metric("Features", len(feature_cols))
            with c3:
                if problem_type.lower() == 'classification':
                    st.metric("Classes", y.nunique())
                else:
                    st.metric("Target Range", f"{y.min():.2f} - {y.max():.2f}")
            with c4:
                st.metric("Memory", f"{X.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # AI Recommendations
            st.markdown("### 🤖 AI Recommendations")
            recommendations = []
            
            # Dataset size recommendations
            if len(X) < 100:
                recommendations.append("⚠️ Small dataset (<100 samples). Consider: KNN, Naive Bayes, or simple models")
            elif len(X) < 1000:
                recommendations.append("👍 Medium dataset. Recommended: Random Forest, SVM, or ensemble methods")
            else:
                recommendations.append("🌟 Large dataset. All models will work well. Try XGBoost/LightGBM for best performance")
            
            # Feature recommendations
            if len(feature_cols) > 50:
                recommendations.append("⚠️ Many features (>50). Consider feature selection or dimensionality reduction")
            
            # Class imbalance check
            if problem_type.lower() == 'classification':
                class_dist = y.value_counts(normalize=True)
                if class_dist.iloc[0] > 0.8:
                    recommendations.append(f"⚠️ Severe class imbalance ({class_dist.iloc[0]*100:.1f}% in majority class)")
                    recommendations.append("💡 Consider: SMOTE, class weights, or ensemble methods")
            
            # Correlation check
            if len(feature_cols) >= 2:
                corr_matrix = X.corr().abs()
                high_corr = (corr_matrix > 0.9).sum().sum() - len(feature_cols)
                if high_corr > 0:
                    recommendations.append(f"⚠️ High correlation detected between {high_corr//2} feature pairs")
                    recommendations.append("💡 Consider removing redundant features")
            
            for rec in recommendations:
                if "⚠️" in rec:
                    st.warning(rec)
                elif "🌟" in rec:
                    st.success(rec)
                else:
                    st.info(rec)
        
        # Train/Test Split Configuration
        st.subheader("🔀 Train/Test Split")
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100
        with col2:
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        with col3:
            random_state = st.number_input("Random State", 0, 100, 42)
        
        # Validate ML parameters
        model_selection_mode = st.radio(
            "Selection Mode:",
            ['🤖 AutoML (Try All Models)', '🎯 Manual Selection', '⚡ Quick Test (Top 3)'],
            horizontal=True
        )
        
        selected_models = []
        
        if model_selection_mode == '🤖 AutoML (Try All Models)':
            st.info("🚀 AutoML will train and compare ALL available models automatically")
            if problem_type.lower() == 'classification':
                selected_models = list(get_classification_models().keys())
            else:
                selected_models = list(get_regression_models().keys())
            st.write(f"**Will train {len(selected_models)} models:** {', '.join(selected_models)}")
        
        elif model_selection_mode == '⚡ Quick Test (Top 3)':
            st.info("⚡ Quick test with 3 most reliable models")
            if problem_type.lower() == 'classification':
                selected_models = ['Logistic Regression', 'Random Forest', 'XGBoost' if XGBOOST_AVAILABLE else 'Gradient Boosting']
            else:
                selected_models = ['Linear Regression', 'Random Forest', 'XGBoost' if XGBOOST_AVAILABLE else 'Gradient Boosting']
            st.write(f"**Models:** {', '.join(selected_models)}")
        
        else:  # Manual Selection
            if problem_type.lower() == 'classification':
                models_dict = get_classification_models()
            else:
                models_dict = get_regression_models()
            
            st.markdown("**Select models to train:**")
            cols = st.columns(3)
            for idx, (model_name, _) in enumerate(models_dict.items()):
                with cols[idx % 3]:
                    if st.checkbox(model_name, value=True, key=f"model_{model_name}"):
                        selected_models.append(model_name)
        
        if not selected_models:
            st.warning("Please select at least one model")
            return
        
        # Validate ML parameters
        is_valid, warnings = validate_ml_parameters(problem_type.lower(), test_size, cv_folds, selected_models)
        if not is_valid:
            st.error("❌ ML parameter validation failed")
            for warning in warnings:
                st.error(warning)
            return
        
        for warning in warnings:
            st.warning(warning)
        
        # Advanced Options
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                scale_features = st.checkbox("Scale Features (Recommended for SVM/Neural Networks)", value=True)
                hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning (Slower)", value=False)
            with col2:
                ensemble_stacking = st.checkbox("Create Ensemble Stack (Advanced)", value=False)
                save_models = st.checkbox("Save Trained Models", value=True)
        
        # Train Models Button
        st.divider()
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            # Prepare data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale if requested
            if scale_features:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
                X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            
            # Get models
            if problem_type.lower() == 'classification':
                models_dict = get_classification_models()
            else:
                models_dict = get_regression_models()
            
            # Train models sequentially with memory management
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, model_name in enumerate(selected_models):
                if model_name not in models_dict:
                    continue
                
                status_text.text(f"Training {model_name}... ({idx+1}/{len(selected_models)})")
                
                model = models_dict[model_name]
                
                # Hyperparameter tuning
                if hyperparameter_tuning and model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
                    st.info(f"🔧 Tuning {model_name}...")
                    param_grid = {}
                    if 'Random Forest' in model_name:
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [None, 10, 20],
                            'min_samples_split': [2, 5]
                        }
                    elif 'XGBoost' in model_name or 'LightGBM' in model_name:
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [3, 5, 7],
                            'learning_rate': [0.01, 0.1, 0.3]
                        }
                    
                    if param_grid:
                        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy' if problem_type.lower() == 'classification' else 'r2', n_jobs=-1)
                        grid_search.fit(X_train, y_train)
                        model = grid_search.best_estimator_
                        st.success(f"✅ Best params: {grid_search.best_params_}")
                            
                            # Train and evaluate
                        result = train_and_evaluate_model(
                                model, X_train, X_test, y_train, y_test,
                                model_name, problem_type.lower(), cv_folds
                            )
                            
                        if 'error' not in result:
                                results.append(result)
                                
                                # Save model if requested
                                if save_models:
                                    st.session_state.trained_models[model_name] = result['model_object']
                            
                            # Clear memory after each model
                        gc.collect()
                        progress_bar.progress((idx + 1) / len(selected_models))
                        
                        status_text.text("✅ Training complete!")
                        st.session_state.model_results = results
                        
                        # Ensemble Stacking
                        if ensemble_stacking and len(results) >= 3:
                            st.info("🔗 Creating ensemble stack...")
                            try:
                                base_models = [(r['model_name'], r['model_object']) for r in results[:3]]
                                if problem_type.lower() == 'classification':
                                    stack = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
                                else:
                                    stack = StackingRegressor(estimators=base_models, final_estimator=Ridge())
                                
                                stack_result = train_and_evaluate_model(
                                    stack, X_train, X_test, y_train, y_test,
                                    'Ensemble Stack', problem_type.lower(), cv_folds
                                )
                                results.append(stack_result)
                                st.session_state.model_results = results
                                st.success("✅ Ensemble model created!")
                            except Exception as e:
                                st.warning(f"Ensemble creation failed: {str(e)}")
                    
                    # Display Results
                    if st.session_state.model_results:
                        st.divider()
                        st.subheader("📊 Model Comparison")
                        
                        results = st.session_state.model_results
                        
                        # Create comparison table
                        if problem_type.lower() == 'classification':
                            comparison_data = []
                            for r in results:
                                if 'error' not in r:
                                    comparison_data.append({
                                        'Model': r['model_name'],
                                        'Accuracy': f"{r['accuracy']*100:.2f}%",
                                        'Precision': f"{r['precision']*100:.2f}%",
                                        'Recall': f"{r['recall']*100:.2f}%",
                                        'F1-Score': f"{r['f1']*100:.2f}%",
                                        'CV Score': f"{r['cv_mean']*100:.2f}%",
                                        'Time (s)': f"{r['training_time']:.2f}"
                                    })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # Best model
                            best_idx = np.argmax([r['accuracy'] for r in results if 'error' not in r])
                            best_model = results[best_idx]
                            st.success(f"🏆 **Best Model:** {best_model['model_name']} with {best_model['accuracy']*100:.2f}% accuracy!")
                            
                        else:  # Regression
                            comparison_data = []
                            for r in results:
                                if 'error' not in r:
                                    comparison_data.append({
                                        'Model': r['model_name'],
                                        'R² Score': f"{r['r2']:.4f}",
                                        'RMSE': f"{r['rmse']:.4f}",
                                        'MAE': f"{r['mae']:.4f}",
                                        'CV R²': f"{r['cv_mean']:.4f}",
                                        'Time (s)': f"{r['training_time']:.2f}"
                                    })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # Best model
                            best_idx = np.argmax([r['r2'] for r in results if 'error' not in r])
                            best_model = results[best_idx]
                            st.success(f"🏆 **Best Model:** {best_model['model_name']} with R² = {best_model['r2']:.4f}!")
                        
                        # Visualizations
                        st.subheader("📈 Performance Visualization")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Performance comparison chart
                            if problem_type.lower() == 'classification':
                                metric_data = pd.DataFrame({
                                    'Model': [r['model_name'] for r in results if 'error' not in r],
                                    'Accuracy': [r['accuracy'] for r in results if 'error' not in r]
                                })
                                fig = px.bar(metric_data, x='Model', y='Accuracy', 
                                            title="Model Accuracy Comparison",
                                            color='Accuracy', color_continuous_scale='Viridis')
                                fig.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                metric_data = pd.DataFrame({
                                    'Model': [r['model_name'] for r in results if 'error' not in r],
                                    'R²': [r['r2'] for r in results if 'error' not in r]
                                })
                                fig = px.bar(metric_data, x='Model', y='R²',
                                            title="Model R² Comparison",
                                            color='R²', color_continuous_scale='Viridis')
                                fig.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Training time comparison
                            time_data = pd.DataFrame({
                                'Model': [r['model_name'] for r in results if 'error' not in r],
                                'Time (s)': [r['training_time'] for r in results if 'error' not in r]
                            })
                            fig = px.bar(time_data, x='Model', y='Time (s)',
                                        title="Training Time Comparison",
                                        color='Time (s)', color_continuous_scale='Blues')
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed Model Analysis
                        st.subheader("🔬 Detailed Model Analysis")
                        selected_model_name = st.selectbox(
                            "Select model for detailed analysis:",
                            [r['model_name'] for r in results if 'error' not in r]
                        )
                        
                        selected_result = next((r for r in results if r['model_name'] == selected_model_name), None)
                        
                        if selected_result:
                            # AI Insights
                            with st.expander("🤖 AI Insights", expanded=True):
                                insights = generate_model_insights(selected_result, problem_type.lower(), X_train, y_train)
                                
                                if insights['insights']:
                                    st.markdown("### 📊 Key Insights")
                                    for insight in insights['insights']:
                                        st.info(insight)
                                
                                if insights['warnings']:
                                    st.markdown("### ⚠️ Warnings")
                                    for warning in insights['warnings']:
                                        st.warning(warning)
                                
                                if insights['recommendations']:
                                    st.markdown("### 💡 Recommendations")
                                    for rec in insights['recommendations']:
                                        st.success(rec)
                            
                            # Feature Importance
                            if 'feature_importance' in selected_result:
                                st.markdown("### 📊 Feature Importance")
                                importance_df = pd.DataFrame({
                                    'Feature': feature_cols,
                                    'Importance': selected_result['feature_importance']
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(importance_df.head(15), x='Importance', y='Feature',
                                            orientation='h', title=f"Top 15 Features - {selected_model_name}")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Confusion Matrix (Classification)
                            if problem_type.lower() == 'classification' and 'confusion_matrix' in selected_result:
                                st.markdown("### 🎯 Confusion Matrix")
                                cm = selected_result['confusion_matrix']
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                plt.title(f"Confusion Matrix - {selected_model_name}")
                                plt.ylabel('True Label')
                                plt.xlabel('Predicted Label')
                                st.pyplot(fig)
                                plt.close()
                            
                            # Predictions vs Actual (Regression)
                            if problem_type.lower() == 'regression':
                                st.markdown("### 📉 Predictions vs Actual")
                                pred_df = pd.DataFrame({
                                    'Actual': y_test,
                                    'Predicted': selected_result['predictions']
                                })
                                fig = px.scatter(pred_df, x='Actual', y='Predicted',
                                                title=f"Predictions vs Actual - {selected_model_name}",
                                                trendline="ols")
                                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                                        y=[y_test.min(), y_test.max()],
                                                        mode='lines', name='Perfect Prediction',
                                                        line=dict(color='red', dash='dash')))
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Export Section
                        st.divider()
                        st.subheader("📥 Export Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Export predictions
                            if st.button("📊 Export Predictions (CSV)"):
                                pred_export = pd.DataFrame({
                                    'Actual': y_test,
                                    f'{selected_model_name}_Predicted': selected_result['predictions']
                                })
                                csv = pred_export.to_csv(index=False)
                                st.download_button(
                                    label="Download Predictions",
                                    data=csv,
                                    file_name=f"predictions_{selected_model_name}.csv",
                                    mime="text/csv"
                                )
                        
                        with col2:
                            # Export trained model
                            if st.button("💾 Export Trained Model"):
                                model_bytes = io.BytesIO()
                                joblib.dump(selected_result['model_object'], model_bytes)
                                model_bytes.seek(0)
                                st.download_button(
                                    label="Download Model (.pkl)",
                                    data=model_bytes,
                                    file_name=f"model_{selected_model_name}.pkl",
                                    mime="application/octet-stream"
                                )
                        
                        with col3:
                            # Export full report
                            if st.button("📄 Export Full Report"):
                                report = f"""
        Machine Learning Model Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Problem Type: {problem_type}
        Target: {target_col}
        Features: {', '.join(feature_cols)}
        Train Size: {len(X_train)} | Test Size: {len(X_test)}
        
        Model: {selected_model_name}
        """
                                if problem_type.lower() == 'classification':
                                    report += f"""
        Accuracy: {selected_result['accuracy']*100:.2f}%
        Precision: {selected_result['precision']*100:.2f}%
        Recall: {selected_result['recall']*100:.2f}%
        F1-Score: {selected_result['f1']*100:.2f}%
        """
                                else:
                                    report += f"""
        R² Score: {selected_result['r2']:.4f}
        RMSE: {selected_result['rmse']:.4f}
        MAE: {selected_result['mae']:.4f}
        """
                                
                                report += f"""
        Cross-Validation Score: {selected_result['cv_mean']:.4f} (±{selected_result['cv_std']:.4f})
        Training Time: {selected_result['training_time']:.2f}s
        """
                                
                                st.download_button(
                                    label="Download Report (.txt)",
                                    data=report,
                                    file_name=f"ml_report_{selected_model_name}.txt",
                                    mime="text/plain"
                                )
                    
                    # Memory management
                    with st.expander("🧠 Memory Management", expanded=False):
                        st.markdown("### Clear Model Cache")
                        if st.button("🗑️ Clear All Trained Models"):
                            clear_model_cache()
                            st.rerun()
                    
                    # "Go to Next Step" button
                    st.markdown("---")
                    if st.button("Go to Next Step", type="primary", use_container_width=True):
                        st.session_state.current_step = min(12, st.session_state.current_step + 1)
                        st.rerun()
        
        

def pivot_table_step(df: pd.DataFrame):
    st.header("Step 7 · Pivot Tables")
    st.markdown(f"**Step 7 of 12**")
    
    # Create two columns layout for main content and pipeline panel
    col_main, col_pipeline = st.columns([3, 1])
    
    with col_main:                
        st.markdown("Create powerful pivot tables to summarize and analyze your data.")
        st.subheader("🔧 Configure Pivot Table")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Rows (Index)")
            row_fields = st.multiselect("Select columns for rows:", options=df.columns.tolist())
            st.markdown("Values (Aggregation)")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_fields = st.multiselect("Select columns to aggregate:", options=numeric_cols)
        with col2:
            st.markdown("Columns")
            column_fields = st.multiselect("Select columns for column headers:", options=df.columns.tolist())
            st.markdown("Aggregation Functions")
            agg_functions = st.multiselect(
                "Select aggregation functions:",
                options=['sum', 'mean', 'median', 'count', 'min', 'max', 'std', 'var'],
                default=['sum']
            )
        with st.expander("⚙️ Advanced Options"):
            a, b, c = st.columns(3)
            with a:
                show_margins = st.checkbox("Show totals (margins)", value=False)
                fill_na = st.checkbox("Fill missing values", value=True)
                fill_value = st.number_input("Fill with value:", value=0.0) if fill_na else None
            with b:
                sort_by_values = st.checkbox("Sort by values", value=False)
                ascending = st.checkbox("Ascending order", value=False)
            with c:
                normalize = st.selectbox("Normalize:", options=['None', 'All', 'Index', 'Columns'])
        table_name = st.text_input("Pivot Table Name:", value=f"Pivot_Table_{len(st.session_state.pivot_tables) + 1}")
        if st.button("🔨 Create Pivot Table", type="primary", use_container_width=True):
            if not row_fields and not column_fields:
                st.error("❌ Please select at least one row or column field!")
            elif not value_fields:
                st.error("❌ Please select at least one value field to aggregate!")
            elif not agg_functions:
                st.error("❌ Please select at least one aggregation function!")
            else:
                try:
                    index = row_fields if row_fields else None
                    columns = column_fields if column_fields else None
                    if len(agg_functions) == 1:
                        pivot = pd.pivot_table(
                            df, values=value_fields, index=index, columns=columns,
                            aggfunc=agg_functions[0], fill_value=fill_value if fill_na else None,
                            margins=show_margins, margins_name='Total'
                        )
                    else:
                        pivot = pd.pivot_table(
                            df, values=value_fields, index=index, columns=columns,
                            aggfunc=agg_functions, fill_value=fill_value if fill_na else None,
                            margins=show_margins, margins_name='Total'
                        )
                    if normalize != 'None':
                        if normalize == 'All':
                            pivot = pivot / pivot.sum().sum() * 100
                        elif normalize == 'Index':
                            pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
                        elif normalize == 'Columns':
                            pivot = pivot.div(pivot.sum(axis=0), axis=1) * 100
                    if sort_by_values and len(pivot.columns) > 0:
                        sort_col = pivot.columns[0]
                        pivot = pivot.sort_values(by=sort_col, ascending=ascending)
                    pivot_info = {
                        'name': table_name,
                        'data': pivot,
                        'config': {
                            'rows': row_fields, 'columns': column_fields,
                            'values': value_fields, 'agg_functions': agg_functions,
                            'normalized': normalize != 'None'
                        }
                    }
                    st.session_state.pivot_tables.append(pivot_info)
                    st.success(f"✅ Pivot table '{table_name}' created successfully!")
                    log_action(f"Created pivot table: {table_name}", snapshot=False)
                except Exception as e:
                    st.error(f"❌ Error creating pivot table: {str(e)}")
        if st.session_state.pivot_tables:
            st.divider()
            st.subheader("📋 Saved Pivot Tables")
            table_names = [pt['name'] for pt in st.session_state.pivot_tables]
            selected_table = st.selectbox("Select pivot table to view:", table_names)
            if selected_table:
                pivot_info = next((pt for pt in st.session_state.pivot_tables if pt['name'] == selected_table), None)
                if pivot_info:
                    pivot_data = pivot_info['data']
                    config = pivot_info['config']
                    with st.expander("📝 Table Configuration"):
                        st.write(f"Rows: {', '.join(config['rows']) if config['rows'] else 'None'}")
                        st.write(f"Columns: {', '.join(config['columns']) if config['columns'] else 'None'}")
                        st.write(f"Values: {', '.join(config['values'])}")
                        st.write(f"Aggregations: {', '.join(config['agg_functions'])}")
                        st.write(f"Normalized: {'Yes' if config['normalized'] else 'No'}")
                    st.dataframe(pivot_data, use_container_width=True, height=400)
                    a, b, c = st.columns(3)
                    with a:
                        st.metric("Rows", pivot_data.shape[0])
                    with b:
                        st.metric("Columns", pivot_data.shape[1])
                    with c:
                        if pivot_data.shape[0] > 0 and pivot_data.shape[1] > 0:
                            st.metric("Total Cells", pivot_data.shape[0] * pivot_data.shape[1])
                    st.subheader("📊 Visualize Pivot Table")
                    viz_type = st.selectbox("Select chart type:", ["Bar Chart", "Line Chart", "Heatmap", "Stacked Bar", "Area Chart"])
                    if st.button("Generate Chart"):
                        try:
                            plot_data = pivot_data.reset_index()
                            fig = None
                            if viz_type == "Bar Chart":
                                fig = px.bar(plot_data, x=plot_data.columns[0], y=plot_data.columns[1:],
                                             title=f"{selected_table} - Bar Chart", barmode='group')
                            elif viz_type == "Line Chart":
                                fig = px.line(plot_data, x=plot_data.columns[0], y=plot_data.columns[1:],
                                              title=f"{selected_table} - Line Chart")
                            elif viz_type == "Heatmap":
                                fig_, ax = plt.subplots(figsize=(10, 8))
                                sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
                                plt.title(f"{selected_table} - Heatmap")
                                st.pyplot(fig_)
                                plt.close()
                            elif viz_type == "Stacked Bar":
                                fig = px.bar(plot_data, x=plot_data.columns[0], y=plot_data.columns[1:],
                                             title=f"{selected_table} - Stacked Bar", barmode='stack')
                            elif viz_type == "Area Chart":
                                fig = px.area(plot_data, x=plot_data.columns[0], y=plot_data.columns[1:],
                                              title=f"{selected_table} - Area Chart")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating visualization: {str(e)}")
                    a, b, c = st.columns(3)
                    with a:
                        csv_buffer = io.StringIO()
                        pivot_data.to_csv(csv_buffer)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"{selected_table}.csv",
                            mime="text/csv"
                        )
                    with b:
                        if st.button("🗑️ Delete This Table", type="secondary"):
                            st.session_state.pivot_tables = [
                                pt for pt in st.session_state.pivot_tables if pt['name'] != selected_table
                            ]
                            st.success(f"Deleted '{selected_table}'")
                            st.rerun()
                    with c:
                        if st.button("🗑️ Clear All Tables"):
                            st.session_state.pivot_tables = []
                            st.success("All pivot tables cleared")
                            st.rerun()
        else:
            st.info("📊 No pivot tables created yet. Configure and create your first pivot table above!")
        
        # "Go to Next Step" button
        st.markdown("---")
        if st.button("Go to Next Step", type="primary", use_container_width=True):
            st.session_state.current_step = min(12, st.session_state.current_step + 1)
            st.rerun()
    

def correlation_insights(df: pd.DataFrame, target: Optional[str] = None):
    st.subheader("📊 Comprehensive Correlation Insights")
    include_categorical = st.checkbox(
        "Include Categorical Columns (will be encoded numerically)",
        value=True
    )
    if include_categorical:
        df_encoded, encoding_info, categorical_cols = encode_categorical_columns(df)
        if categorical_cols:
            st.info(f"Encoded {len(categorical_cols)} categorical columns for analysis")
        if encoding_info:
            with st.expander("📋 View Categorical Encoding Details"):
                for col, mapping in encoding_info.items():
                    st.write(f"{col}:")
                    st.json(mapping)
        df_analysis = df_encoded
    else:
        df_analysis = df
        encoding_info = {}
        categorical_cols = []
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for correlation analysis!")
        return
    pearson_corr, spearman_corr = compute_correlations(df_analysis[numeric_cols])
    st.markdown("### 🔹 Pearson Correlation Heatmap (Linear)")
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    sns.heatmap(pearson_corr, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Pearson Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()
    st.markdown("### 🔹 Spearman Correlation Heatmap (Monotonic)")
    st.info("💡 Spearman correlation is better for ordinal/ranked data.")
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    sns.heatmap(spearman_corr, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Spearman Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    if target and target in numeric_cols:
        st.markdown(f"### 🎯 Correlation with Target Column: `{target}`")
        a, b = st.columns(2)
        with a:
            st.markdown("#### Pearson Correlation")
            corr_with_target_p = pearson_corr[target].sort_values(ascending=False)
            corr_with_target_p = corr_with_target_p[corr_with_target_p.index != target]
            if len(corr_with_target_p) > 0:
                fig3 = px.bar(
                    corr_with_target_p.reset_index(),
                    x='index', y=target, color=target,
                    title=f"Pearson Correlation with {target}",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig3, use_container_width=True)
        with b:
            st.markdown("#### Spearman Correlation")
            corr_with_target_s = spearman_corr[target].sort_values(ascending=False)
            corr_with_target_s = corr_with_target_s[corr_with_target_s.index != target]
            if len(corr_with_target_s) > 0:
                fig4 = px.bar(
                    corr_with_target_s.reset_index(),
                    x='index', y=target, color=target,
                    title=f"Spearman Correlation with {target}",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig4, use_container_width=True)
    threshold = st.slider("Correlation Threshold:", 0.5, 0.95, 0.7, 0.05)
    corr_pairs = pearson_corr.unstack().reset_index()
    corr_pairs.columns = ['Feature_1', 'Feature_2', 'Correlation']
    corr_pairs = corr_pairs[
        (abs(corr_pairs['Correlation']) > threshold) &
        (corr_pairs['Feature_1'] != corr_pairs['Feature_2'])
    ]
    corr_pairs['pair'] = corr_pairs.apply(lambda x: tuple(sorted([x['Feature_1'], x['Feature_2']])), axis=1)
    corr_pairs = corr_pairs.drop_duplicates(subset='pair').drop('pair', axis=1)
    corr_pairs = corr_pairs.sort_values(by='Correlation', key=abs, ascending=False)
    if len(corr_pairs) > 0:
        st.markdown(f"### 🔍 Highly Correlated Feature Pairs (>|{threshold}|)")
        st.dataframe(corr_pairs.reset_index(drop=True), use_container_width=True)
        top_pairs = corr_pairs.head(10)
        fig5 = px.bar(
            top_pairs,
            x='Correlation',
            y=top_pairs['Feature_1'] + ' ↔ ' + top_pairs['Feature_2'],
            orientation='h',
            title="Top 10 Highly Correlated Pairs",
            color='Correlation',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.success(f"🎉 No highly correlated pairs above threshold |{threshold}|!")

def analysis_and_insights_step(df: pd.DataFrame):
    st.header("Step 10 · Analysis & Insights")
    st.markdown(f"**Step 10 of 12**")
    
    # Create two columns layout for main content and pipeline panel
    col_main, col_pipeline = st.columns([3, 1])
    
    with col_main:
        # BOTH TABS VISIBLE - using st.tabs() which shows all tabs by default
        tab1, tab2 = st.tabs(["🔍 Quick Insights", "📈 Correlation Analysis"])
        
        with tab1:
            st.subheader("🤖 AI-Powered Quick Insights")
            st.markdown("Get instant insights about your data quality, patterns, and recommendations")
            
            if st.button("🔍 Generate Quick Insights", type="primary"):
                with st.spinner("Analyzing your data..."):
                    insights = generate_quick_insights(df)
                    
                    # Display narratives
                    if insights['narrative']:
                        st.markdown("### 📊 Key Findings")
                        for item in insights['narrative']:
                            st.info(item)
                    
                    # Display warnings
                    if insights['warnings']:
                        st.markdown("### ⚠️ Warnings")
                        for item in insights['warnings']:
                            st.warning(item)
                    
                    # Display recommendations
                    if insights['recommendations']:
                        st.markdown("### 💡 Recommendations")
                        for item in insights['recommendations']:
                            st.success(item)
                    
                    # Visual insights
                    st.markdown("### 📉 Visual Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Missing data visualization
                        missing_data = df.isnull().sum()
                        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                        if len(missing_data) > 0:
                            fig = px.bar(
                                x=missing_data.values,
                                y=missing_data.index,
                                orientation='h',
                                title="Missing Values by Column",
                                labels={'x': 'Count', 'y': 'Column'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Class imbalance for categorical
                        cat_cols = df.select_dtypes(include=['object', 'category']).columns
                        if len(cat_cols) > 0:
                            col_to_check = cat_cols[0]
                            if 2 <= df[col_to_check].nunique() <= 10:
                                value_counts = df[col_to_check].value_counts()
                                fig = px.pie(
                                    values=value_counts.values,
                                    names=value_counts.index,
                                    title=f"Distribution of {col_to_check}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            analysis_mode = st.radio(
                "Select Analysis Mode:",
                ["Automatic Analysis (All Columns)", "Target-Based Analysis"]
            )
            if analysis_mode == "Automatic Analysis (All Columns)":
                if st.button("Run Automatic Correlation Analysis"):
                    try:
                        correlation_insights(df, target=None)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                target_column = st.selectbox("Select target column:", [""] + list(df.columns))
                if st.button("Run Target-Based Correlation Analysis"):
                    if target_column:
                        try:
                            correlation_insights(df, target=target_column)
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.warning("⚠️ Please select a target column!")
        
        # "Go to Next Step" button
        st.markdown("---")
        if st.button("Go to Next Step", type="primary", use_container_width=True):
            st.session_state.current_step = min(12, st.session_state.current_step + 1)
            st.rerun()
    


def visualization_playground_step(df: pd.DataFrame):
    """
    🎮 Visual Playground
    An Excel-like interactive chart builder where users can play with different visualizations.
    """
    st.header("Step 11 · Visual Playground")
    st.markdown(f"**Step 11 of 12**")
    
    # Create two columns layout for main content and pipeline panel
    col_main, col_pipeline = st.columns([3, 1])
    
    with col_main:
        st.markdown(
            "Build and play with interactive charts just like you would in Excel — "
            "pick your data, choose a chart type, and explore!"
        )

        if df is None or df.empty:
            st.warning("No data available. Please upload a dataset first.")
            return

        # Layout: left for configuration, right for live preview
        config_col, preview_col = st.columns([1, 2])

        with config_col:
            st.subheader("🛠 Chart Setup")

            # Basic selections
            all_columns = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

            x_axis = st.selectbox(
                "X-axis column",
                options=["(index)"] + all_columns,
                help="Select what goes on the horizontal axis. Choose '(index)' to use row index."
            )

            # Allow multiple y columns (like Excel multi-series)
            y_axis = st.multiselect(
                "Y-axis column(s)",
                options=numeric_cols,
                help="Select one or more numeric columns to plot."
            )

            chart_type = st.selectbox(
                "Chart type",
                [
                    "Line",
                    "Bar",
                    "Stacked Bar",
                    "Area",
                    "Scatter",
                    "Histogram",
                    "Box Plot",
                    "Violin Plot",
                    "Heatmap"
                ]
            )

            color_by = st.selectbox(
                "Color / Group by (optional)",
                options=["(none)"] + all_columns,
                help="Use a column to create color groups / series."
            )

            # Optional filters
            st.subheader("🎯 Optional Filters")
            with st.expander("Add simple filters", expanded=False):
                filter_column = st.selectbox(
                    "Filter column (optional)",
                    options=["(none)"] + all_columns,
                    key="viz_filter_col"
                )
                filter_value = None
                if filter_column != "(none)":
                    unique_vals = df[filter_column].dropna().unique().tolist()
                    # Convert numpy types to python native for Streamlit
                    unique_vals = [v.item() if hasattr(v, "item") else v for v in unique_vals]
                    filter_value = st.selectbox(
                        "Keep only rows where value equals:",
                        options=unique_vals,
                        key="viz_filter_val"
                    )

            st.subheader("📏 Display Options")
            show_data_table = st.checkbox("Show data used for chart", value=False)
            use_log_y = st.checkbox("Log-scale Y axis (where applicable)", value=False)

            play_button = st.button("🚀 Generate Visualization", type="primary", use_container_width=True)

        with preview_col:
            st.subheader("📊 Live Preview")

            if not y_axis and chart_type not in ["Histogram", "Heatmap"]:
                st.info("Select at least one Y-axis column to see a chart.")
                return

            if play_button:
                plot_df = df.copy()

                # Apply filter if set
                if filter_column != "(none)" and filter_value is not None:
                    plot_df = plot_df[plot_df[filter_column] == filter_value]

                if plot_df.empty:
                    st.warning("Filter removed all rows. Try relaxing the filter.")
                    return

                # Build base x-axis
                if x_axis == "(index)":
                    plot_df = plot_df.reset_index()
                    x_col_name = "index"
                else:
                    x_col_name = x_axis

                try:
                    fig = None

                    if chart_type in ["Line", "Bar", "Stacked Bar", "Area"]:
                        if not y_axis:
                            st.warning("Please select at least one numeric column for the Y-axis.")
                            return

                        # Melt to long format if multiple series
                        long_df = plot_df.melt(
                            id_vars=[x_col_name] + ([color_by] if color_by != "(none)" else []),
                            value_vars=y_axis,
                            var_name="Series",
                            value_name="Value"
                        )

                        if chart_type == "Line":
                            fig = px.line(
                                long_df,
                                x=x_col_name,
                                y="Value",
                                color="Series" if color_by == "(none)" else color_by,
                                markers=True,
                                title="Line Chart"
                            )
                        elif chart_type == "Bar":
                            fig = px.bar(
                                long_df,
                                x=x_col_name,
                                y="Value",
                                color="Series" if color_by == "(none)" else color_by,
                                barmode="group",
                                title="Bar Chart"
                            )
                        elif chart_type == "Stacked Bar":
                            fig = px.bar(
                                long_df,
                                x=x_col_name,
                                y="Value",
                                color="Series" if color_by == "(none)" else color_by,
                                barmode="stack",
                                title="Stacked Bar Chart"
                            )
                        elif chart_type == "Area":
                            fig = px.area(
                                long_df,
                                x=x_col_name,
                                y="Value",
                                color="Series" if color_by == "(none)" else color_by,
                                title="Area Chart"
                            )

                        if use_log_y:
                            fig.update_yaxes(type="log")

                    elif chart_type == "Scatter":
                        if len(y_axis) != 1:
                            st.warning("Scatter plot supports exactly one Y-axis column.")
                            return
                        fig = px.scatter(
                            plot_df,
                            x=x_col_name,
                            y=y_axis[0],
                            color=None if color_by == "(none)" else color_by,
                            title="Scatter Plot",
                            trendline="ols"
                        )
                        if use_log_y:
                            fig.update_yaxes(type="log")

                    elif chart_type == "Histogram":
                        numeric_options = numeric_cols or plot_df.select_dtypes(include=[np.number]).columns.tolist()
                        if not numeric_options:
                            st.warning("No numeric columns available for histogram.")
                            return
                        hist_col = st.selectbox("Numeric column for histogram", options=numeric_options, key="viz_hist_col")
                        bins = st.slider("Number of bins", 5, 100, 30)
                        fig = px.histogram(
                            plot_df,
                            x=hist_col,
                            nbins=bins,
                            color=None if color_by == "(none)" else color_by,
                            title="Histogram"
                        )

                    elif chart_type == "Box Plot":
                        if not y_axis:
                            st.warning("Select at least one Y-axis for box plot.")
                            return
                        fig = px.box(
                            plot_df,
                            x=None if color_by == "(none)" else color_by,
                            y=y_axis,
                            title="Box Plot"
                        )

                    elif chart_type == "Violin Plot":
                        if len(y_axis) != 1:
                            st.warning("Violin plot supports exactly one Y-axis column.")
                            return
                        fig = px.violin(
                            plot_df,
                            x=None if color_by == "(none)" else color_by,
                            y=y_axis[0],
                            box=True,
                            points="all",
                            title="Violin Plot"
                        )

                    elif chart_type == "Heatmap":
                        num_cols = numeric_cols or plot_df.select_dtypes(include=[np.number]).columns.tolist()
                        if len(num_cols) < 2:
                            st.warning("Need at least two numeric columns for a heatmap.")
                            return
                        corr = plot_df[num_cols].corr()
                        fig = px.imshow(
                            corr,
                            text_auto=True,
                            aspect="auto",
                            title="Correlation Heatmap"
                        )

                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)

                    if show_data_table:
                        st.markdown("### 🔍 Data used for this chart")
                        st.dataframe(plot_df[[x_col_name] + y_axis].head(100), use_container_width=True)

                except Exception as e:
                    st.error(f"Error generating visualization: {e}")
        
        # "Go to Next Step" button
        st.markdown("---")
        if st.button("Go to Next Step", type="primary", use_container_width=True):
            st.session_state.current_step = min(12, st.session_state.current_step + 1)
            st.rerun()
    

def final_overview_and_export(df: pd.DataFrame):
    st.header("Step 12 · Final & Export")
    st.markdown(f"**Step 12 of 12**")
    
    # Create two columns layout for main content and pipeline panel
    col_main, col_pipeline = st.columns([3, 1])
    
    with col_main:
        if st.session_state.original_df is not None:
            st.subheader("Before vs After Comparison")
            a, b = st.columns(2)
            with a:
                st.write("Original Dataset")
                orig_df = st.session_state.original_df
                st.metric("Rows", f"{orig_df.shape[0]:,}")
                st.metric("Columns", f"{orig_df.shape[1]:,}")
                orig_missing = orig_df.isnull().sum().sum()
                orig_total = orig_df.shape[0] * orig_df.shape[1]
                orig_missing_pct = (orig_missing / orig_total) * 100 if orig_total > 0 else 0
                st.metric("Missing %", f"{orig_missing_pct:.1f}%")
            with b:
                st.write("Cleaned Dataset")
                st.metric("Rows", f"{df.shape[0]:,}")
                st.metric("Columns", f"{df.shape[1]:,}")
                curr_missing = df.isnull().sum().sum()
                curr_total = df.shape[0] * df.shape[1]
                curr_missing_pct = (curr_missing / curr_total) * 100 if curr_total > 0 else 0
                st.metric("Missing %", f"{curr_missing_pct:.1f}%")
        
        show_overview_metrics(df)
        
        # Generate Comprehensive Report
        with st.expander("📄 Generate Comprehensive Report", expanded=False):
            st.markdown("**Create a detailed summary report of your cleaning process**")
            
            if st.button("📝 Generate Report"):
                report_text = f"""
# Data Cleaning Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Summary
- **Original Shape**: {st.session_state.original_df.shape if st.session_state.original_df is not None else 'N/A'}
- **Final Shape**: {df.shape}
- **Rows Changed**: {st.session_state.original_df.shape[0] - df.shape[0] if st.session_state.original_df is not None else 0}
- **Columns Changed**: {st.session_state.original_df.shape[1] - df.shape[1] if st.session_state.original_df is not None else 0}

## Data Quality Score
"""
                if st.session_state.data_quality_score:
                    score = st.session_state.data_quality_score
                    report_text += f"""
- **Overall Score**: {score['total_score']}/100
- **Completeness**: {score['completeness']}/40
- **Consistency**: {score['consistency']}/30
- **Uniqueness**: {score['uniqueness']}/20
- **Validity**: {score['validity']}/10
"""
                
                report_text += f"""

## Cleaning Actions Applied
Total Actions: {len(st.session_state.action_log)}

"""
                for action in st.session_state.action_log[-20:]:  # Last 20 actions
                    report_text += f"- {action}\n"
                
                report_text += f"""

## Final Data Types
"""
                for col in df.columns:
                    report_text += f"- {col}: {df[col].dtype}\n"
                
                st.text_area("Report Content", report_text, height=400)
                st.download_button(
                    label="📥 Download Report",
                    data=report_text,
                    file_name=f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        st.subheader("📥 Export Cleaned Dataset")
        filename = st.text_input(
            "Filename:",
            value=f"cleaned_{st.session_state.filename}" if st.session_state.filename else "cleaned_dataset.csv"
        )
        if not filename.endswith('.csv'):
            filename += '.csv'
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        st.download_button(
            label="📥 Download Cleaned Dataset",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )
        
        st.subheader("🔄 Reset Options")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset to Original Dataset"):
                if st.session_state.original_df is not None:
                    st.session_state.df = st.session_state.original_df.copy()
                    init_history_on_upload("Reset to original")
                    log_action("Reset dataset to original", snapshot=True)
                    st.rerun()
        with col2:
            if st.button("🧾 Clear Action Log"):
                st.session_state.action_log = []
                st.rerun()
        
        # Memory Management
        with st.expander("🧠 Memory Management", expanded=False):
            st.markdown("### Optimize Memory Usage")
            if st.button("⚡ Optimize DataFrame Memory"):
                with st.spinner("Optimizing memory usage..."):
                    df_optimized, stats = optimize_dataframe_memory(df)
                    st.session_state.df = df_optimized
                    st.session_state.memory_optimized = True
                    
                    st.success(f"✅ Memory optimized! Saved {stats['savings_mb']} MB ({stats['savings_pct']}%)")
                    st.write(f"**Original:** {stats['original_memory_mb']} MB")
                    st.write(f"**Optimized:** {stats['optimized_memory_mb']} MB")
                    st.write(f"**Changes:** {len(stats['changes'])} columns optimized")
                    
                    log_action(f"Memory optimization: Saved {stats['savings_mb']} MB", snapshot=True)
                    st.rerun()
            
            st.markdown("### Clear Cache")
            if st.button("🗑️ Clear All Caches"):
                clear_model_cache()
                st.cache_data.clear()
                gc.collect()
                st.success("✅ All caches cleared!")


# ---------------------------------------------------------------------
# Enhanced Sidebar Navigation
# ---------------------------------------------------------------------
def sidebar_navigation():
    st.sidebar.header("🧹 Smart Data Cleaning Pipeline")
    
    # Check dependencies on first load
    if not st.session_state.dependencies_checked:
        dependencies = check_dependencies()
        missing = [name for name, available in dependencies.items() if not available]
        if missing:
            with st.sidebar.expander("⚠️ Missing Dependencies", expanded=True):
                st.warning(f"Missing packages: {', '.join(missing)}")
                st.info("Some features may be unavailable. Install missing packages for full functionality.")
        st.session_state.dependencies_checked = True
    
    # Steps definition
    steps = [
        "Step 1 · Upload Data",
        "Step 2 · Data Types",
        "Step 3 · Text Cleaning",
        "Step 4 · Missing Values",
        "Step 5 · Outliers",
        "Step 6 · Columns (Add/Drop)",
        "Step 7 · Pivot Tables",
        "Step 8 · Feature Engineering",
        "Step 9 · ML & AutoML",
        "Step 10 · Analysis & Insights",
        "Step 11 · Visual Playground",
        "Step 12 · Final & Export"
    ]
    
    current_step = st.sidebar.radio("Navigation:", steps, index=st.session_state.current_step - 1)
    st.session_state.current_step = steps.index(current_step) + 1
    
    # Data summary when data is loaded
    if st.session_state.df is not None:
        st.sidebar.divider()
        st.sidebar.markdown("### 📊 Data Summary")
        
        df = st.session_state.df
        st.sidebar.metric("Rows", f"{df.shape[0]:,}")
        st.sidebar.metric("Columns", f"{df.shape[1]:,}")
        
        # Missing values percentage
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        st.sidebar.metric("Missing %", f"{missing_pct:.1f}%")
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.sidebar.metric("Memory (MB)", f"{memory_mb:.2f}")
        
        # Next recommended step
        st.sidebar.divider()
        st.sidebar.markdown("### 🎯 Next Recommended Step")
        
        # Simple logic to suggest next step
        if missing_pct > 20:
            st.sidebar.info("Missing Values treatment recommended")
        elif len(df.select_dtypes(include=['object', 'string']).columns) > 0:
            st.sidebar.info("Text cleaning recommended")
        else:
            next_step_idx = min(st.session_state.current_step, 11)
            st.sidebar.info(f"Continue with {steps[next_step_idx]}")

# ---------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------
def main():
    st.title("🧹 Smart Data Cleaning Pipeline")
    st.markdown("**Professional-grade data cleaning with AI-powered insights**")
    
    # Initialize dependencies check
    if not st.session_state.dependencies_checked:
        check_dependencies()
    
    sidebar_navigation()
    
    if st.session_state.current_step == 1:
        st.header("Step 1 · Upload Data")
        st.markdown(f"**Step 1 of 12**")
        
        # Create two columns layout for main content and pipeline panel
        col_main, col_pipeline = st.columns([3, 1])
        
        with col_main:
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            
            # Character encoding support
            encoding_options = ['auto', 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            selected_encoding = st.selectbox("File Encoding:", encoding_options, index=0)
            
            # Large file handling
            if uploaded_file is not None:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                if file_size_mb > 100:
                    st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). Processing may take longer.")
                    
                    # Sampling option for very large files
                    use_sampling = st.checkbox("Use sampling for large file (first 10,000 rows)", value=False)
                    sample_size = 10000 if use_sampling else None
                else:
                    use_sampling = False
                    sample_size = None
            
            if uploaded_file is not None:
                try:
                    # Read with selected encoding
                    if selected_encoding == 'auto':
                        # Try multiple encodings
                        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                        df = None
                        for enc in encodings_to_try:
                            try:
                                uploaded_file.seek(0)
                                if use_sampling and sample_size:
                                    # Read sample for large files
                                    df = pd.read_csv(uploaded_file, encoding=enc, nrows=sample_size)
                                else:
                                    df = pd.read_csv(uploaded_file, encoding=enc)
                                st.success(f"Successfully read with {enc} encoding")
                                break
                            except:
                                continue
                        
                        if df is None:
                            st.error("Could not read file with any encoding. Try manual encoding selection.")
                            return
                    else:
                        uploaded_file.seek(0)
                        if use_sampling and sample_size:
                            df = pd.read_csv(uploaded_file, encoding=selected_encoding, nrows=sample_size)
                        else:
                            df = pd.read_csv(uploaded_file, encoding=selected_encoding)
                    
                    st.session_state.df = df.copy()
                    st.session_state.original_df = df.copy()
                    st.session_state.filename = uploaded_file.name
                    
                    # Memory optimization option
                    with st.expander("⚡ Memory Optimization", expanded=False):
                        optimize_memory = st.checkbox("Optimize memory usage (recommended for large datasets)", value=file_size_mb > 10)
                        if optimize_memory:
                            with st.spinner("Optimizing memory usage..."):
                                df_optimized, stats = optimize_dataframe_memory(df)
                                st.session_state.df = df_optimized
                                st.success(f"✅ Memory optimized! Saved {stats['savings_mb']} MB")
                                st.write(f"**Original:** {stats['original_memory_mb']} MB")
                                st.write(f"**Optimized:** {stats['optimized_memory_mb']} MB")
                                df = df_optimized
                    
                    init_history_on_upload(f"Upload: {uploaded_file.name}")
                    st.success(f"✅ Uploaded: {uploaded_file.name}")
                    log_action(f"Uploaded dataset: {uploaded_file.name} ({df.shape[0]} rows, {df.shape[1]} columns)", snapshot=False)
                    
                    # Calculate initial data quality score
                    with st.spinner("Calculating data quality score..."):
                        quality_score = calculate_data_quality_score(df)
                        st.session_state.data_quality_score = quality_score
                    
                    show_overview_metrics(df)
                    
                    # Data Quality Dashboard
                    with st.expander("🎯 Data Quality Assessment", expanded=True):
                        st.markdown("### Overall Data Quality Score")
                        score = quality_score['total_score']
                        
                        # Quality gauge
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total Score", f"{score:.1f}/100")
                        with col2:
                            st.metric("Completeness", f"{quality_score['completeness']:.1f}/40")
                        with col3:
                            st.metric("Consistency", f"{quality_score['consistency']:.1f}/30")
                        with col4:
                            st.metric("Uniqueness", f"{quality_score['uniqueness']:.1f}/20")
                        with col5:
                            st.metric("Validity", f"{quality_score['validity']:.1f}/10")
                        
                        # Score interpretation
                        if score >= 80:
                            st.success("🌟 Excellent quality! Your data is in great shape.")
                        elif score >= 60:
                            st.info("👍 Good quality with room for improvement.")
                        elif score >= 40:
                            st.warning("⚠️ Fair quality - significant cleaning recommended.")
                        else:
                            st.error("❌ Poor quality - extensive cleaning required.")
                        
                        # Quality breakdown
                        st.markdown("### Quality Breakdown")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(f"**Missing Data**: {quality_score['missing_percentage']:.1f}%")
                            st.write(f"**Duplicate Rows**: {quality_score['duplicate_rows']:,} ({quality_score['duplicate_percentage']:.1f}%)")
                        with c2:
                            # Create quality gauge chart
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=score,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Quality Score"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 40], 'color': "lightcoral"},
                                        {'range': [40, 60], 'color': "lightyellow"},
                                        {'range': [60, 80], 'color': "lightgreen"},
                                        {'range': [80, 100], 'color': "darkgreen"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 90
                                    }
                                }
                            ))
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Duplicate Detection
                    with st.expander("🔍 Duplicate Analysis", expanded=False):
                        if quality_score['duplicate_rows'] > 0:
                            st.warning(f"Found {quality_score['duplicate_rows']} duplicate rows")
                            if st.button("View Duplicates"):
                                duplicates = detect_duplicates(df)
                                st.dataframe(duplicates.head(20), use_container_width=True)
                            
                            if st.button("Remove All Duplicates"):
                                df_cleaned = df.drop_duplicates()
                                removed = len(df) - len(df_cleaned)
                                st.session_state.df = df_cleaned
                                st.success(f"✅ Removed {removed} duplicate rows")
                                log_action(f"Removed {removed} duplicate rows", snapshot=True)
                                st.rerun()
                        else:
                            st.success("✨ No duplicate rows found!")
                    
                    # Auto Profiling Report (optional)
                    if PROFILING_AVAILABLE:
                        with st.expander("📊 Generate Detailed Profiling Report (Optional)", expanded=False):
                            st.markdown("**Generate a comprehensive HTML report using ydata-profiling**")
                            st.warning("⚠️ This may take a while for large datasets")
                            
                            if st.button("Generate Profile Report"):
                                with st.spinner("Generating comprehensive profile... This may take a few minutes."):
                                    try:
                                        profile = ProfileReport(df, title="Data Profiling Report", minimal=True)
                                        profile_html = profile.to_html()
                                        st.download_button(
                                            label="📥 Download Profile Report (HTML)",
                                            data=profile_html,
                                            file_name="data_profile.html",
                                            mime="text/html"
                                        )
                                        st.success("✅ Profile report generated!")
                                    except Exception as e:
                                        st.error(f"Error generating report: {str(e)}")
                    else:
                        with st.expander("📊 Advanced Profiling (Optional)", expanded=False):
                            graceful_fallback('ydata_profiling', 'Detailed Profiling Report')
                    
                    st.subheader("📋 Data Preview")
                    st.dataframe(df.head(20), use_container_width=True)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("➡️ Start Cleaning Pipeline", type="primary", use_container_width=True):
                            st.session_state.current_step = 2
                            st.rerun()
                    with c2:
                        if st.button("📊 Load Example Dataset"):
                            example_df = px.data.iris()
                            st.session_state.df = example_df.copy()
                            st.session_state.original_df = example_df.copy()
                            st.session_state.filename = "iris_example.csv"
                            init_history_on_upload("Load example: iris")
                            quality_score = calculate_data_quality_score(example_df)
                            st.session_state.data_quality_score = quality_score
                            log_action("Loaded example Iris dataset", snapshot=True)
                            st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    st.info("Try selecting a different encoding or check the file format.")
            else:
                st.info("👆 Please upload a CSV file to get started!")
                st.markdown("### ✨ Enhanced Features")
                features = {
                    "🤖 Power BI-Style Undo/Redo": "Track every change with detailed history and revert any step",
                    "🔍 Comprehensive Bulk Operation Logging": "See every individual change in multi-column operations",
                    "📊 All UI Elements Visible": "No hidden sections - everything expanded by default",
                    "🔧 Enhanced Data Type Detection": "Smart detection with mixed type handling and confidence scores",
                    "🎯 ML Parameter Validation": "Automatic validation for anomaly detection and model training",
                    "⚡ Performance Optimized": "Vectorized operations and memory management for large datasets",
                    "🔄 Feature Engineering State Sync": "Automatic state management when columns change",
                    "🔒 Security Enhanced": "Safe formula evaluation and input validation"
                }
                for feature, desc in features.items():
                    st.markdown(f"**{feature}**: {desc}")
        

    
    elif st.session_state.df is not None:
        df = st.session_state.df
        
        # Create two columns layout for all other steps
        col_main, col_pipeline = st.columns([3, 1])
        
        with col_main:
            step = st.session_state.current_step
            
            if step == 2:
                st.header("Step 2 · Data Types")
                st.markdown(f"**Step 2 of 12**")
                data_type_management_step(df)
            elif step == 3:
                st.header("Step 3 · Text Cleaning")
                st.markdown(f"**Step 3 of 12**")
                text_cleaning_step(df)
            elif step == 4:
                missing_values_treatment_step(df)
            elif step == 5:
                outlier_cleaning_step(df)
            elif step == 6:
                drop_columns_step(df)
            elif step == 7:
                pivot_table_step(df)
            elif step == 8:
                feature_engineering_step(df)
            elif step == 9:
                modeling_and_automl_step(df)
            elif step == 10:
                analysis_and_insights_step(df)
            elif step == 11:
                visualization_playground_step(df)
            elif step == 12:
                try:
                    final_overview_and_export(df)
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col_pipeline:
            render_enhanced_action_log_ui()
    else:
        st.warning("Please upload a dataset first!")
        if st.button("← Go to Upload"):
            st.session_state.current_step = 1
            st.rerun()

if __name__ == "__main__":
    main()             