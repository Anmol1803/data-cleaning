# Step 5: Outliers
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple

# Import utilities
from ..utils.dependency_utils import SKLEARN_AVAILABLE, graceful_fallback
from ..utils.history_utils import log_action

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
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(contamination=contamination, random_state=42)
            predictions = model.fit_predict(data)
        else:  # LOF
            from sklearn.neighbors import LocalOutlierFactor
            model = LocalOutlierFactor(contamination=contamination)
            predictions = model.fit_predict(data)
        
        anomaly_indices = data.index[predictions == -1]
        return anomaly_indices
    except Exception as e:
        st.error(f"Error in ML detection: {str(e)}")
        return None

def step5_outliers():
    st.header("Step 5 · Outliers")
    st.markdown(f"**Step 5 of 12**")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first!")
        return
    
    df = st.session_state.df
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