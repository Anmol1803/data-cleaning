# Step 8: Feature Engineering
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Import utilities
from ..utils.dependency_utils import SKLEARN_AVAILABLE, graceful_fallback
from ..utils.history_utils import log_action, log_bulk_action
from ..utils.memory_utils import auto_select_essential_columns, sync_feature_engineering_state

def step8_features():
    st.header("Step 8 · Feature Engineering")
    st.markdown(f"**Step 8 of 12**")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first!")
        return
    
    df = st.session_state.df
    
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
                            from sklearn.preprocessing import LabelEncoder
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
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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
                        from sklearn.preprocessing import PolynomialFeatures
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