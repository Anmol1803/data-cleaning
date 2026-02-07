# Step 6: Columns Add/Drop
import streamlit as st
import pandas as pd
import numpy as np

# Import utilities
from ..utils.history_utils import log_action, log_bulk_action
from ..utils.memory_utils import sync_feature_engineering_state

def step6_columns():
    st.header("Step 6 · Columns (Add/Drop)")
    st.markdown(f"**Step 6 of 12**")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first!")
        return
    
    df = st.session_state.df
    
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