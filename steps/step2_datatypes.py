# Step 2: Data Type Management
import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple

# Import utilities
from ..utils.data_quality_utils import detect_data_types, convert_column_type
from ..utils.history_utils import log_action, log_bulk_action
from ..components.column_stats import show_column_stats_card

def step2_datatypes():
    st.header("🔧 Data Type Management")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first!")
        return
    
    df = st.session_state.df
    
    # Smart Type Detection - EXPANDED BY DEFAULT
    with st.expander("🤖 Smart Type Detection", expanded=True):
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
    
    # "Go to Next Step" button
    st.markdown("---")
    if st.button("Go to Next Step", type="primary", use_container_width=True):
        st.session_state.current_step = min(12, st.session_state.current_step + 1)
        st.rerun()