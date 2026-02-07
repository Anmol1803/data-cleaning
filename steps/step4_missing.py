import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ast
from typing import List, Dict, Tuple, Any

# Import utilities
from ..utils.history_utils import log_action, log_bulk_action
from ..components.column_stats import show_column_stats_card
from ..utils.data_quality_utils import fill_missing_contextually

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

def step4_missing():
    st.header("Step 4 · Missing Values")
    st.markdown(f"**Step 4 of 12**")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first!")
        return
    
    df = st.session_state.df
    
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