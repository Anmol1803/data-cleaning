# Step 3: Text Cleaning
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Import utilities
from ..utils.dependency_utils import RAPIDFUZZ_AVAILABLE, graceful_fallback
from ..utils.history_utils import log_action
from ..components.column_stats import show_column_stats_card

if RAPIDFUZZ_AVAILABLE:
    from rapidfuzz import fuzz

def find_fuzzy_duplicates(df: pd.DataFrame, column: str, threshold: int = 80) -> pd.DataFrame:
    """Find fuzzy duplicate values in a column with validation"""
    if not RAPIDFUZZ_AVAILABLE:
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

def step3_textclean():
    st.header("📝 Text Cleaning Tools")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first!")
        return
    
    df = st.session_state.df
    text_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    if not text_cols:
        st.info("No text/categorical columns found in dataset.")
        return
    
    col_choice = st.selectbox("Select column to clean:", text_cols)
    
    # Fuzzy Duplicate Detection - EXPANDED BY DEFAULT
    with st.expander("🔍 Fuzzy Duplicate Detection", expanded=True):
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
    with st.expander("📊 Text Profile", expanded=True):
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
    
    # "Go to Next Step" button
    st.markdown("---")
    if st.button("Go to Next Step", type="primary", use_container_width=True):
        st.session_state.current_step = min(12, st.session_state.current_step + 1)
        st.rerun()