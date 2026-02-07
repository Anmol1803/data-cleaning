# Column statistics card
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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