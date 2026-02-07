# Data overview cards
import streamlit as st
import pandas as pd
import plotly.express as px

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