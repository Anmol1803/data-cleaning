# Step 1: Upload Data
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Import utilities
from ..utils.history_utils import init_history_on_upload, log_action
from ..utils.memory_utils import optimize_dataframe_memory
from ..utils.data_quality_utils import calculate_data_quality_score, detect_duplicates
from ..utils.dependency_utils import graceful_fallback, PROFILING_AVAILABLE
from ..components.overview_metrics import show_overview_metrics

# Optional profiling import
if PROFILING_AVAILABLE:
    from ydata_profiling import ProfileReport

def step1_upload():
    st.header("Step 1 · Upload Data")
    st.markdown(f"**Step 1 of 12**")
    
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