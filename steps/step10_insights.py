# Step 10: Analysis & Insights
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Optional

# Import utilities
from ..utils.history_utils import log_action

def encode_categorical_columns(df: pd.DataFrame):
    """Encode categorical columns for correlation analysis"""
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

def compute_correlations(df: pd.DataFrame):
    """Compute correlation matrices"""
    pearson_corr = df.corr(numeric_only=True, method='pearson')
    spearman_corr = df.corr(numeric_only=True, method='spearman')
    return pearson_corr, spearman_corr

def correlation_insights(df: pd.DataFrame, target: Optional[str] = None):
    """Display comprehensive correlation insights"""
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

def step10_insights():
    st.header("Step 10 · Analysis & Insights")
    st.markdown(f"**Step 10 of 12**")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first!")
        return
    
    df = st.session_state.df
    
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