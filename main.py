# Main navigation file
import warnings
warnings.filterwarnings('ignore')

import io
import json
import ast
import gc
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any, Union
from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Initialize session state at the very top
if 'dependencies_checked' not in st.session_state:
    st.session_state.dependencies_checked = False

# Import session state initialization
from .session_state import *

# Import utilities
from .utils.dependency_utils import check_dependencies, graceful_fallback
from .utils.history_utils import init_history_on_upload, log_action, render_enhanced_action_log_ui
from .utils.memory_utils import optimize_dataframe_memory, clear_model_cache
from .utils.data_quality_utils import calculate_data_quality_score, detect_duplicates

# Import components
from .components.sidebar import sidebar_navigation
from .components.overview_metrics import show_overview_metrics
from .components.column_stats import show_column_stats_card
from .components.pipeline_panel import render_enhanced_action_log_ui as render_pipeline_panel

# Import all steps
from .steps.step1_upload import step1_upload
from .steps.step2_datatypes import step2_datatypes
from .steps.step3_textclean import step3_textclean
from .steps.step4_missing import step4_missing
from .steps.step5_outliers import step5_outliers
from .steps.step6_columns import step6_columns
from .steps.step7_pivot import step7_pivot
from .steps.step8_features import step8_features
from .steps.step9_ml import step9_ml
from .steps.step10_insights import step10_insights
from .steps.step11_visual import step11_visual
from .steps.step12_export import step12_export

# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Data Cleaning Pipeline",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------
# Add this at the beginning of your main() function or before checking dependencies
def main():
    # Initialize session state
    if 'dependencies_checked' not in st.session_state:
        st.session_state.dependencies_checked = False
    
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # Initialize other session state variables used in step9_ml
    if 'feature_engineering_columns' not in st.session_state:
        st.session_state.feature_engineering_columns = {'selected': []}
    
    if 'model_results' not in st.session_state:
        st.session_state.model_results = []
    
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    
    if 'training_done' not in st.session_state:
        st.session_state.training_done = False
    
    if 'current_problem_type' not in st.session_state:
        st.session_state.current_problem_type = None
    
    if 'stability_analysis' not in st.session_state:
        st.session_state.stability_analysis = {}
    
    # Now check dependencies
    if not st.session_state.dependencies_checked:
        # Your dependency checking code here
        st.session_state.dependencies_checked = True
    
    sidebar_navigation()
    
    # Create two columns layout for all steps
    col_main, col_pipeline = st.columns([3, 1])
    
    with col_main:
        if st.session_state.current_step == 1:
            step1_upload()
        
        elif st.session_state.df is not None:
            step = st.session_state.current_step
            
            if step == 2:
                step2_datatypes()
            elif step == 3:
                step3_textclean()
            elif step == 4:
                step4_missing()
            elif step == 5:
                step5_outliers()
            elif step == 6:
                step6_columns()
            elif step == 7:
                step7_pivot()
            elif step == 8:
                step8_features()
            elif step == 9:
                step9_ml()
            elif step == 10:
                step10_insights()
            elif step == 11:
                step11_visual()
            elif step == 12:
                try:
                    step12_export()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please upload a dataset first!")
            if st.button("← Go to Upload"):
                st.session_state.current_step = 1
                st.rerun()
    
    with col_pipeline:
        # Only render pipeline panel if we have data
        if st.session_state.df is not None and st.session_state.current_step > 1:
            render_enhanced_action_log_ui()

if __name__ == "__main__":
    main()