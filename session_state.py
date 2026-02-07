# Session state manager
import streamlit as st

# ---------------------------------------------------------------------
# Enhanced Session state initialization
# ---------------------------------------------------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'action_log' not in st.session_state:
    st.session_state.action_log = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'future' not in st.session_state:
    st.session_state.future = []
if 'action_seq' not in st.session_state:
    st.session_state.action_seq = 0
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'pivot_tables' not in st.session_state:
    st.session_state.pivot_tables = []
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}
if 'data_quality_score' not in st.session_state:
    st.session_state.data_quality_score = None
if 'feature_pipelines' not in st.session_state:
    st.session_state.feature_pipelines = {}
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = []
if 'feature_engineering_columns' not in st.session_state:
    st.session_state.feature_engineering_columns = {'selected': [], 'excluded': []}
if 'detailed_changes' not in st.session_state:
    st.session_state.detailed_changes = []
if 'memory_optimized' not in st.session_state:
    st.session_state.memory_optimized = False
if 'active_expanders' not in st.session_state:
    st.session_state.active_expanders = {}
if 'dependencies_checked' not in st.session_state:
    st.session_state.dependencies_checked = False