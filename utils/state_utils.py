# utils/state_utils.py
import streamlit as st

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'dependencies_checked': False,
        'current_step': 1,
        'df': None,
        'feature_engineering_columns': {'selected': []},
        'model_results': [],
        'trained_models': {},
        'training_done': False,
        'current_problem_type': None,
        'stability_analysis': {},
        # Add any other session state variables you use
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value