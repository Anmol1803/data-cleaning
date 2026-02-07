"""
Smart Data Cleaning Pipeline - Modularized Version

A comprehensive 12-step data cleaning and machine learning pipeline built with Streamlit.
This modular version organizes the original monolithic code into maintainable components.

Structure:
- main.py: Main entry point and navigation
- session_state.py: Centralized session state management
- utils/: Shared utility functions
- components/: Reusable UI components
- steps/: Individual step implementations (Step 1-12)
"""

__version__ = "1.0.0"
__author__ = "Smart Data Cleaning Pipeline"

# Export main components for easy access
from .main import main
from .session_state import *

# Export utilities
from .utils.dependency_utils import check_dependencies, graceful_fallback
from .utils.history_utils import (
    init_history_on_upload, log_action, log_bulk_action,
    undo_last, redo_next, revert_to_action, render_enhanced_action_log_ui
)
from .utils.memory_utils import optimize_dataframe_memory, clear_model_cache
from .utils.data_quality_utils import calculate_data_quality_score, detect_duplicates

# Export components
from .components.sidebar import sidebar_navigation
from .components.overview_metrics import show_overview_metrics
from .components.column_stats import show_column_stats_card

# Export all steps
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