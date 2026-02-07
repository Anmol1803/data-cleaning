# History and undo/redo functions
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import streamlit as st
import pandas as pd
import numpy as np

# Import from other utility files
from .memory_utils import sync_feature_engineering_state
from .memory_utils import reset_feature_engineering_state

# ---------------------------------------------------------------------
# History and time utilities
# ---------------------------------------------------------------------
def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _now_ts() -> float:
    return datetime.now().timestamp()

def _snapshot(label: str, detailed_changes: Optional[List[Dict]] = None) -> None:
    """Enhanced snapshot with detailed change tracking"""
    if st.session_state.df is not None:
        # Increment action sequence
        st.session_state.action_seq += 1
        
        # Create history entry with enhanced details
        history_entry = {
            'id': st.session_state.action_seq,
            'label': label,
            'time': _now_str(),
            'timestamp': _now_ts(),
            'df': st.session_state.df.copy(deep=True),
            'detailed_changes': detailed_changes or [],
            'step': st.session_state.current_step
        }
        
        # Add to history and clear redo stack
        st.session_state.history.append(history_entry)
        st.session_state.future = []  # Clear redo stack on new action

def init_history_on_upload(label: str) -> None:
    """Initialize history only when a new dataset is uploaded."""
    if 'df' not in st.session_state or st.session_state.get('dataset_label') != label:
        # Clear all states on new dataset
        st.session_state.history = []
        st.session_state.future = []
        st.session_state.action_seq = 0
        st.session_state.action_log = []
        st.session_state.detailed_changes = []
        st.session_state.pivot_tables = []
        st.session_state.trained_models = {}
        st.session_state.model_results = []
        st.session_state.dataset_label = label
        
        # Reset feature engineering
        reset_feature_engineering_state()
        
        # Take initial snapshot
        if st.session_state.df is not None:
            _snapshot(label, [{'type': 'dataset_upload', 'details': f'Uploaded: {label}'}])

def log_bulk_action(base_message: str, changes: List[Dict]) -> None:
    """Log bulk operations with individual change tracking"""
    timestamp = _now_str()
    
    # Add main action
    st.session_state.action_log.append(f"[{timestamp}] {base_message}")
    
    # Add individual changes
    for change in changes:
        col = change.get('column', 'Unknown')
        operation = change.get('operation', 'Unknown')
        details = change.get('details', '')
        st.session_state.action_log.append(f"  └─ {col}: {operation} {details}")
    
    # Store detailed changes for history
    st.session_state.detailed_changes = changes

def log_action(message: str, snapshot: bool = False, detailed_changes: Optional[List[Dict]] = None) -> None:
    """Enhanced log action with detailed change tracking"""
    timestamp = _now_str()
    st.session_state.action_log.append(f"[{timestamp}] {message}")
    
    if snapshot:
        _snapshot(message, detailed_changes or st.session_state.detailed_changes)
        # Clear detailed changes after snapshot
        st.session_state.detailed_changes = []

def undo_last() -> None:
    """Enhanced undo with detailed tracking"""
    if len(st.session_state.history) <= 1:
        st.warning("No more steps to undo.")
        return
    
    last = st.session_state.history.pop()
    st.session_state.future.append(last)
    prev = st.session_state.history[-1]
    st.session_state.df = prev['df'].copy(deep=True)
    
    # Log undo action
    change_count = len(last.get('detailed_changes', []))
    log_message = f"Undone: {last['label']} ({change_count} changes)"
    st.session_state.action_log.append(f"[{_now_str()}] ↩️ {log_message}")
    st.success(log_message)
    
    # Sync feature engineering state
    if st.session_state.df is not None:
        sync_feature_engineering_state(st.session_state.df)

def redo_next() -> None:
    """Enhanced redo with detailed tracking"""
    if not st.session_state.future:
        st.warning("Nothing to redo.")
        return
    
    nxt = st.session_state.future.pop()
    st.session_state.history.append({
        'id': nxt['id'],
        'label': nxt['label'],
        'time': _now_str(),
        'timestamp': _now_ts(),
        'df': nxt['df'].copy(deep=True),
        'detailed_changes': nxt.get('detailed_changes', []),
        'step': nxt.get('step', st.session_state.current_step)
    })
    st.session_state.df = nxt['df'].copy(deep=True)
    
    # Log redo action
    change_count = len(nxt.get('detailed_changes', []))
    log_message = f"Redone: {nxt['label']} ({change_count} changes)"
    st.session_state.action_log.append(f"[{_now_str()}] ↪️ {log_message}")
    st.success(log_message)
    
    # Sync feature engineering state
    if st.session_state.df is not None:
        sync_feature_engineering_state(st.session_state.df)

def revert_to_action(action_id: int) -> None:
    """Revert to specific action with detailed tracking"""
    ids = [h['id'] for h in st.session_state.history]
    if action_id not in ids:
        st.error("Selected action not found in history.")
        return
    
    idx = ids.index(action_id)
    trimmed = st.session_state.history[:idx+1]
    removed = st.session_state.history[idx+1:]
    
    # Calculate total changes being reverted
    total_changes = sum(len(h.get('detailed_changes', [])) for h in removed)
    
    st.session_state.history = trimmed
    st.session_state.future = removed
    st.session_state.df = st.session_state.history[-1]['df'].copy(deep=True)
    
    # Log revert action
    log_message = f"Reverted to step #{action_id}: {st.session_state.history[-1]['label']} ({len(removed)} steps, {total_changes} changes)"
    st.session_state.action_log.append(f"[{_now_str()}] ⏪ {log_message}")
    st.success(log_message)
    
    # Sync feature engineering state
    if st.session_state.df is not None:
        sync_feature_engineering_state(st.session_state.df)

def render_enhanced_action_log_ui():
    """Enhanced action log UI with Power BI-style features"""
    st.markdown("#### 📝 Applied Steps Timeline")
    
    if not st.session_state.history:
        st.info("No steps yet.")
        return
    
    # Timeline view
    hist_rows = []
    for h in st.session_state.history:
        change_count = len(h.get('detailed_changes', []))
        hist_rows.append({
            "Step": f"#{h['id']}",
            "Action": h['label'],
            "Time": h['time'],
            "Changes": f"{change_count}",
            "Status": "✅"
        })
    
    hist_df = pd.DataFrame(hist_rows)
    st.dataframe(hist_df, use_container_width=True, height=240)
    
    # Step selection with details
    st.markdown("---")
    st.markdown("#### 🔍 Step Details")
    
    step_options = [f"#{h['id']} - {h['label']}" for h in st.session_state.history]
    
    # FIXED: Use unique key based on current step
    selected_step = st.selectbox(
        "Select step to view details:", 
        step_options, 
        key=f"step_select_pipeline_{st.session_state.current_step}"
    )
    
    if selected_step:
        step_id = int(selected_step.split('#')[1].split(' ')[0])
        selected_history = next((h for h in st.session_state.history if h['id'] == step_id), None)
        
        if selected_history:
            st.write(f"**Action:** {selected_history['label']}")
            st.write(f"**Time:** {selected_history['time']}")
            
            # Show detailed changes
            detailed_changes = selected_history.get('detailed_changes', [])
            if detailed_changes:
                st.markdown("**Individual Changes:**")
                for change in detailed_changes:
                    col = change.get('column', 'Unknown')
                    operation = change.get('operation', 'Unknown')
                    details = change.get('details', '')
                    st.write(f"  • **{col}**: {operation} {details}")
            else:
                st.info("No detailed change tracking for this step")
    
    # Enhanced controls - USING UNIQUE KEYS BASED ON CURRENT STEP
    st.markdown("---")
    st.markdown("#### ⚡ Controls")
    
    # FIXED: Use current_step in key instead of time.time()
    if st.button("↩️ Undo Last Step", use_container_width=True, key=f"undo_{st.session_state.current_step}"):
        undo_last()
        st.rerun()
    
    if st.button("↪️ Redo Next", use_container_width=True, key=f"redo_{st.session_state.current_step}"):
        redo_next()
        st.rerun()
    
    if st.session_state.history:
        revert_options = [f"#{h['id']} - {h['label'][:30]}..." for h in st.session_state.history]
        
        # FIXED: Unique key for revert selectbox
        revert_to = st.selectbox(
            "Revert to:", 
            revert_options, 
            key=f"revert_select_pipeline_{st.session_state.current_step}"
        )
        
        # FIXED: Unique key for revert button
        if st.button("⏪ Revert to Selected Step", use_container_width=True, key=f"revert_{st.session_state.current_step}"):
            step_id = int(revert_to.split('#')[1].split(' ')[0])
            revert_to_action(step_id)
            st.rerun()
    
    # FIXED: Unique key for clear button
    if st.button("🗑️ Clear Entire History", use_container_width=True, type="secondary", key=f"clear_{st.session_state.current_step}"):
        st.session_state.action_log = []
        st.session_state.detailed_changes = []
        st.rerun()
    
    # Export action log
    st.markdown("---")
    
    # FIXED: Unique key for export button
    if st.button("📥 Export Action Log", use_container_width=True, key=f"export_{st.session_state.current_step}"):
        log_text = "\n".join(st.session_state.action_log[-100:])  # Last 100 entries
        st.download_button(
            label="Download Log",
            data=log_text,
            file_name=f"action_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )