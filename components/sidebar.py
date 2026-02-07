# Navigation sidebar
import streamlit as st
from ..utils.dependency_utils import check_dependencies

def sidebar_navigation():
    st.sidebar.header("🧹 Smart Data Cleaning Pipeline")
    
    # Check dependencies on first load
    if not st.session_state.dependencies_checked:
        dependencies = check_dependencies()
        missing = [name for name, available in dependencies.items() if not available]
        if missing:
            with st.sidebar.expander("⚠️ Missing Dependencies", expanded=True):
                st.warning(f"Missing packages: {', '.join(missing)}")
                st.info("Some features may be unavailable. Install missing packages for full functionality.")
        st.session_state.dependencies_checked = True
    
    # Steps definition
    steps = [
        "Step 1 · Upload Data",
        "Step 2 · Data Types",
        "Step 3 · Text Cleaning",
        "Step 4 · Missing Values",
        "Step 5 · Outliers",
        "Step 6 · Columns (Add/Drop)",
        "Step 7 · Pivot Tables",
        "Step 8 · Feature Engineering",
        "Step 9 · ML & AutoML",
        "Step 10 · Analysis & Insights",
        "Step 11 · Visual Playground",
        "Step 12 · Final & Export"
    ]
    
    current_step = st.sidebar.radio("Navigation:", steps, index=st.session_state.current_step - 1)
    st.session_state.current_step = steps.index(current_step) + 1
    
    # Data summary when data is loaded
    if st.session_state.df is not None:
        st.sidebar.divider()
        st.sidebar.markdown("### 📊 Data Summary")
        
        df = st.session_state.df
        st.sidebar.metric("Rows", f"{df.shape[0]:,}")
        st.sidebar.metric("Columns", f"{df.shape[1]:,}")
        
        # Missing values percentage
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        st.sidebar.metric("Missing %", f"{missing_pct:.1f}%")
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.sidebar.metric("Memory (MB)", f"{memory_mb:.2f}")
        
        # Next recommended step
        st.sidebar.divider()
        st.sidebar.markdown("### 🎯 Next Recommended Step")
        
        # Simple logic to suggest next step
        if missing_pct > 20:
            st.sidebar.info("Missing Values treatment recommended")
        elif len(df.select_dtypes(include=['object', 'string']).columns) > 0:
            st.sidebar.info("Text cleaning recommended")
        else:
            next_step_idx = min(st.session_state.current_step, 11)
            st.sidebar.info(f"Continue with {steps[next_step_idx]}")