import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Interactive Data Cleaning Pipeline",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'action_log' not in st.session_state:
    st.session_state.action_log = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'filename' not in st.session_state:
    st.session_state.filename = None

# Utility functions

def log_action(message):
    """Add action to log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.action_log.append(f"[{timestamp}] {message}")


def show_overview_metrics(df):
    """Display dataset overview with metrics and charts"""
    st.header("📊 Dataset Overview")

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", f"{df.shape[1]:,}")
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory (MB)", f"{memory_mb:.2f}")
    with col4:
        total_missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        st.metric("Missing (%)", f"{missing_pct:.1f}%")

    # Data types distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Types Distribution")
        dtype_counts = df.dtypes.value_counts()
        # Convert dtype objects to strings for JSON serialization
        fig = px.pie(values=dtype_counts.values.tolist(), 
                     names=[str(x) for x in dtype_counts.index],
                     title="Column Types")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Missing Values by Column")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=True)

        if len(missing_data) > 0:
            # Convert to lists for JSON serialization
            fig = px.bar(x=missing_data.values.tolist(), 
                        y=[str(x) for x in missing_data.index],
                        orientation='h', title="Missing Values Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 No missing values found!")


def show_column_stats_card(df, col):
    """Display detailed column statistics in a card format"""
    col_data = df[col]

    with st.container():
        st.subheader(f"📋 Column: {col}")

        # Basic stats in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Count", f"{col_data.count():,}")
        with col2:
            st.metric("Missing", f"{col_data.isnull().sum():,}")
        with col3:
            st.metric("Unique", f"{col_data.nunique():,}")
        with col4:
            missing_pct = (col_data.isnull().sum() / len(df)) * 100 if len(df) > 0 else 0
            st.metric("Missing %", f"{missing_pct:.1f}%")

        # Type-specific statistics
        if pd.api.types.is_numeric_dtype(col_data):
            st.subheader("📈 Numerical Statistics")
            col1, col2 = st.columns(2)

            with col1:
                stats = col_data.describe()
                # Safely extract median (50%) if present
                median = stats.get('50%') if '50%' in stats.index else col_data.median()
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Median', 'Std', 'Min', 'Max'],
                    'Value': [stats.get('mean', np.nan), median, stats.get('std', np.nan),
                              stats.get('min', np.nan), stats.get('max', np.nan)]
                })
                st.dataframe(stats_df, use_container_width=True)

            with col2:
                # Distribution plot - convert to list for JSON serialization
                clean_data = col_data.dropna().tolist()
                temp_df = pd.DataFrame({col: clean_data})
                fig = px.histogram(temp_df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("📝 Categorical Statistics")
            value_counts = col_data.value_counts().head(10)

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(value_counts, use_container_width=True)
            with col2:
                if len(value_counts) > 1:
                    # Convert to lists for JSON serialization
                    fig = px.bar(x=[str(x) for x in value_counts.index], 
                                y=value_counts.values.tolist(),
                                title=f"Top Values in {col}")
                    st.plotly_chart(fig, use_container_width=True)


def data_type_management_step(df):
    """Step 2: Data Type Management"""
    st.header("🔧 Data Type Management")

    # Show current data types
    dtype_df = pd.DataFrame({
        'Column': df.columns,
        'Current Type': df.dtypes.astype(str),
        'Sample Values': [str(df[col].dropna().head(3).tolist()) for col in df.columns]
    })

    st.subheader("Current Data Types")
    st.dataframe(dtype_df, use_container_width=True)

    # Column selection
    col_to_change = st.selectbox("Select column to modify:",
                                [''] + list(df.columns))

    if col_to_change:
        show_column_stats_card(df, col_to_change)

        st.subheader("Change Data Type")
        new_type = st.selectbox("Select new data type:",
                               ['int', 'float', 'string', 'datetime', 'category', 'bool', 'custom'])

        if new_type == 'custom':
            custom_type = st.text_input("Enter pandas dtype:")
            new_type = custom_type

        if st.button(f"Convert {col_to_change} to {new_type}"):
            success, message = convert_column_type(df, col_to_change, new_type)
            if success:
                st.success(f"✅ {message}")
                log_action(f"Changed {col_to_change} to {new_type}")
                st.rerun()
            else:
                st.error(f"❌ {message}")


def convert_column_type(df, col_name, new_type):
    """Convert column data type with error handling - FIXED VERSION"""
    try:
        if new_type == 'int':
            # First convert to float to handle any non-numeric values
            temp = pd.to_numeric(df[col_name], errors='coerce')
            # Round to nearest integer before converting
            temp = temp.round(0)
            # Convert to Int64 (nullable integer)
            df[col_name] = temp.astype('Int64')
        elif new_type == 'float':
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        elif new_type == 'string':
            df[col_name] = df[col_name].astype(str)
        elif new_type == 'datetime':
            df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
        elif new_type == 'category':
            df[col_name] = df[col_name].astype('category')
        elif new_type == 'bool':
            # Handle common boolean representations
            df[col_name] = df[col_name].replace({"True": True, "False": False, "true": True, "false": False,
                                                  "1": True, "0": False, 1: True, 0: False}).astype(bool)
        else:
            df[col_name] = df[col_name].astype(new_type)
        return True, f"Successfully converted to {new_type}"
    except Exception as e:
        return False, f"Conversion failed: {str(e)}"


def text_cleaning_step(df):
    """Step 3: Text Cleaning Tools - FIXED INDENTATION"""
    st.header("📝 Text Cleaning Tools")

    text_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    if not text_cols:
        st.info("No text/categorical columns found in dataset.")
    else:
        col_choice = st.selectbox("Select column to clean:", text_cols)

        # --- Extra spaces & special characters ---
        if st.checkbox("Remove extra spaces"):
            df[col_choice] = df[col_choice].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            st.success(f"Extra spaces removed from '{col_choice}'")

        if st.checkbox("Remove special characters (keep letters/numbers/spaces)"):
            df[col_choice] = df[col_choice].astype(str).str.replace(r"[^A-Za-z0-9\s]+", "", regex=True)
            st.success(f"Special characters removed from '{col_choice}'")

        # --- Case conversion ---
        case_action = st.radio("Case Conversion:", ["None", "Lowercase", "Uppercase", "Title Case"])
        if case_action == "Lowercase":
            df[col_choice] = df[col_choice].astype(str).str.lower()
            st.success(f"Converted '{col_choice}' to lowercase")
        elif case_action == "Uppercase":
            df[col_choice] = df[col_choice].astype(str).str.upper()
            st.success(f"Converted '{col_choice}' to UPPERCASE")
        elif case_action == "Title Case":
            df[col_choice] = df[col_choice].astype(str).str.title()
            st.success(f"Converted '{col_choice}' to Title Case")

        st.subheader("🔎 Preview After Cleaning")
        st.write(df[[col_choice]].head(10))

        # --- Replace Text ---
        st.subheader("🔄 Replace Text")
        find_text = st.text_input("Find text:", key=f"find_{col_choice}")
        replace_text = st.text_input("Replace with:", key=f"replace_{col_choice}")

        if st.button("Replace", key=f"replace_btn_{col_choice}"):
            if find_text != "":
                df[col_choice] = df[col_choice].astype(str).str.replace(find_text, replace_text, regex=False)
                st.success(f"Replaced '{find_text}' with '{replace_text}' in '{col_choice}'")
                log_action(f"Replaced '{find_text}' with '{replace_text}' in column '{col_choice}'")
                st.rerun()
            else:
                st.warning("Please enter text to find.")


def missing_values_treatment_step(df):
    """Step 4: Missing Values Treatment - FIXED DEPRECATED METHODS"""
    st.header("🔧 Missing Values Treatment")

    # Find columns with missing values
    missing_cols = df.columns[df.isnull().any()].tolist()

    if not missing_cols:
        st.success("🎉 No missing values found!")
        return

    # Missing values overview
    missing_summary = pd.DataFrame({
        'Column': missing_cols,
        'Missing Count': [df[col].isnull().sum() for col in missing_cols],
        'Missing %': [df[col].isnull().sum() / len(df) * 100 for col in missing_cols]
    }).sort_values('Missing Count', ascending=False)

    st.subheader("Missing Values Summary")
    st.dataframe(missing_summary, use_container_width=True)

    # Column selection for treatment
    col_to_clean = st.selectbox("Select column to clean:",
                               [''] + missing_cols)

    if col_to_clean:
        show_column_stats_card(df, col_to_clean)

        st.subheader("Treatment Options")
        is_numeric = pd.api.types.is_numeric_dtype(df[col_to_clean])

        # Create treatment buttons in columns
        col1, col2, col3, col4 = st.columns(4)

        if is_numeric:
            with col1:
                if st.button("Fill with Mean"):
                    df[col_to_clean].fillna(df[col_to_clean].mean(), inplace=True)
                    st.success("Filled with mean")
                    log_action(f"Filled {col_to_clean} with mean")
                    st.rerun()

                if st.button("Fill with Min"):
                    df[col_to_clean].fillna(df[col_to_clean].min(), inplace=True)
                    st.success("Filled with minimum")
                    log_action(f"Filled {col_to_clean} with min")
                    st.rerun()

            with col2:
                if st.button("Fill with Median"):
                    df[col_to_clean].fillna(df[col_to_clean].median(), inplace=True)
                    st.success("Filled with median")
                    log_action(f"Filled {col_to_clean} with median")
                    st.rerun()

                if st.button("Fill with Max"):
                    df[col_to_clean].fillna(df[col_to_clean].max(), inplace=True)
                    st.success("Filled with maximum")
                    log_action(f"Filled {col_to_clean} with max")
                    st.rerun()
        else:
            with col1:
                if st.button("Fill with Most Frequent"):
                    most_frequent = df[col_to_clean].mode().iloc[0] if not df[col_to_clean].mode().empty else 'Unknown'
                    df[col_to_clean].fillna(most_frequent, inplace=True)
                    st.success("Filled with most frequent value")
                    log_action(f"Filled {col_to_clean} with most frequent")
                    st.rerun()

            with col2:
                if st.button("Show Unique Values"):
                    st.subheader("Value Frequency")
                    value_counts = df[col_to_clean].value_counts(dropna=False)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.dataframe(value_counts.head(20))
                    with col_b:
                        # Convert to lists for JSON serialization
                        fig = px.bar(x=[str(x) for x in value_counts.head(10).index],
                                   y=value_counts.head(10).values.tolist(),
                                   title="Top 10 Values")
                        st.plotly_chart(fig, use_container_width=True)

        with col3:
            if st.button("Forward Fill"):
                # FIXED: Use ffill() instead of fillna(method='ffill')
                df[col_to_clean] = df[col_to_clean].ffill()
                st.success("Forward filled")
                log_action(f"Forward filled {col_to_clean}")
                st.rerun()

            if st.button("Drop Rows"):
                original_len = len(df)
                df.dropna(subset=[col_to_clean], inplace=True)
                dropped = original_len - len(df)
                st.success(f"Dropped {dropped} rows")
                log_action(f"Dropped {dropped} rows from {col_to_clean}")
                st.rerun()

        with col4:
            if st.button("Backward Fill"):
                # FIXED: Use bfill() instead of fillna(method='bfill')
                df[col_to_clean] = df[col_to_clean].bfill()
                st.success("Backward filled")
                log_action(f"Backward filled {col_to_clean}")
                st.rerun()

            # Custom value input
            custom_value = st.text_input("Custom fill value:", key=f"custom_{col_to_clean}")
            if st.button("Fill with Custom") and custom_value != "":
                # Try to cast custom value to appropriate dtype if numeric column
                try:
                    if pd.api.types.is_numeric_dtype(df[col_to_clean]):
                        cast_val = float(custom_value)
                    else:
                        cast_val = custom_value
                    df[col_to_clean].fillna(cast_val, inplace=True)
                    st.success(f"Filled with '{custom_value}'")
                    log_action(f"Filled {col_to_clean} with custom value: {custom_value}")
                    st.rerun()
                except Exception:
                    st.error("Could not cast custom value to column dtype. Filled as string instead.")
                    df[col_to_clean].fillna(custom_value, inplace=True)
                    log_action(f"Filled {col_to_clean} with custom value as string: {custom_value}")
                    st.rerun()


def outlier_cleaning_step(df):
    """Step 5: Outlier Detection and Custom Cleaning"""
    st.header("🔧 Outlier Detection & Custom Cleaning")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns found for outlier detection.")
        return

    selected_col = st.selectbox("Select column for outlier analysis:",
                               [''] + numeric_cols)

    if selected_col:
        col_data = df[selected_col].dropna()

        # Outlier detection methods
        method = st.radio("Outlier Detection Method:",
                         ['IQR Method', 'Z-Score Method', 'Custom Range'])

        # Defaults
        lower_bound = None
        upper_bound = None
        min_val = None
        max_val = None

        if method == 'IQR Method':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]

        elif method == 'Z-Score Method':
            z_threshold = st.slider("Z-Score Threshold:", 1.0, 4.0, 3.0, 0.1)
            # compute z-scores aligned with original index
            mean = col_data.mean()
            std = col_data.std()
            z_scores = (df[selected_col] - mean).abs() / std
            outliers = df[z_scores > z_threshold]

        else:  # Custom Range
            min_val = st.number_input("Minimum value:", value=float(col_data.min()))
            max_val = st.number_input("Maximum value:", value=float(col_data.max()))
            outliers = df[(df[selected_col] < min_val) | (df[selected_col] > max_val)]

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Outliers Found", len(outliers))
            st.metric("Outlier %", f"{(len(outliers)/len(df)*100) if len(df)>0 else 0:.1f}%")

        with col2:
            # Visualization - convert to clean data for JSON serialization
            clean_data = df[selected_col].dropna().tolist()
            temp_df = pd.DataFrame({selected_col: clean_data})
            fig = px.box(temp_df, y=selected_col, title=f"Box Plot of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)

        if len(outliers) > 0:
            st.subheader("Outlier Treatment")
            treatment = st.radio("Treatment method:",
                               ['Remove Outliers', 'Cap to Bounds', 'Keep Outliers'])

            if st.button("Apply Treatment"):
                if treatment == 'Remove Outliers':
                    original_len = len(df)
                    df.drop(outliers.index, inplace=True)
                    removed = original_len - len(df)
                    st.success(f"Removed {removed} outliers")
                    log_action(f"Removed {removed} outliers from {selected_col}")
                    st.rerun()

                elif treatment == 'Cap to Bounds':
                    if method == 'IQR Method' and lower_bound is not None and upper_bound is not None:
                        df[selected_col] = df[selected_col].clip(lower_bound, upper_bound)
                    elif method == 'Custom Range' and min_val is not None and max_val is not None:
                        df[selected_col] = df[selected_col].clip(min_val, max_val)
                    st.success("Outliers capped to bounds")
                    log_action(f"Capped outliers in {selected_col}")
                    st.rerun()


def drop_columns_step(df):
    """Step 6: Drop Columns"""
    st.header("🗑️ Drop Columns")

    if len(df.columns) == 0:
        st.warning("No columns to drop!")
        return

    st.subheader("Current Columns")

    # Create a dataframe showing column info
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Missing': [df[col].isnull().sum() for col in df.columns],
        'Unique': [df[col].nunique() for col in df.columns]
    })

    st.dataframe(col_info, use_container_width=True)

    # Multi-select for columns to drop
    cols_to_drop = st.multiselect("Select columns to drop:", df.columns.tolist())

    if cols_to_drop:
        st.warning(f"You are about to drop {len(cols_to_drop)} columns: {cols_to_drop}")

        if st.button("Confirm Drop Columns"):
            df.drop(columns=cols_to_drop, inplace=True)
            st.success(f"✅ Dropped {len(cols_to_drop)} columns")
            log_action(f"Dropped columns: {cols_to_drop}")
            st.rerun()


def final_overview_and_export(df):
    """Final step: Overview and Export"""
    st.header("📋 Final Overview & Export")

    # Show before/after comparison if original exists
    if st.session_state.original_df is not None:
        st.subheader("Before vs After Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Original Dataset**")
            orig_df = st.session_state.original_df
            st.metric("Rows", f"{orig_df.shape[0]:,}")
            st.metric("Columns", f"{orig_df.shape[1]:,}")
            orig_missing = orig_df.isnull().sum().sum()
            orig_total = orig_df.shape[0] * orig_df.shape[1]
            orig_missing_pct = (orig_missing / orig_total) * 100 if orig_total > 0 else 0
            st.metric("Missing %", f"{orig_missing_pct:.1f}%")

        with col2:
            st.write("**Cleaned Dataset**")
            st.metric("Rows", f"{df.shape[0]:,}")
            st.metric("Columns", f"{df.shape[1]:,}")
            curr_missing = df.isnull().sum().sum()
            curr_total = df.shape[0] * df.shape[1]
            curr_missing_pct = (curr_missing / curr_total) * 100 if curr_total > 0 else 0
            st.metric("Missing %", f"{curr_missing_pct:.1f}%")

    # Current dataset overview
    show_overview_metrics(df)

    # Export section
    st.subheader("📥 Export Cleaned Dataset")

    filename = st.text_input("Filename:",
                            value=f"cleaned_{st.session_state.filename}" if st.session_state.filename else "cleaned_dataset.csv")

    if not filename.endswith('.csv'):
        filename += '.csv'

    # Convert dataframe to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="📥 Download Cleaned Dataset",
        data=csv_data,
        file_name=filename,
        mime="text/csv"
    )

    if st.button("🔁 Reset to Original Dataset"):
        if st.session_state.original_df is not None:
            st.session_state.df = st.session_state.original_df.copy()
            log_action("Reset dataset to original")
            st.rerun()

    if st.button("🧾 Clear Action Log"):
        st.session_state.action_log = []
        st.rerun()


def sidebar_navigation():
    """Sidebar navigation and action log"""
    st.sidebar.header("🧹 Data Cleaning Pipeline")

    # Step navigation
    steps = [
        "📤 Upload Data",
        "🔧 Data Types",
        "📝 Text Cleaning",
        "🔧 Missing Values",
        "🔧 Outliers & Custom",
        "🗑️ Drop Columns",
        "📋 Final & Export"
    ]

    current_step = st.sidebar.radio("Navigation:", steps, index=st.session_state.current_step-1)
    st.session_state.current_step = steps.index(current_step) + 1

    # Action log
    if st.session_state.action_log:
        with st.sidebar.expander("📝 Action Log", expanded=False):
            for action in reversed(st.session_state.action_log[-25:]):
                st.text(action)

    # Dataset info
    if st.session_state.df is not None:
        st.sidebar.subheader("📊 Current Dataset")
        st.sidebar.metric("Rows", f"{st.session_state.df.shape[0]:,}")
        st.sidebar.metric("Columns", f"{st.session_state.df.shape[1]:,}")
        missing_pct = (st.session_state.df.isnull().sum().sum() /
                      (st.session_state.df.shape[0] * st.session_state.df.shape[1])) * 100 if st.session_state.df.shape[0]>0 else 0
        st.sidebar.metric("Missing %", f"{missing_pct:.1f}%")


def main():
    """Main application function"""
    st.title("🧹 Interactive Data Cleaning Pipeline")
    st.markdown("Transform your messy data into clean, analysis-ready datasets with this guided pipeline.")

    sidebar_navigation()

    # Step 1: Data Upload
    if st.session_state.current_step == 1:
        st.header("📤 Upload Your Dataset")

        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df.copy()
                st.session_state.original_df = df.copy()
                st.session_state.filename = uploaded_file.name

                st.success(f"✅ Uploaded: {uploaded_file.name}")
                log_action(f"Uploaded dataset: {uploaded_file.name} ({df.shape[0]} rows, {df.shape[1]} columns)")

                show_overview_metrics(df)

                # Sample data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(), use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("➡️ Start Cleaning Pipeline"):
                        st.session_state.current_step = 2
                        st.rerun()
                with col2:
                    if st.button("🔁 Load Example Dataset"):
                        example_df = px.data.iris()
                        st.session_state.df = example_df.copy()
                        st.session_state.original_df = example_df.copy()
                        st.session_state.filename = "iris_example.csv"
                        log_action("Loaded example Iris dataset")
                        st.rerun()

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

        else:
            st.info("👆 Please upload a CSV file to get started!")

    # Steps 2-7: Only show if data is loaded
    elif st.session_state.df is not None:
        df = st.session_state.df

        # Navigation controls
        nav_col1, nav_col2 = st.columns([1, 1])
        with nav_col1:
            if st.button("⬅️ Previous"):
                st.session_state.current_step = max(1, st.session_state.current_step - 1)
                st.rerun()
        with nav_col2:
            if st.button("Next ➡️"):
                st.session_state.current_step = min(7, st.session_state.current_step + 1)
                st.rerun()

        if st.session_state.current_step == 2:
            data_type_management_step(df)
        elif st.session_state.current_step == 3:
            text_cleaning_step(df)
        elif st.session_state.current_step == 4:
            missing_values_treatment_step(df)
        elif st.session_state.current_step == 5:
            outlier_cleaning_step(df)
        elif st.session_state.current_step == 6:
            drop_columns_step(df)
        elif st.session_state.current_step == 7:
            final_overview_and_export(df)

    else:
        st.warning("Please upload a dataset first!")
        if st.button("← Go to Upload"):
            st.session_state.current_step = 1
            st.rerun()


if __name__ == "__main__":
    main()