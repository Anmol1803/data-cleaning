# Step 11: Visual Playground
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def step11_visual():
    """
    🎮 Visual Playground
    An Excel-like interactive chart builder where users can play with different visualizations.
    """
    st.header("Step 11 · Visual Playground")
    st.markdown(f"**Step 11 of 12**")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first!")
        return
    
    df = st.session_state.df
    
    st.markdown(
        "Build and play with interactive charts just like you would in Excel — "
        "pick your data, choose a chart type, and explore!"
    )

    # Layout: left for configuration, right for live preview
    config_col, preview_col = st.columns([1, 2])

    with config_col:
        st.subheader("🛠 Chart Setup")

        # Basic selections
        all_columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        x_axis = st.selectbox(
            "X-axis column",
            options=["(index)"] + all_columns,
            help="Select what goes on the horizontal axis. Choose '(index)' to use row index."
        )

        # Allow multiple y columns (like Excel multi-series)
        y_axis = st.multiselect(
            "Y-axis column(s)",
            options=numeric_cols,
            help="Select one or more numeric columns to plot."
        )

        chart_type = st.selectbox(
            "Chart type",
            [
                "Line",
                "Bar",
                "Stacked Bar",
                "Area",
                "Scatter",
                "Histogram",
                "Box Plot",
                "Violin Plot",
                "Heatmap"
            ]
        )

        color_by = st.selectbox(
            "Color / Group by (optional)",
            options=["(none)"] + all_columns,
            help="Use a column to create color groups / series."
        )

        # Optional filters
        st.subheader("🎯 Optional Filters")
        with st.expander("Add simple filters", expanded=False):
            filter_column = st.selectbox(
                "Filter column (optional)",
                options=["(none)"] + all_columns,
                key="viz_filter_col"
            )
            filter_value = None
            if filter_column != "(none)":
                unique_vals = df[filter_column].dropna().unique().tolist()
                # Convert numpy types to python native for Streamlit
                unique_vals = [v.item() if hasattr(v, "item") else v for v in unique_vals]
                filter_value = st.selectbox(
                    "Keep only rows where value equals:",
                    options=unique_vals,
                    key="viz_filter_val"
                )

        st.subheader("📏 Display Options")
        show_data_table = st.checkbox("Show data used for chart", value=False)
        use_log_y = st.checkbox("Log-scale Y axis (where applicable)", value=False)

        play_button = st.button("🚀 Generate Visualization", type="primary", use_container_width=True)

    with preview_col:
        st.subheader("📊 Live Preview")

        if not y_axis and chart_type not in ["Histogram", "Heatmap"]:
            st.info("Select at least one Y-axis column to see a chart.")
            return

        if play_button:
            plot_df = df.copy()

            # Apply filter if set
            if filter_column != "(none)" and filter_value is not None:
                plot_df = plot_df[plot_df[filter_column] == filter_value]

            if plot_df.empty:
                st.warning("Filter removed all rows. Try relaxing the filter.")
                return

            # Build base x-axis
            if x_axis == "(index)":
                plot_df = plot_df.reset_index()
                x_col_name = "index"
            else:
                x_col_name = x_axis

            try:
                fig = None

                if chart_type in ["Line", "Bar", "Stacked Bar", "Area"]:
                    if not y_axis:
                        st.warning("Please select at least one numeric column for the Y-axis.")
                        return

                    # Melt to long format if multiple series
                    long_df = plot_df.melt(
                        id_vars=[x_col_name] + ([color_by] if color_by != "(none)" else []),
                        value_vars=y_axis,
                        var_name="Series",
                        value_name="Value"
                    )

                    if chart_type == "Line":
                        fig = px.line(
                            long_df,
                            x=x_col_name,
                            y="Value",
                            color="Series" if color_by == "(none)" else color_by,
                            markers=True,
                            title="Line Chart"
                        )
                    elif chart_type == "Bar":
                        fig = px.bar(
                            long_df,
                            x=x_col_name,
                            y="Value",
                            color="Series" if color_by == "(none)" else color_by,
                            barmode="group",
                            title="Bar Chart"
                        )
                    elif chart_type == "Stacked Bar":
                        fig = px.bar(
                            long_df,
                            x=x_col_name,
                            y="Value",
                            color="Series" if color_by == "(none)" else color_by,
                            barmode="stack",
                            title="Stacked Bar Chart"
                        )
                    elif chart_type == "Area":
                        fig = px.area(
                            long_df,
                            x=x_col_name,
                            y="Value",
                            color="Series" if color_by == "(none)" else color_by,
                            title="Area Chart"
                        )

                    if use_log_y:
                        fig.update_yaxes(type="log")

                elif chart_type == "Scatter":
                    if len(y_axis) != 1:
                        st.warning("Scatter plot supports exactly one Y-axis column.")
                        return
                    fig = px.scatter(
                        plot_df,
                        x=x_col_name,
                        y=y_axis[0],
                        color=None if color_by == "(none)" else color_by,
                        title="Scatter Plot",
                        trendline="ols"
                    )
                    if use_log_y:
                        fig.update_yaxes(type="log")

                elif chart_type == "Histogram":
                    numeric_options = numeric_cols or plot_df.select_dtypes(include=[np.number]).columns.tolist()
                    if not numeric_options:
                        st.warning("No numeric columns available for histogram.")
                        return
                    hist_col = st.selectbox("Numeric column for histogram", options=numeric_options, key="viz_hist_col")
                    bins = st.slider("Number of bins", 5, 100, 30)
                    fig = px.histogram(
                        plot_df,
                        x=hist_col,
                        nbins=bins,
                        color=None if color_by == "(none)" else color_by,
                        title="Histogram"
                    )

                elif chart_type == "Box Plot":
                    if not y_axis:
                        st.warning("Select at least one Y-axis for box plot.")
                        return
                    fig = px.box(
                        plot_df,
                        x=None if color_by == "(none)" else color_by,
                        y=y_axis,
                        title="Box Plot"
                    )

                elif chart_type == "Violin Plot":
                    if len(y_axis) != 1:
                        st.warning("Violin plot supports exactly one Y-axis column.")
                        return
                    fig = px.violin(
                        plot_df,
                        x=None if color_by == "(none)" else color_by,
                        y=y_axis[0],
                        box=True,
                        points="all",
                        title="Violin Plot"
                    )

                elif chart_type == "Heatmap":
                    num_cols = numeric_cols or plot_df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(num_cols) < 2:
                        st.warning("Need at least two numeric columns for a heatmap.")
                        return
                    corr = plot_df[num_cols].corr()
                    fig = px.imshow(
                        corr,
                        text_auto=True,
                        aspect="auto",
                        title="Correlation Heatmap"
                    )

                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

                if show_data_table:
                    st.markdown("### 🔍 Data used for this chart")
                    st.dataframe(plot_df[[x_col_name] + y_axis].head(100), use_container_width=True)

            except Exception as e:
                st.error(f"Error generating visualization: {e}")
    
    # "Go to Next Step" button
    st.markdown("---")
    if st.button("Go to Next Step", type="primary", use_container_width=True):
        st.session_state.current_step = min(12, st.session_state.current_step + 1)
        st.rerun()