# Step 7: Pivot Tables
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Import utilities
from ..utils.history_utils import log_action

def step7_pivot():
    st.header("Step 7 · Pivot Tables")
    st.markdown(f"**Step 7 of 12**")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first!")
        return
    
    df = st.session_state.df
                
    st.markdown("Create powerful pivot tables to summarize and analyze your data.")
    st.subheader("🔧 Configure Pivot Table")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Rows (Index)")
        row_fields = st.multiselect("Select columns for rows:", options=df.columns.tolist())
        st.markdown("Values (Aggregation)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        value_fields = st.multiselect("Select columns to aggregate:", options=numeric_cols)
    with col2:
        st.markdown("Columns")
        column_fields = st.multiselect("Select columns for column headers:", options=df.columns.tolist())
        st.markdown("Aggregation Functions")
        agg_functions = st.multiselect(
            "Select aggregation functions:",
            options=['sum', 'mean', 'median', 'count', 'min', 'max', 'std', 'var'],
            default=['sum']
        )
    with st.expander("⚙️ Advanced Options"):
        a, b, c = st.columns(3)
        with a:
            show_margins = st.checkbox("Show totals (margins)", value=False)
            fill_na = st.checkbox("Fill missing values", value=True)
            fill_value = st.number_input("Fill with value:", value=0.0) if fill_na else None
        with b:
            sort_by_values = st.checkbox("Sort by values", value=False)
            ascending = st.checkbox("Ascending order", value=False)
        with c:
            normalize = st.selectbox("Normalize:", options=['None', 'All', 'Index', 'Columns'])
    table_name = st.text_input("Pivot Table Name:", value=f"Pivot_Table_{len(st.session_state.pivot_tables) + 1}")
    if st.button("🔨 Create Pivot Table", type="primary", use_container_width=True):
        if not row_fields and not column_fields:
            st.error("❌ Please select at least one row or column field!")
        elif not value_fields:
            st.error("❌ Please select at least one value field to aggregate!")
        elif not agg_functions:
            st.error("❌ Please select at least one aggregation function!")
        else:
            try:
                index = row_fields if row_fields else None
                columns = column_fields if column_fields else None
                if len(agg_functions) == 1:
                    pivot = pd.pivot_table(
                        df, values=value_fields, index=index, columns=columns,
                        aggfunc=agg_functions[0], fill_value=fill_value if fill_na else None,
                        margins=show_margins, margins_name='Total'
                    )
                else:
                    pivot = pd.pivot_table(
                        df, values=value_fields, index=index, columns=columns,
                        aggfunc=agg_functions, fill_value=fill_value if fill_na else None,
                        margins=show_margins, margins_name='Total'
                    )
                if normalize != 'None':
                    if normalize == 'All':
                        pivot = pivot / pivot.sum().sum() * 100
                    elif normalize == 'Index':
                        pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
                    elif normalize == 'Columns':
                        pivot = pivot.div(pivot.sum(axis=0), axis=1) * 100
                if sort_by_values and len(pivot.columns) > 0:
                    sort_col = pivot.columns[0]
                    pivot = pivot.sort_values(by=sort_col, ascending=ascending)
                pivot_info = {
                    'name': table_name,
                    'data': pivot,
                    'config': {
                        'rows': row_fields, 'columns': column_fields,
                        'values': value_fields, 'agg_functions': agg_functions,
                        'normalized': normalize != 'None'
                    }
                }
                st.session_state.pivot_tables.append(pivot_info)
                st.success(f"✅ Pivot table '{table_name}' created successfully!")
                log_action(f"Created pivot table: {table_name}", snapshot=False)
            except Exception as e:
                st.error(f"❌ Error creating pivot table: {str(e)}")
    if st.session_state.pivot_tables:
        st.divider()
        st.subheader("📋 Saved Pivot Tables")
        table_names = [pt['name'] for pt in st.session_state.pivot_tables]
        selected_table = st.selectbox("Select pivot table to view:", table_names)
        if selected_table:
            pivot_info = next((pt for pt in st.session_state.pivot_tables if pt['name'] == selected_table), None)
            if pivot_info:
                pivot_data = pivot_info['data']
                config = pivot_info['config']
                with st.expander("📝 Table Configuration"):
                    st.write(f"Rows: {', '.join(config['rows']) if config['rows'] else 'None'}")
                    st.write(f"Columns: {', '.join(config['columns']) if config['columns'] else 'None'}")
                    st.write(f"Values: {', '.join(config['values'])}")
                    st.write(f"Aggregations: {', '.join(config['agg_functions'])}")
                    st.write(f"Normalized: {'Yes' if config['normalized'] else 'No'}")
                st.dataframe(pivot_data, use_container_width=True, height=400)
                a, b, c = st.columns(3)
                with a:
                    st.metric("Rows", pivot_data.shape[0])
                with b:
                    st.metric("Columns", pivot_data.shape[1])
                with c:
                    if pivot_data.shape[0] > 0 and pivot_data.shape[1] > 0:
                        st.metric("Total Cells", pivot_data.shape[0] * pivot_data.shape[1])
                st.subheader("📊 Visualize Pivot Table")
                viz_type = st.selectbox("Select chart type:", ["Bar Chart", "Line Chart", "Heatmap", "Stacked Bar", "Area Chart"])
                if st.button("Generate Chart"):
                    try:
                        plot_data = pivot_data.reset_index()
                        fig = None
                        if viz_type == "Bar Chart":
                            fig = px.bar(plot_data, x=plot_data.columns[0], y=plot_data.columns[1:],
                                         title=f"{selected_table} - Bar Chart", barmode='group')
                        elif viz_type == "Line Chart":
                            fig = px.line(plot_data, x=plot_data.columns[0], y=plot_data.columns[1:],
                                          title=f"{selected_table} - Line Chart")
                        elif viz_type == "Heatmap":
                            fig_, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
                            plt.title(f"{selected_table} - Heatmap")
                            st.pyplot(fig_)
                            plt.close()
                        elif viz_type == "Stacked Bar":
                            fig = px.bar(plot_data, x=plot_data.columns[0], y=plot_data.columns[1:],
                                         title=f"{selected_table} - Stacked Bar", barmode='stack')
                        elif viz_type == "Area Chart":
                            fig = px.area(plot_data, x=plot_data.columns[0], y=plot_data.columns[1:],
                                          title=f"{selected_table} - Area Chart")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
                a, b, c = st.columns(3)
                with a:
                    csv_buffer = io.StringIO()
                    pivot_data.to_csv(csv_buffer)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"{selected_table}.csv",
                        mime="text/csv"
                    )
                with b:
                    if st.button("🗑️ Delete This Table", type="secondary"):
                        st.session_state.pivot_tables = [
                            pt for pt in st.session_state.pivot_tables if pt['name'] != selected_table
                        ]
                        st.success(f"Deleted '{selected_table}'")
                        st.rerun()
                with c:
                    if st.button("🗑️ Clear All Tables"):
                        st.session_state.pivot_tables = []
                        st.success("All pivot tables cleared")
                        st.rerun()
    else:
        st.info("📊 No pivot tables created yet. Configure and create your first pivot table above!")
    
    # "Go to Next Step" button
    st.markdown("---")
    if st.button("Go to Next Step", type="primary", use_container_width=True):
        st.session_state.current_step = min(12, st.session_state.current_step + 1)
        st.rerun()