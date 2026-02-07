# steps/step12_export.py

import streamlit as st
import pandas as pd
import json
import io
import zipfile
import gc
from datetime import datetime

from ..utils.history_utils import log_action
from ..utils.memory_utils import optimize_dataframe_memory, clear_model_cache
from ..components.overview_metrics import show_overview_metrics


# ---------------------------------------------------------------------
# Helper: Generate cleaning report
# ---------------------------------------------------------------------
def generate_cleaning_report(original_df, cleaned_df, action_log, quality_score):
    return {
        "timestamp": datetime.now().isoformat(),
        "original_shape": original_df.shape if original_df is not None else None,
        "cleaned_shape": cleaned_df.shape,
        "rows_removed": (
            original_df.shape[0] - cleaned_df.shape[0]
            if original_df is not None else None
        ),
        "columns_removed": (
            list(set(original_df.columns) - set(cleaned_df.columns))
            if original_df is not None else []
        ),
        "columns_added": (
            list(set(cleaned_df.columns) - set(original_df.columns))
            if original_df is not None else []
        ),
        "missing_values_removed": (
            int(original_df.isnull().sum().sum() - cleaned_df.isnull().sum().sum())
            if original_df is not None else None
        ),
        "duplicates_removed": (
            int(original_df.duplicated().sum() - cleaned_df.duplicated().sum())
            if original_df is not None else None
        ),
        "data_quality_score": quality_score,
        "total_actions": len(action_log),
        "recent_actions": action_log[-50:]
    }


# ---------------------------------------------------------------------
# Step 12: Final & Export
# ---------------------------------------------------------------------
def step12_export():
    st.header("Step 12 · Final & Export")
    st.markdown("**Step 12 of 12**")

    # -----------------------------
    # Validation
    # -----------------------------
    if st.session_state.df is None:
        st.warning("No dataset found. Please complete previous steps.")
        return

    df = st.session_state.df
    original_df = st.session_state.original_df
    action_log = st.session_state.action_log
    quality_score = st.session_state.data_quality_score

    # -----------------------------
    # Before vs After
    # -----------------------------
    if original_df is not None:
        st.subheader("📊 Before vs After")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Original Rows", original_df.shape[0])
            st.metric("Original Columns", original_df.shape[1])
            st.metric(
                "Original Missing",
                int(original_df.isnull().sum().sum())
            )

        with c2:
            st.metric("Final Rows", df.shape[0])
            st.metric("Final Columns", df.shape[1])
            st.metric(
                "Final Missing",
                int(df.isnull().sum().sum())
            )

    show_overview_metrics(df)
    st.divider()

    # -----------------------------
    # Export Options
    # -----------------------------
    st.subheader("📤 Export Options")

    export_formats = st.multiselect(
        "Select formats",
        ["CSV", "Excel", "Parquet", "JSON Report"],
        default=["CSV"]
    )

    add_timestamp = st.checkbox("Add timestamp to filenames", value=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""

    # -----------------------------
    # Individual Exports
    # -----------------------------
    if "CSV" in export_formats:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Download CSV",
            csv_buf.getvalue(),
            file_name=f"cleaned_data_{timestamp}.csv",
            mime="text/csv"
        )

    if "Excel" in export_formats:
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Cleaned_Data")
        excel_buf.seek(0)

        st.download_button(
            "📥 Download Excel",
            excel_buf,
            file_name=f"cleaned_data_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if "Parquet" in export_formats:
        pq_buf = io.BytesIO()
        df.to_parquet(pq_buf, index=False)
        pq_buf.seek(0)

        st.download_button(
            "📥 Download Parquet",
            pq_buf,
            file_name=f"cleaned_data_{timestamp}.parquet",
            mime="application/octet-stream"
        )

    if "JSON Report" in export_formats and original_df is not None:
        report = generate_cleaning_report(
            original_df, df, action_log, quality_score
        )
        st.download_button(
            "📥 Download Cleaning Report",
            json.dumps(report, indent=2),
            file_name=f"cleaning_report_{timestamp}.json",
            mime="application/json"
        )

    st.divider()

    # -----------------------------
    # ZIP Export
    # -----------------------------
    if st.button("📦 Export All as ZIP", type="primary"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("cleaned_data.csv", df.to_csv(index=False))

            if original_df is not None:
                report = generate_cleaning_report(
                    original_df, df, action_log, quality_score
                )
                zf.writestr(
                    "cleaning_report.json",
                    json.dumps(report, indent=2)
                )

        zip_buffer.seek(0)

        st.download_button(
            "📥 Download ZIP",
            zip_buffer,
            file_name=f"export_bundle_{timestamp}.zip",
            mime="application/zip"
        )

    st.divider()

    # -----------------------------
    # Maintenance
    # -----------------------------
    st.subheader("🧠 Maintenance")

    if st.button("⚡ Optimize Memory"):
        df_opt, stats = optimize_dataframe_memory(df)
        st.session_state.df = df_opt
        log_action(
            f"Memory optimized: saved {stats['savings_mb']} MB",
            snapshot=True
        )
        st.success("Memory optimized")
        st.rerun()

    if st.button("🗑 Clear Cache"):
        clear_model_cache()
        if hasattr(st, "cache_data"):
            st.cache_data.clear()
        gc.collect()
        st.success("Cache cleared")

    if st.button("🏁 Complete Pipeline", type="primary"):
        st.success("Pipeline completed successfully 🎉")
        st.balloons()
