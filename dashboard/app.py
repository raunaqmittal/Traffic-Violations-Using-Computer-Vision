"""
Traffic Violation Analytics Dashboard — Streamlit app.

Launch: python app.py --dashboard
   OR:  streamlit run dashboard/app.py
"""

import sys
import os
from pathlib import Path
import streamlit as st
import pandas as pd

# Ensure project root is on path when launched standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.schema import init_db
from src.analytics.stats import violation_summary, search
from src.database.repository import export_csv

st.set_page_config(
    page_title="Traffic Violation System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

_DB_PATH = str(Path(__file__).parent.parent / "artifacts" / "violations.db")


@st.cache_resource
def get_engine():
    return init_db(_DB_PATH)


@st.cache_data(ttl=30)
def fetch_summary(_engine):
    return violation_summary(_engine)


@st.cache_data(ttl=30)
def fetch_rows(_engine, vtype, plate, date_from, date_to, status):
    return search(_engine, violation_type=vtype or None, plate_number=plate or None,
                  date_from=date_from, date_to=date_to, status=status or None)


def main():
    engine = get_engine()

    # ── Sidebar ────────────────────────────────────────────────────────────
    st.sidebar.title("🚦 Filters")
    violation_types = ["", "helmet", "seatbelt", "triple_riding", "wrong_side",
                       "stop_line", "red_light", "illegal_parking"]
    vtype = st.sidebar.selectbox("Violation Type", violation_types)
    plate = st.sidebar.text_input("Plate Number (partial ok)")
    date_from = st.sidebar.text_input("Date From (YYYY-MM-DD)")
    date_to = st.sidebar.text_input("Date To (YYYY-MM-DD)")
    statuses = ["", "auto_flagged", "review", "indeterminate"]
    status = st.sidebar.selectbox("Status", statuses)

    if st.sidebar.button("🔄 Refresh"):
        st.cache_data.clear()

    # ── Header ─────────────────────────────────────────────────────────────
    st.title("🚦 Traffic Violation Detection System")
    st.caption("Real-time violation analytics and evidence management")

    summary = fetch_summary(engine)
    total = summary["total"]
    by_type: dict = summary["by_type"]

    # ── Top KPI row ────────────────────────────────────────────────────────
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Violations", total)
    kpi_cols[1].metric("Helmet", by_type.get("helmet", 0))
    kpi_cols[2].metric("Red Light", by_type.get("red_light", 0))
    kpi_cols[3].metric("Wrong Side", by_type.get("wrong_side", 0))

    st.divider()

    # ── Charts ─────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Violations by Type")
        if by_type:
            type_df = pd.DataFrame(
                {"Violation": list(by_type.keys()), "Count": list(by_type.values())}
            ).sort_values("Count", ascending=False)
            st.bar_chart(type_df.set_index("Violation"))
        else:
            st.info("No data yet.")

    with col_right:
        st.subheader("Violations Over Time")
        by_date = summary.get("by_date", [])
        if by_date:
            date_df = pd.DataFrame(by_date).set_index("date")
            st.line_chart(date_df["count"])
        else:
            st.info("No data yet.")

    st.divider()

    # ── Data table ─────────────────────────────────────────────────────────
    st.subheader("Violation Records")
    rows = fetch_rows(engine, vtype, plate, date_from or None, date_to or None, status)

    if rows:
        df = pd.DataFrame(rows)
        display_cols = [
            "id", "timestamp", "violation_type", "status", "confidence",
            "plate_number", "camera_id", "vehicle_id", "is_blurry",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, height=400)

        # Evidence image viewer
        if "evidence_image_path" in df.columns:
            selected_id = st.selectbox(
                "View evidence image for record ID",
                options=df["id"].tolist(),
                index=0,
            )
            selected_row = df[df["id"] == selected_id].iloc[0]
            img_path = selected_row.get("evidence_image_path")
            if pd.notna(img_path) and isinstance(img_path, str) and Path(img_path).exists():
                st.image(img_path, caption=f"Evidence for ID {selected_id}", use_container_width=True)
            else:
                st.info("No evidence image saved for this record.")

        # CSV export
        csv_path = str(Path(_DB_PATH).parent / "export.csv")
        if st.button("⬇️ Export to CSV"):
            export_csv(engine, csv_path, limit=10000)
            with open(csv_path, "rb") as f:
                st.download_button(
                    label="Download CSV",
                    data=f,
                    file_name="violations_export.csv",
                    mime="text/csv",
                )
    else:
        st.info("No records match the current filters.")


if __name__ == "__main__":
    main()
