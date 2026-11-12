from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# AIRFLY INSIGHTS | Airline Operations Analytics
# Integrated Streamlit application based on the AirFly Colab
# analysis notebook. Keep this file beside the sample CSV.
# ============================================================

st.set_page_config(
    page_title="AirFly Insights | Airline Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "flight_data_2024_sample.csv"
CHART_THEME = "plotly_dark"

st.markdown("""
<style>
.block-container {padding-top: 1.25rem; padding-bottom: 2.2rem; max-width: 1600px;}
.hero {padding: 1.55rem 1.8rem; border-radius: 18px; color: white;
       background: linear-gradient(115deg,#10243e 0%,#164e63 55%,#0f766e 100%);
       margin-bottom: 1.1rem;}
.hero h1 {font-size: 2.25rem; font-weight: 800; margin: 0;}
.hero p {color: #d8edf3; margin: .45rem 0 0 0; font-size: 1rem;}
div[data-testid="stMetric"] {background: var(--secondary-background-color);
 border: 1px solid rgba(128,128,128,.20); padding: 13px 15px; border-radius: 13px;}
.insight {background: var(--secondary-background-color); border-left: 4px solid #0d9488;
 padding: 12px 15px; border-radius: 8px; margin: 7px 0;}
.small-note {color: #8b98a9; font-size: .86rem;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def first_existing(frame, candidates):
    return next((c for c in candidates if c in frame.columns), None)


@st.cache_data(show_spinner="Loading and preparing flight records...")
def load_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    required = ["fl_date", "op_unique_carrier", "origin", "dest"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))

    data["fl_date"] = pd.to_datetime(data["fl_date"], errors="coerce")
    numeric_cols = [
        "year", "month", "day_of_month", "day_of_week", "cancelled", "diverted",
        "crs_dep_time", "dep_time", "dep_delay", "crs_arr_time", "arr_time",
        "arr_delay", "air_time", "actual_elapsed_time", "distance",
        "carrier_delay", "weather_delay", "nas_delay", "security_delay",
        "late_aircraft_delay",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    for flag in ["cancelled", "diverted"]:
        if flag not in data.columns:
            data[flag] = 0
        data[flag] = data[flag].fillna(0).astype("int8")

    if "cancellation_code" not in data.columns:
        data["cancellation_code"] = np.nan
    data.loc[data["cancelled"].eq(0), "cancellation_code"] = "Not Cancelled"

    data = data.drop_duplicates().copy()
    data["flight_status"] = np.select(
        [data["cancelled"].eq(1), data["diverted"].eq(1)],
        ["Cancelled", "Diverted"], default="Completed"
    )
    data["year"] = data["fl_date"].dt.year
    data["month"] = data["fl_date"].dt.month
    data["month_name"] = data["fl_date"].dt.strftime("%B")
    data["weekday_name"] = data["fl_date"].dt.day_name()
    data["route"] = data["origin"].astype(str) + " → " + data["dest"].astype(str)

    if "crs_dep_time" in data.columns:
        dep = data["crs_dep_time"]
        data["departure_hour"] = (dep // 100).where(dep.between(0, 2359))
        data.loc[~data["departure_hour"].between(0, 23), "departure_hour"] = np.nan
    else:
        data["departure_hour"] = np.nan

    # Matches the notebook's delay-category definitions.
    arr = data["arr_delay"] if "arr_delay" in data.columns else pd.Series(np.nan, index=data.index)
    data["delay_category"] = np.select(
        [data["cancelled"].eq(1), data["diverted"].eq(1), arr.le(15),
         arr.between(16, 60), arr.between(61, 180), arr.gt(180)],
        ["Cancelled", "Diverted", "On Time", "Minor Delay", "Major Delay", "Severe Delay"],
        default="Unknown",
    )
    if "distance" in data.columns:
        data["distance_category"] = pd.cut(
            data["distance"], bins=[0, 500, 1500, np.inf],
            labels=["Short Haul", "Medium Haul", "Long Haul"]
        )
    return data


def make_excel_report(filtered: pd.DataFrame, kpis: pd.DataFrame,
                      airline: pd.DataFrame, causes: pd.DataFrame,
                      weekday: pd.DataFrame, hourly: pd.DataFrame,
                      severity: pd.DataFrame, cancellations: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        kpis.to_excel(writer, sheet_name="KPIs", index=False)
        airline.to_excel(writer, sheet_name="Airline Performance", index=False)
        causes.to_excel(writer, sheet_name="Delay Causes", index=False)
        weekday.to_excel(writer, sheet_name="Weekday Analysis", index=False)
        hourly.to_excel(writer, sheet_name="Hourly Analysis", index=False)
        severity.to_excel(writer, sheet_name="Delay Severity", index=False)
        cancellations.to_excel(writer, sheet_name="Cancellation Reasons", index=False)
        filtered.to_excel(writer, sheet_name="Filtered Data", index=False)
    return output.getvalue()


# --------------------------- Load ----------------------------
if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE.name}. Upload the CSV beside AirFly_app.py in the GitHub repository.")
    st.stop()
try:
    df = load_data(str(DATA_FILE))
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

# --------------------------- Header --------------------------
st.markdown("""
<div class="hero">
  <h1>✈️ AirFly Insights</h1>
  <p>Airline Operations Analytics · Flight performance, delays, routes and cancellations</p>
</div>
""", unsafe_allow_html=True)
st.caption("Interactive analysis of the supplied flight sample. Use the sidebar to filter all sections consistently.")

# --------------------------- Filters -------------------------
st.sidebar.title("Dashboard Controls")
st.sidebar.caption("Choose the records included in the analysis.")
airlines = sorted(df["op_unique_carrier"].dropna().astype(str).unique().tolist())
months_present = sorted(df["month"].dropna().astype(int).unique().tolist())
statuses = ["Completed", "Cancelled", "Diverted"]
sel_airlines = st.sidebar.multiselect("Airline", airlines, default=airlines)
month_names = {m: pd.Timestamp(year=2000, month=m, day=1).strftime("%B") for m in months_present}
sel_month_names = st.sidebar.multiselect("Month", [month_names[m] for m in months_present], default=[month_names[m] for m in months_present])
month_lookup = {v: k for k, v in month_names.items()}
sel_months = [month_lookup[m] for m in sel_month_names]
sel_status = st.sidebar.multiselect("Flight status", statuses, default=statuses)

filtered = df[
    df["op_unique_carrier"].astype(str).isin(sel_airlines)
    & df["month"].isin(sel_months)
    & df["flight_status"].isin(sel_status)
].copy()
if filtered.empty:
    st.warning("No flights match these filters. Select at least one airline, month and status.")
    st.stop()

# --------------------------- KPIs ----------------------------
total = len(filtered)
completed_mask = filtered["flight_status"].eq("Completed")
completed_count = int(completed_mask.sum())
cancelled_count = int(filtered["flight_status"].eq("Cancelled").sum())
diverted_count = int(filtered["flight_status"].eq("Diverted").sum())
if "arr_delay" in filtered.columns:
    valid_arrivals = filtered.loc[completed_mask & filtered["arr_delay"].notna()]
    on_time_pct = valid_arrivals["arr_delay"].le(15).mean() * 100 if len(valid_arrivals) else np.nan
    avg_arr_delay = valid_arrivals["arr_delay"].mean() if len(valid_arrivals) else np.nan
    delayed_count = int(valid_arrivals["arr_delay"].gt(15).sum())
else:
    valid_arrivals = pd.DataFrame()
    on_time_pct = avg_arr_delay = np.nan
    delayed_count = 0
cancel_pct = cancelled_count / total * 100 if total else 0

a, b, c, d, e, f = st.columns(6)
a.metric("Total Flights", f"{total:,}")
b.metric("Completed", f"{completed_count:,}")
c.metric("On-time Arrival", "—" if pd.isna(on_time_pct) else f"{on_time_pct:.1f}%")
d.metric("Avg Arrival Delay", "—" if pd.isna(avg_arr_delay) else f"{avg_arr_delay:.1f} min")
e.metric("Cancellation Rate", f"{cancel_pct:.2f}%")
f.metric("Diverted Flights", f"{diverted_count:,}")
st.caption("On-time rate uses completed flights with recorded arrival delay; on-time means arrival delay ≤ 15 minutes. Cancelled/diverted flights are excluded from this KPI.")

# ---------------------- Shared analysis tables --------------
month_order = list(range(1, 13))
month_labels = {m: pd.Timestamp(year=2000, month=m, day=1).strftime("%b") for m in month_order}
monthly = (filtered.dropna(subset=["month"]).groupby("month").size().reindex(month_order, fill_value=0)
           .rename("Total Flights").rename_axis("Month Number").reset_index())
monthly = monthly[monthly["Month Number"].isin(sel_months)]
monthly["Month"] = monthly["Month Number"].map(month_labels)

if "op_unique_carrier" in filtered:
    agg = {"Total Flights": ("op_unique_carrier", "size"),
           "Cancelled Flights": ("cancelled", "sum"),
           "Diverted Flights": ("diverted", "sum")}
    if "dep_delay" in filtered: agg["Avg Departure Delay (min)"] = ("dep_delay", "mean")
    if "arr_delay" in filtered: agg["Avg Arrival Delay (min)"] = ("arr_delay", "mean")
    airline_summary = filtered.groupby("op_unique_carrier").agg(**agg).reset_index().rename(columns={"op_unique_carrier": "Airline"})
    airline_summary["Cancellation Rate (%)"] = np.where(
        airline_summary["Total Flights"].gt(0),
        airline_summary["Cancelled Flights"] / airline_summary["Total Flights"] * 100, 0)
    airline_summary = airline_summary.sort_values("Avg Arrival Delay (min)" if "Avg Arrival Delay (min)" in airline_summary else "Total Flights", ascending=False)
else:
    airline_summary = pd.DataFrame()

# Delay causes: use the same five fields and sum logic as the notebook.
delay_cols = {"Carrier": "carrier_delay", "Weather": "weather_delay", "NAS": "nas_delay",
              "Security": "security_delay", "Late Aircraft": "late_aircraft_delay"}
available_delay_cols = {label: col for label, col in delay_cols.items() if col in filtered.columns}
if available_delay_cols:
    cause_table = pd.DataFrame({"Delay Cause": list(available_delay_cols),
        "Total Delay Minutes": [pd.to_numeric(filtered[col], errors="coerce").sum() for col in available_delay_cols.values()]})
    cause_table = cause_table.sort_values("Total Delay Minutes", ascending=False)
else:
    cause_table = pd.DataFrame(columns=["Delay Cause", "Total Delay Minutes"])

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_table = (filtered[filtered["flight_status"].eq("Completed")]
    .groupby("weekday_name").agg(Total_Flights=("weekday_name", "size"),
        Average_Departure_Delay=("dep_delay", "mean") if "dep_delay" in filtered else ("weekday_name", "size"),
        Average_Arrival_Delay=("arr_delay", "mean") if "arr_delay" in filtered else ("weekday_name", "size"))
    .reindex(weekday_order).rename_axis("Weekday").reset_index())

hourly_table = (filtered[filtered["flight_status"].eq("Completed")]
    .dropna(subset=["departure_hour"])
    .groupby("departure_hour").agg(Total_Flights=("departure_hour", "size"),
        Average_Departure_Delay=("dep_delay", "mean") if "dep_delay" in filtered else ("departure_hour", "size"),
        Average_Arrival_Delay=("arr_delay", "mean") if "arr_delay" in filtered else ("departure_hour", "size"))
    .reset_index())

severity_order = ["On Time", "Minor Delay", "Major Delay", "Severe Delay", "Unknown"]
severity_table = (filtered.loc[filtered["flight_status"].eq("Completed"), "delay_category"]
    .value_counts().reindex(severity_order, fill_value=0).rename_axis("Delay Category").reset_index(name="Flights"))

cancelled_rows = filtered[filtered["cancelled"].eq(1)].copy()
cancellation_map = {"A": "Carrier", "B": "Weather", "C": "National Airspace System", "D": "Security"}
if not cancelled_rows.empty:
    cancelled_rows["Cancellation Reason"] = cancelled_rows.get("cancellation_code", pd.Series(index=cancelled_rows.index, dtype=object)).map(cancellation_map).fillna("Unknown")
    cancellation_table = cancelled_rows["Cancellation Reason"].value_counts().rename_axis("Cancellation Reason").reset_index(name="Flights")
else:
    cancellation_table = pd.DataFrame(columns=["Cancellation Reason", "Flights"])

# Monthly cancellation rate calculated on all selected records in each month.
cancellation_monthly = (filtered.dropna(subset=["month"]).groupby("month")
    .agg(Total_Flights=("cancelled", "size"), Cancelled_Flights=("cancelled", "sum")).reset_index())
cancellation_monthly["Cancellation Rate (%)"] = np.where(
    cancellation_monthly["Total_Flights"].gt(0),
    cancellation_monthly["Cancelled_Flights"] / cancellation_monthly["Total_Flights"] * 100, 0)
cancellation_monthly["Month"] = cancellation_monthly["month"].map(month_labels)
cancellation_monthly = cancellation_monthly.sort_values("month")

# -------------------------- Tabs -----------------------------
tabs = st.tabs(["Overview", "Airline Performance", "Airports & Routes", "Delay Analytics", "Cancellations", "Reports"])

with tabs[0]:
    st.subheader("Executive Overview")
    left, right = st.columns(2)
    with left:
        fig = px.line(monthly, x="Month", y="Total Flights", markers=True, text="Total Flights",
                      title="Monthly Flight Traffic", template=CHART_THEME)
        fig.update_traces(line=dict(width=3), marker=dict(size=8), textposition="top center")
        fig.update_layout(height=430, xaxis_title="Month", yaxis_title="Number of flights")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        status_counts = filtered["flight_status"].value_counts().rename_axis("Flight Status").reset_index(name="Flights")
        fig = px.pie(status_counts, names="Flight Status", values="Flights", hole=.55,
                     title="Flight Status Distribution", template=CHART_THEME)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Automated Business Insights")
    insights = []
    airline_counts = filtered["op_unique_carrier"].value_counts()
    if not airline_counts.empty:
        insights.append(f"**Highest flight volume:** {airline_counts.index[0]} with {airline_counts.iloc[0]:,} flights in the selected records.")
    month_counts = filtered["month_name"].value_counts()
    if not month_counts.empty:
        insights.append(f"**Busiest month:** {month_counts.index[0]} in the selected records.")
    route_counts = filtered["route"].value_counts()
    if not route_counts.empty:
        insights.append(f"**Busiest route:** {route_counts.index[0]} with {route_counts.iloc[0]:,} flights.")
    if not cause_table.empty:
        insights.append(f"**Largest recorded delay contributor:** {cause_table.iloc[0]['Delay Cause']} ({cause_table.iloc[0]['Total Delay Minutes']:,.0f} minutes).")
    insights.append(f"**Cancellation rate:** {cancel_pct:.2f}% of the selected flight records.")
    if not pd.isna(avg_arr_delay):
        insights.append(f"**Average arrival delay:** {avg_arr_delay:.2f} minutes among completed flights with recorded arrival delay.")
    for insight in insights:
        st.markdown(f'<div class="insight">{insight}</div>', unsafe_allow_html=True)

with tabs[1]:
    st.subheader("Airline Performance")
    left, right = st.columns(2)
    with left:
        airline_volume = filtered.groupby("op_unique_carrier").size().rename("Total Flights").reset_index().sort_values("Total Flights")
        fig = px.bar(airline_volume, x="Total Flights", y="op_unique_carrier", orientation="h", text="Total Flights",
                     title="Flight Volume by Airline", template=CHART_THEME, color="Total Flights", color_continuous_scale="Teal")
        fig.update_layout(height=500, coloraxis_showscale=False, yaxis_title="Airline")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if "dep_delay" in filtered.columns and "arr_delay" in filtered.columns:
            delay_comparison = filtered.groupby("op_unique_carrier").agg(
                Departure_Delay=("dep_delay", "mean"), Arrival_Delay=("arr_delay", "mean")).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=delay_comparison["op_unique_carrier"], y=delay_comparison["Departure_Delay"], name="Departure Delay"))
            fig.add_trace(go.Bar(x=delay_comparison["op_unique_carrier"], y=delay_comparison["Arrival_Delay"], name="Arrival Delay"))
            fig.update_layout(title="Average Departure vs Arrival Delay by Airline", barmode="group", template=CHART_THEME,
                              height=500, xaxis_title="Airline", yaxis_title="Average Delay (minutes)")
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Airline Performance Summary")
    st.dataframe(airline_summary.round(2), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Airport & Route Intelligence")
    left, right = st.columns(2)
    with left:
        origin_counts = filtered["origin"].value_counts().rename("Departures")
        dest_counts = filtered["dest"].value_counts().rename("Arrivals")
        airport_counts = pd.concat([origin_counts, dest_counts], axis=1).fillna(0)
        airport_counts["Total Movements"] = airport_counts["Departures"] + airport_counts["Arrivals"]
        airport_top = airport_counts.nlargest(15, "Total Movements").reset_index(names="Airport").sort_values("Total Movements")
        fig = px.bar(airport_top, x="Total Movements", y="Airport", orientation="h", color="Total Movements",
                     title="Top 15 Busiest Airports", template=CHART_THEME, color_continuous_scale="Viridis",
                     hover_data=["Departures", "Arrivals"])
        fig.update_layout(height=650, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        route_top = (filtered.groupby("route").agg(Total_Flights=("route", "size"),
            Average_Arrival_Delay=("arr_delay", "mean") if "arr_delay" in filtered else ("route", "size"))
            .reset_index().nlargest(15, "Total_Flights").sort_values("Total_Flights"))
        fig = px.bar(route_top, x="Total_Flights", y="route", orientation="h", color="Average_Arrival_Delay",
                     title="Top 15 Most Popular Flight Routes", template=CHART_THEME, color_continuous_scale="Turbo",
                     hover_data=["Average_Arrival_Delay"])
        fig.update_layout(height=650, xaxis_title="Number of flights", yaxis_title="Flight route")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Airport movements count each selected flight once at its origin and once at its destination.")

with tabs[3]:
    st.subheader("Delay Analytics")
    left, right = st.columns(2)
    with left:
        if not cause_table.empty:
            fig = px.bar(cause_table.sort_values("Total Delay Minutes"), x="Total Delay Minutes", y="Delay Cause",
                         orientation="h", text_auto=".0f", title="Total Delay Minutes by Cause",
                         template=CHART_THEME, color="Total Delay Minutes", color_continuous_scale="Teal")
            fig.update_layout(height=450, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Delay-cause fields are unavailable in this dataset.")
    with right:
        fig = px.pie(severity_table, names="Delay Category", values="Flights", hole=.55,
                     title="Arrival Delay Severity — Completed Flights", template=CHART_THEME)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Average Arrival Delay by Weekday")
    weekday_plot = weekday_table.dropna(subset=["Average_Arrival_Delay"])
    if not weekday_plot.empty:
        fig = px.bar(weekday_plot, x="Weekday", y="Average_Arrival_Delay", color="Average_Arrival_Delay",
                     title="Average Arrival Delay by Weekday", template=CHART_THEME,
                     color_continuous_scale="Turbo", hover_data=["Total_Flights"])
        fig.update_layout(height=430, yaxis_title="Average arrival delay (minutes)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No recorded arrival delays are available for weekday analysis.")

    st.markdown("#### Average Departure Delay by Scheduled Departure Hour")
    if not hourly_table.empty:
        fig = px.line(hourly_table, x="departure_hour", y="Average_Departure_Delay", markers=True,
                      title="Average Departure Delay by Hour", template=CHART_THEME,
                      hover_data=["Total_Flights"])
        fig.update_layout(height=430, xaxis=dict(dtick=1), xaxis_title="Scheduled departure hour",
                          yaxis_title="Average departure delay (minutes)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid scheduled departure-hour values are available.")
    with st.expander("View detailed weekday and hourly tables"):
        st.markdown("**Weekday analysis**")
        st.dataframe(weekday_table.round(2), use_container_width=True, hide_index=True)
        st.markdown("**Hourly analysis**")
        st.dataframe(hourly_table.round(2), use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Cancellation Analytics")
    left, right = st.columns(2)
    with left:
        if not cancellation_monthly.empty:
            fig = px.bar(cancellation_monthly, x="Month", y="Cancellation Rate (%)", color="Cancellation Rate (%)",
                         text_auto=".2f", title="Monthly Flight Cancellation Rate", template=CHART_THEME,
                         color_continuous_scale="Reds")
            fig.update_layout(height=450, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if not cancellation_table.empty:
            fig = px.bar(cancellation_table, x="Cancellation Reason", y="Flights", color="Flights", text_auto=True,
                         title="Flight Cancellations by Reason", template=CHART_THEME, color_continuous_scale="Reds")
            fig.update_layout(height=450, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cancelled flights are present in the current filter selection.")
    st.markdown("#### Cancellation Reason Summary")
    if not cancellation_table.empty:
        st.dataframe(cancellation_table, use_container_width=True, hide_index=True)
    else:
        st.write("No cancellation records to summarize.")
    st.caption("Cancellation codes: A = Carrier, B = Weather, C = National Airspace System, D = Security. Unmapped or missing codes are shown as Unknown.")

with tabs[5]:
    st.subheader("Reports & Downloads")
    st.write("Export the currently filtered records and the associated analysis tables.")
    kpi_report = pd.DataFrame({"Metric": ["Total Flights", "Completed Flights", "Cancelled Flights", "Diverted Flights",
        "On-Time Arrival Percentage", "Average Arrival Delay (minutes)", "Delayed Flights (>15 min)", "Cancellation Percentage"],
        "Value": [total, completed_count, cancelled_count, diverted_count,
                  None if pd.isna(on_time_pct) else round(on_time_pct, 2),
                  None if pd.isna(avg_arr_delay) else round(avg_arr_delay, 2), delayed_count, round(cancel_pct, 2)]})
    st.markdown("**KPI summary**")
    st.dataframe(kpi_report, use_container_width=True, hide_index=True)
    excel_bytes = make_excel_report(filtered, kpi_report, airline_summary, cause_table,
                                    weekday_table, hourly_table, severity_table, cancellation_table)
    st.download_button("⬇️ Download Complete Excel Analysis Report", data=excel_bytes,
                       file_name="AirFly_Insights_Analysis_Report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
    st.download_button("⬇️ Download Filtered Flight Data (CSV)",
                       data=filtered.to_csv(index=False).encode("utf-8"),
                       file_name="airfly_filtered_flights.csv", mime="text/csv", use_container_width=True)
    st.download_button("⬇️ Download Airline Performance Summary (CSV)",
                       data=airline_summary.to_csv(index=False).encode("utf-8"),
                       file_name="airfly_airline_performance.csv", mime="text/csv", use_container_width=True)
    st.caption("The Excel workbook contains KPIs, airline performance, delay causes, weekday analysis, hourly analysis, delay severity, cancellation reasons and filtered data.")

st.divider()
st.caption("AirFly Insights · Airline Operations Analytics · Python · Pandas · Plotly · Streamlit")
