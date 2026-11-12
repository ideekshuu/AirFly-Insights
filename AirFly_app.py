from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# AIRFLY INSIGHTS | Airline Operations Analytics
# Keep this file in the same Colab folder as the CSV dataset.
# ------------------------------------------------------------

st.set_page_config(
    page_title="AirFly Insights",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path(__file__).resolve().parent / "flight_data_2024_sample.csv"

# ---------- Styling ----------
st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
    .hero {
        padding: 1.35rem 1.5rem; border-radius: 18px;
        background: linear-gradient(115deg, #10243e 0%, #164e63 55%, #0f766e 100%);
        color: white; margin-bottom: 1rem;
    }
    .hero h1 {margin: 0; font-size: 2.1rem; font-weight: 750;}
    .hero p {margin: .45rem 0 0 0; color: #d8edf3; font-size: 1rem;}
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,.18);
        padding: 14px 16px; border-radius: 14px;
    }
    .section-note {color: #778397; font-size: .88rem; margin-top: -8px;}
    .insight {
        background: var(--secondary-background-color);
        border-left: 4px solid #0d9488; padding: 12px 15px;
        border-radius: 9px; margin: 7px 0;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- Data loading ----------
@st.cache_data(show_spinner="Loading flight dataset...")
def load_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)

    required = ["fl_date", "op_unique_carrier", "origin", "dest"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    data["fl_date"] = pd.to_datetime(data["fl_date"], errors="coerce")

    numeric_cols = [
        "cancelled", "diverted", "dep_delay", "arr_delay",
        "crs_dep_time", "distance", "air_time", "carrier_delay",
        "weather_delay", "nas_delay", "security_delay",
        "late_aircraft_delay"
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "cancelled" not in data.columns:
        data["cancelled"] = 0
    if "diverted" not in data.columns:
        data["diverted"] = 0

    data["cancelled"] = data["cancelled"].fillna(0).astype(int)
    data["diverted"] = data["diverted"].fillna(0).astype(int)

    data["flight_status"] = np.select(
        [data["cancelled"].eq(1), data["diverted"].eq(1)],
        ["Cancelled", "Diverted"],
        default="Completed"
    )

    data["year"] = data["fl_date"].dt.year
    data["month"] = data["fl_date"].dt.month
    data["month_name"] = data["fl_date"].dt.strftime("%b")
    data["weekday_name"] = data["fl_date"].dt.day_name()
    data["route"] = data["origin"].astype(str) + " → " + data["dest"].astype(str)

    if "crs_dep_time" in data.columns:
        data["departure_hour"] = (data["crs_dep_time"] // 100).where(
            data["crs_dep_time"].between(0, 2359)
        )
        data.loc[~data["departure_hour"].between(0, 23), "departure_hour"] = np.nan
    else:
        data["departure_hour"] = np.nan

    if "arr_delay" in data.columns:
        data["delay_category"] = np.select(
            [
                data["flight_status"].eq("Cancelled"),
                data["flight_status"].eq("Diverted"),
                data["arr_delay"].isna(),
                data["arr_delay"].le(15),
                data["arr_delay"].le(60),
                data["arr_delay"].le(180),
            ],
            ["Cancelled", "Diverted", "Unknown", "On Time (≤15 min)",
             "Minor (16–60 min)", "Major (61–180 min)"],
            default="Severe (>180 min)"
        )
    else:
        data["delay_category"] = data["flight_status"]

    return data.drop_duplicates().copy()


# ---------- Load file with a helpful error ----------
try:
    if not DATA_FILE.exists():
        st.error(
            "Dataset not found. Upload `flight_data_2024_sample.csv` to the "
            "same Colab folder as `AirFly_app.py`, then rerun the app."
        )
        st.stop()
    df = load_data(str(DATA_FILE))
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>✈️ AirFly Insights</h1>
  <p>Airline Operations Analytics · Flight performance, delays, routes and cancellations</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar filters ----------
st.sidebar.title("Dashboard Controls")
st.sidebar.caption("Filter the flight records to explore performance.")

airline_values = sorted(df["op_unique_carrier"].dropna().astype(str).unique())
month_order = list(range(1, 13))
available_months = [m for m in month_order if m in df["month"].dropna().unique()]
status_values = ["Completed", "Cancelled", "Diverted"]

selected_airlines = st.sidebar.multiselect(
    "Airline", airline_values, default=airline_values
)
month_labels = {m: pd.Timestamp(year=2024, month=m, day=1).strftime("%B")
                for m in available_months}
selected_month_labels = st.sidebar.multiselect(
    "Month", [month_labels[m] for m in available_months],
    default=[month_labels[m] for m in available_months]
)
selected_status = st.sidebar.multiselect(
    "Flight status", status_values, default=status_values
)

month_lookup = {v: k for k, v in month_labels.items()}
selected_months = [month_lookup[x] for x in selected_month_labels]

filtered = df[
    df["op_unique_carrier"].astype(str).isin(selected_airlines)
    & df["month"].isin(selected_months)
    & df["flight_status"].isin(selected_status)
].copy()

if filtered.empty:
    st.warning("No flights match the selected filters. Choose at least one airline, month and status.")
    st.stop()

# ---------- KPI calculations ----------
total = len(filtered)
completed = filtered["flight_status"].eq("Completed").sum()
cancelled = filtered["flight_status"].eq("Cancelled").sum()
diverted = filtered["flight_status"].eq("Diverted")

valid_arrivals = filtered.loc[
    filtered["flight_status"].eq("Completed") & filtered["arr_delay"].notna()
] if "arr_delay" in filtered.columns else pd.DataFrame()

if not valid_arrivals.empty:
    on_time_pct = (valid_arrivals["arr_delay"].le(15).mean()) * 100
    avg_arr_delay = valid_arrivals["arr_delay"].mean()
    delayed_count = valid_arrivals["arr_delay"].gt(15).sum()
else:
    on_time_pct = np.nan
    avg_arr_delay = np.nan
    delayed_count = 0

cancel_pct = (cancelled / total * 100) if total else 0
divert_pct = (diverted.sum() / total * 100) if total else 0

st.subheader("Flight Operations Snapshot")
st.caption("On-time performance uses completed flights with a recorded arrival delay; on time is defined as arrival delay ≤ 15 minutes.")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Flights", f"{total:,}")
k2.metric("Completed", f"{completed:,}")
k3.metric("On-time Rate", "—" if pd.isna(on_time_pct) else f"{on_time_pct:.1f}%")
k4.metric("Avg Arrival Delay", "—" if pd.isna(avg_arr_delay) else f"{avg_arr_delay:.1f} min")
k5.metric("Cancellation Rate", f"{cancel_pct:.2f}%")
k6.metric("Diverted Flights", f"{int(diverted.sum()):,}")

# ---------- Chart theme ----------
template = "plotly_dark"

# ---------- Main charts ----------
st.divider()
st.subheader("Traffic & Airline Performance")
left, right = st.columns(2)

with left:
    monthly = (
        filtered.dropna(subset=["month"])
        .groupby(["month", "month_name"], as_index=False)
        .size().rename(columns={"size": "Flights"})
        .sort_values("month")
    )
    if not monthly.empty:
        fig = px.bar(monthly, x="month_name", y="Flights",
                     title="Monthly Flight Traffic", text_auto=True,
                     template=template, color="Flights",
                     color_continuous_scale="Teal")
        fig.update_layout(xaxis_title="Month", yaxis_title="Number of flights",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

with right:
    airline_volume = (
        filtered.groupby("op_unique_carrier", as_index=False)
        .size().rename(columns={"size": "Flights"})
        .sort_values("Flights", ascending=True)
    )
    fig = px.bar(airline_volume, x="Flights", y="op_unique_carrier",
                 orientation="h", title="Flight Volume by Airline",
                 text="Flights", template=template, color="Flights",
                 color_continuous_scale="Blues")
    fig.update_layout(xaxis_title="Number of flights", yaxis_title="Airline",
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Flight Status & Delay Analysis")
left, right = st.columns(2)

with left:
    status_counts = filtered["flight_status"].value_counts().rename_axis("Status").reset_index(name="Flights")
    fig = px.pie(status_counts, names="Status", values="Flights",
                 title="Flight Status Distribution", hole=0.55,
                 template=template)
    st.plotly_chart(fig, use_container_width=True)

with right:
    delay_counts = filtered["delay_category"].value_counts().rename_axis("Delay Category").reset_index(name="Flights")
    fig = px.bar(delay_counts.sort_values("Flights"), x="Flights",
                 y="Delay Category", orientation="h",
                 title="Arrival Delay Severity", text="Flights",
                 template=template, color="Flights",
                 color_continuous_scale="Sunset")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Delay causes ----------
delay_cols = [
    "carrier_delay", "weather_delay", "nas_delay",
    "security_delay", "late_aircraft_delay"
]
available_delay_cols = [c for c in delay_cols if c in filtered.columns]

st.divider()
st.subheader("Why Are Flights Delayed?")
if available_delay_cols:
    delay_totals = filtered[available_delay_cols].fillna(0).sum().rename_axis("Cause").reset_index(name="Delay Minutes")
    cause_names = {
        "carrier_delay": "Carrier",
        "weather_delay": "Weather",
        "nas_delay": "National Air System",
        "security_delay": "Security",
        "late_aircraft_delay": "Late Aircraft"
    }
    delay_totals["Cause"] = delay_totals["Cause"].map(cause_names).fillna(delay_totals["Cause"])
    delay_totals = delay_totals.sort_values("Delay Minutes", ascending=True)
    fig = px.bar(delay_totals, x="Delay Minutes", y="Cause",
                 orientation="h", text_auto=".0f",
                 title="Total Delay Minutes by Cause",
                 template=template, color="Delay Minutes",
                 color_continuous_scale="Teal")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Delay-cause columns are not available in this dataset.")

# ---------- Airports and routes ----------
st.divider()
st.subheader("Airport & Route Intelligence")
left, right = st.columns(2)

with left:
    airport_counts = pd.concat(
        [filtered["origin"].value_counts(), filtered["dest"].value_counts()],
        axis=1
    ).fillna(0)
    airport_counts.columns = ["Departures", "Arrivals"]
    airport_counts["Movements"] = airport_counts.sum(axis=1)
    airport_top = airport_counts.nlargest(12, "Movements").reset_index(names="Airport")
    fig = px.bar(airport_top.sort_values("Movements"), x="Movements",
                 y="Airport", orientation="h", text="Movements",
                 title="Top 12 Airports by Total Movements",
                 template=template, color="Movements",
                 color_continuous_scale="Blues")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    route_top = (
        filtered.groupby("route", as_index=False).size()
        .rename(columns={"size": "Flights"})
        .nlargest(12, "Flights").sort_values("Flights")
    )
    fig = px.bar(route_top, x="Flights", y="route", orientation="h",
                 text="Flights", title="Top 12 Busiest Routes",
                 template=template, color="Flights",
                 color_continuous_scale="Teal")
    fig.update_layout(xaxis_title="Number of flights", yaxis_title="Route",
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Airline summary table ----------
st.divider()
st.subheader("Airline Performance Table")
agg = {
    "Total Flights": ("op_unique_carrier", "size"),
    "Cancelled Flights": ("cancelled", "sum"),
    "Diverted Flights": ("diverted", "sum"),
}
if "dep_delay" in filtered.columns:
    agg["avg_dep_delay"] = ("dep_delay", "mean")
if "arr_delay" in filtered.columns:
    agg["avg_arr_delay"] = ("arr_delay", "mean")

airline_summary = filtered.groupby("op_unique_carrier").agg(**agg).reset_index()
airline_summary = airline_summary.rename(columns={
    "op_unique_carrier": "Airline",
    "avg_dep_delay": "Avg Departure Delay (min)",
    "avg_arr_delay": "Avg Arrival Delay (min)"
})
st.dataframe(airline_summary, use_container_width=True, hide_index=True)

# ---------- Automated insights ----------
st.divider()
st.subheader("Automated Business Insights")
insights = []

top_airline = filtered["op_unique_carrier"].value_counts()
if not top_airline.empty:
    insights.append(f"**Highest flight volume:** {top_airline.index[0]} with {top_airline.iloc[0]:,} flights in the selected data.")

top_route = filtered["route"].value_counts()
if not top_route.empty:
    insights.append(f"**Busiest route:** {top_route.index[0]} with {top_route.iloc[0]:,} flights.")

if available_delay_cols:
    leading_cause = filtered[available_delay_cols].fillna(0).sum().idxmax()
    insights.append(f"**Largest recorded delay contributor:** {cause_names.get(leading_cause, leading_cause)}.")

if not pd.isna(avg_arr_delay):
    insights.append(f"**Average arrival delay:** {avg_arr_delay:.1f} minutes across completed flights with recorded arrival delay.")

insights.append(f"**Cancellation rate:** {cancel_pct:.2f}% of the currently filtered flight records.")
for item in insights:
    st.markdown(f'<div class="insight">{item}</div>', unsafe_allow_html=True)

# ---------- Downloads ----------
st.divider()
st.subheader("Export Filtered Results")
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Filtered Flight Data (CSV)",
    data=csv_bytes,
    file_name="airfly_filtered_flights.csv",
    mime="text/csv",
)

summary_csv = airline_summary.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Airline Summary (CSV)",
    data=summary_csv,
    file_name="airfly_airline_summary.csv",
    mime="text/csv",
)

st.caption("AirFly Insights · Airline Operations Analytics · Built with Python, Pandas, Plotly and Streamlit")
