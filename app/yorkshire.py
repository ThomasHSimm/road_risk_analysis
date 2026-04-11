"""
Yorkshire Road Risk Explorer
-----------------------------
Streamlit app for exploring road risk scores across the Yorkshire network.

Run with:
    streamlit run app/yorkshire.py

Requires:
    pip install streamlit folium streamlit-folium
"""

import json
import sys
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Yorkshire Road Risk Explorer",
    page_icon="🛣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from road_risk.config import _ROOT
except ImportError:
    # Fallback — try common locations
    for candidate in [
        Path(__file__).parent.parent,
        Path.home() / "Documents/GitHub/road-risk-analysis",
        Path.home() / "road-risk-analysis",
    ]:
        if (candidate / "data/models/risk_scores.parquet").exists():
            _ROOT = candidate
            break
    else:
        _ROOT = None

RISK_PATH     = _ROOT / "data/models/risk_scores.parquet"         if _ROOT else None
OR_PATH       = _ROOT / "data/processed/shapefiles/openroads_yorkshire.parquet" if _ROOT else None
NET_PATH      = _ROOT / "data/features/network_features.parquet"  if _ROOT else None
DTC_PATH      = Path(__file__).parent / "data/raw/dvsa/dtc_summary.csv"

# Also check uploads folder for dtc_summary
if not DTC_PATH.exists():
    DTC_PATH = Path(__file__).parent.parent / "data/raw/dvsa/dtc_summary.csv"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f1117; }
    [data-testid="stSidebar"] * { color: #e8e8e8 !important; }
    .metric-card {
        background: #1a1d27;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 3px solid #e05252;
    }
    .metric-card .label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 22px; font-weight: 600; color: #f0f0f0; }
    .metric-card .sub   { font-size: 12px; color: #aaa; margin-top: 2px; }
    h1 { font-size: 1.6rem !important; }
    .stSelectbox label, .stMultiSelect label, .stSlider label { font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_risk():
    if RISK_PATH is None or not RISK_PATH.exists():
        return None
    return pd.read_parquet(RISK_PATH)

@st.cache_data
def load_openroads():
    if OR_PATH is None or not OR_PATH.exists():
        return None
    gdf = gpd.read_parquet(OR_PATH)
    return gdf.to_crs("EPSG:4326")

@st.cache_data
def load_network():
    if NET_PATH is None or not NET_PATH.exists():
        return None
    return pd.read_parquet(NET_PATH)

@st.cache_data
def load_dtc():
    # Try multiple locations
    for p in [DTC_PATH,
              Path(__file__).parent.parent / "data/raw/dvsa/dtc_summary.csv"]:
        if p.exists():
            dtc = pd.read_csv(p)
            # Filter to Yorkshire bbox
            return dtc[
                dtc["latitude"].between(53.3, 54.7) &
                dtc["longitude"].between(-2.4, 0.2)
            ].copy()
    return None

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RISK_PALETTE = [
    (0,   "#2166ac"),   # lowest — blue
    (20,  "#74add1"),
    (40,  "#e0f3f8"),
    (60,  "#fee090"),
    (80,  "#f46d43"),
    (95,  "#d73027"),
    (99,  "#a50026"),   # top 1% — dark red
]

def risk_colour(pct):
    """Map risk percentile (0-100) to hex colour."""
    if pd.isna(pct):
        return "#555555"
    for i, (thresh, col) in enumerate(RISK_PALETTE):
        if pct <= thresh:
            return RISK_PALETTE[max(0, i-1)][1]
    return RISK_PALETTE[-1][1]

def road_weight(road_class):
    """Line weight by road classification."""
    return {"Motorway": 4, "A Road": 3, "B Road": 2}.get(road_class, 1.5)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛣 Yorkshire Road Risk")
    st.caption("Exposure-adjusted collision risk · OS Open Roads · 2015–2024")
    st.divider()

    st.subheader("Filters")

    road_classes = st.multiselect(
        "Road classification",
        ["Motorway", "A Road", "B Road", "Classified Unnumbered", "Unclassified"],
        default=["Motorway", "A Road", "B Road"],
    )

    risk_tier = st.select_slider(
        "Show risk tier",
        options=["Top 1%", "Top 5%", "Top 10%", "Top 25%", "All roads"],
        value="Top 10%",
    )
    tier_map = {"Top 1%": 99, "Top 5%": 95, "Top 10%": 90, "Top 25%": 75, "All roads": 0}
    min_percentile = tier_map[risk_tier]

    year_options = [2019, 2021, 2023]
    selected_year = st.selectbox("Model year", year_options, index=2)

    st.divider()
    show_centres = st.toggle("Show DVSA test centres", value=True)
    show_legend  = st.toggle("Show legend", value=True)

    st.divider()
    st.caption("**About the model**")
    st.caption(
        "Risk scores from a Poisson GLM + XGBoost trained on 102k collisions "
        "2015–2024. Exposure offset = log(AADT × length × 365 / 1M vehicle-km). "
        "Higher percentile = more collisions than expected given traffic volume."
    )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
risk    = load_risk()
openroads = load_openroads()
net     = load_network()
dtc     = load_dtc()

data_ok = risk is not None and openroads is not None

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("Yorkshire Road Risk Explorer")

if not data_ok:
    st.error(
        "Could not load model outputs. Make sure you have run the pipeline "
        "(`model.py --stage collision`) and that `_ROOT` points to the project root."
    )
    st.info(f"Expected risk scores at: `{RISK_PATH}`")
    st.stop()

# ---------------------------------------------------------------------------
# Filter & join data
# ---------------------------------------------------------------------------
risk_year = risk[risk["year"] == selected_year].copy()

# Aggregate per link (take most recent year already filtered)
risk_agg = risk_year.groupby("link_id").agg(
    risk_percentile=("risk_percentile", "max"),
    predicted_glm=("predicted_glm", "mean"),
    residual_glm=("residual_glm", "mean"),
    collision_count=("collision_count", "sum"),
    estimated_aadt=("estimated_aadt", "mean"),
).reset_index()

# Join geometry
map_gdf = openroads[["link_id", "geometry", "road_classification",
                      "road_name", "link_length_km", "form_of_way"]].merge(
    risk_agg, on="link_id", how="inner"
)

# Join network features for click panel
if net is not None:
    map_gdf = map_gdf.merge(
        net[["link_id", "betweenness_relative", "degree_mean",
             "dist_to_major_km", "speed_limit_mph"]],
        on="link_id", how="left"
    )

# Apply filters
if road_classes:
    map_gdf = map_gdf[map_gdf["road_classification"].isin(road_classes)]

if min_percentile > 0:
    map_gdf = map_gdf[map_gdf["risk_percentile"] >= min_percentile]

# Clip to Yorkshire proper
cx = map_gdf.geometry.centroid.x
cy = map_gdf.geometry.centroid.y
in_yorks = cx.between(-2.3, 0.2) & cy.between(53.3, 54.65)
map_gdf = map_gdf[in_yorks].copy()

# ---------------------------------------------------------------------------
# Summary metrics (top bar)
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Road links shown</div>
        <div class="value">{len(map_gdf):,}</div>
        <div class="sub">{risk_tier} · {selected_year}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    med_pct = map_gdf["risk_percentile"].median()
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Median risk percentile</div>
        <div class="value">{med_pct:.0f}</div>
        <div class="sub">of all 705k Yorkshire links</div>
    </div>""", unsafe_allow_html=True)

with col3:
    col_links = map_gdf[map_gdf["collision_count"] > 0]
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Links with collisions</div>
        <div class="value">{len(col_links):,}</div>
        <div class="sub">in {selected_year} model year</div>
    </div>""", unsafe_allow_html=True)

with col4:
    mean_aadt = map_gdf["estimated_aadt"].median()
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Median estimated AADT</div>
        <div class="value">{mean_aadt:,.0f}</div>
        <div class="sub">vehicles/day</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
map_col, info_col = st.columns([3, 1])

with map_col:
    m = folium.Map(
        location=[53.95, -1.3],
        zoom_start=9,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    # Road links
    # Sample for performance if very large
    MAX_LINKS = 8000
    plot_gdf = map_gdf if len(map_gdf) <= MAX_LINKS else map_gdf.sample(MAX_LINKS, random_state=42)

    if len(map_gdf) > MAX_LINKS:
        st.caption(f"⚠ Showing {MAX_LINKS:,} of {len(map_gdf):,} links for performance. Zoom in or apply tighter filters.")

    for _, row in plot_gdf.iterrows():
        pct = row.get("risk_percentile", np.nan)
        colour = risk_colour(pct)
        weight = road_weight(row.get("road_classification", ""))

        # Build tooltip
        road_name = row.get("road_name", "") or "Unnamed"
        road_class = row.get("road_classification", "Unknown")
        aadt = row.get("estimated_aadt", np.nan)
        residual = row.get("residual_glm", np.nan)
        col_count = row.get("collision_count", 0)

        tooltip = (
            f"<b>{road_name}</b> · {road_class}<br>"
            f"Risk percentile: <b>{pct:.0f}</b><br>"
            f"Est. AADT: {aadt:,.0f} veh/day<br>"
            f"Collisions ({selected_year}): {int(col_count)}<br>"
            f"Excess risk: {residual:+.3f}"
        )

        try:
            coords = [(c[1], c[0]) for c in row.geometry.coords]
        except Exception:
            # Handle MultiLineString
            continue

        folium.PolyLine(
            coords,
            color=colour,
            weight=weight,
            opacity=0.85,
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(m)

    # DVSA test centres
    if show_centres and dtc is not None:
        centre_group = folium.FeatureGroup(name="DVSA Test Centres")
        for _, row in dtc.iterrows():
            pass_rate = row.get("pass", np.nan)
            n_tests = row.get("totalTestCount", 0)
            name = row.get("name", "Unknown")

            popup_html = f"""
            <b>{name}</b><br>
            Pass rate: <b>{pass_rate:.1%}</b><br>
            Total tests: {int(n_tests):,}
            """

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=8,
                color="white",
                weight=2,
                fill=True,
                fill_color="#f5c842",
                fill_opacity=0.9,
                tooltip=folium.Tooltip(name, sticky=False),
                popup=folium.Popup(popup_html, max_width=200),
            ).add_to(centre_group)

        centre_group.add_to(m)

    # Legend
    if show_legend:
        legend_html = """
        <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                    background:#1a1a2e; padding:12px 16px; border-radius:8px;
                    border:1px solid #444; font-size:12px; color:#ddd;">
            <b style="font-size:13px;">Risk Percentile</b><br>
            <span style="color:#a50026">⬛</span> Top 1%<br>
            <span style="color:#d73027">⬛</span> Top 1–5%<br>
            <span style="color:#f46d43">⬛</span> Top 5–20%<br>
            <span style="color:#fee090">⬛</span> Middle 40–80%<br>
            <span style="color:#74add1">⬛</span> Lower 20–40%<br>
            <span style="color:#2166ac">⬛</span> Lowest 20%<br>
            <br>
            <span style="color:#f5c842">●</span> DVSA test centre
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)

    map_data = st_folium(m, height=600, use_container_width=True)

# ---------------------------------------------------------------------------
# Info panel (right column)
# ---------------------------------------------------------------------------
with info_col:
    st.subheader("Road details")

    if map_data and map_data.get("last_object_clicked_tooltip"):
        st.markdown(map_data["last_object_clicked_tooltip"], unsafe_allow_html=True)
    else:
        st.caption("Hover over a road link to see details, or click to select.")

    st.divider()

    # Top 10 highest risk links currently shown
    st.subheader(f"Top 10 highest risk")
    if len(map_gdf) > 0:
        top10 = (
            map_gdf[["road_name", "road_classification", "risk_percentile",
                      "estimated_aadt", "collision_count"]]
            .sort_values("risk_percentile", ascending=False)
            .head(10)
        )
        top10["road_name"] = top10["road_name"].fillna("Unnamed")
        top10["risk_percentile"] = top10["risk_percentile"].round(0).astype(int)
        top10["estimated_aadt"] = top10["estimated_aadt"].round(0).astype(int)
        st.dataframe(
            top10.rename(columns={
                "road_name": "Road",
                "road_classification": "Class",
                "risk_percentile": "Pct",
                "estimated_aadt": "AADT",
                "collision_count": "Cols",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # Pass rate context if centres visible
    if show_centres and dtc is not None:
        st.subheader("Yorkshire test centres")
        yorks_dtc = dtc.copy()
        yorks_dtc = yorks_dtc[yorks_dtc["longitude"].between(-2.3, 0.2)]
        yorks_dtc = yorks_dtc[["name", "pass", "totalTestCount"]].sort_values("pass")
        yorks_dtc["pass"] = (yorks_dtc["pass"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            yorks_dtc.rename(columns={"name": "Centre", "pass": "Pass rate",
                                       "totalTestCount": "Tests"}),
            use_container_width=True,
            hide_index=True,
            height=250,
        )

# ---------------------------------------------------------------------------
# Road classification breakdown chart
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Risk distribution by road class")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    if len(map_gdf) > 0:
        by_class = (
            map_gdf.groupby("road_classification")["risk_percentile"]
            .agg(["median", "count"])
            .reset_index()
            .sort_values("median", ascending=False)
            .rename(columns={"road_classification": "Road class",
                              "median": "Median risk percentile",
                              "count": "Links"})
        )
        st.dataframe(by_class, use_container_width=True, hide_index=True)

with chart_col2:
    st.caption("""
    **Interpreting the risk percentile**

    The percentile is calculated across all 705,672 Yorkshire road links.
    A road in the **top 1%** (≥99th percentile) has more collisions than
    99% of all Yorkshire roads *given its traffic volume*.

    A road with high traffic volume (e.g. a motorway) may have a *lower*
    percentile than a quiet rural B-road if the B-road has disproportionate
    collisions per vehicle-km.

    The **excess risk** (residual) shows observed minus model-predicted
    collisions. Positive = more dangerous than the model expects.
    """)
