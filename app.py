import requests
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

API_URL = os.getenv("API_URL", "https://c-api-adh4.onrender.com/predict")
# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IT5006 Chicago Crime Dashboard",
    page_icon="Current Activity",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER: FORMAT NUMBERS (NEW) ---
def format_big_number(num):
    """Formats large numbers (e.g., 1,200,000 -> 1.2M, 45,000 -> 45.0K)."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return f"{num:,}"

# --- CSS STYLING ---
st.markdown("""
<style>
    /* 1. RESPONSIVE Metric Value Styling */
    [data-testid="stMetricValue"] {
        font-size: clamp(18px, 1.8vw, 26px) !important; 
        font-weight: bold !important;
        word-wrap: break-word !important;       
        white-space: pre-wrap !important;       
        line-height: 1.2 !important;            
        height: auto !important;                
        min-height: 50px !important;            
    }
    
    [data-testid="stMetricLabel"] {
        font-size: clamp(12px, 1.2vw, 14px) !important;
        width: 100% !important;
        white-space: normal !important;
    }
    
    .section-header {
        font-size: 24px;
        font-weight: bold;
        margin-top: 35px;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e0e0e0;
        color: #333;
    }
    
    .status-box {
        padding: 10px;
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        border-radius: 4px;
        margin-bottom: 15px;
        font-weight: bold;
        color: #0f52ba;
    }
    
    div.stButton > button {
        margin-top: 28px; 
        width: 100%;
    }
    
    .benchmark-text {
        text-align: center;
        font-size: 13px;
        color: #666;
        margin-top: -10px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER: CUSTOM METRIC CARD (UPDATED) ---
def custom_metric(label, value):
    # FIX: Changed white-space to 'normal' so text wraps if it's too long
    st.markdown(f"""
    <div style="
        background-color: #f9f9f9; 
        border: 1px solid #e0e0e0; 
        border-radius: 8px; 
        padding: 10px; 
        text-align: center;
        margin-bottom: 10px;">
        <div style="font-size: 14px; color: #666; margin-bottom: 2px;">{label}</div>
        <div style="font-size: 22px; font-weight: bold; color: #333; white-space: nowrap;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADING ---
@st.cache_data
def load_crime_data():
    try:
        df = pd.read_csv('Crime_Dataset_Lite_small.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month_name()
        df['Hour'] = df['Date'].dt.hour
        df['DayOfWeek'] = df['Date'].dt.day_name()

        for col in ['Community Area', 'Beat', 'District', 'Ward']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        return df
    except FileNotFoundError:
        st.error("❌ Crime_Dataset_Lite.csv not found.")
        return pd.DataFrame()

@st.cache_data
def load_census_data():
    census_dict = {}
    for year in range(2015, 2025):
        fname = f"CCA_{year}.geojson"
        if os.path.exists(fname):
            try:
                gdf = gpd.read_file(fname)
                df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
                
                if 'GEOID' in df.columns:
                    df['Community Area'] = pd.to_numeric(df['GEOID'], errors='coerce').fillna(0).astype(int)
                elif 'OBJECTID' in df.columns:
                    df['Community Area'] = pd.to_numeric(df['OBJECTID'], errors='coerce').fillna(0).astype(int)
                elif len(df) == 77:
                     df['Community Area'] = range(1, 78)

                # Metrics calculations (Kept original logic)
                if 'TOT_POP' in df.columns: pop = df['TOT_POP'].replace(0, 1)
                else: pop = 1
                
                inc_cols = ['HCUND20K', 'HC20Kto49K', 'HC50Kto75K', 'HCOV75K']
                if all(c in df.columns for c in inc_cols):
                    df['Calculated_HH'] = df[inc_cols].sum(axis=1)
                elif 'TOT_HH' in df.columns:
                    df['Calculated_HH'] = df['TOT_HH']
                else:
                    df['Calculated_HH'] = 1
                hh = df['Calculated_HH'].replace(0, 1)

                if 'WHITE' in df.columns:
                    df['Pct_White'] = (df['WHITE'] / pop) * 100
                    df['Pct_Black'] = (df['BLACK'] / pop) * 100
                    df['Pct_Hispanic'] = (df['HISP'] / pop) * 100
                    df['Pct_Asian'] = (df['ASIAN'] / pop) * 100
                
                # --- UPDATE 1: Create aliases for superior model ---
                if 'Pct_Black' in df.columns:
                    df['Black_Pct'] = df['Pct_Black']
                else:
                    df['Black_Pct'] = 0
                
                low_inc_cols = ['HCUND20K', 'HC20Kto49K']
                if all(c in df.columns for c in low_inc_cols):
                    df['Pct_LowIncome'] = (df[low_inc_cols].sum(axis=1) / hh) * 100
                else: df['Pct_LowIncome'] = 0 

                df['Pct_HighIncome'] = 0
                df['Wealth_Label'] = "Wealth (>$75k)"
                if 'INC_GT_150' in df.columns:
                    df['Pct_HighIncome'] = (df['INC_GT_150'] / hh) * 100
                    df['Wealth_Label'] = "Wealth (>$150k)"
                elif 'HCOV150K' in df.columns:
                    df['Pct_HighIncome'] = (df['HCOV150K'] / hh) * 100
                    df['Wealth_Label'] = "Wealth (>$150k)"
                elif 'HCOV75K' in df.columns:
                    df['Pct_HighIncome'] = (df['HCOV75K'] / hh) * 100
                    df['Wealth_Label'] = "Wealth (>$75k)"

                if 'MEDINC' in df.columns: df['Median_Income'] = df['MEDINC']
                elif 'MED_INC' in df.columns: df['Median_Income'] = df['MED_INC']
                else: df['Median_Income'] = 0

                if 'MED_HV' in df.columns: df['Median_HomeVal'] = df['MED_HV']
                else: df['Median_HomeVal'] = 0

                if 'UNEMP' in df.columns and 'IN_LBFRC' in df.columns:
                    df['Labor_Force'] = df['IN_LBFRC'].replace(0, 1)
                    df['Pct_Unemp'] = (df['UNEMP'] / df['Labor_Force']) * 100
                else: 
                    df['Pct_Unemp'] = 0
                    df['Labor_Force'] = 1
                
                # --- UPDATE 2: Create alias for Unemployment Rate ---
                df['Unemployment_Rate'] = df['Pct_Unemp']

                pop_25 = 1
                if 'POP_25OV' in df.columns: pop_25 = df['POP_25OV'].replace(0, 1)
                elif 'AGE_25_UP' in df.columns: pop_25 = df['AGE_25_UP'].replace(0, 1)
                df['Pop_Over25'] = pop_25
                
                df['Pct_NoHS'] = 0
                df['Pct_Bach'] = 0
                lt_hs_cols = [c for c in df.columns if c.upper() in ['LT_HS', 'EDU_LESS_HS', 'NOT_HS_GRAD']]
                if lt_hs_cols: df['Pct_NoHS'] = (df[lt_hs_cols[0]] / pop_25) * 100
                bach_cols = [c for c in df.columns if c.upper() in ['BACH', 'EDU_BACH', 'BACHELORS_OR_MORE']]
                if bach_cols: df['Pct_Bach'] = (df[bach_cols[0]] / pop_25) * 100

                if 'FOR_BORN' in df.columns: df['Pct_ForeignBorn'] = (df['FOR_BORN'] / pop) * 100
                else: df['Pct_ForeignBorn'] = 0

                if 'NO_VEH' in df.columns: df['Pct_NoVeh'] = (df['NO_VEH'] / hh) * 100
                else: df['Pct_NoVeh'] = 0

                if 'POP_HH' in df.columns: df['Avg_HH_Size'] = df['POP_HH'] / hh
                else: df['Avg_HH_Size'] = 0

                # Added new features to column list
                cols = ['Community Area', 'Pct_White', 'Pct_Black', 'Pct_Hispanic', 'Pct_Asian', 
                        'Pct_LowIncome', 'Pct_HighIncome', 'Median_Income', 'Median_HomeVal', 
                        'Pct_Unemp', 'Labor_Force', 
                        'Pct_NoHS', 'Pct_Bach', 'Pop_Over25',
                        'Pct_ForeignBorn', 'Pct_NoVeh', 'Avg_HH_Size',
                        'TOT_POP', 'Calculated_HH', 
                        'WHITE', 'BLACK', 'HISP', 'ASIAN', 'Wealth_Label',
                        'Unemployment_Rate', 'Black_Pct'] # Added here
                
                avail = [c for c in cols if c in df.columns]
                if 'Community Area' in avail:
                    census_dict[year] = df[avail].set_index('Community Area').fillna(0)
            except: pass

    if 2015 in census_dict and 2016 in census_dict:
        df15 = census_dict[2015]
        df16 = census_dict[2016]
        idx = df15.index.intersection(df16.index)
        census_dict[2014] = df15.loc[idx].copy() 

    return census_dict

@st.cache_data
def load_geography(level):
    try:
        if level == 'Community Area':
            gdf = gpd.read_file('Boundaries.geojson')
            id_col = 'area_num_1' if 'area_num_1' in gdf.columns else 'area_numbe'
            gdf['geometry_id'] = pd.to_numeric(gdf[id_col], errors='coerce').fillna(0).astype(int)
            gdf['name'] = gdf['community'].str.title()
        elif level == 'Police Beat':
            gdf = gpd.read_file('Boundaries_beat.geojson')
            gdf['geometry_id'] = pd.to_numeric(gdf['beat_num'], errors='coerce').fillna(0).astype(int)
            gdf['name'] = "Beat " + gdf['geometry_id'].astype(str)
        elif level == 'Police District':
            gdf = gpd.read_file('Boundaries_district.geojson')
            gdf['geometry_id'] = pd.to_numeric(gdf['dist_num'], errors='coerce').fillna(0).astype(int)
            gdf['name'] = "District " + gdf['geometry_id'].astype(str)
        elif level == 'Ward':
            gdf = gpd.read_file('Boundaries_ward.geojson')
            gdf['geometry_id'] = pd.to_numeric(gdf['ward'], errors='coerce').fillna(0).astype(int)
            gdf['name'] = "Ward " + gdf['geometry_id'].astype(str)
        return gdf 
    except FileNotFoundError:
        st.error(f"❌ GeoJSON for {level} not found.")
        return gpd.GeoDataFrame()

# --- HELPER: CLUSTERING (UPDATED) ---
def run_clustering(census_df):
    if census_df.empty: return census_df
    
    # --- UPDATE 3: Use the Superior Feature Set ---
    features = ['Median_Income', 'Unemployment_Rate', 'Black_Pct']
    
    # Ensure all features exist
    features = [f for f in features if f in census_df.columns]
    
    X = census_df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # --- UPDATE 4: Smart Sort by Income Descending (0 = Highest Income) ---
    temp_df = pd.DataFrame({'label': labels, 'score': census_df['Median_Income']})
    # Sort descending so Rank 0 is the highest income (Prosperous)
    rank = temp_df.groupby('label')['score'].mean().sort_values(ascending=False).index
    
    remap = {old_label: new_label for new_label, old_label in enumerate(rank)}
    census_df = census_df.copy()
    census_df['Cluster'] = pd.Series(labels, index=X.index).map(remap)
    return census_df

# --- HELPER: BENCHMARK CALCULATOR ---
def calculate_benchmark(df, metric, denominator=None, method='mean'):
    if method == 'median':
        return df[metric].median()
    elif method == 'mean' and denominator:
        try:
            total_numerator = (df[metric] / 100 * df[denominator]).sum()
            total_denominator = df[denominator].sum()
            if total_denominator == 0: return 0
            return (total_numerator / total_denominator) * 100
        except:
            return df[metric].mean() 
    return df[metric].mean()

def render_model_findings():
    st.title("Model Lab")
    st.caption("Phase 2 results are presented here as an interactive diagnostics page rather than a plain summary page.")

    # =========================
    # 1. Core metrics
    # =========================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", "LightGBM")
    c2.metric("Best Accuracy", "0.52")
    c3.metric("Best Macro F1", "0.31")
    c4.metric("Models Compared", "4")

    st.markdown("---")

    # =========================
    # 2. Model comparison data
    # =========================
    model_df = pd.DataFrame({
        "Model": ["Logistic Regression", "KNN", "Random Forest", "LightGBM"],
        "Accuracy": [0.28, 0.35, 0.45, 0.52],
        "Macro F1": [0.12, 0.18, 0.24, 0.31]
    })

    model_long = model_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Macro F1"],
        var_name="Metric",
        value_name="Score"
    )

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Model Performance Comparison")
        fig_perf = px.bar(
            model_long,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            text="Score",
            title="Overall Accuracy and Macro F1 Across Models"
        )
        fig_perf.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_perf.update_layout(yaxis_title="Score", xaxis_title="")
        st.plotly_chart(fig_perf, use_container_width=True)

    with right:
        st.subheader("Performance Table")
        st.dataframe(
            model_df.style.format({
                "Accuracy": "{:.2f}",
                "Macro F1": "{:.2f}"
            }),
            use_container_width=True
        )

        st.subheader("Interpretation")
        st.write(
            "LightGBM achieved the strongest overall performance in the Chicago setting, "
            "suggesting that non-linear tree-based models are better able to capture the "
            "interaction between time, space, transit accessibility, and neighborhood structure."
        )

    st.markdown("---")

    # =========================
    # 3. Prediction task + features
    # =========================
    left2, right2 = st.columns([1.05, 1])

    with left2:
        st.subheader("Prediction Task")
        st.write(
            "The modeling task is to predict one of six aggregated crime categories for a Chicago incident."
        )

        task_df = pd.DataFrame({
            "Target Class": ["Violent", "Property", "Sexual", "Vice", "Public Order", "Other"]
        })
        st.dataframe(task_df, use_container_width=True, hide_index=True)

        st.subheader("Training / Test Scale")
        scale_df = pd.DataFrame({
            "Split": ["Train", "Test"],
            "Rows": [2200656, 550164]
        })
        st.dataframe(scale_df, use_container_width=True, hide_index=True)

    with right2:
        st.subheader("Feature Groups")
        st.markdown("""
        **Temporal**
        - `hour`
        - `day_of_week`
        - `month`
        - `is_weekend`

        **Spatial / Transit**
        - `community_area_id`
        - `distance_to_nearest_station`
        - `stations_within_500m`

        **Neighborhood Structure**
        - `community_type`
        """)

        st.subheader("Why These Features Matter")
        st.write(
            "These features directly reflect the main Phase 1 findings: crime patterns vary over time, "
            "across neighborhoods, and with transit accessibility."
        )

    st.markdown("---")

    # =========================
    # 4. Robustness-style diagnostics from notebook outputs
    # =========================
    d1, d2 = st.columns(2)

    temporal_diag = pd.DataFrame({
        "Context": ["Weekend", "Weekday"],
        "Accuracy": [0.2622, 0.2755]
    })

    spatial_diag = pd.DataFrame({
        "Context": ["<= 500m to Station", "> 500m to Station"],
        "Accuracy": [0.4127, 0.2204]
    })

    with d1:
        st.subheader("Temporal Diagnostic")
        fig_temp = px.bar(
            temporal_diag,
            x="Context",
            y="Accuracy",
            text="Accuracy",
            title="Weekend vs Weekday Accuracy"
        )
        fig_temp.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig_temp.update_layout(yaxis_tickformat=".0%", yaxis_title="Accuracy", xaxis_title="")
        st.plotly_chart(fig_temp, use_container_width=True)

    with d2:
        st.subheader("Transit Diagnostic")
        fig_spatial = px.bar(
            spatial_diag,
            x="Context",
            y="Accuracy",
            text="Accuracy",
            title="Accuracy by Transit Proximity"
        )
        fig_spatial.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig_spatial.update_layout(yaxis_tickformat=".0%", yaxis_title="Accuracy", xaxis_title="")
        st.plotly_chart(fig_spatial, use_container_width=True)

    st.markdown("---")

    # =========================
    # 5. Key takeaways
    # =========================
    k1, k2 = st.columns(2)

    with k1:
        st.subheader("What We Learned")
        st.markdown("""
        - LightGBM is the strongest model among the four tested models  
        - Model performance improves when richer local context is included  
        - Accuracy is noticeably higher near transit hubs  
        - Predictive power is present, but still uneven across contexts  
        """)

    with k2:
        st.subheader("Why This Matters for Phase 3")
        st.success(
            "Phase 3 extends Phase 2 from static model evaluation to an interactive diagnostics and analytics interface."
        )

    st.info(
        "Current Model Lab is built from Phase 2 notebook outputs. Once the final processed dataset or exported model artifacts are available, this page can be extended into a full Prediction Studio."
    )


def render_generalization_limitations():
    st.title("Robustness Lab")
    st.caption("Phase 3 extends Phase 2 by examining what happens when the model is pushed beyond the main Chicago setting.")

    # =========================
    # 1. Top summary cards
    # =========================
    c1, c2, c3 = st.columns(3)
    c1.metric("Chicago Internal (Full Model)", "0.52")
    c2.metric("Chicago 2025 (Time-only)", "0.12")
    c3.metric("Texas Spatial Test (Time-only)", "0.11")

    st.markdown("---")

    # =========================
    # 2. Generalization comparison
    # =========================
    robustness_df = pd.DataFrame({
        "Evaluation Setting": [
            "Chicago Internal\n(Full Feature Model)",
            "Chicago 2025\n(Time-only Model)",
            "Texas NIBRS\n(Time-only Model)"
        ],
        "Accuracy": [0.52, 0.12, 0.11],
        "Type": [
            "In-sample / local",
            "Out-of-time",
            "Cross-region"
        ]
    })

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Generalization Comparison")
        fig_robust = px.bar(
            robustness_df,
            x="Evaluation Setting",
            y="Accuracy",
            color="Type",
            text="Accuracy",
            title="Accuracy Across Evaluation Contexts"
        )
        fig_robust.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_robust.update_layout(
            yaxis_tickformat=".0%",
            yaxis_title="Accuracy",
            xaxis_title=""
        )
        st.plotly_chart(fig_robust, use_container_width=True)

    with right:
        st.subheader("Evaluation Settings")
        st.dataframe(
            robustness_df.style.format({"Accuracy": "{:.2f}"}),
            use_container_width=True
        )

        st.subheader("Immediate Reading")
        st.write(
            "The internal Chicago model performs much better than the more portable time-only versions. "
            "This suggests that predictive performance depends heavily on local spatial and neighborhood context."
        )

    st.markdown("---")

    # =========================
    # 3. Important clarification
    # =========================
    st.subheader("Important Clarification")
    st.info(
        "The 0.12 and 0.11 results come from a more universal time-only model, not from the full Chicago feature model. "
        "So these results should be interpreted as evidence that local context matters a lot, rather than as proof that the full model simply 'fails everywhere'."
    )

    expl1, expl2 = st.columns([1.05, 1])

    with expl1:
        st.subheader("What Was Removed in the Time-only Setup")
        removed_df = pd.DataFrame({
            "Removed Feature Group": [
                "Community area information",
                "Transit accessibility features",
                "Neighborhood socioeconomic structure"
            ],
            "Why it matters": [
                "Captures local spatial heterogeneity",
                "Captures opportunity structure near stations",
                "Captures structural differences across neighborhoods"
            ]
        })
        st.dataframe(removed_df, use_container_width=True, hide_index=True)

    with expl2:
        st.subheader("Inputs Retained in the Time-only Setup")
        retained_df = pd.DataFrame({
            "Retained Input": ["hour", "day_of_week", "month", "is_weekend"]
        })
        st.dataframe(retained_df, use_container_width=True, hide_index=True)

        st.write(
            "After removing spatial, transit, and neighborhood information, the model becomes more portable, "
            "but much less predictive."
        )

    st.markdown("---")

    # =========================
    # 4. Context sensitivity
    # =========================
    st.subheader("Context Sensitivity Diagnostics")

    temporal_diag = pd.DataFrame({
        "Context": ["Weekend", "Weekday"],
        "Accuracy": [0.2622, 0.2755],
        "Diagnostic": "Temporal context"
    })

    transit_diag = pd.DataFrame({
        "Context": ["<= 500m to Station", "> 500m to Station"],
        "Accuracy": [0.4127, 0.2204],
        "Diagnostic": "Transit context"
    })

    d1, d2 = st.columns(2)

    with d1:
        fig_temp = px.bar(
            temporal_diag,
            x="Context",
            y="Accuracy",
            text="Accuracy",
            title="Weekend vs Weekday Accuracy"
        )
        fig_temp.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig_temp.update_layout(
            yaxis_tickformat=".0%",
            yaxis_title="Accuracy",
            xaxis_title=""
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with d2:
        fig_transit = px.bar(
            transit_diag,
            x="Context",
            y="Accuracy",
            text="Accuracy",
            title="Accuracy by Transit Proximity"
        )
        fig_transit.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig_transit.update_layout(
            yaxis_tickformat=".0%",
            yaxis_title="Accuracy",
            xaxis_title=""
        )
        st.plotly_chart(fig_transit, use_container_width=True)

    st.markdown("---")


    # =========================
    # 5. Interpretation + deployment positioning
    # =========================
    i1, i2 = st.columns(2)

    with i1:
        st.subheader("What We Learn from Robustness Tests")
        st.markdown("""
        - Model performance is much stronger in the original Chicago context  
        - Removing local spatial features causes a sharp drop in predictive power  
        - Temporal signals alone are not enough for strong transferability  
        - Transit-rich and location-specific settings appear more predictable  
        """)

    with i2:
        st.subheader("Deployment Positioning")
        st.markdown("""
        - Chicago-focused analytics prototype  
        - Local decision-support interface  
        - Useful for exploratory analysis and resource discussion  
        - Not suitable for blind cross-city deployment  
        - Not a universal predictive policing engine  
        """)

    st.markdown("---")

    # =========================
    # 6. Ethics and responsible use
    # =========================
    e1, e2 = st.columns(2)

    with e1:
        st.subheader("Ethical Concerns")
        st.markdown("""
        - Historical crime records may reflect reporting bias  
        - Policing intensity may distort observed crime distributions  
        - Predictive systems may reinforce over-policing in already monitored areas  
        - Performance gaps across contexts can create misleading confidence  
        """)

    with e2:
        st.subheader("Responsible Use Notes")
        st.markdown("""
        - Use model outputs as analytical support, not automatic decisions  
        - Interpret predictions together with local knowledge  
        - Avoid direct transfer of Chicago-trained logic to other regions  
        - Treat robustness checks as part of deployment governance  
        """)

    st.warning(
        "The strongest conclusion from this page is not that prediction is impossible, but that it is highly localized. "
        "Phase 3 therefore positions the prototype as a Chicago-focused analytics and decision-support system."
    )

def render_system_view():
    st.title("System View")
    st.caption("This page explains how Phase 1, Phase 2, and Phase 3 are connected into one end-to-end analytics workflow.")

    # =========================
    # 1. Top overview cards
    # =========================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Project Phases", "3")
    c2.metric("Core Data Sources", "3")
    c3.metric("Model Types", "4")
    c4.metric("Current App Modules", "4")

    st.markdown("---")

    # =========================
    # 2. End-to-end pipeline
    # =========================
    st.subheader("End-to-End Project Pipeline")

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.markdown("### Phase 1")
        st.success("EDA & Pattern Discovery")
        st.markdown("""
        - Temporal trends  
        - Spatial hotspot analysis  
        - Transit-related patterns  
        - Socioeconomic clustering  
        """)

    with p2:
        st.markdown("### Phase 2")
        st.success("Modeling & Evaluation")
        st.markdown("""
        - Crime-category prediction  
        - Logistic Regression  
        - KNN / Random Forest  
        - LightGBM comparison  
        """)

    with p3:
        st.markdown("### Phase 3")
        st.success("Interactive Analytics Studio")
        st.markdown("""
        - EDA Dashboard  
        - Model Lab  
        - Robustness Lab  
        - System View  
        """)

    with p4:
        st.markdown("### Final Positioning")
        st.success("Responsible Use")
        st.markdown("""
        - Chicago-focused prototype  
        - Local decision support  
        - Not universal deployment  
        - Robustness-aware interpretation  
        """)

    st.markdown("---")

    # =========================
    # 3. Data sources and inputs
    # =========================
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Core Data Sources")
        source_df = pd.DataFrame({
            "Data Source": [
                "Chicago Crime Dataset",
                "CTA Transit / Station Accessibility Data",
                "Socioeconomic / Community Area Data"
            ],
            "Role in Project": [
                "Main incident-level crime records",
                "Captures transit proximity and opportunity structure",
                "Captures neighborhood structural differences"
            ]
        })
        st.dataframe(source_df, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Main Inputs Used in the Model")
        input_df = pd.DataFrame({
            "Feature Group": [
                "Temporal",
                "Temporal",
                "Temporal",
                "Temporal",
                "Spatial / Transit",
                "Spatial / Transit",
                "Spatial / Transit",
                "Neighborhood"
            ],
            "Feature": [
                "hour",
                "day_of_week",
                "month",
                "is_weekend",
                "community_area_id",
                "distance_to_nearest_station",
                "stations_within_500m",
                "community_type"
            ]
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # =========================
    # 4. Feature engineering flow
    # =========================
    st.subheader("Feature Engineering Flow")

    f1, f2, f3, f4, f5 = st.columns(5)

    with f1:
        st.markdown("**1. Crime Aggregation**")
        st.write("Raw crime types were aggregated into six broader categories.")

    with f2:
        st.markdown("**2. Temporal Extraction**")
        st.write("Incident timestamps were transformed into hour, weekday, month, and weekend indicators.")

    with f3:
        st.markdown("**3. Transit Features**")
        st.write("Station distance and station density features were added to capture transit accessibility.")

    with f4:
        st.markdown("**4. Community Structure**")
        st.write("Neighborhood socioeconomic information was clustered into community archetypes.")

    with f5:
        st.markdown("**5. Final Model Matrix**")
        st.write("All engineered features were merged into the final training table used in Phase 2.")

    st.markdown("---")

    # =========================
    # 5. How the app is organized
    # =========================
    st.subheader("How Phase 3 Organizes the Project")

    app1, app2, app3, app4 = st.columns(4)

    with app1:
        st.markdown("### EDA Dashboard")
        st.write("Explores crime patterns through maps, KPIs, temporal charts, and neighborhood profiles.")

    with app2:
        st.markdown("### Model Lab")
        st.write("Presents model comparison, feature groups, train/test scale, and diagnostic views.")

    with app3:
        st.markdown("### Robustness Lab")
        st.write("Examines out-of-time, cross-region, and context-sensitivity results.")

    with app4:
        st.markdown("### System View")
        st.write("Explains the architecture, data flow, modeling flow, and deployment positioning.")

    st.markdown("---")

    # =========================
    # 6. Deployment-oriented interpretation
    # =========================
    d1, d2 = st.columns(2)

    with d1:
        st.subheader("What Phase 3 Adds")
        st.markdown("""
        - Converts earlier outputs into an interactive analytics interface  
        - Integrates EDA, modeling, and robustness analysis in one system  
        - Improves explainability for demos, reports, and presentations  
        - Moves the project closer to a deployment-oriented prototype  
        """)

    with d2:
        st.subheader("What Phase 3 Does Not Claim")
        st.markdown("""
        - It does not claim universal transferability  
        - It does not support blind cross-city deployment  
        - It does not replace human judgment  
        - It does not remove the need for fairness and context checks  
        """)

    st.info(
        "System View helps frame the whole project as an end-to-end Chicago crime analytics workflow, rather than as a disconnected set of EDA charts and model results."
    )

def render_prediction_studio():
    st.title("Prediction Studio")
    st.caption(
        "This page is the next integration layer of Phase 3. It is designed for scenario-based prediction and model deployment, and is now connected to the local inference API."
    )

    # =========================
    # 1. Top overview
    # =========================
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Mode", "Live Inference")
    c2.metric("Model Status", "Model API connected")
    c3.metric("Target", "Top-3 crime categories")

    st.markdown("---")

    # =========================
    # 2. Scenario input form
    # =========================
    st.subheader("Scenario Input")

    left, right = st.columns(2)

    with left:
        hour = st.slider("Hour of day", min_value=0, max_value=23, value=14)
        day_of_week = st.selectbox(
            "Day of week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=4
        )
        month = st.selectbox(
            "Month",
            list(range(1, 13)),
            index=5
        )
        is_weekend = st.radio("Is weekend?", ["No", "Yes"], horizontal=True)

    with right:
        community_area_id = st.number_input(
            "Community area ID",
            min_value=1,
            max_value=77,
            value=32,
            step=1
        )
        distance_to_nearest_station = st.number_input(
            "Distance to nearest station (meters)",
            min_value=0.0,
            max_value=5000.0,
            value=400.0,
            step=50.0
        )
        stations_within_500m = st.number_input(
            "Stations within 500m",
            min_value=0,
            max_value=10,
            value=1,
            step=1
        )
        community_type = st.selectbox(
            "Community type",
            ["Distressed", "Prosperous"],
            index=0
        )

    st.markdown("---")

    # =========================
    # 3. Scenario summary
    # =========================
    st.subheader("Scenario Summary")
    summary_df = pd.DataFrame({
        "Input": [
            "Hour",
            "Day of week",
            "Month",
            "Weekend",
            "Community area ID",
            "Distance to nearest station",
            "Stations within 500m",
            "Community type"
        ],
        "Value": [
            hour,
            day_of_week,
            month,
            is_weekend,
            community_area_id,
            f"{distance_to_nearest_station:.0f} m",
            stations_within_500m,
            community_type
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # =========================
    # 4. Real prediction mode
    # =========================
    st.subheader("Real Prediction Output")

    run_demo = st.button("Run Real Prediction")

    if run_demo:
        DAY_MAP = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6,
        }

        payload = {
            "hour": int(hour),
            "day_of_week": DAY_MAP[day_of_week],
            "month": int(month),
            "is_weekend": 1 if is_weekend == "Yes" else 0,
            "community_area_id": int(community_area_id),
            "distance_to_nearest_station": float(distance_to_nearest_station),
            "stations_within_500m": int(stations_within_500m),
            "community_type": community_type,
        }

        try:
            with st.spinner("Running model inference..."):
                res = requests.post(
                    API_URL,
                    json=payload,
                    timeout=30
                )
                res.raise_for_status()
                result = res.json()

            st.success("Prediction complete")

            r1, r2 = st.columns([1.2, 1])

            with r1:
                st.markdown("### Top-3 Crime Categories")

                prob_df = pd.DataFrame(result["top_3_predictions"])
                prob_df["Probability"] = prob_df["probability"].map(lambda x: f"{x:.2%}")
                prob_df = prob_df.rename(columns={"label": "Crime Category"})

                st.dataframe(
                    prob_df[["Crime Category", "Probability"]],
                    use_container_width=True,
                    hide_index=True
                )

                fig_pred = px.bar(
                    prob_df,
                    x="Crime Category",
                    y="probability",
                    text="Probability",
                    title="Predicted Probability Distribution"
                )
                fig_pred.update_traces(textposition="outside")
                fig_pred.update_layout(
                    yaxis_tickformat=".0%",
                    yaxis_title="Probability",
                    xaxis_title=""
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            with r2:
                st.markdown("### Prediction Summary")
                st.metric("Most Likely Category", result["predicted_label"])
                st.metric("Predicted Probability", f"{result['predicted_probability']:.2%}")
                st.caption(f"Model: {result['model_version']}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")

    # =========================
    # 5. Integration roadmap
    # =========================
    st.subheader("Integration Roadmap")
    roadmap_df = pd.DataFrame({
        "Step": [
            "Export trained LightGBM model",
            "Export label encoder",
            "Export feature columns and risk maps",
            "Connect model inference to dashboard",
            "Replace prototype outputs with real predictions"
        ],
        "Status": [
            "Completed",
            "Completed",
            "Partially completed",
            "Completed",
            "Completed"
        ]
    })
    st.dataframe(roadmap_df, use_container_width=True, hide_index=True)

    st.info(
        "Prediction Studio is now connected to the local inference API and can generate real Top-3 category predictions from scenario inputs."
    )

# --- MAIN APP ---
df_crime = load_crime_data()
census_data = load_census_data()

# --- PAGE NAVIGATION ---
st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
page = st.sidebar.radio(
    "Go to page:",
    ["EDA Dashboard", "Model Lab", "Robustness Lab", "System View", "Prediction Studio"]
)

if page == "Model Lab":
    render_model_findings()
    st.stop()

if page == "Robustness Lab":
    render_generalization_limitations()
    st.stop()

if page == "System View":
    render_system_view()
    st.stop()

if page == "Prediction Studio":
    render_prediction_studio()
    st.stop()
    
# --- INIT STATE ---
if 'selected_id' not in st.session_state:
    st.session_state.selected_id = None

# --- SIDEBAR ---
st.sidebar.header("Filter Controls") 
geo_level = st.sidebar.radio("Geography Level:", ('Community Area', 'Police District', 'Police Beat', 'Ward'))
years = st.sidebar.slider("Year Range:", int(df_crime['Year'].min()), int(df_crime['Year'].max()), (2020, 2024))

st.sidebar.subheader("Time Filters")
all_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
sel_months = st.sidebar.multiselect("Select Months (Optional):", all_months, default=[])

st.sidebar.subheader("Crime Filters")
cats = sorted(df_crime['Crime_Category'].unique().tolist())
sel_cat = st.sidebar.selectbox("Category:", ["All"] + cats)
sel_sub = []
if sel_cat != "All":
    subs = df_crime[df_crime['Crime_Category'] == sel_cat]['Primary Type'].unique().tolist()
    sel_sub = st.sidebar.multiselect("Subtypes:", sorted(subs))

# --- PROFILE SELECTOR ---
valid_communities = None
census_year_data = None
cluster_names = {0: "Affluent / High SES", 1: "Working Class / Mixed", 2: "Vulnerable / Low SES"}

if geo_level == 'Community Area':
    mid_year = max(2014, min(2024, int((years[0] + years[1]) / 2)))
    if mid_year in census_data:
        census_year_data = run_clustering(census_data[mid_year])
        st.sidebar.markdown("---")
        st.sidebar.subheader("Neighborhood Profile")
        sel_cluster = st.sidebar.selectbox("Filter by Archetype:", ["All Neighborhoods"] + list(cluster_names.values()))
        if sel_cluster != "All Neighborhoods":
            target_cluster = [k for k, v in cluster_names.items() if v == sel_cluster][0]
            census_filtered = census_year_data[census_year_data['Cluster'] == target_cluster]
            valid_communities = census_filtered.index.tolist()

# --- FILTER DATA ---
mask = (df_crime['Year'].between(years[0], years[1]))
if sel_months: mask &= (df_crime['Month'].isin(sel_months))
if sel_cat != "All": mask &= (df_crime['Crime_Category'] == sel_cat)
if sel_sub: mask &= (df_crime['Primary Type'].isin(sel_sub))
df_filtered = df_crime[mask]

if valid_communities is not None:
    df_filtered = df_filtered[df_filtered['Community Area'].isin(valid_communities)]

# --- PREP MAP ---
merge_key = {'Community Area': 'Community Area', 'Police Beat': 'Beat', 'Police District': 'District', 'Ward': 'Ward'}[geo_level]
map_agg = df_filtered.groupby(merge_key).agg(Total=('ID', 'count'), Arrest=('Arrest', 'sum')).reset_index()
map_agg['Efficiency'] = (map_agg['Arrest'] / map_agg['Total']) * 100
map_agg['geometry_id'] = map_agg[merge_key]

gdf = load_geography(geo_level).reset_index(drop=True).merge(map_agg, on='geometry_id', how='left').fillna(0)
if geo_level == 'Community Area' and census_year_data is not None:
    gdf = gdf.merge(census_year_data, left_on='geometry_id', right_index=True, how='left').fillna(0)
if valid_communities is not None and geo_level == 'Community Area':
    gdf = gdf[gdf['geometry_id'].isin(valid_communities)].reset_index(drop=True)

# --- VISUALS ---
st.title("IT5006 Chicago Crime Dashboard")

if years[0] == years[1]: year_text = f"{years[0]}"
else: year_text = f"{years[0]} - {years[1]}"

st.markdown(f"**Analyzing:** {sel_cat} | **Year:** {year_text} | **Level:** {geo_level}")

col1, col2 = st.columns([2.5, 1])

with col1:
    c_map_controls = st.columns([2.5, 1])
    with c_map_controls[0]:
        metric = st.radio("Metric:", ('Total Volume', 'Arrest Efficiency %'), horizontal=True)
    with c_map_controls[1]:
        if st.button("Clear Map Selection"):
            st.session_state.selected_id = None
            st.rerun()

    col = 'Total' if metric == 'Total Volume' else 'Efficiency'
    scale = 'Reds' if metric == 'Total Volume' else 'Greens'
    
    if not gdf.empty:
        h_data = {'Total':True, 'Efficiency':':.1f'}
        if 'Median_Income' in gdf.columns: h_data.update({'Median_Income': ':$,.0f'})
        
        # Base Map
        fig = px.choropleth_map(gdf, geojson=gdf.geometry, locations=gdf.index, color=col,
                                color_continuous_scale=scale, range_color=(0, gdf[col].quantile(0.95)),
                                map_style="carto-positron", zoom=9.5, center={"lat": 41.85, "lon": -87.65},
                                opacity=0.6, hover_name='name', hover_data=h_data)
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=600)
        
        # Highlight Layer
        if st.session_state.selected_id is not None:
            sel_row = gdf[gdf['geometry_id'] == st.session_state.selected_id]
            if not sel_row.empty:
                fig.add_trace(go.Choroplethmap(
                    geojson=sel_row.geometry.__geo_interface__,
                    locations=sel_row.index,
                    z=[1] * len(sel_row),
                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                    marker_line_width=4,      
                    marker_line_color='red', 
                    showscale=False,
                    hoverinfo='skip',
                    name='Selected'
                ))

        # Capture Click
        sel = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
        if sel and sel['selection']['points']:
            idx = sel['selection']['points'][0]['point_index']
            clicked_id = gdf.iloc[idx]['geometry_id']
            if st.session_state.selected_id == clicked_id:
                st.session_state.selected_id = None
            else:
                st.session_state.selected_id = clicked_id
            st.rerun()
    else:
        st.warning("No data matches filters.")
        sel = None

# --- DETERMINE VIEW ---
sel_id = st.session_state.selected_id
sel_name = "City-Wide"
sel_row = None

if sel_id is not None:
    if sel_id in gdf['geometry_id'].values:
        sel_row = gdf[gdf['geometry_id'] == sel_id].iloc[0]
        sel_name = sel_row['name']
        df_view = df_filtered[df_filtered[merge_key] == sel_id]
    else:
        st.session_state.selected_id = None
        df_view = df_filtered
else:
    df_view = df_filtered

with col2:
    if sel_id:
        st.markdown(f'<div class="status-box">📍 Viewing: {sel_name}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box">🌎 Viewing: City-Wide</div>', unsafe_allow_html=True)
        
    tot = len(df_view)
    eff = (df_view['Arrest'].sum() / tot * 100) if tot > 0 else 0
    pop_val = sel_row['TOT_POP'] if sel_row is not None and 'TOT_POP' in sel_row else 0
    if pop_val == 0 and census_year_data is not None: pop_val = census_year_data['TOT_POP'].sum()
    rate_per_1k = (tot / pop_val * 1000) if pop_val > 0 else 0

    c1, c2, c3 = st.columns(3)
    # FIX: Applied format_big_number here
    with c1: custom_metric("Incidents", format_big_number(tot))
    with c2: custom_metric("Rate / 1k", f"{rate_per_1k:.1f}")
    with c3: custom_metric("Arrest %", f"{eff:.1f}%")

    st.markdown("#### Top Crime Types")
    if not df_view.empty:
        top_crimes = df_view['Primary Type'].value_counts().head(5)
        for i, (crime, count) in enumerate(top_crimes.items(), 1):
            st.markdown(f"<div style='font-size: 14px; margin-bottom: 4px;'><b>{i}. {crime}</b>: {count:,}</div>", unsafe_allow_html=True)
    else:
        st.write("No data.")

    if geo_level == 'Community Area' and sel_row is not None:
        st.markdown("---")
        st.markdown("#### Demographics")
        labels = ['White', 'Black', 'Hispanic', 'Asian']
        values = [sel_row['Pct_White'], sel_row['Pct_Black'], sel_row['Pct_Hispanic'], sel_row.get('Pct_Asian', 0)]
        
        color_map = {'White': '#1f77b4', 'Black': '#d62728', 'Hispanic': '#2ca02c', 'Asian': '#ff7f0e'}
        marker_colors = [color_map[l] for l in labels]

        fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6, marker=dict(colors=marker_colors))])
        
        # Centered Legend
        fig_donut.update_layout(
            showlegend=True, 
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), 
            margin=dict(t=0, b=0, l=0, r=0), 
            height=200
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
        if census_year_data is not None:
            tot_pop_city = census_year_data['TOT_POP'].sum()
            avg_white = (census_year_data['WHITE'].sum() / tot_pop_city) * 100
            avg_black = (census_year_data['BLACK'].sum() / tot_pop_city) * 100
            avg_hisp = (census_year_data['HISP'].sum() / tot_pop_city) * 100
            avg_asian = (census_year_data['ASIAN'].sum() / tot_pop_city) * 100
            st.caption(f"**Vs. City Avg:** White: {avg_white:.1f}% | Black: {avg_black:.1f}% | Hisp: {avg_hisp:.1f}% | Asian: {avg_asian:.1f}%")

# --- SOCIOECONOMIC DASHBOARD (UI FIX: RANKINGS, SMART COLORS & >150k DEFAULT) ---
if geo_level == 'Community Area' and sel_row is not None and census_year_data is not None:
    st.markdown("---") 
    st.markdown('<div class="section-header">Socioeconomic Profile (Selected vs. City Benchmark)</div>', unsafe_allow_html=True)
    
    # 1. Helper to calculate Rank (e.g., "1st", "15th")
    def get_rank_str(df, metric, current_val, high_is_rank_1=True):
        try:
            # Rank the entire city data for this metric
            ranks = df[metric].rank(ascending=not high_is_rank_1, method='min')
            r = ranks[df[metric] == current_val].iloc[0]
            
            n = int(r)
            if 11 <= (n % 100) <= 13: suffix = 'th'
            else: suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
            return f"({n}{suffix})"
        except:
            return ""

    # 2. UI RENDERER (Fixed: Added benchmark_label back)
    def render_box(col, label, val, benchmark_val, rank_str, suffix="", prefix="", benchmark_label="Avg", is_bad_high=False):
        import textwrap 
        
        # Color Logic
        diff = val - benchmark_val
        pct_diff = (diff / benchmark_val) * 100 if benchmark_val != 0 else 0
        
        # Neutral Zone: If within 5% of city average, stay Black
        if abs(pct_diff) < 5: 
            color = "#333333" 
        elif is_bad_high:
            color = "#d62728" if diff > 0 else "#2ca02c" # Red if High (Bad)
        else:
            color = "#2ca02c" if diff > 0 else "#d62728" # Green if High (Good)

        # HTML Block
        html_code = textwrap.dedent(f"""
            <div style="background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                <div style="font-size: 14px; color: #555; margin-bottom: 5px;">
                    {label}
                </div>
                <div style="font-size: 24px; font-weight: bold; color: {color}; margin-bottom: 8px;">
                    {prefix}{val:,.1f}{suffix} <span style="font-size: 16px; color: #888; font-weight: normal;">{rank_str}</span>
                </div>
                <div style="font-size: 12px; color: #666; border-top: 1px solid #eee; padding-top: 5px;">
                    City {benchmark_label}: <b>{prefix}{benchmark_val:,.1f}{suffix}</b>
                </div>
            </div>
        """)
        
        with col:
            st.markdown(html_code, unsafe_allow_html=True)

    # --- RENDER METRICS ---
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    
    # 1. Median Income
    bm_inc = calculate_benchmark(census_year_data, 'Median_Income', method='median')
    rank_inc = get_rank_str(census_year_data, 'Median_Income', sel_row['Median_Income'], high_is_rank_1=True)
    render_box(r1c1, "Median Income", sel_row['Median_Income'], bm_inc, rank_inc, prefix="$", is_bad_high=False)
    
    # 2. Home Value
    bm_hv = calculate_benchmark(census_year_data, 'Median_HomeVal', method='median')
    rank_hv = get_rank_str(census_year_data, 'Median_HomeVal', sel_row['Median_HomeVal'], high_is_rank_1=True)
    render_box(r1c2, "Median Home Value", sel_row['Median_HomeVal'], bm_hv, rank_hv, prefix="$", is_bad_high=False)
    
    # 3. Poverty
    bm_low = calculate_benchmark(census_year_data, 'Pct_LowIncome', denominator='Calculated_HH', method='mean')
    rank_low = get_rank_str(census_year_data, 'Pct_LowIncome', sel_row['Pct_LowIncome'], high_is_rank_1=False)
    render_box(r1c3, "Poverty (<$50k)", sel_row['Pct_LowIncome'], bm_low, rank_low, suffix="%", is_bad_high=True)
    
    # 4. Wealth (Updated Default to >$150k)
    wealth_label = sel_row.get('Wealth_Label', "Wealth (>$150k)")
    bm_high = calculate_benchmark(census_year_data, 'Pct_HighIncome', denominator='Calculated_HH', method='mean')
    rank_high = get_rank_str(census_year_data, 'Pct_HighIncome', sel_row['Pct_HighIncome'], high_is_rank_1=True)
    render_box(r1c4, wealth_label, sel_row['Pct_HighIncome'], bm_high, rank_high, suffix="%", is_bad_high=False)

    r2c1, r2c2, r2c3 = st.columns(3)
    
    # 5. No High School
    bm_nohs = calculate_benchmark(census_year_data, 'Pct_NoHS', denominator='Pop_Over25', method='mean')
    rank_nohs = get_rank_str(census_year_data, 'Pct_NoHS', sel_row['Pct_NoHS'], high_is_rank_1=False)
    render_box(r2c1, "No High School Diploma", sel_row['Pct_NoHS'], bm_nohs, rank_nohs, suffix="%", is_bad_high=True)
    
    # 6. Bachelors
    bm_bach = calculate_benchmark(census_year_data, 'Pct_Bach', denominator='Pop_Over25', method='mean')
    rank_bach = get_rank_str(census_year_data, 'Pct_Bach', sel_row['Pct_Bach'], high_is_rank_1=True)
    render_box(r2c2, "Bachelor's Degree+", sel_row['Pct_Bach'], bm_bach, rank_bach, suffix="%", is_bad_high=False)
    
    # 7. Unemployment
    bm_unemp = calculate_benchmark(census_year_data, 'Pct_Unemp', denominator='Labor_Force', method='mean')
    rank_unemp = get_rank_str(census_year_data, 'Pct_Unemp', sel_row['Pct_Unemp'], high_is_rank_1=False)
    render_box(r2c3, "Unemployment Rate", sel_row['Pct_Unemp'], bm_unemp, rank_unemp, suffix="%", is_bad_high=True)

    r3c1, r3c2, r3c3 = st.columns(3)
    
    # 8. Immigrant Pop
    bm_fb = calculate_benchmark(census_year_data, 'Pct_ForeignBorn', denominator='TOT_POP', method='mean')
    rank_fb = get_rank_str(census_year_data, 'Pct_ForeignBorn', sel_row['Pct_ForeignBorn'], high_is_rank_1=True)
    render_box(r3c1, "Immigrant Population", sel_row['Pct_ForeignBorn'], bm_fb, rank_fb, suffix="%", is_bad_high=False)
    
    # 9. HH Size
    bm_hh = calculate_benchmark(census_year_data, 'Avg_HH_Size', denominator='Calculated_HH', method='mean')
    rank_hh = get_rank_str(census_year_data, 'Avg_HH_Size', sel_row['Avg_HH_Size'], high_is_rank_1=True)
    render_box(r3c2, "Average Household Size", sel_row['Avg_HH_Size'], bm_hh, rank_hh, benchmark_label="Avg")
    
    # 10. No Vehicle
    bm_noveh = calculate_benchmark(census_year_data, 'Pct_NoVeh', denominator='Calculated_HH', method='mean')
    rank_noveh = get_rank_str(census_year_data, 'Pct_NoVeh', sel_row['Pct_NoVeh'], high_is_rank_1=False)
    render_box(r3c3, "Households with No Vehicles", sel_row['Pct_NoVeh'], bm_noveh, rank_noveh, suffix="%", is_bad_high=True)

# --- ROW 3: TRENDS & HOURLY ---
st.markdown("---")
rc1, rc2 = st.columns(2)
with rc1:
    st.subheader("Monthly Trends") 
    if not df_view.empty:
        trend = df_view.groupby(pd.Grouper(key='Date', freq='ME')).size().reset_index(name='Count')
        fig = px.line(trend, x='Date', y='Count', markers=True, color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig, use_container_width=True)
with rc2:
    st.subheader("24-Hour Profile") 
    if not df_view.empty:
        prof = df_view.groupby('Hour').size().reset_index(name='Count')
        fig = px.bar(prof, x='Hour', y='Count')
        fig.update_layout(xaxis=dict(type='category'))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Temporal Heatmap") 
if not df_view.empty:
    heat = df_view.groupby(['DayOfWeek', 'Hour']).size().reset_index(name='Count')
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig = px.density_heatmap(heat, x='Hour', y='DayOfWeek', z='Count', color_continuous_scale='Reds',
                             category_orders={'DayOfWeek': days}, nbinsx=24, nbinsy=7)
    fig.update_traces(xgap=3, ygap=3)
    fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1, showgrid=False), yaxis=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True)
