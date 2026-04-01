# Chicago Crime & Socioeconomic Dashboard
### IT5006 Fundamentals of Data Analytics (Team 10) | AY25/26 Semester 2

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This interactive dashboard is phase 1 of a project for **NUS IT5006**, designed to analyze crime patterns in Chicago using the Chicago Crime Dataset (2018 - 2024). Streamlit RAM limited the amount of data which could be processed by the dashboard.

Chicago Community Area (CCA) datasets and Chicago boundaries geojson files were also utilised to supplement the Chicago Crime Dataset.

## Key Features

### 1. Interactive Geospatial Analysis
* **Multi-Level Granularity:** Switch views between **Community Areas**, **Police Districts**, **Police Beats**, and **Wards**.
* **Dynamic Choropleth Maps:** Visualize **Crime Volume** or **Arrest Efficiency %** across the city using interactive Plotly maps.
* **Point-and-Click Deep Dive:** Click any region on the map to instantly filter all downstream metrics for that specific location. Selected socioeconomic parameters for Community Areas are available.

### 2. Machine Learning Integration (Neighborhood Archetypes)
* Uses **K-Means Clustering** to categorize Chicago's 77 community areas into three distinct socioeconomic profiles:
    * **Affluent / High SES**
    * **Working Class / Mixed**
    * **Vulnerable / Low SES**
* **Model Features:** Clustered based on `Median_Income`, `Unemployment_Rate`, and `Black_Pct`.

### 3. Socioeconomic Benchmarking
* **Comparative Metrics:** Automatically compares a selected neighborhood's stats against the **City-Wide Average**.
* **Key Indicators:**
    * **Economic:** Median Income, Home Value, Poverty Rates.
    * **Education:** High School & Bachelor's degree attainment.
    * **Infrastructure:** Vehicle ownership & Household size.

### 4. Temporal Crime Trends
* **Heatmaps:** Day-of-Week vs. Hour-of-Day density maps to identify peak crime windows.
* **Trendlines:** Monthly crime volume analysis from 2018 to 2024.
* **24-Hour Profile:** Hourly distribution of criminal activity.

## Project Structure
```text
├── data/
│   ├── Crime_Dataset_Lite.zip       # Main crime dataset (2018-2024)
│   ├── Boundaries.geojson           # Community Area boundaries
│   ├── Boundaries_beat.geojson      # Police Beat boundaries
│   ├── Boundaries_district.geojson  # Police District boundaries
│   ├── Boundaries_ward.geojson      # Ward boundaries
│   └── CCA_20xx.geojson             # Annual Census/Demographic data
├── app.py                           # Main dashboard application
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
