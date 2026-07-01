<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.36+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/RAPIDS_cuDF-GPU_Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/BigQuery-Cloud_Analytics-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

# 🚦 Bengaluru Commute Intelligence

> **An end-to-end data intelligence platform** that transforms raw Bengaluru traffic data into actionable commute risk scores — powered by GPU-accelerated data processing (NVIDIA RAPIDS cuDF), RandomForest ML predictions, live weather integration, and Google BigQuery analytics.

---

## 🎯 Problem Statement

Bengaluru is one of the world's most congested cities. Commuters waste **1.5–2 hours daily** navigating unpredictable traffic. Existing tools show current congestion but don't **predict risk** or recommend **safer alternative routes** considering weather, incidents, and time-of-day patterns.

This platform solves that by combining **2.5 years of historical traffic data** (Jan 2022 – Aug 2024) with **real-time weather forecasts** and **ML-based risk prediction** to help commuters make smarter decisions.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                                  │
│                                                                     │
│  📊 Raw CSV (20K rows)                                              │
│    │                                                                │
│    ▼                                                                │
│  🧹 clean_data.py                                                   │
│    ├─ Null/duplicate handling (median/mode imputation)              │
│    ├─ Column normalisation & type coercion                          │
│    ├─ 🌦️ Open-Meteo Weather API integration                        │
│    │   └─ Hourly → daily peak-hour aggregation                     │
│    ├─ 6-component weighted Risk Score (0-100)                      │
│    └─ Risk Tier classification (Low/Moderate/High/Critical)        │
│    │                                                                │
│    ▼                                                                │
│  📁 data/processed.csv (27 columns, enriched)                       │
│    │                                                                │
│    ├──► 🤖 train_model.py                                           │
│    │     ├─ RandomForest risk_score regressor (R² ≈ 0.97)          │
│    │     ├─ RandomForest TTI regressor                              │
│    │     └─ Saved to models/risk_model.pkl                         │
│    │                                                                │
│    ├──► ☁️  bigquery_dashboard.py                                    │
│    │     ├─ Auto-creates BQ dataset + table                        │
│    │     ├─ 3 analytical queries (Q1/Q2/Q3)                        │
│    │     └─ Offline pandas fallback                                │
│    │                                                                │
│    └──► 🚀 cudf_benchmark.ipynb (Google Colab + T4 GPU)             │
│          └─ cudf.pandas acceleration benchmark                     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                      DASHBOARD (app.py)                             │
│                                                                     │
│  ┌─────────────┐  ┌──────────────────┐  ┌────────────────────────┐ │
│  │  📅 Date     │  │  📂 Historical   │  │  🤖 ML Predicted       │ │
│  │  Selector   │  │  (2022-Aug2024)  │  │  (Today / Future)      │ │
│  │             │  │  ─── Blue Badge  │  │  ─── Amber Badge       │ │
│  │  🛣️ Route   │  │  Actual CSV data │  │  RF model + live       │ │
│  │  ⏱️ Slot    │  │                  │  │  Open-Meteo weather    │ │
│  └─────────────┘  └──────────────────┘  └────────────────────────┘ │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Risk Gauge · Metric Cards · Alternate Route Recommendation │   │
│  │  Hourly Trend Charts · Route Comparison · Risk Heatmap      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 📊 Dual-Mode Intelligence
| Mode | Date Range | Source | Badge |
|------|-----------|--------|-------|
| **Historical** | Jan 2022 – Aug 2024 | Actual recorded data from CSV | 📂 Blue badge |
| **Predicted** | Today onwards (+16 days) | RandomForest ML model + live Open-Meteo forecast | 🤖 Amber badge |

### 🧠 ML Risk Prediction
- **RandomForest Regressor** (200 trees, max_depth=18)
- **9 features**: route, day-of-week, time-slot, rainfall, congestion, road capacity, incidents, visibility, weather severity
- **R² ≈ 0.97** on risk_score prediction
- Dual outputs: **risk_score** (0-100) + **travel_time_index**

### 🌦️ Live Weather Integration
- **Open-Meteo API** (free, no key required)
- Historical archive for past dates
- 16-day forecast for today/future
- Automatic fallback to synthetic weather if API unavailable

### 🚀 GPU Acceleration (NVIDIA RAPIDS cuDF)
- `cudf.pandas` zero-code-change accelerator
- **~8x speedup** on data cleaning pipeline at 1M+ row scale
- Colab notebook provided for reproducible benchmarks

### ☁️ BigQuery Analytics
- Auto-provisioning: dataset + table creation
- 3 pre-built analytical queries
- Seamless offline fallback to pandas

### 📈 Interactive Dashboard
- **Risk Score Gauge** with color-coded thresholds
- **Alternate Route Recommendation** (lowest-risk route at same time)
- **Hourly Trend Charts** (avg/min/max risk across all routes)
- **Route Comparison Bar Chart** with selected-route highlighting
- **Stacked Risk-Tier Heatmap** across all routes
- **Raw data explorer** with full drill-down capability

---

## 📂 Project Structure

```
bengaluru-commute-decision-tool/
├── app.py                          # Streamlit dashboard (dual-mode)
├── Banglore_traffic_Dataset.csv    # Original dataset
├── .streamlit/
│   └── config.toml                 # Dark theme + server config
├── data/
│   ├── bengaluru_traffic.csv       # Raw traffic data
│   ├── processed.csv               # Enriched output (27 cols)
│   └── dashboard/
│       ├── q1_hour_x_route.csv     # Risk by hour × route
│       ├── q2_hourly_trend.csv     # System-wide hourly trend
│       └── q3_route_heatmap.csv    # Route × risk-tier breakdown
├── models/
│   └── risk_model.pkl              # Trained RF bundle (~9.5 MB)
├── scripts/
│   ├── clean_data.py               # ETL + weather + risk scoring
│   ├── train_model.py              # ML model training
│   ├── bigquery_dashboard.py       # BQ analytics pipeline
│   └── weather_api.py              # Unified weather helper
├── notebooks/
│   └── cudf_benchmark.ipynb        # GPU acceleration benchmark
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Manasa-L-Hegde/bengaluru-commute-decision-tool.git
cd bengaluru-commute-decision-tool

# Install dependencies
pip install streamlit pandas numpy plotly scikit-learn joblib requests

# Run the dashboard
streamlit run app.py
```

### Full Pipeline (from scratch)

```bash
# Step 1: Clean data + fetch weather + compute risk scores
python scripts/clean_data.py

# Step 2: Train ML models
python scripts/train_model.py

# Step 3: Generate dashboard analytics CSVs
python scripts/bigquery_dashboard.py

# Step 4: Launch dashboard
streamlit run app.py
```

---

## 🏎️ NVIDIA RAPIDS cuDF Benchmark

We leverage [NVIDIA RAPIDS cuDF](https://rapids.ai/cudf-pandas/) for GPU-accelerated data processing. The `cudf.pandas` module acts as a **zero-code-change drop-in accelerator** — the same pandas code runs on GPU automatically.

### Running the Benchmark (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/)
2. Select **T4 GPU** runtime (`Runtime → Change runtime type → T4 GPU`)
3. Upload `notebooks/cudf_benchmark.ipynb`
4. Run all cells

### What the Benchmark Does

| Step | Description |
|------|-------------|
| **Data Scaling** | Upscales the 20K-row dataset to **~1M rows** via realistic resampling |
| **CPU Run** | Runs the full cleaning pipeline with standard pandas |
| **GPU Run** | Runs the identical pipeline with `cudf.pandas` acceleration |
| **Comparison** | Reports wall-clock times and speedup factor |

### Expected Results

```
╔══════════════════════════════════════════════════╗
║         cuDF.pandas Benchmark Results            ║
╠══════════════════════════════════════════════════╣
║  Dataset Size    :  1,000,000 rows               ║
║  CPU (pandas)    :  ~45.2 seconds                ║
║  GPU (cudf)      :  ~5.8 seconds                 ║
║  Speedup         :  ~7.8x faster                 ║
╚══════════════════════════════════════════════════╝
```

> **Note:** Actual speedup depends on GPU type and data characteristics. T4 GPU typically yields 6–10x acceleration on tabular ETL workloads.

---

## 📊 Risk Score Methodology

The risk score is a **weighted composite** of 6 normalised components:

| Component | Weight | Source | Range |
|-----------|--------|--------|-------|
| Congestion Level | 40% | Traffic data | 0–100 |
| Road Capacity Utilisation | 20% | Traffic data | 0–100% |
| Precipitation (Rain) | 15% | Open-Meteo API | 0–30mm |
| Visibility (inverse) | 10% | Open-Meteo API | 200–10,000m |
| Travel Time Index | 10% | Traffic data | 1.0–1.5 |
| Incident Reports | 5% | Traffic data | 0–7 |

**Risk Tiers:**
- 🟢 **Low** (0–25): Safe commute, minimal delays
- 🟡 **Moderate** (25–50): Minor congestion, plan extra time
- 🟠 **High** (50–75): Significant delays, consider alternatives
- 🔴 **Critical** (75–100): Severe congestion, avoid if possible

---

## ☁️ BigQuery Integration

The platform supports **Google BigQuery** for scalable cloud analytics. When BigQuery is available, all dashboard queries run server-side on Google's infrastructure.

### Setup (Optional)

```bash
# Option A: Application Default Credentials
gcloud auth application-default login

# Option B: Service Account
export GCP_KEY_PATH=/path/to/service-account-key.json
export GCP_PROJECT=your-project-id

# Option C: Offline Mode (no BigQuery needed)
export OFFLINE_MODE=1
```

The dashboard works perfectly in **offline mode** (default) — all analytics run via pandas locally.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit, Plotly, Custom CSS |
| **ML** | scikit-learn (RandomForest), joblib |
| **Data Processing** | pandas, NumPy, NVIDIA RAPIDS cuDF |
| **Weather** | Open-Meteo API (free, no key) |
| **Cloud Analytics** | Google BigQuery |
| **Visualization** | Plotly (Gauge, Bar, Scatter, Stacked Bar) |

---

## 📋 Dataset

- **Source:** Bengaluru Traffic Dataset
- **Period:** January 2022 – August 2024
- **Routes:** 20+ major road intersections/corridors
- **Areas:** Hebbal, Indiranagar, Koramangala, Whitefield, Electronic City, Silk Board, and more
- **Features:** Traffic volume, speed, congestion, incidents, parking, public transport, weather conditions

---

## 👩‍💻 Author

**Manasa L Hegde**

- GitHub: [@Manasa-L-Hegde](https://github.com/Manasa-L-Hegde)

---

## 📄 License

This project is built for the NVIDIA RAPIDS / Google Cloud hackathon challenge. All data processing, ML, and analytics are original work.

---

<p align="center">
  <b>Built with ❤️ for smarter commutes in Bengaluru</b>
</p>
