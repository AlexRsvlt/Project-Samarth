# 🌾 Project Samarth - Agricultural Intelligence System

**AI-powered Q&A system for cross-ministry agricultural data analysis in India**

> Breaking data silos between IMD and MoAFW to deliver actionable insights for policymakers and researchers.

---

## 🎯 Problem Statement

Government agricultural data in India exists in **isolated silos**:

- **IMD (Meteorological Department):** Rainfall data by meteorological subdivisions  
- **MoAFW (Agriculture Ministry):** Crop data by states and districts  

➡️ These datasets use **different naming conventions and granularities**, making cross-domain analysis difficult.  

**Project Samarth** bridges this gap through:
- Intelligent geographic mapping  
- Smart data aggregation  
- Natural language querying  

---

## ✨ Key Features

- 🗺️ **Automatic Geographic Mapping** – Handles state ↔ subdivision mismatches  
- 🧮 **Smart Data Aggregation** – Merges rainfall and crop data seamlessly  
- 📊 **Statistical Analysis** – Correlations, trends, and significance testing  
- 💬 **Natural Language Queries** – Ask questions in plain English  
- 📈 **Interactive Visualizations** – Auto-generated charts and summaries  
- 🔍 **117 Years of Rainfall Data** – 1901–2017  
- ✅ **Source Attribution** – Cites IMD and MoAFW  

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CSV files: `crop_production.csv`, `rainfall_india.csv`

### Installation
```bash
git clone https://github.com/<your-username>/project-samarth.git
cd project-samarth
pip install -r requirements.txt
```
### Run
```bash
python app_enterprise.py
```
Visit http://localhost:5000
 in your browser.

### Project Structure
project_samarth/
├── data/
│   ├── crop_production.csv
│   └── rainfall_india.csv
├── templates/
│   └── index_enterprise.html
├── geo_mapper.py
├── data_aggregator.py
├── query_router.py
├── data_loader.py
├── app_enterprise.py
└── requirements.txt

### Architecture
User Query → Query Router → Geo Mapper → Smart Aggregator → Statistical Analyzer → Response Generator

### Core Modules
- geo_mapper.py – Maps rainfall subdivisions to states
- data_aggregator.py – Aggregates and analyzes data
- query_router.py – Parses and classifies natural language queries
- app_enterprise.py – Flask-based backend
- index_enterprise.html – Frontend interface

### Data Sources
1. Rainfall: Indian Meteorological Department (IMD), 1901–2017
2. Crop Production: Ministry of Agriculture & Farmers Welfare (MoAFW), 2001–2005

### Built With
- Flask – Web framework
- Pandas – Data manipulation
- NumPy – Numerical computing
- SciPy – Statistical analysis
- Chart.js – Visualization

### Acknowledgments
Project Samarth demonstrates how AI and data integration can transform government datasets into accessible, actionable intelligence for the agricultural sector.

#### Empowering policymakers. Enabling data-driven agriculture.
cd project-samarth
pip install -r requirements.txt
