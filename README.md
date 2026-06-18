# 🚀 AutoML Pro – AI-Powered Machine Learning Intelligence Platform

AutoML Pro is an end-to-end Machine Learning dashboard built with **Streamlit**, designed to simplify the complete ML workflow from data exploration to model evaluation. The platform combines traditional AutoML capabilities with AI-powered dataset auditing and profiling using **Google Gemini**, enabling users to identify data quality issues before training models.

## 🌐 Live Application

Live Demo: https://automl-dharshan2412.streamlit.app/

---

## ✨ Features

### 📊 Exploratory Data Analysis (EDA)

* Interactive dataset preview
* Dynamic histograms
* Correlation heatmaps
* Numerical and categorical data exploration
* Instant visual insights using Plotly

### 🛡️ AI-Powered Trust Auditor

Before model training, the Trust Auditor automatically performs:

* Missing value analysis
* Duplicate row detection
* Class imbalance detection
* High-cardinality feature identification
* Constant column detection
* Target leakage detection
* Sample size evaluation

The findings are analyzed by **Google Gemini**, which generates:

* Dataset health verdict
* Risk prioritization
* Actionable recommendations
* Training readiness assessment

### 📋 Intelligent Data Profiling

Automatically generates a comprehensive dataset profile:

* Missing value percentages
* Column cardinality analysis
* Skewness detection
* Outlier identification
* Duplicate analysis
* Dataset health indicators

Gemini then produces a concise AI-generated narrative summary explaining the overall condition of the dataset.

### 🤖 Automated Model Training

Supports both Classification and Regression workflows.

#### Classification Models

* Logistic Regression
* Random Forest Classifier
* Decision Tree Classifier
* K-Nearest Neighbors
* XGBoost Classifier

#### Regression Models

* Linear Regression
* Random Forest Regressor
* Decision Tree Regressor
* K-Nearest Neighbors Regressor
* XGBoost Regressor

### 📈 Model Performance Dashboard

* Model leaderboard
* Automatic best-model selection
* Performance comparison charts
* Accuracy-based ranking
* Download trained models as `.pkl`

### 🔍 Feature Importance Analysis

For supported tree-based models:

* Feature ranking
* Importance visualization
* Top feature identification

### ⚡ AI Insights

Generate AI-powered insights about:

* Dataset quality
* Risk assessment
* Data health summaries
* Training recommendations

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Data Processing

* Pandas
* NumPy

### Visualization

* Plotly

### Machine Learning

* Scikit-Learn
* XGBoost

### AI Integration

* Google Gemini API

### Reporting

* ReportLab

### Model Serialization

* Joblib

---

## 📂 Project Structure

```text
Auto_ml/
│
├── data/
│   └── Sample datasets
│
├── src/
│   ├── main.py
│   ├── ml_utility.py
│   ├── audit_agent.py
│   ├── profiling_agent.py
│   └── report_generator.py
│
├── .env.example
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Dharshang2412/Auto_ml.git
cd Auto_ml
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Example configuration file:

```env
GEMINI_API_KEY=YOUR_API_KEY_HERE
```

---

## ▶️ Run Application

```bash
https://automl-dharshan2412.streamlit.app/
```

---

## 📸 Workflow

1. Upload a CSV or Excel dataset
2. Explore data using EDA tools
3. Run Trust Auditor for data quality assessment
4. Generate AI-powered profiling report
5. Train multiple ML models automatically
6. Compare model performance
7. Analyze feature importance
8. Download the best-performing model

---

## 🎯 Key Highlights

* End-to-End AutoML Pipeline
* AI-Powered Dataset Auditing
* Intelligent Data Profiling
* Automated Model Benchmarking
* Interactive Visual Analytics
* Gemini-Powered Recommendations
* Classification & Regression Support
* Streamlit-Based User Interface


## 👨‍💻 Author

**Dharshan G**

GitHub: https://github.com/Dharshang2412

---

## 📄 License

This project is licensed under the MIT License.
