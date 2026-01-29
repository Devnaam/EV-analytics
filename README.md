# 🚗 Electric Vehicle Analytics & Performance Prediction

A comprehensive machine learning project that analyzes electric vehicle data and predicts performance metrics using advanced regression models.

## 📊 Project Overview

This project leverages data analytics and machine learning to analyze the Electric Vehicle Population dataset from Kaggle. It predicts vehicle electric range based on manufacturer characteristics, model year, and market trends.

### Key Highlights
- **Dataset**: 36,590 electric vehicles (2016-2021)
- **Best Model**: XGBoost Regressor with **94.7% accuracy (R²)**
- **Prediction Error**: Mean Absolute Error of **8.56 miles**
- **Manufacturers Analyzed**: 15 major EV manufacturers including Tesla, Nissan, Chevrolet, etc.

---

## 🎯 Objectives

1. Analyze electric vehicle data using statistical and analytical techniques
2. Clean and preprocess raw EV data for machine learning
3. Identify key factors influencing EV performance
4. Build predictive models for EV electric range
5. Evaluate and compare different machine learning algorithms
6. Visualize EV performance trends and insights

---

## 📁 Project Structure

```
EV-Analytics-Project/
│
├── data/
│   └── Electric_Vehicle_Population_Data.csv
│
├── visualizations/
│   ├── 01_distributions.png
│   ├── 02_range_by_year.png
│   ├── 03_manufacturer_analysis.png
│   ├── 04_range_by_manufacturer.png
│   ├── 05_boxplots.png
│   ├── 06_model_comparison.png
│   ├── 07_actual_vs_predicted.png
│   ├── 08_residual_plots.png
│   └── 09_feature_importance.png
│
├── models/
│   ├── best_model_xgboost.pkl
│   └── model_comparison_results.csv
│
├── ev_analytics.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies Used

### Programming Language
- **Python 3.10+**

### Libraries
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, xgboost
- **Development**: Jupyter Notebook (optional)

### Machine Learning Models
1. Linear Regression (Baseline)
2. Random Forest Regressor
3. XGBoost Regressor ⭐ (Best Performance)

---

## 📈 Dataset Information

**Source**: [Kaggle - Electric Vehicle Population Data](https://www.kaggle.com/datasets/ratikkakkar/electric-vehicle-population-data)

### Key Attributes
- Manufacturer / Make
- Model & Model Year
- Electric Range
- Electric Vehicle Type
- State & County
- Vehicle specifications

### Dataset Statistics (After Cleaning)
- **Total Vehicles**: 36,590
- **Model Years**: 2016 - 2021
- **Average Range**: 215.72 miles
- **Range Std Dev**: 61.94 miles
- **Min Range**: 57 miles
- **Max Range**: 337 miles

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/EV-Analytics-Project.git
cd EV-Analytics-Project
```

2. **Create virtual environment**

```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download dataset**

Download the dataset from Kaggle

Place Electric_Vehicle_Population_Data.csv in the data/ folder

5. **Run the project**

```bash
python ev_analytics.py
```

---

## 📊 Project Methodology
Phase 1: Data Collection
Loaded Electric Vehicle Population dataset

Initial inspection of 112,634 records

Phase 2: Data Cleaning
Removed duplicates and missing values

Filtered to Battery Electric Vehicles (BEV) only

Filtered model years 2016+

Removed extreme outliers using IQR method

Final dataset: 36,590 samples

Phase 3: Exploratory Data Analysis
Distribution analysis of electric range

Trend analysis by model year

Manufacturer comparison

Statistical summaries

Generated 5 EDA visualizations

Phase 4: Feature Engineering
Created new features:

Vehicle_Age: Current year - Model Year

Years_Since_2016: Normalized year feature

Make_Encoded: Label-encoded manufacturer

Manufacturer_Tier: Categorized by average range (Low/Medium/High)

Manufacturer_Market_Share: Percentage of dataset per manufacturer

Phase 5: Model Development
Trained three regression models:

Linear Regression: Baseline model

Random Forest: Ensemble tree-based model

XGBoost: Gradient boosting model

Phase 6: Model Evaluation
Split data: 80% training, 20% testing

Evaluation metrics: R² Score, MAE, RMSE

Feature importance analysis

Residual analysis

🏆 Results
Model Performance Comparison
Model	R² Score	MAE (miles)	RMSE (miles)
Linear Regression	0.8110	21.78	26.70
Random Forest	0.9467	8.56	14.19
XGBoost ⭐	0.9468	8.56	14.17
Best Model: XGBoost
Accuracy: 94.68% (R² Score)

Average Prediction Error: 8.56 miles

Performance: Excellent generalization on test data

Feature Importance
Most influential factors in predicting EV range:

Manufacturer identity (Make_Encoded)

Manufacturer tier (range capability category)

Model year progression (Years_Since_2016)

Market share of manufacturer

📊 Key Insights
Range Evolution: Electric vehicle range has improved steadily from 2016 to 2021

Manufacturer Impact: Tesla leads in average range (300+ miles), followed by Chevrolet and Jaguar

Market Trends: Battery Electric Vehicles (BEVs) dominate the market over PHEVs

Prediction Accuracy: XGBoost can predict EV range within ±8.56 miles on average

📸 Visualizations
The project generates 9 professional visualizations:

Distribution Analysis: Range and model year distributions

Trend Analysis: Average range by year with standard deviation

Manufacturer Analysis: Top manufacturers by count and average range

Range Distribution: Comparison across top 5 manufacturers

Box Plots: Outlier detection and range by year

Model Comparison: R², MAE, RMSE comparison charts

Actual vs Predicted: Scatter plots for all three models

Residual Plots: Error distribution analysis

Feature Importance: Importance ranking for tree-based models

🔮 Future Enhancements
Integration of real-time vehicle telemetry data

Battery health and degradation prediction

Cost and emission impact analysis

Deployment as a web application using Flask/Streamlit

Deep learning models (Neural Networks)

Time series forecasting for future range improvements

📝 Limitations
Dataset limited to vehicles registered up to 2021

No real-time sensor or driving condition data

Battery degradation not considered

Limited to top 15 manufacturers

Geographic data not fully utilized

👨‍💻 Author
Your Name

GitHub: @yourusername

LinkedIn: Your LinkedIn

Email: your.email@example.com

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Dataset: Kaggle - Electric Vehicle Population Data

Libraries: scikit-learn, XGBoost, pandas, matplotlib, seaborn

Guidance: Final Year Project Supervisor

📚 References
Scikit-learn Documentation: https://scikit-learn.org/

XGBoost Documentation: https://xgboost.readthedocs.io/

Electric Vehicle Market Research Papers

Kaggle Dataset: Electric Vehicle Population Data

⭐ If you found this project useful, please consider giving it a star!

---
