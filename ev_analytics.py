"""
Electric Vehicle Analytics & Performance Prediction
Final Year Project
Dataset: Electric Vehicle Population Data (Kaggle)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
import os
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("🚗 ELECTRIC VEHICLE ANALYTICS & PERFORMANCE PREDICTION 🚗")
print("="*70)

# ============================================
# PHASE 1: DATA LOADING & INSPECTION
# ============================================

def load_data(filepath):
    """Load the EV dataset"""
    print("\n📂 PHASE 1: DATA LOADING & INSPECTION")
    print("-" * 70)
    
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df

def inspect_data(df):
    """Initial data inspection"""
    print("\n📋 Dataset Overview:")
    print(f"Columns: {list(df.columns)}\n")
    
    print("First 5 rows:")
    print(df.head())
    
    print("\n📊 Data Types:")
    print(df.dtypes)
    
    print("\n🔍 Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print(f"\n🔄 Duplicate Rows: {df.duplicated().sum()}")
    
    print("\n📈 Statistical Summary (Numerical Columns):")
    print(df.describe())
    
    return df

# ============================================
# PHASE 2: DATA CLEANING & PREPROCESSING
# ============================================

def clean_data(df):
    """Clean and preprocess the dataset"""
    print("\n" + "="*70)
    print("🧹 PHASE 2: DATA CLEANING & PREPROCESSING")
    print("-" * 70)
    
    df_cleaned = df.copy()
    initial_rows = len(df_cleaned)
    
    # 1. Remove duplicates
    df_cleaned = df_cleaned.drop_duplicates()
    print(f"\n✅ Removed {initial_rows - len(df_cleaned)} duplicate rows")
    
    # 2. Focus on Battery Electric Vehicles (BEV) - they have better data
    if 'Electric Vehicle Type' in df_cleaned.columns:
        print(f"\nOriginal EV types distribution:")
        print(df_cleaned['Electric Vehicle Type'].value_counts())
        df_cleaned = df_cleaned[df_cleaned['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)']
        print(f"✅ Filtered to Battery Electric Vehicles (BEV) only")
    
    # 3. Filter for vehicles with valid Electric Range (> 0)
    if 'Electric Range' in df_cleaned.columns:
        before = len(df_cleaned)
        df_cleaned = df_cleaned[df_cleaned['Electric Range'] > 0]
        print(f"✅ Removed {before - len(df_cleaned)} vehicles with 0 electric range")
    
    # 4. Filter for recent model years (2016 onwards for better data quality)
    if 'Model Year' in df_cleaned.columns:
        before = len(df_cleaned)
        df_cleaned = df_cleaned[df_cleaned['Model Year'] >= 2016]
        print(f"✅ Filtered to Model Year >= 2016 ({len(df_cleaned)} vehicles)")
    
    # 5. Keep only top manufacturers (to ensure enough samples per category)
    if 'Make' in df_cleaned.columns:
        top_makes = df_cleaned['Make'].value_counts().head(15).index
        df_cleaned = df_cleaned[df_cleaned['Make'].isin(top_makes)]
        print(f"✅ Kept top 15 manufacturers: {list(top_makes)}")
    
    # 6. Select relevant columns
    columns_to_keep = ['Make', 'Model', 'Model Year', 'Electric Range', 
                       'Electric Vehicle Type', 'State', 'County']
    
    columns_to_keep = [col for col in columns_to_keep if col in df_cleaned.columns]
    df_cleaned = df_cleaned[columns_to_keep]
    
    # 7. Handle missing values
    df_cleaned = df_cleaned.dropna(subset=['Electric Range', 'Model Year', 'Make'])
    print(f"✅ Removed rows with missing critical values")
    
    # 8. Remove outliers using IQR method (only extreme outliers)
    def remove_outliers_iqr(df, column, multiplier=3):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        before = len(df)
        df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        removed = before - len(df_filtered)
        if removed > 0:
            print(f"✅ Removed {removed} extreme outliers from {column}")
        return df_filtered
    
    if 'Electric Range' in df_cleaned.columns:
        df_cleaned = remove_outliers_iqr(df_cleaned, 'Electric Range', multiplier=3)
    
    print(f"\n✅ Final cleaned dataset shape: {df_cleaned.shape}")
    print(f"✅ Model years: {df_cleaned['Model Year'].min()} - {df_cleaned['Model Year'].max()}")
    print(f"✅ Number of unique manufacturers: {df_cleaned['Make'].nunique()}")
    
    return df_cleaned

# ============================================
# PHASE 3: EXPLORATORY DATA ANALYSIS
# ============================================

def perform_eda(df):
    """Perform comprehensive EDA"""
    print("\n" + "="*70)
    print("📊 PHASE 3: EXPLORATORY DATA ANALYSIS")
    print("-" * 70)
    
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Distribution of Electric Range
    print("\n📈 Creating visualizations...")
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['Electric Range'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    plt.xlabel('Electric Range (miles)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Electric Range', fontsize=14, fontweight='bold')
    plt.axvline(df['Electric Range'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {df["Electric Range"].mean():.1f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(df['Model Year'], bins=len(df['Model Year'].unique()), 
             color='#e74c3c', edgecolor='black', alpha=0.7)
    plt.xlabel('Model Year', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Model Year', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/01_distributions.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 01_distributions.png")
    plt.close()
    
    # 2. Range by Model Year
    plt.figure(figsize=(12, 6))
    year_stats = df.groupby('Model Year')['Electric Range'].agg(['mean', 'median', 'std']).reset_index()
    
    plt.plot(year_stats['Model Year'], year_stats['mean'], marker='o', linewidth=2.5, 
             markersize=10, color='#2ecc71', label='Mean Range')
    plt.plot(year_stats['Model Year'], year_stats['median'], marker='s', linewidth=2.5, 
             markersize=10, color='#e74c3c', label='Median Range', linestyle='--')
    plt.fill_between(year_stats['Model Year'], 
                     year_stats['mean'] - year_stats['std'],
                     year_stats['mean'] + year_stats['std'],
                     alpha=0.2, color='#2ecc71')
    
    plt.xlabel('Model Year', fontsize=12)
    plt.ylabel('Electric Range (miles)', fontsize=12)
    plt.title('Average Electric Range by Model Year (with Std Dev)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/02_range_by_year.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 02_range_by_year.png")
    plt.close()
    
    # 3. Top manufacturers by count and average range
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Count
    top_makes = df['Make'].value_counts().head(10)
    axes[0].barh(top_makes.index, top_makes.values, color='#9b59b6')
    axes[0].set_xlabel('Number of Vehicles', fontsize=12)
    axes[0].set_ylabel('Manufacturer', fontsize=12)
    axes[0].set_title('Top 10 EV Manufacturers by Count', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    
    # Average Range
    avg_range_by_make = df.groupby('Make')['Electric Range'].mean().sort_values(ascending=True).tail(10)
    axes[1].barh(avg_range_by_make.index, avg_range_by_make.values, color='#e67e22')
    axes[1].set_xlabel('Average Electric Range (miles)', fontsize=12)
    axes[1].set_ylabel('Manufacturer', fontsize=12)
    axes[1].set_title('Top 10 Manufacturers by Average Range', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/03_manufacturer_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 03_manufacturer_analysis.png")
    plt.close()
    
    # 4. Range distribution by top manufacturers
    top_5_makes = df['Make'].value_counts().head(5).index
    df_top5 = df[df['Make'].isin(top_5_makes)]
    
    plt.figure(figsize=(12, 6))
    for make in top_5_makes:
        data = df_top5[df_top5['Make'] == make]['Electric Range']
        plt.hist(data, bins=20, alpha=0.5, label=make, edgecolor='black')
    
    plt.xlabel('Electric Range (miles)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Electric Range Distribution by Top 5 Manufacturers', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/04_range_by_manufacturer.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 04_range_by_manufacturer.png")
    plt.close()
    
    # 5. Box plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].boxplot(df['Electric Range'].dropna(), vert=True)
    axes[0].set_ylabel('Electric Range (miles)', fontsize=12)
    axes[0].set_title('Box Plot: Electric Range', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plot by year
    years = sorted(df['Model Year'].unique())
    data_by_year = [df[df['Model Year'] == year]['Electric Range'].values for year in years]
    axes[1].boxplot(data_by_year, labels=years)
    axes[1].set_xlabel('Model Year', fontsize=12)
    axes[1].set_ylabel('Electric Range (miles)', fontsize=12)
    axes[1].set_title('Electric Range by Model Year', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/05_boxplots.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 05_boxplots.png")
    plt.close()
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"Total vehicles analyzed: {len(df)}")
    print(f"Average Electric Range: {df['Electric Range'].mean():.2f} miles")
    print(f"Median Electric Range: {df['Electric Range'].median():.2f} miles")
    print(f"Range Std Dev: {df['Electric Range'].std():.2f} miles")
    print(f"Min Range: {df['Electric Range'].min():.0f} miles")
    print(f"Max Range: {df['Electric Range'].max():.0f} miles")
    print(f"Most common manufacturer: {df['Make'].mode()[0]}")
    print(f"Model years covered: {df['Model Year'].min()} - {df['Model Year'].max()}")
    print(f"Number of different manufacturers: {df['Make'].nunique()}")
    print(f"Number of different models: {df['Model'].nunique()}")
    
    print("\n✅ EDA completed! Check 'visualizations' folder.")

# ============================================
# PHASE 4: FEATURE ENGINEERING
# ============================================

def feature_engineering(df):
    """Create new features and prepare data for modeling"""
    print("\n" + "="*70)
    print("⚙️ PHASE 4: FEATURE ENGINEERING")
    print("-" * 70)
    
    df_featured = df.copy()
    
    # 1. Create age of vehicle feature
    current_year = 2026
    df_featured['Vehicle_Age'] = current_year - df_featured['Model Year']
    print("✅ Created feature: Vehicle_Age")
    
    # 2. Years since 2016 (normalized year feature)
    df_featured['Years_Since_2016'] = df_featured['Model Year'] - 2016
    print("✅ Created feature: Years_Since_2016")
    
    # 3. Encode manufacturer
    le_make = LabelEncoder()
    df_featured['Make_Encoded'] = le_make.fit_transform(df_featured['Make'])
    print("✅ Encoded feature: Make")
    
    # 4. Create manufacturer tier based on average range
    make_avg_range = df_featured.groupby('Make')['Electric Range'].mean()
    make_tier = {}
    for make, avg_range in make_avg_range.items():
        if avg_range >= 200:
            make_tier[make] = 2  # High Range
        elif avg_range >= 100:
            make_tier[make] = 1  # Medium Range
        else:
            make_tier[make] = 0  # Low Range
    
    df_featured['Manufacturer_Tier'] = df_featured['Make'].map(make_tier)
    print("✅ Created feature: Manufacturer_Tier (0=Low, 1=Medium, 2=High)")
    
    # 5. Create manufacturer market share feature
    make_counts = df_featured['Make'].value_counts()
    make_share = (make_counts / len(df_featured)) * 100
    df_featured['Manufacturer_Market_Share'] = df_featured['Make'].map(make_share)
    print("✅ Created feature: Manufacturer_Market_Share (%)")
    
    print(f"\n✅ Feature engineering completed!")
    print(f"Total features now: {df_featured.shape[1]}")
    print(f"\nFeature list: {list(df_featured.columns)}")
    
    return df_featured

# ============================================
# PHASE 5: MODEL DEVELOPMENT
# ============================================

def prepare_modeling_data(df):
    """Prepare features and target for modeling"""
    print("\n" + "="*70)
    print("🤖 PHASE 5: MODEL DEVELOPMENT")
    print("-" * 70)
    
    # Select features for modeling (NO Model Year or Vehicle_Age to avoid perfect correlation)
    feature_columns = ['Years_Since_2016', 'Make_Encoded', 'Manufacturer_Tier', 
                       'Manufacturer_Market_Share']
    
    X = df[feature_columns]
    y = df['Electric Range']
    
    print(f"\n📊 Features selected for modeling:")
    for col in feature_columns:
        print(f"   - {col}")
    
    print(f"\n📊 Target variable: Electric Range")
    print(f"📊 Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Check variance in features
    print(f"\n📊 Feature variance check:")
    for col in feature_columns:
        print(f"   {col}: std={X[col].std():.3f}, range=[{X[col].min():.1f}, {X[col].max():.1f}]")
    
    print(f"\n📊 Target variance: std={y.std():.3f}, range=[{y.min():.1f}, {y.max():.1f}]")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"\n✅ Train set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models"""
    print("\n" + "-" * 70)
    print("🏋️ Training Models...")
    print("-" * 70)
    
    models = {}
    results = {}
    
    # 1. Linear Regression
    print("\n1️⃣ Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    models['Linear Regression'] = lr_model
    results['Linear Regression'] = {
        'R2 Score': r2_score(y_test, lr_pred),
        'MAE': mean_absolute_error(y_test, lr_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'predictions': lr_pred
    }
    print(f"✅ Linear Regression trained - R²: {results['Linear Regression']['R2 Score']:.4f}")
    
    # 2. Random Forest Regressor
    print("\n2️⃣ Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    models['Random Forest'] = rf_model
    results['Random Forest'] = {
        'R2 Score': r2_score(y_test, rf_pred),
        'MAE': mean_absolute_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'predictions': rf_pred,
        'feature_importance': rf_model.feature_importances_
    }
    print(f"✅ Random Forest trained - R²: {results['Random Forest']['R2 Score']:.4f}")
    
    # 3. XGBoost Regressor
    print("\n3️⃣ Training XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {
        'R2 Score': r2_score(y_test, xgb_pred),
        'MAE': mean_absolute_error(y_test, xgb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
        'predictions': xgb_pred,
        'feature_importance': xgb_model.feature_importances_
    }
    print(f"✅ XGBoost trained - R²: {results['XGBoost']['R2 Score']:.4f}")
    
    return models, results

# ============================================
# PHASE 6: MODEL EVALUATION & VISUALIZATION
# ============================================

def evaluate_models(results, y_test, feature_columns):
    """Evaluate and compare model performance"""
    print("\n" + "="*70)
    print("📈 PHASE 6: MODEL EVALUATION & RESULTS")
    print("-" * 70)
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'R² Score': [results[model]['R2 Score'] for model in results.keys()],
        'MAE': [results[model]['MAE'] for model in results.keys()],
        'RMSE': [results[model]['RMSE'] for model in results.keys()]
    })
    
    print("\n📊 MODEL PERFORMANCE COMPARISON:")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70)
    
    # Find best model
    best_model = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']
    best_r2 = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'R² Score']
    best_mae = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'MAE']
    best_rmse = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'RMSE']
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   R² Score: {best_r2:.4f}")
    print(f"   MAE: {best_mae:.4f} miles")
    print(f"   RMSE: {best_rmse:.4f} miles")
    
    # Visualize model comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models_list = list(results.keys())
    r2_scores = [results[model]['R2 Score'] for model in models_list]
    mae_scores = [results[model]['MAE'] for model in models_list]
    rmse_scores = [results[model]['RMSE'] for model in models_list]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    axes[0].bar(models_list, r2_scores, color=colors)
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('R² Score Comparison (Higher is Better)', fontsize=14, fontweight='bold')
    axes[0].set_ylim([min(r2_scores) - 0.05, 1])
    for i, v in enumerate(r2_scores):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    
    axes[1].bar(models_list, mae_scores, color=colors)
    axes[1].set_ylabel('Mean Absolute Error (miles)', fontsize=12)
    axes[1].set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    for i, v in enumerate(mae_scores):
        axes[1].text(i, v + 1, f'{v:.2f}', ha='center', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    
    axes[2].bar(models_list, rmse_scores, color=colors)
    axes[2].set_ylabel('Root Mean Squared Error (miles)', fontsize=12)
    axes[2].set_title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    for i, v in enumerate(rmse_scores):
        axes[2].text(i, v + 1, f'{v:.2f}', ha='center', fontweight='bold')
    axes[2].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig('visualizations/06_model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: 06_model_comparison.png")
    plt.close()
    
    # Actual vs Predicted plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (model_name, color) in enumerate(zip(models_list, colors)):
        predictions = results[model_name]['predictions']
        axes[idx].scatter(y_test, predictions, alpha=0.6, color=color, edgecolors='black', s=50)
        axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                       'r--', lw=3, label='Perfect Prediction')
        axes[idx].set_xlabel('Actual Range (miles)', fontsize=11)
        axes[idx].set_ylabel('Predicted Range (miles)', fontsize=11)
        axes[idx].set_title(f'{model_name}\nR² = {results[model_name]["R2 Score"]:.3f}, MAE = {results[model_name]["MAE"]:.2f}', 
                           fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/07_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 07_actual_vs_predicted.png")
    plt.close()
    
    # Residual plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (model_name, color) in enumerate(zip(models_list, colors)):
        predictions = results[model_name]['predictions']
        residuals = y_test - predictions
        
        axes[idx].scatter(predictions, residuals, alpha=0.6, color=color, edgecolors='black', s=50)
        axes[idx].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[idx].set_xlabel('Predicted Range (miles)', fontsize=11)
        axes[idx].set_ylabel('Residuals (miles)', fontsize=11)
        axes[idx].set_title(f'{model_name} - Residual Plot', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/08_residual_plots.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 08_residual_plots.png")
    plt.close()
    
    # Feature importance for tree-based models
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random Forest feature importance
    rf_importance = results['Random Forest']['feature_importance']
    rf_features = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': rf_importance
    }).sort_values('Importance', ascending=True)
    
    axes[0].barh(rf_features['Feature'], rf_features['Importance'], color='#2ecc71')
    axes[0].set_xlabel('Importance', fontsize=12)
    axes[0].set_title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
    
    # XGBoost feature importance
    xgb_importance = results['XGBoost']['feature_importance']
    xgb_features = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': xgb_importance
    }).sort_values('Importance', ascending=True)
    
    axes[1].barh(xgb_features['Feature'], xgb_features['Importance'], color='#e74c3c')
    axes[1].set_xlabel('Importance', fontsize=12)
    axes[1].set_title('XGBoost - Feature Importance', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/09_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 09_feature_importance.png")
    plt.close()
    
    return comparison_df, best_model

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function"""
    
    # Phase 1: Load Data
    df = load_data('data/Electric_Vehicle_Population_Data.csv')
    df = inspect_data(df)
    
    # Phase 2: Clean Data
    df_cleaned = clean_data(df)
    
    # Check if we have enough data
    if len(df_cleaned) < 500:
        print(f"\n⚠️ WARNING: Dataset too small ({len(df_cleaned)} samples)")
        print("⚠️ This may lead to overfitting. Consider relaxing filter criteria.")
    
    # Phase 3: EDA
    perform_eda(df_cleaned)
    
    # Phase 4: Feature Engineering
    df_featured = feature_engineering(df_cleaned)
    
    # Phase 5: Prepare and Train Models
    X_train, X_test, y_train, y_test, feature_columns = prepare_modeling_data(df_featured)
    models, results = train_models(X_train, X_test, y_train, y_test)
    
    # Phase 6: Evaluate Models
    comparison_df, best_model = evaluate_models(results, y_test, feature_columns)
    
    # Save results to CSV
    comparison_df.to_csv('models/model_comparison_results.csv', index=False)
    print("\n✅ Results saved to: models/model_comparison_results.csv")
    
    # Save the best model
    import pickle
    best_model_obj = models[best_model]
    with open(f'models/best_model_{best_model.replace(" ", "_").lower()}.pkl', 'wb') as f:
        pickle.dump(best_model_obj, f)
    print(f"✅ Best model saved to: models/best_model_{best_model.replace(' ', '_').lower()}.pkl")
    
    print("\n" + "="*70)
    print("🎉 PROJECT COMPLETED SUCCESSFULLY! 🎉")
    print("="*70)
    print("\n📁 Output Files Generated:")
    print("   📊 9 Visualizations in 'visualizations/' folder")
    print("   📈 Model comparison results in 'models/' folder")
    print("   🤖 Best trained model saved in 'models/' folder")
    print("\n" + "="*70)
    
    # Final model interpretation
    print("\n📝 MODEL INTERPRETATION:")
    print(f"   - Best performing model: {best_model}")
    print(f"   - Most important features: {feature_columns}")
    print(f"   - The model can predict EV range with {comparison_df[comparison_df['Model']==best_model]['R² Score'].values[0]*100:.1f}% accuracy")
    print("="*70)

if __name__ == "__main__":
    main()
