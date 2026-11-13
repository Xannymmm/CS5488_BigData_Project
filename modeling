import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set chart style for better presentation in PPT
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150 

# ==========================================
# 1. Define Functions (For C's parallel testing)
# ==========================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def build_features(df):
    """Perform feature engineering."""
    print("Starting feature engineering...")
    start_time = time.time()
    
    # 1. Time features
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 2. Spatial distance feature
    df['distance_km'] = haversine_km(
        df['pickup_latitude'], df['pickup_longitude'], 
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    # 3. Spatial clustering feature (K-Means)
    coords = df[['pickup_latitude', 'pickup_longitude']].values
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    df['pickup_cluster'] = kmeans.fit_predict(coords)
    
    print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds")
    return df

def train_and_evaluate(df, n_jobs=1, n_estimators=20):
    """
    Train Random Forest and evaluate. 
    Exposed 'n_jobs' for C's parallel benchmarking.
    """
    print(f"\nStarting model training (n_jobs={n_jobs}, n_estimators={n_estimators})...")
    
    features = ['hour', 'dayofweek', 'is_weekend', 'distance_km', 'pickup_cluster']
    X = df[features]
    y = np.log1p(df['trip_duration'])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=20,               
        n_jobs=n_jobs, 
        random_state=42
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Model training completed in {train_time:.2f} seconds")
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"Validation RMSLE: {rmse:.4f}")
    
    return model, features, train_time, y_val, preds

# ==========================================
# 2. Main Execution Block
# ==========================================
if __name__ == "__main__":
    
    # --- IMPORTANT: Update this path to your local file location ---
    file_path = 'train_clean_v2.parquet' 
    
    print("Loading data...")
    df_raw = pd.read_parquet(file_path)
    print(f"Data loaded successfully. Total rows: {len(df_raw)}")
    
    # Execute Feature Engineering
    df_feat = build_features(df_raw)
    
    # Execute Model Training (Single Core baseline)
    model, feature_cols, time_taken, y_val, preds = train_and_evaluate(df_feat, n_jobs=1, n_estimators=20)
    
    # ==========================================
    # 3. Generate Visualizations for PPT
    # ==========================================
    print("\nGenerating charts...")
    
    # Chart 1: Feature Importance
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importance (Random Forest)', fontsize=14)
    plt.xlabel('Relative Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close() # Close plot to save memory

    # Chart 2: Actual vs Predicted
    sample_idx = np.random.choice(len(y_val), 10000, replace=False)
    sample_y = y_val.iloc[sample_idx]
    sample_preds = preds[sample_idx]

    plt.figure(figsize=(6, 6))
    plt.scatter(sample_y, sample_preds, alpha=0.2, color='blue', s=10)
    plt.plot([sample_y.min(), sample_y.max()], [sample_y.min(), sample_y.max()], 'r--', lw=2)
    plt.xlabel('Actual log(Trip Duration)', fontsize=12)
    plt.ylabel('Predicted log(Trip Duration)', fontsize=12)
    plt.title('Actual vs Predicted Trip Duration', fontsize=14)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=300)
    plt.close()
    
    print("Charts saved successfully as 'feature_importance.png' and 'actual_vs_predicted.png'.")
    
