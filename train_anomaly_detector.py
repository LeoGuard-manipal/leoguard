"""
LeoGuard - Anomaly Detection Model
Detects unusual sensor readings that indicate potential industrial emissions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

# Input/Output paths
DATA_FILE = 'data/synthetic_sensor_data.csv'
MODEL_OUTPUT_DIR = 'models/anomaly_detection'
VISUALIZATION_OUTPUT_DIR = 'visualizations'

# Model configuration
CONTAMINATION_RATE = 0.05  # Expect ~5% of data to be anomalies
RANDOM_STATE = 42  # For reproducibility

# Features to use for anomaly detection
FEATURES = ['co2_ppm', 'temperature_c', 'humidity_percent']

# Thresholds for alerts
CO2_SPIKE_THRESHOLD = 480  # ppm - Alert if CO2 > this value
CO2_RATE_OF_CHANGE_THRESHOLD = 30  # ppm/hour - Alert if increases this fast

# ============================================
# DATA LOADING
# ============================================

def load_data(filepath):
    """Load sensor data from CSV"""
    
    print("\n📂 Loading data...")
    
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found at {filepath}")
        print("   Make sure you ran: python generate_synthetic_data.py")
        return None
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ Data loaded successfully!")
    print(f"   Records: {len(df)}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Columns: {', '.join(df.columns)}")
    
    return df

# ============================================
# DATA PREPROCESSING
# ============================================

def preprocess_data(df):
    """
    Prepare data for model training
    
    Why preprocessing?
    - Different features have different scales
    - CO2: 0-600 ppm
    - Temperature: 20-40 °C
    - Humidity: 0-100 %
    
    Without scaling, CO2 (larger numbers) would dominate
    Scaling makes all features equally important
    """
    
    print("\n🔧 Preprocessing data...")
    
    # Check for missing values
    missing = df[FEATURES].isnull().sum()
    if missing.any():
        print(f"⚠️  Missing values found:")
        print(missing[missing > 0])
        print("   Filling with forward fill method...")
        df[FEATURES] = df[FEATURES].fillna(method='ffill')
    
    print(f"✅ Data preprocessed")
    
    return df

# ============================================
# MODEL TRAINING
# ============================================

def train_anomaly_detector(df):
    """
    Train Isolation Forest model
    
    How it works:
    1. Randomly select features and values
    2. Split data recursively
    3. Points that are "easy to isolate" = anomalies
    4. Points that are "hard to isolate" = normal
    """
    
    print("\n🤖 Training anomaly detection model...")
    print(f"   Features: {FEATURES}")
    print(f"   Samples: {len(df)}")
    print(f"   Contamination rate: {CONTAMINATION_RATE*100}%")
    
    # Extract features
    X = df[FEATURES].values
    
    # Standardize features (scaling to 0-1 range)
    # This is IMPORTANT for machine learning
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Data standardized (mean=0, std=1)")
    
    # Train Isolation Forest
    model = IsolationForest(
        contamination=CONTAMINATION_RATE,  # Expect 5% anomalies
        random_state=RANDOM_STATE,
        n_estimators=100  # Number of trees in the forest
    )
    
    # Fit the model
    predictions = model.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
    anomaly_scores = model.score_samples(X_scaled)  # Score for each point
    
    print(f"✅ Model training complete!")
    print(f"   Anomalies detected: {(predictions == -1).sum()}")
    print(f"   Normal readings: {(predictions == 1).sum()}")
    
    return model, scaler, predictions, anomaly_scores

# ============================================
# MODEL EVALUATION
# ============================================

def evaluate_model(df, predictions, anomaly_scores):
    """
    Analyze what the model found
    """
    
    print("\n📊 Model Evaluation:")
    print("="*60)
    
    # Add predictions to dataframe for analysis
    df['prediction'] = predictions  # -1 = anomaly, 1 = normal
    df['anomaly_score'] = anomaly_scores
    
    # Get anomalies
    anomalies = df[df['prediction'] == -1]
    
    print(f"\n🔴 ANOMALIES DETECTED: {len(anomalies)}")
    print(f"   Anomaly percentage: {len(anomalies)/len(df)*100:.1f}%")
    
    # Analyze anomaly characteristics
    print(f"\n📈 Anomaly Characteristics:")
    print(f"   CO2 range in anomalies:")
    print(f"      Min: {anomalies['co2_ppm'].min():.1f} ppm")
    print(f"      Max: {anomalies['co2_ppm'].max():.1f} ppm")
    print(f"      Mean: {anomalies['co2_ppm'].mean():.1f} ppm")
    
    print(f"   Temperature range in anomalies:")
    print(f"      Min: {anomalies['temperature_c'].min():.1f} °C")
    print(f"      Max: {anomalies['temperature_c'].max():.1f} °C")
    print(f"      Mean: {anomalies['temperature_c'].mean():.1f} °C")
    
    print(f"\n📈 Normal Data Characteristics:")
    normal = df[df['prediction'] == 1]
    print(f"   CO2 range in normal data:")
    print(f"      Min: {normal['co2_ppm'].min():.1f} ppm")
    print(f"      Max: {normal['co2_ppm'].max():.1f} ppm")
    print(f"      Mean: {normal['co2_ppm'].mean():.1f} ppm")
    
    print("\n" + "="*60)
    
    return df

# ============================================
# VISUALIZATION
# ============================================

def create_visualizations(df, model, scaler):
    """
    Create helpful charts to understand the model
    """
    
    print("\n📊 Creating visualizations...")
    
    # Create output directory
    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
    
    # -------- PLOT 1: CO2 Over Time with Anomalies Highlighted --------
    print("   Creating plot 1: CO2 Timeline...")
    
    fig, ax = plt.subplots(figsize=(16, 5))
    
    # Plot normal points
    normal = df[df['prediction'] == 1]
    ax.plot(normal['timestamp'], normal['co2_ppm'], 
            color='blue', alpha=0.5, linewidth=1, label='Normal')
    
    # Plot anomalies (highlighted)
    anomalies = df[df['prediction'] == -1]
    ax.scatter(anomalies['timestamp'], anomalies['co2_ppm'], 
              color='red', s=100, marker='X', label='Anomaly', zorder=5)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('CO₂ (ppm)', fontsize=12)
    ax.set_title('LeoGuard - CO₂ Anomaly Detection Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{VISUALIZATION_OUTPUT_DIR}/01_co2_anomalies_timeline.png', dpi=100, bbox_inches='tight')
    print(f"   ✅ Saved: 01_co2_anomalies_timeline.png")
    
    # -------- PLOT 2: Feature Distribution --------
    print("   Creating plot 2: Feature Distributions...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    features_to_plot = ['co2_ppm', 'temperature_c', 'humidity_percent']
    titles = ['CO₂ (ppm)', 'Temperature (°C)', 'Humidity (%)']
    
    for i, (feature, title) in enumerate(zip(features_to_plot, titles)):
        normal_vals = df[df['prediction'] == 1][feature]
        anomaly_vals = df[df['prediction'] == -1][feature]
        
        axes[i].hist(normal_vals, bins=30, alpha=0.7, color='blue', label='Normal')
        axes[i].hist(anomaly_vals, bins=30, alpha=0.7, color='red', label='Anomaly')
        axes[i].set_xlabel(title, fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].set_title(f'Distribution: {title}', fontsize=11, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{VISUALIZATION_OUTPUT_DIR}/02_feature_distributions.png', dpi=100, bbox_inches='tight')
    print(f"   ✅ Saved: 02_feature_distributions.png")
    
    # -------- PLOT 3: Anomaly Scores --------
    print("   Creating plot 3: Anomaly Scores...")
    
    fig, ax = plt.subplots(figsize=(16, 5))
    
    ax.plot(df['timestamp'], df['anomaly_score'], 
            color='purple', alpha=0.7, linewidth=1, label='Anomaly Score')
    ax.axhline(y=df[df['prediction']==-1]['anomaly_score'].max(), 
              color='red', linestyle='--', linewidth=2, label='Anomaly Threshold')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Anomaly Score', fontsize=12)
    ax.set_title('Anomaly Scores Over Time (Lower = More Anomalous)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{VISUALIZATION_OUTPUT_DIR}/03_anomaly_scores.png', dpi=100, bbox_inches='tight')
    print(f"   ✅ Saved: 03_anomaly_scores.png")
    
    # -------- PLOT 4: 3D Scatter (All Features) --------
    print("   Creating plot 4: 3D Feature Space...")
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    normal_data = df[df['prediction'] == 1]
    anomaly_data = df[df['prediction'] == -1]
    
    ax.scatter(normal_data['co2_ppm'], normal_data['temperature_c'], normal_data['humidity_percent'],
              c='blue', marker='o', s=20, alpha=0.5, label='Normal')
    ax.scatter(anomaly_data['co2_ppm'], anomaly_data['temperature_c'], anomaly_data['humidity_percent'],
              c='red', marker='X', s=100, label='Anomaly')
    
    ax.set_xlabel('CO₂ (ppm)', fontsize=10)
    ax.set_ylabel('Temperature (°C)', fontsize=10)
    ax.set_zlabel('Humidity (%)', fontsize=10)
    ax.set_title('3D Feature Space - Anomaly Detection', fontsize=12, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{VISUALIZATION_OUTPUT_DIR}/04_3d_feature_space.png', dpi=100, bbox_inches='tight')
    print(f"   ✅ Saved: 04_3d_feature_space.png")
    
    print(f"\n✅ All visualizations created in '{VISUALIZATION_OUTPUT_DIR}/' folder")

# ============================================
# MODEL SAVING
# ============================================

def save_model(model, scaler):
    """
    Save trained model and scaler to disk
    
    Why save?
    - Reuse the model later without retraining
    - Share with team
    - Deploy to production
    """
    
    print("\n💾 Saving model...")
    
    # Create output directory
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    # Save model
    model_path = f'{MODEL_OUTPUT_DIR}/isolation_forest_model.pkl'
    joblib.dump(model, model_path)
    print(f"   ✅ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = f'{MODEL_OUTPUT_DIR}/feature_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ Scaler saved: {scaler_path}")
    
    # Save model info
    info_path = f'{MODEL_OUTPUT_DIR}/model_info.txt'
    with open(info_path, 'w') as f:
        f.write("LeoGuard - Anomaly Detection Model\n")
        f.write("="*50 + "\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Algorithm: Isolation Forest\n")
        f.write(f"Contamination Rate: {CONTAMINATION_RATE*100}%\n")
        f.write(f"Features Used: {', '.join(FEATURES)}\n")
        f.write(f"Training Samples: {len(df)}\n")
    
    print(f"   ✅ Info saved: {info_path}")

# ============================================
# SAMPLE PREDICTIONS
# ============================================

def predict_new_readings(model, scaler):
    """
    Demonstrate how to use the model for predictions
    """
    
    print("\n🔮 Sample Predictions (How to use the model):")
    print("="*60)
    
    # Example sensor readings
    test_readings = [
        {'co2': 415, 'temp': 32, 'humidity': 45},  # Normal
        {'co2': 520, 'temp': 36, 'humidity': 35},  # Anomaly
        {'co2': 410, 'temp': 28, 'humidity': 50},  # Normal
    ]
    
    for i, reading in enumerate(test_readings, 1):
        # Prepare data
        sample = np.array([[reading['co2'], reading['temp'], reading['humidity']]])
        sample_scaled = scaler.transform(sample)
        
        # Predict
        prediction = model.predict(sample_scaled)[0]
        anomaly_score = model.score_samples(sample_scaled)[0]
        
        # Interpret
        label = "🔴 ANOMALY" if prediction == -1 else "🟢 NORMAL"
        
        print(f"\nExample {i}: {label}")
        print(f"   Input: CO₂={reading['co2']} ppm, Temp={reading['temp']}°C, Humidity={reading['humidity']}%")
        print(f"   Anomaly Score: {anomaly_score:.4f}")
        print(f"   Interpretation: {'Unusual pattern detected!' if prediction == -1 else 'Normal pattern'}")

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == '__main__':
    
    print("\n" + "="*60)
    print("🚀 LeoGuard - Anomaly Detection Model Training")
    print("="*60)
    
    # Step 1: Load data
    df = load_data(DATA_FILE)
    if df is None:
        exit(1)
    
    # Step 2: Preprocess
    df = preprocess_data(df)
    
    # Step 3: Train model
    model, scaler, predictions, anomaly_scores = train_anomaly_detector(df)
    
    # Step 4: Evaluate
    df = evaluate_model(df, predictions, anomaly_scores)
    
    # Step 5: Visualize
    create_visualizations(df, model, scaler)
    
    # Step 6: Save
    save_model(model, scaler)
    
    # Step 7: Demo predictions
    predict_new_readings(model, scaler)
    
    print("\n" + "="*60)
    print("✨ Model training complete!")
    print("="*60)
    print("\n📌 Next steps:")
    print("   1. Review visualizations in 'visualizations/' folder")
    print("   2. Train prediction model: python train_predictor.py")
    print("   3. Integrate models with backend API")
    print("\n")
