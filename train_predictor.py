"""
LeoGuard - Time Series Prediction Model
Predicts future CO2 levels based on historical trends
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime, timedelta

# ============================================
# CONFIGURATION
# ============================================

# Input/Output paths
DATA_FILE = 'data/synthetic_sensor_data.csv'
MODEL_OUTPUT_DIR = 'models/prediction'
VISUALIZATION_OUTPUT_DIR = 'visualizations'

# Model configuration
LAGS = 4  # Use last 4 readings (120 minutes of history with 30-min intervals)
TEST_SIZE = 0.2  # 20% for testing, 80% for training
RANDOM_STATE = 42

# Features
TARGET = 'co2_ppm'
FEATURES_FOR_PREDICTION = ['co2_ppm', 'temperature_c', 'humidity_percent']

# ============================================
# DATA LOADING
# ============================================

def load_data(filepath):
    """Load and prepare time series data"""
    
    print("\n📂 Loading data...")
    
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found at {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Data loaded!")
    print(f"   Records: {len(df)}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

# ============================================
# FEATURE ENGINEERING
# ============================================

def create_lagged_features(df, lags=4):
    """
    Create lagged features for time series prediction
    
    Example with lags=3:
    - co2_lag1: CO2 from 30 minutes ago
    - co2_lag2: CO2 from 60 minutes ago
    - co2_lag3: CO2 from 90 minutes ago
    - temp_lag1: Temperature from 30 minutes ago
    - etc.
    """
    
    print(f"\n🔧 Creating {lags} lagged features...")
    
    df_lagged = df.copy()
    
    # Create lag features for each feature
    for feature in FEATURES_FOR_PREDICTION:
        for lag in range(1, lags + 1):
            lag_col_name = f'{feature}_lag{lag}'
            df_lagged[lag_col_name] = df_lagged[feature].shift(lag)
    
    # Remove rows with NaN (first few rows will have NaN due to shifting)
    df_lagged = df_lagged.dropna()
    
    print(f"✅ Lagged features created!")
    print(f"   New features: {lags * len(FEATURES_FOR_PREDICTION)}")
    print(f"   Rows after dropping NaN: {len(df_lagged)}")
    
    return df_lagged

# ============================================
# DATA SPLITTING
# ============================================

def split_data(df, test_size=0.2):
    """
    Split data into training and testing sets
    
    Important: For time series, we DON'T randomize!
    We use temporal order:
    - Training: Earlier data
    - Testing: Later data (to simulate real prediction)
    """
    
    print(f"\n✂️  Splitting data...")
    
    # Get lag column names
    lag_cols = [col for col in df.columns if 'lag' in col]
    
    # Separate features (X) and target (y)
    X = df[lag_cols]
    y = df[TARGET]
    
    # Temporal split (NOT random!)
    split_idx = int(len(df) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"✅ Data split!")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Split date: {df['timestamp'].iloc[split_idx]}")
    
    return X_train, X_test, y_train, y_test, X.columns

# ============================================
# MODEL TRAINING
# ============================================

def train_models(X_train, y_train):
    """
    Train multiple models and compare
    
    We'll train 2 models:
    1. Linear Regression (simple, fast)
    2. Random Forest (more complex, potentially better)
    """
    
    print("\n🤖 Training prediction models...")
    
    models = {}
    
    # -------- Model 1: Linear Regression --------
    print("\n   [1/2] Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    models['linear_regression'] = lr_model
    print("        ✅ Linear Regression trained")
    
    # -------- Model 2: Random Forest --------
    print("\n   [2/2] Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1  # Use all CPU cores
    )
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model
    print("        ✅ Random Forest trained")
    
    print("\n✅ All models trained!")
    
    return models

# ============================================
# MODEL EVALUATION
# ============================================

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Evaluate all models and compare performance
    
    Metrics:
    - MAE (Mean Absolute Error): Average prediction error in ppm
    - RMSE (Root Mean Squared Error): Penalizes large errors more
    - R² Score: How well model fits (0-1, higher is better)
    """
    
    print("\n📊 Model Evaluation:")
    print("="*70)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name.upper()}")
        print("-" * 70)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Training metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Testing metrics
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"Training Metrics:")
        print(f"   MAE:  {train_mae:.2f} ppm  (Average error)")
        print(f"   RMSE: {train_rmse:.2f} ppm (Root mean squared error)")
        print(f"   R²:   {train_r2:.4f}    (Fit quality: 0-1, higher better)")
        
        print(f"\nTesting Metrics (What we actually care about):")
        print(f"   MAE:  {test_mae:.2f} ppm  (Average error)")
        print(f"   RMSE: {test_rmse:.2f} ppm (Root mean squared error)")
        print(f"   R²:   {test_r2:.4f}    (Fit quality: 0-1, higher better)")
        
        # Store results
        results[model_name] = {
            'model': model,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'y_test_pred': y_test_pred
        }
    
    print("\n" + "="*70)
    
    return results

# ============================================
# VISUALIZATION
# ============================================

def create_visualizations(results, X_test, y_test):
    """Create prediction visualization charts"""
    
    print("\n📊 Creating visualizations...")
    
    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
    
    # -------- PLOT 1: Predictions vs Actual (All Models) --------
    print("   Creating plot 1: Predictions vs Actual...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    test_indices = range(len(y_test))
    
    for idx, (model_name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        y_pred = result['y_test_pred']
        mae = result['test_mae']
        rmse = result['test_rmse']
        r2 = result['test_r2']
        
        # Plot actual vs predicted
        ax.plot(test_indices, y_test.values, 'b-', linewidth=2, label='Actual CO₂', alpha=0.7)
        ax.plot(test_indices, y_pred, 'r--', linewidth=2, label='Predicted CO₂', alpha=0.7)
        
        # Fill between for error visualization
        ax.fill_between(test_indices, y_test.values, y_pred, alpha=0.2, color='gray')
        
        ax.set_xlabel('Test Sample Index', fontsize=11)
        ax.set_ylabel('CO₂ (ppm)', fontsize=11)
        ax.set_title(f'{model_name.upper()} - Predictions vs Actual (MAE: {mae:.2f} ppm, R²: {r2:.4f})', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{VISUALIZATION_OUTPUT_DIR}/05_prediction_comparison.png', dpi=100, bbox_inches='tight')
    print(f"   ✅ Saved: 05_prediction_comparison.png")
    
    # -------- PLOT 2: Prediction Errors --------
    print("   Creating plot 2: Prediction Errors...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (model_name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        y_pred = result['y_test_pred']
        errors = y_test.values - y_pred
        
        ax.hist(errors, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Prediction Error (ppm)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{model_name.upper()} - Error Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{VISUALIZATION_OUTPUT_DIR}/06_prediction_errors.png', dpi=100, bbox_inches='tight')
    print(f"   ✅ Saved: 06_prediction_errors.png")
    
    # -------- PLOT 3: Residuals Over Time --------
    print("   Creating plot 3: Residuals Over Time...")
    
    fig, ax = plt.subplots(figsize=(16, 5))
    
    for model_name, result in results.items():
        y_pred = result['y_test_pred']
        residuals = y_test.values - y_pred
        ax.plot(test_indices, residuals, marker='o', linewidth=1, label=model_name, alpha=0.7)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Test Sample Index', fontsize=11)
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
    ax.set_title('Prediction Residuals Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{VISUALIZATION_OUTPUT_DIR}/07_residuals_over_time.png', dpi=100, bbox_inches='tight')
    print(f"   ✅ Saved: 07_residuals_over_time.png")
    
    print(f"\n✅ All visualizations created!")

# ============================================
# MODEL SELECTION & SAVING
# ============================================

def select_best_model(results):
    """
    Select the best performing model based on test R² score
    """
    
    print("\n🏆 Selecting best model...")
    
    best_model_name = max(results.keys(), 
                         key=lambda x: results[x]['test_r2'])
    best_result = results[best_model_name]
    
    print(f"\n   Best Model: {best_model_name.upper()}")
    print(f"   Test R² Score: {best_result['test_r2']:.4f}")
    print(f"   Test MAE: {best_result['test_mae']:.2f} ppm")
    print(f"   Test RMSE: {best_result['test_rmse']:.2f} ppm")
    
    return best_model_name, best_result['model']

def save_models(results, best_model_name):
    """Save all models and best model"""
    
    print("\n💾 Saving models...")
    
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    # Save all models
    for model_name, result in results.items():
        model_path = f'{MODEL_OUTPUT_DIR}/{model_name}.pkl'
        joblib.dump(result['model'], model_path)
        print(f"   ✅ {model_name}: {model_path}")
    
    # Save best model
    best_model_path = f'{MODEL_OUTPUT_DIR}/best_predictor_model.pkl'
    joblib.dump(results[best_model_name]['model'], best_model_path)
    print(f"   ✅ Best model: {best_model_path}")
    
    # Save model info
    info_path = f'{MODEL_OUTPUT_DIR}/model_info.txt'
    with open(info_path, 'w') as f:
        f.write("LeoGuard - CO₂ Prediction Model\n")
        f.write("="*50 + "\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Lags Used: {LAGS}\n")
        f.write(f"Features: {', '.join(FEATURES_FOR_PREDICTION)}\n")
        f.write(f"\nBest Model Performance:\n")
        f.write(f"  Test R²: {results[best_model_name]['test_r2']:.4f}\n")
        f.write(f"  Test MAE: {results[best_model_name]['test_mae']:.2f} ppm\n")
        f.write(f"  Test RMSE: {results[best_model_name]['test_rmse']:.2f} ppm\n")
    
    print(f"   ✅ Model info: {info_path}")

# ============================================
# DEMO PREDICTIONS
# ============================================

def demo_predictions(best_model, feature_names):
    """
    Demonstrate how to use the model for real predictions
    """
    
    print("\n🔮 Demo: How to Use the Prediction Model")
    print("="*60)
    
    # Example: Last hour of data
    print("\nScenario: You have sensor readings from the last 2 hours")
    print("         (4 readings with 30-minute intervals)")
    print("\nHistorical readings:")
    
    readings = [
        {'co2': 410, 'temp': 31, 'humidity': 45},
        {'co2': 412, 'temp': 32, 'humidity': 44},
        {'co2': 415, 'temp': 33, 'humidity': 42},
        {'co2': 420, 'temp': 34, 'humidity': 40},
    ]
    
    for i, reading in enumerate(readings, 1):
        time_ago = (4-i) * 30
        print(f"   {time_ago} min ago: CO₂={reading['co2']} ppm, Temp={reading['temp']}°C, Humidity={reading['humidity']}%")
    
    # Prepare features for prediction
    feature_array = np.array([
        readings[3]['co2'], readings[3]['temp'], readings[3]['humidity'],
        readings[2]['co2'], readings[2]['temp'], readings[2]['humidity'],
        readings[1]['co2'], readings[1]['temp'], readings[1]['humidity'],
        readings[0]['co2'], readings[0]['temp'], readings[0]['humidity'],
    ]).reshape(1, -1)
    
    # Make prediction
    predicted_co2 = best_model.predict(feature_array)[0]
    
    print(f"\n✨ Prediction:")
    print(f"   Predicted CO₂ in 30 minutes: {predicted_co2:.1f} ppm")
    print(f"   Current CO₂: {readings[-1]['co2']} ppm")
    print(f"   Change: {predicted_co2 - readings[-1]['co2']:+.1f} ppm")
    
    if predicted_co2 > 480:
        print(f"\n   🔴 ALERT: CO₂ approaching critical threshold!")
    elif predicted_co2 > 450:
        print(f"\n   ⚠️  WARNING: CO₂ elevation detected")
    else:
        print(f"\n   🟢 OK: CO₂ levels normal")
    
    print("\n" + "="*60)

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == '__main__':
    
    print("\n" + "="*60)
    print("🚀 LeoGuard - Time Series Prediction Model Training")
    print("="*60)
    
    # Step 1: Load data
    df = load_data(DATA_FILE)
    if df is None:
        exit(1)
    
    # Step 2: Create lagged features
    df_lagged = create_lagged_features(df, lags=LAGS)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test, feature_names = split_data(df_lagged, test_size=TEST_SIZE)
    
    # Step 4: Train models
    models = train_models(X_train, y_train)
    
    # Step 5: Evaluate
    results = evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # Step 6: Visualize
    create_visualizations(results, X_test, y_test)
    
    # Step 7: Select best model
    best_model_name, best_model = select_best_model(results)
    
    # Step 8: Save
    save_models(results, best_model_name)
    
    # Step 9: Demo
    demo_predictions(best_model, feature_names)
    
    print("\n" + "="*60)
    print("✨ Prediction model training complete!")
    print("="*60)
    print("\n📌 Next steps:")
    print("   1. Review visualizations in 'visualizations/' folder")
    print("   2. Download satellite data: python download_satellite_data.py")
    print("   3. Integrate models with backend API")
    print("\n")
