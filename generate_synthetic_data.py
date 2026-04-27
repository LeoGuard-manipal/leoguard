"""
LeoGuard - Synthetic Sensor Data Generator
Generates realistic 30-day sensor data for training ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ============================================
# CONFIGURATION
# ============================================

# Time configuration
START_DATE = '2026-04-27'
DAYS = 30
FREQUENCY = '30min'  # Data point every 30 minutes

# Sensor configurations
CO2_BASELINE = 410  # ppm (baseline atmospheric CO2)
CO2_DAILY_AMPLITUDE = 30  # ppm (daily variation)
CO2_NOISE_STD = 5  # Standard deviation of random noise
CO2_SPIKE_PROBABILITY = 0.002  # Probability of industrial spike
CO2_SPIKE_MAGNITUDE = [50, 150]  # Range of spike values

TEMP_BASELINE = 32  # °C (UAE average)
TEMP_DAILY_AMPLITUDE = 8  # °C variation
TEMP_NOISE_STD = 2  # Random noise

HUMIDITY_MIN = 20  # %
HUMIDITY_MAX = 60  # %

ZONE_NAME = 'Dubai_Industrial_Zone_1'

# ============================================
# GENERATE DATA
# ============================================

def generate_synthetic_data():
    """Generate realistic 30-day sensor dataset"""
    
    print("🔧 Generating LeoGuard synthetic dataset...")
    print(f"   Period: {START_DATE} for {DAYS} days")
    print(f"   Frequency: {FREQUENCY}\n")
    
    # Create time array
    dates = pd.date_range(START_DATE, periods=int(DAYS * 24 * 2), freq=FREQUENCY)
    print(f"✓ Created {len(dates)} timestamps")
    
    # -------- CO2 Data Generation --------
    print("📊 Generating CO2 data...")
    
    # Daily sine wave (CO2 higher during day, lower at night)
    daily_cycle = np.sin(np.arange(len(dates)) * 2 * np.pi / 48) * CO2_DAILY_AMPLITUDE
    
    # Random noise
    noise = np.random.normal(0, CO2_NOISE_STD, len(dates))
    
    # Industrial spikes (sudden emissions)
    spikes = np.zeros(len(dates))
    spike_count = 0
    for i in range(len(dates)):
        if np.random.random() < CO2_SPIKE_PROBABILITY:
            # Create a spike that lasts 5-10 readings
            spike_duration = np.random.randint(5, 10)
            spike_magnitude = np.random.uniform(CO2_SPIKE_MAGNITUDE[0], CO2_SPIKE_MAGNITUDE[1])
            spikes[i:min(i + spike_duration, len(dates))] = spike_magnitude
            spike_count += 1
    
    # Combine all components
    co2_values = CO2_BASELINE + daily_cycle + noise + spikes
    
    print(f"✓ Generated CO2 data")
    print(f"  - Baseline: {CO2_BASELINE} ppm")
    print(f"  - Range: {co2_values.min():.1f} - {co2_values.max():.1f} ppm")
    print(f"  - Spikes detected: {spike_count}")
    
    # -------- Temperature Data --------
    print("📊 Generating temperature data...")
    
    # Daily temperature cycle (warmer during day, cooler at night)
    temp_daily_cycle = np.sin(np.arange(len(dates)) * 2 * np.pi / 48) * TEMP_DAILY_AMPLITUDE
    temp_noise = np.random.normal(0, TEMP_NOISE_STD, len(dates))
    temp_values = TEMP_BASELINE + temp_daily_cycle + temp_noise
    
    print(f"✓ Generated temperature data")
    print(f"  - Range: {temp_values.min():.1f} - {temp_values.max():.1f} °C")
    
    # -------- Humidity Data --------
    print("📊 Generating humidity data...")
    
    # Humidity inversely correlated with temperature (realistic)
    humidity_values = np.random.uniform(HUMIDITY_MIN, HUMIDITY_MAX, len(dates))
    # Inverse correlation with temp
    humidity_values = humidity_values - (temp_values - TEMP_BASELINE) * 0.8
    humidity_values = np.clip(humidity_values, HUMIDITY_MIN, HUMIDITY_MAX)
    
    print(f"✓ Generated humidity data")
    print(f"  - Range: {humidity_values.min():.1f} - {humidity_values.max():.1f} %")
    
    # -------- Create DataFrame --------
    print("\n📋 Creating dataset...")
    
    df = pd.DataFrame({
        'timestamp': dates,
        'co2_ppm': np.round(co2_values, 2),
        'temperature_c': np.round(temp_values, 2),
        'humidity_percent': np.round(humidity_values, 2),
        'zone': ZONE_NAME
    })
    
    print(f"✓ Created DataFrame with {len(df)} records")
    
    return df

# ============================================
# DATA ANALYSIS
# ============================================

def analyze_data(df):
    """Analyze and print dataset statistics"""
    
    print("\n" + "="*60)
    print("📈 DATASET STATISTICS")
    print("="*60)
    
    print(f"\nTotal records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Zone: {df['zone'].iloc[0]}")
    
    print("\n🔬 CO2 (ppm):")
    print(f"   Mean: {df['co2_ppm'].mean():.2f}")
    print(f"   Std Dev: {df['co2_ppm'].std():.2f}")
    print(f"   Min: {df['co2_ppm'].min():.2f}")
    print(f"   Max: {df['co2_ppm'].max():.2f}")
    print(f"   Q25: {df['co2_ppm'].quantile(0.25):.2f}")
    print(f"   Q50: {df['co2_ppm'].quantile(0.50):.2f}")
    print(f"   Q75: {df['co2_ppm'].quantile(0.75):.2f}")
    
    print("\n🌡️  Temperature (°C):")
    print(f"   Mean: {df['temperature_c'].mean():.2f}")
    print(f"   Std Dev: {df['temperature_c'].std():.2f}")
    print(f"   Min: {df['temperature_c'].min():.2f}")
    print(f"   Max: {df['temperature_c'].max():.2f}")
    
    print("\n💧 Humidity (%):")
    print(f"   Mean: {df['humidity_percent'].mean():.2f}")
    print(f"   Std Dev: {df['humidity_percent'].std():.2f}")
    print(f"   Min: {df['humidity_percent'].min():.2f}")
    print(f"   Max: {df['humidity_percent'].max():.2f}")
    
    print("\n" + "="*60)

# ============================================
# SAVE DATA
# ============================================

def save_data(df, filename='data/synthetic_sensor_data.csv'):
    """Save dataset to CSV file"""
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df.to_csv(filename, index=False)
    file_size = os.path.getsize(filename) / 1024  # Size in KB
    
    print(f"\n✅ Dataset saved to: {filename}")
    print(f"   File size: {file_size:.2f} KB")
    print(f"   Records: {len(df)}")

# ============================================
# SAMPLE DATA
# ============================================

def print_sample_data(df, n_rows=10):
    """Print sample rows from dataset"""
    
    print("\n" + "="*60)
    print("📝 SAMPLE DATA (First 10 rows)")
    print("="*60 + "\n")
    
    print(df.head(n_rows).to_string(index=False))
    
    print("\n" + "="*60)
    print("📝 ANOMALY EXAMPLES (Spike events)")
    print("="*60 + "\n")
    
    # Find rows with high CO2 (potential spikes)
    threshold = df['co2_ppm'].quantile(0.90)
    anomalies = df[df['co2_ppm'] > threshold].head(5)
    
    print(anomalies.to_string(index=False))

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌍 LeoGuard - Synthetic Data Generator")
    print("="*60 + "\n")
    
    # Generate data
    df = generate_synthetic_data()
    
    # Analyze
    analyze_data(df)
    
    # Print samples
    print_sample_data(df)
    
    # Save
    save_data(df)
    
    print("\n✨ Done! Your synthetic dataset is ready for training!\n")
