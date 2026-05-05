"""
Auto-generate satellite data CSV files for LeoGuard Earth Engine integration.
Run this script to populate the data/satellite/ directory with sample data.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sentinel5p_data(output_path='data/satellite/sentinel5p_co2_data.csv', days=30):
    """
    Generate Sentinel-5P CO2 data CSV file.
    
    Args:
        output_path: Path to save the CSV file
        days: Number of days of data to generate (default: 30)
    """
    print(f"Generating Sentinel-5P CO2 data ({days} days)...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Manipal, India coordinates
    latitude = 13.3441
    longitude = 74.7421
    
    # Generate data
    data = []
    np.random.seed(42)  # For reproducibility
    
    for date in dates:
        # CO2 levels vary between 410-420 ppb with some noise
        co2_ppb = 412 + np.random.normal(0, 2)
        
        # Cloud fraction varies between 0-0.3 (clear to somewhat cloudy)
        cloud_fraction = np.random.uniform(0.05, 0.30)
        
        # QA value (quality assurance) between 0.85-0.95
        qa_value = 0.90 + np.random.uniform(-0.05, 0.05)
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'latitude': latitude,
            'longitude': longitude,
            'co2_column_density_ppb': round(co2_ppb, 2),
            'cloud_fraction': round(cloud_fraction, 3),
            'qa_value': round(qa_value, 3),
            'location_name': 'Manipal_India'
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df)} rows in {output_path}")
    print(f"  Columns: {', '.join(df.columns)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    
    return df


def generate_satellite_vs_ground_comparison(output_path='data/satellite/satellite_vs_ground_comparison.csv', days=30):
    """
    Generate satellite vs ground station comparison CSV file.
    
    Args:
        output_path: Path to save the CSV file
        days: Number of days of data to generate (default: 30)
    """
    print(f"Generating satellite vs ground comparison data ({days} days)...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate dates with hourly intervals
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Manipal, India coordinates
    location_name = 'Manipal_India'
    latitude = 13.3441
    longitude = 74.7421
    
    # Generate data
    data = []
    np.random.seed(42)  # For reproducibility
    
    for date in dates:
        # Satellite CO2 (ppb) - typically 410-420 ppb
        satellite_co2_ppb = 415 + np.random.normal(0, 2.5)
        
        # Ground station CO2 (ppm) - related to satellite but with some variation
        # Add correlation but also some measurement difference
        ground_station_co2_ppm = satellite_co2_ppb / 1000 * 1.005 + np.random.normal(0, 1.5)
        
        # Temperature varies throughout day (20-32°C)
        hour = date.hour
        temp_base = 26 + 6 * np.sin((hour - 6) * np.pi / 12)  # Sinusoidal pattern
        temperature_celsius = temp_base + np.random.normal(0, 0.5)
        
        # Humidity higher at night, lower during day (50-75%)
        humidity_base = 62 + 12 * np.sin((hour - 12) * np.pi / 12)
        humidity_percent = humidity_base + np.random.normal(0, 2)
        
        # Wind speed varies throughout day (1-6 m/s)
        wind_speed_ms = 3 + 2 * np.sin(hour * np.pi / 12) + np.random.uniform(-0.5, 0.5)
        
        # Air Quality Index (50-150, lower is better)
        aqi = 100 + int(np.random.normal(0, 20))
        
        # Calculate correlation quality
        co2_diff = abs(satellite_co2_ppb - ground_station_co2_ppm * 1000)
        if co2_diff < 5:
            correlation = 'Strong correlation'
        elif co2_diff < 10:
            correlation = 'High agreement'
        elif co2_diff < 15:
            correlation = 'Good correlation'
        elif co2_diff < 20:
            correlation = 'Moderate correlation'
        else:
            correlation = 'Weak correlation'
        
        data.append({
            'date': date.strftime('%Y-%m-%d %H:%M:%S'),
            'location': location_name,
            'latitude': latitude,
            'longitude': longitude,
            'satellite_co2_ppb': round(satellite_co2_ppb, 2),
            'ground_station_co2_ppm': round(ground_station_co2_ppm, 2),
            'temperature_celsius': round(temperature_celsius, 1),
            'humidity_percent': round(humidity_percent, 1),
            'wind_speed_ms': round(wind_speed_ms, 2),
            'aqi': aqi,
            'correlation_notes': correlation
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df)} rows in {output_path}")
    print(f"  Columns: {', '.join(df.columns)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  CO2 range (satellite): {df['satellite_co2_ppb'].min():.2f} - {df['satellite_co2_ppb'].max():.2f} ppb")
    print(f"  Temperature range: {df['temperature_celsius'].min():.1f} - {df['temperature_celsius'].max():.1f}°C")
    print()
    
    return df


def validate_generated_files(sentinel_path='data/satellite/sentinel5p_co2_data.csv',
                           comparison_path='data/satellite/satellite_vs_ground_comparison.csv'):
    """
    Validate the generated CSV files.
    
    Args:
        sentinel_path: Path to Sentinel-5P CSV
        comparison_path: Path to comparison CSV
    """
    print("Validating generated files...")
    print("=" * 70)
    
    # Validate Sentinel-5P data
    if os.path.exists(sentinel_path):
        df1 = pd.read_csv(sentinel_path)
        print(f"\n✓ {sentinel_path}")
        print(f"  Shape: {df1.shape[0]} rows × {df1.shape[1]} columns")
        print(f"  Columns: {list(df1.columns)}")
        print(f"\n  Sample data:")
        print(df1.head(3).to_string(index=False))
        print()
    else:
        print(f"✗ {sentinel_path} not found")
    
    # Validate comparison data
    if os.path.exists(comparison_path):
        df2 = pd.read_csv(comparison_path)
        print(f"\n✓ {comparison_path}")
        print(f"  Shape: {df2.shape[0]} rows × {df2.shape[1]} columns")
        print(f"  Columns: {list(df2.columns)}")
        print(f"\n  Sample data:")
        print(df2.head(3).to_string(index=False))
        print()
    else:
        print(f"✗ {comparison_path} not found")
    
    print("=" * 70)
    print("✓ Validation complete!")


def main():
    """Main execution."""
    print("\n" + "=" * 70)
    print("LeoGuard - Satellite Data CSV Generator")
    print("=" * 70 + "\n")
    
    try:
        # Generate both CSV files
        sentinel_df = generate_sentinel5p_data(days=30)
        comparison_df = generate_satellite_vs_ground_comparison(days=15)
        
        # Validate generated files
        validate_generated_files()
        
        print("\n✓ All CSV files generated successfully!")
        print("\nNext steps:")
        print("  1. Run: python test_earth_engine.py")
        print("  2. Run: python fetch_satellite_data.py")
        print("  3. Commit changes: git add data/ && git commit -m 'Add generated satellite data'")
        print("  4. Push: git push origin main\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
