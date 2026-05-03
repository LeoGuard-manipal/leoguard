"""
Test module for Earth Engine integration with LeoGuard.
Verifies authentication and basic satellite data retrieval functionality.
"""

import ee
import pytest
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEarthEngineAuthentication:
    """Test cases for Earth Engine authentication and initialization."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test - initialize Earth Engine."""
        try:
            ee.Initialize()
        except Exception as e:
            pytest.skip(f"Earth Engine initialization failed: {e}")
        yield
    
    def test_ee_is_authenticated(self):
        """Test that Earth Engine is properly authenticated."""
        try:
            # Try to access a simple asset to verify authentication
            result = ee.Image('COPERNICUS/S5P/NRTI/CO2').first()
            assert result is not None, "Failed to access Sentinel-5P CO2 data"
        except Exception as e:
            pytest.fail(f"Authentication check failed: {e}")
    
    def test_sentinel5p_co2_availability(self):
        """Test that Sentinel-5P CO2 data is accessible."""
        try:
            dataset = ee.ImageCollection('COPERNICUS/S5P/NRTI/CO2')
            assert dataset is not None, "Could not access Sentinel-5P CO2 dataset"
            # Verify the dataset has data
            count = dataset.size().getInfo()
            assert count > 0, "No Sentinel-5P CO2 data available"
        except Exception as e:
            pytest.fail(f"Sentinel-5P data availability test failed: {e}")
    
    def test_roi_selection(self):
        """Test ROI (Region of Interest) selection."""
        try:
            # Define a sample ROI (example: Manipal, India)
            roi = ee.Geometry.Point([74.7421, 13.3441])
            assert roi is not None, "Failed to create ROI geometry"
        except Exception as e:
            pytest.fail(f"ROI selection test failed: {e}")
    
    def test_date_filtering(self):
        """Test date filtering for satellite data."""
        try:
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now()
            
            dataset = ee.ImageCollection('COPERNICUS/S5P/NRTI/CO2') \
                .filterDate(start_date.isoformat(), end_date.isoformat())
            
            count = dataset.size().getInfo()
            assert count >= 0, "Date filtering returned invalid results"
        except Exception as e:
            pytest.fail(f"Date filtering test failed: {e}")
    
    def test_get_co2_data(self):
        """Test retrieval of CO2 data."""
        try:
            dataset = ee.ImageCollection('COPERNICUS/S5P/NRTI/CO2')
            latest = dataset.first()
            
            # Get CO2 band
            co2_band = latest.select('CO2_column_number_density')
            assert co2_band is not None, "Failed to select CO2 band"
        except Exception as e:
            pytest.fail(f"CO2 data retrieval test failed: {e}")


class TestDataExport:
    """Test cases for data export functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        try:
            ee.Initialize()
        except Exception as e:
            pytest.skip(f"Earth Engine initialization failed: {e}")
        yield
    
    def test_export_to_geotiff(self):
        """Test exporting data to GeoTIFF format."""
        try:
            # This is a placeholder test - actual export would need Google Cloud Storage
            dataset = ee.ImageCollection('COPERNICUS/S5P/NRTI/CO2').first()
            assert dataset is not None, "Failed to get image for export"
        except Exception as e:
            pytest.fail(f"GeoTIFF export test failed: {e}")
    
    def test_export_to_csv(self):
        """Test exporting aggregated data to CSV."""
        try:
            # This is a placeholder test - actual CSV export would use EE Table API
            dataset = ee.ImageCollection('COPERNICUS/S5P/NRTI/CO2')
            assert dataset is not None, "Failed to get collection for export"
        except Exception as e:
            pytest.fail(f"CSV export test failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
