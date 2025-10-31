"""
Custom Data Loader for Project Samarth
Handles the actual crop and rainfall datasets
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional

class CropDataLoader:
    """Load and process the actual crop production dataset"""
    
    def __init__(self, filepath: str):
        """Initialize with crop dataset"""
        self.df = pd.read_csv(filepath)
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        # Identify crop columns (they contain crop names)
        self.identify_crop_columns()
        
        # Standardize state names
        self.df['State'] = self.df['State'].str.strip()
        self.df['District'] = self.df['District'].str.strip()
        
        # Get unique values
        self.states = sorted(self.df['State'].unique())
        self.districts = sorted(self.df['District'].unique())
        self.years = sorted(self.df['Year'].unique())
        
        print(f"✓ Loaded crop data: {len(self.df)} rows")
        print(f"  States: {len(self.states)}")
        print(f"  Districts: {len(self.districts)}")
        print(f"  Years: {self.years}")
        print(f"  Crops identified: {len(self.crop_columns)}")
    
    def identify_crop_columns(self):
        """Identify which columns contain crop production data"""
        # Skip non-crop columns
        skip_cols = ['State', 'District', 'Year']
        
        # Crop columns are those that aren't State/District/Year
        all_cols = self.df.columns.tolist()
        
        # Find production columns (they often have "Production" or crop names)
        production_cols = []
        area_cols = []
        yield_cols = []
        
        for col in all_cols:
            if col in skip_cols:
                continue
            
            col_lower = col.lower()
            
            # Check if it's production, area, or yield
            if 'production' in col_lower:
                # Extract crop name
                crop_name = col.replace('Production', '').strip()
                if crop_name and 'Tonn' not in crop_name:
                    production_cols.append(col)
            elif 'area' in col_lower and 'hectare' in col_lower:
                area_cols.append(col)
            elif 'yield' in col_lower and 'tonn' in col_lower:
                yield_cols.append(col)
            elif col_lower in ['cashewnut', 'coconut', 'ginger', 'sugarcane', 'sweet pot.']:
                # Direct crop name columns
                production_cols.append(col)
        
        # Create mapping of crop names
        self.crop_columns = {}
        
        # Parse production columns to get crop names
        for col in production_cols:
            # Extract crop name from column
            crop_name = self._extract_crop_name(col)
            if crop_name:
                self.crop_columns[crop_name] = {
                    'production': col,
                    'area': self._find_matching_column(col, area_cols, 'Area'),
                    'yield': self._find_matching_column(col, yield_cols, 'Yield')
                }
        
        # If no "Production" suffix, try direct crop columns
        if not self.crop_columns:
            for col in all_cols:
                if col not in skip_cols and not any(x in col for x in ['Area', 'Yield']):
                    crop_name = col.strip()
                    self.crop_columns[crop_name] = {
                        'production': col,
                        'area': None,
                        'yield': None
                    }
    
    def _extract_crop_name(self, column_name: str) -> str:
        """Extract crop name from column"""
        # Remove common suffixes
        crop = column_name.replace('Production', '').replace('(Tonn/Hectare)', '')
        crop = crop.replace('Area (Hectare)', '').replace('Yield', '')
        crop = crop.strip()
        return crop if crop else None
    
    def _find_matching_column(self, base_col: str, search_cols: List[str], suffix: str) -> Optional[str]:
        """Find matching area/yield column for a production column"""
        crop_name = self._extract_crop_name(base_col)
        if not crop_name:
            return None
        
        for col in search_cols:
            if crop_name in col and suffix in col:
                return col
        return None
    
    def get_crop_production(self, state: str, district: Optional[str] = None, 
                           crop: Optional[str] = None, year: Optional[int] = None) -> pd.DataFrame:
        """Get crop production data with filters"""
        query = self.df.copy()
        
        # Apply filters
        if state:
            query = query[query['State'].str.contains(state, case=False, na=False)]
        
        if district:
            query = query[query['District'].str.contains(district, case=False, na=False)]
        
        if year:
            query = query[query['Year'] == year]
        
        return query
    
    def get_crop_list(self) -> List[str]:
        """Get list of available crops"""
        return list(self.crop_columns.keys())
    
    def normalize_state_name(self, state: str) -> str:
        """Normalize state name for matching with rainfall data"""
        # Remove numbers and extra formatting
        state = re.sub(r'^\d+\.\s*', '', state)  # Remove leading numbers
        state = state.strip()
        return state


class RainfallDataLoader:
    """Load and process the actual rainfall dataset (1901-2017)"""
    
    def __init__(self, filepath: str):
        """Initialize with rainfall dataset"""
        self.df = pd.read_csv(filepath)
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip().str.upper()
        
        # Standardize subdivision names
        self.df['SUBDIVISION'] = self.df['SUBDIVISION'].str.strip()
        
        # Month columns
        self.month_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        
        # Seasonal columns
        self.seasonal_cols = {
            'Winter': 'JF',
            'Pre-Monsoon': 'MAM',
            'Monsoon': 'JJAS',
            'Post-Monsoon': 'OND'
        }
        
        # Get unique values
        self.subdivisions = sorted(self.df['SUBDIVISION'].unique())
        self.years = sorted(self.df['YEAR'].unique())
        
        print(f"✓ Loaded rainfall data: {len(self.df)} rows")
        print(f"  Subdivisions: {len(self.subdivisions)}")
        print(f"  Year range: {self.years[0]} - {self.years[-1]}")
    
    def normalize_subdivision_name(self, subdivision: str) -> str:
        """Normalize subdivision name for matching with crop data"""
        # Convert "Andaman & Nicobar Islands" to match crop format
        return subdivision.strip()
    
    def find_matching_subdivision(self, state_name: str) -> Optional[str]:
        """Find matching subdivision for a crop data state name"""
        # Remove leading numbers from state name
        clean_state = re.sub(r'^\d+\.\s*', '', state_name).strip()
        
        # Try exact match first
        for subdiv in self.subdivisions:
            if subdiv.lower() == clean_state.lower():
                return subdiv
        
        # Try partial match
        for subdiv in self.subdivisions:
            if clean_state.lower() in subdiv.lower() or subdiv.lower() in clean_state.lower():
                return subdiv
        
        return None


class DataIntegrator:
    """Integrate crop and rainfall datasets"""
    
    def __init__(self, crop_loader: CropDataLoader, rainfall_loader: RainfallDataLoader):
        self.crop_loader = crop_loader
        self.rainfall_loader = rainfall_loader
        
        # Create mapping between state names
        self.state_mapping = self._create_state_mapping()
        
        print(f"\n✓ Created state mapping: {len(self.state_mapping)} matches")
    
    def _create_state_mapping(self) -> Dict[str, str]:
        """Create mapping between crop states and rainfall subdivisions"""
        mapping = {}
        
        for state in self.crop_loader.states:
            # Normalize and find match
            clean_state = self.crop_loader.normalize_state_name(state)
            matched_subdiv = self.rainfall_loader.find_matching_subdivision(clean_state)
            
            if matched_subdiv:
                mapping[state] = matched_subdiv
        
        return mapping
    
    def get_integrated_data(self, state: str, year: int) -> Dict:
        """Get both crop and rainfall data for a state and year"""
        result = {
            'state': state,
            'year': year,
            'crop_data': None,
            'rainfall_data': None
        }
        
        # Get crop data
        crop_data = self.crop_loader.get_crop_production(state=state, year=year)
        if not crop_data.empty:
            result['crop_data'] = crop_data
        
        # Get rainfall data
        if state in self.state_mapping:
            subdivision = self.state_mapping[state]
            rainfall_data = self.rainfall_loader.df[
                (self.rainfall_loader.df['SUBDIVISION'] == subdivision) &
                (self.rainfall_loader.df['YEAR'] == year)
            ]
            if not rainfall_data.empty:
                result['rainfall_data'] = rainfall_data
        
        return result
    
    def analyze_crop_rainfall_correlation(self, state: str, crop: str) -> Dict:
        """Analyze correlation between rainfall and crop production"""
        if state not in self.state_mapping:
            return {'error': 'State not found in rainfall data'}
        
        subdivision = self.state_mapping[state]
        
        # Get crop data for all years
        crop_data = self.crop_loader.get_crop_production(state=state)
        
        if crop not in self.crop_loader.crop_columns:
            return {'error': f'Crop {crop} not found'}
        
        # Get production column
        prod_col = self.crop_loader.crop_columns[crop]['production']
        
        # Get rainfall data for matching years
        years = crop_data['Year'].unique()
        rainfall_data = self.rainfall_loader.df[
            (self.rainfall_loader.df['SUBDIVISION'] == subdivision) &
            (self.rainfall_loader.df['YEAR'].isin(years))
        ]
        
        if rainfall_data.empty or crop_data.empty:
            return {'error': 'Insufficient data for correlation'}
        
        # Merge datasets
        merged = crop_data[['Year', prod_col]].merge(
            rainfall_data[['YEAR', 'ANNUAL']], 
            left_on='Year', 
            right_on='YEAR'
        )
        
        # Remove NaN values
        merged = merged.dropna(subset=[prod_col, 'ANNUAL'])
        
        if len(merged) < 3:
            return {'error': 'Insufficient overlapping data'}
        
        # Calculate correlation
        correlation = merged[prod_col].corr(merged['ANNUAL'])
        
        return {
            'state': state,
            'crop': crop,
            'subdivision': subdivision,
            'correlation': round(correlation, 3),
            'data_points': len(merged),
            'years': merged['Year'].tolist(),
            'production': merged[prod_col].tolist(),
            'rainfall': merged['ANNUAL'].tolist()
        }