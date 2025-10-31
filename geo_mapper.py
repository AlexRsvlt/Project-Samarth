"""
Geographic Mapping System for Project Samarth
Handles mapping between rainfall subdivisions, states, and crop districts
Solves the cross-ministry data integration challenge
"""

import pandas as pd
from typing import List, Dict, Optional, Set
import re

class GeographicMapper:
    """
    Maps between different administrative boundaries:
    - Rainfall Subdivisions (IMD) ↔ States ↔ Districts (Agriculture)
    Handles one-to-many and many-to-many relationships
    """
    
    def __init__(self):
        """Initialize with comprehensive mapping"""
        
        # Define subdivision to state mappings
        self.subdivision_to_state = {
            # Direct state matches 
            'Kerala': ['Kerala'],  
            'Tamil Nadu': ['Tamil Nadu'],
            'Bihar': ['Bihar'],
            'Chhattisgarh': ['Chhattisgarh'],
            'Jharkhand': ['Jharkhand'],
            'Orissa': ['Odisha', 'Orissa'],
            'Punjab': ['Punjab'],
            'Arunachal Pradesh': ['Arunachal Pradesh'],
            'Lakshadweep': ['Lakshadweep'],
            'Uttarakhand': ['Uttarakhand'],
            
            # Complex mappings - one state to multiple subdivisions
            'Coastal Andhra Pradesh': ['Andhra Pradesh'],
            'Rayalaseema': ['Andhra Pradesh'],
            'Telangana': ['Telangana', 'Andhra Pradesh'],  # Telangana was part of AP
            
            'Coastal Karnataka': ['Karnataka'],
            'North Interior Karnataka': ['Karnataka'],
            'South Interior Karnataka': ['Karnataka'],
            
            'Gangetic West Bengal': ['West Bengal'],
            'Sub Himalayan West Bengal & Sikkim': ['West Bengal', 'Sikkim'],
            
            'Konkan & Goa': ['Maharashtra', 'Goa'],
            'Madhya Maharashtra': ['Maharashtra'],
            'Marathwada': ['Maharashtra'],
            'Vidarbha': ['Maharashtra'],
            
            'West Rajasthan': ['Rajasthan'],
            'East Rajasthan': ['Rajasthan'],
            
            'East Madhya Pradesh': ['Madhya Pradesh'],
            'West Madhya Pradesh': ['Madhya Pradesh'],
            
            'East Uttar Pradesh': ['Uttar Pradesh'],
            'West Uttar Pradesh': ['Uttar Pradesh'],
            
            'Andaman & Nicobar Islands': ['Andaman and Nicobar Islands'],
            'Assam & EMeghalaya': ['Assam', 'Meghalaya'],
            'Haryana Delhi & Chandigarh': ['Haryana', 'Delhi', 'Chandigarh'],
            'Jammu & Kashmi': ['Jammu and Kashmir'],
            'Himachal Pradesh': ['Himachal Pradesh'],

        }
        
        # Reverse mapping: state to subdivisions
        self.state_to_subdivisions = {}
        for subdiv, states in self.subdivision_to_state.items():
            for state in states:
                state_clean = self._normalize_name(state)
                if state_clean not in self.state_to_subdivisions:
                    self.state_to_subdivisions[state_clean] = []
                self.state_to_subdivisions[state_clean].append(subdiv)
        
        # Common name variations for robust matching
        self.name_variations = {
            'andaman': ['andaman & nicobar islands', 'andaman and nicobar islands'],
            'a&n': ['andaman & nicobar islands'],
            'ap': ['andhra pradesh', 'arunachal pradesh'],
            'wb': ['west bengal'],
            'tn': ['tamil nadu'],
            'up': ['uttar pradesh'],
            'mp': ['madhya pradesh'],
            'hp': ['himachal pradesh'],
            'jk': ['jammu & kashmir', 'jammu and kashmir'],
            'uk': ['uttarakhand'],
            'odisha': ['orissa'],
            'orissa': ['odisha'],
        }
    
    def _normalize_name(self, name: str) -> str:
        """Normalize geographic names for comparison"""
        if not name:
            return ""
        
        # Remove leading numbers (from crop data: "1. Andaman and Nicobar")
        name = re.sub(r'^\d+\.\s*', '', name)
        
        # Convert to lowercase, strip whitespace
        name = name.lower().strip()
        
        # Standardize & to and
        name = name.replace(' & ', ' and ')
        
        return name
    
    def find_state_from_query(self, query: str) -> Optional[str]:
        """Extract state name from natural language query"""
        query_lower = query.lower()
        
        # Try direct state names first
        for state in self.state_to_subdivisions.keys():
            if state in query_lower:
                return state
        
        # Try name variations
        for variation, possible_names in self.name_variations.items():
            if variation in query_lower:
                # Return the first matching state
                for name in possible_names:
                    if name in self.state_to_subdivisions:
                        return name
        
        return None
    
    def find_subdivision_from_query(self, query: str) -> Optional[str]:
        """Extract rainfall subdivision from query"""
        query_lower = query.lower()
        
        # Try exact match first (case-insensitive)
        for subdiv in self.subdivision_to_state.keys():
            if subdiv.lower() in query_lower:
                return subdiv
        
        # Try partial matches
        for subdiv in self.subdivision_to_state.keys():
            subdiv_words = subdiv.lower().split()
            # Check if all significant words are in query
            significant_words = [w for w in subdiv_words if len(w) > 3 and w not in ['and', 'the']]
            if significant_words and all(w in query_lower for w in significant_words):
                return subdiv
        
        return None
    
    def get_subdivisions_for_state(self, state_name: str) -> List[str]:
        """Get all rainfall subdivisions that correspond to a state"""
        state_normalized = self._normalize_name(state_name)
        return self.state_to_subdivisions.get(state_normalized, [])
    
    def get_states_for_subdivision(self, subdivision: str) -> List[str]:
        """Get all states that correspond to a subdivision"""
        return self.subdivision_to_state.get(subdivision.upper(), [])
    
    def aggregate_subdivision_data(self, rainfall_df: pd.DataFrame, state_name: str) -> Dict:
        """
        Aggregate rainfall data from multiple subdivisions to state level
        Handles cases where one state maps to multiple subdivisions
        """
        subdivisions = self.get_subdivisions_for_state(state_name)
        
        if not subdivisions:
            return {'error': f'No subdivisions found for {state_name}'}
        
        # Filter rainfall data for relevant subdivisions
        state_data = rainfall_df[rainfall_df['SUBDIVISION'].isin(subdivisions)].copy()
        
        if state_data.empty:
            return {'error': f'No data found for subdivisions: {subdivisions}'}
        
        # Aggregate by year (average across subdivisions)
        month_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                     'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        
        aggregated = state_data.groupby('YEAR').agg({
            'ANNUAL': 'mean',
            'JF': 'mean',
            'MAM': 'mean',
            'JJAS': 'mean',
            'OND': 'mean',
            **{col: 'mean' for col in month_cols}
        }).reset_index()
        
        return {
            'state': state_name,
            'subdivisions_used': subdivisions,
            'data': aggregated,
            'aggregation_method': 'mean across subdivisions'
        }
    
    def match_crop_and_rainfall(self, crop_state: str, rainfall_subdivisions: List[str]) -> Dict:
        """
        Determine if a crop data state matches any rainfall subdivisions
        Returns matching information and confidence score
        """
        state_normalized = self._normalize_name(crop_state)
        
        # Get expected subdivisions for this state
        expected_subdivisions = self.get_subdivisions_for_state(crop_state)
        
        # Check for matches
        matches = []
        for subdiv in rainfall_subdivisions:
            if subdiv in expected_subdivisions:
                matches.append(subdiv)
        
        confidence = len(matches) / len(expected_subdivisions) if expected_subdivisions else 0
        
        return {
            'crop_state': crop_state,
            'expected_subdivisions': expected_subdivisions,
            'available_subdivisions': matches,
            'confidence': confidence,
            'can_integrate': len(matches) > 0
        }
    
    def create_integrated_dataset(self, crop_df: pd.DataFrame, rainfall_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create integrated dataset joining crop and rainfall data
        Handles geographic mapping automatically
        """
        integrated_records = []
        
        # Get unique crop states
        crop_states = crop_df['State'].unique()
        
        for state in crop_states:
            # Get subdivisions for this state
            subdivisions = self.get_subdivisions_for_state(state)
            
            if not subdivisions:
                continue
            
            # Get crop data for this state
            state_crop_data = crop_df[crop_df['State'] == state]
            
            # Get and aggregate rainfall data
            state_rainfall = rainfall_df[rainfall_df['SUBDIVISION'].isin(subdivisions)]
            
            if state_rainfall.empty:
                continue
            
            # Aggregate rainfall by year
            rainfall_agg = state_rainfall.groupby('YEAR').agg({
                'ANNUAL': 'mean',
                'JJAS': 'mean'
            }).reset_index()
            
            # Get unique years from crop data
            crop_years = state_crop_data['Year'].unique()
            
            for year in crop_years:
                # Check if rainfall data exists for this year
                year_rainfall = rainfall_agg[rainfall_agg['YEAR'] == year]
                
                if year_rainfall.empty:
                    continue
                
                # Get crop data for this year
                year_crop = state_crop_data[state_crop_data['Year'] == year]
                
                for _, crop_row in year_crop.iterrows():
                    record = {
                        'State': state,
                        'District': crop_row['District'],
                        'Year': year,
                        'Annual_Rainfall_mm': year_rainfall['ANNUAL'].values[0],
                        'Monsoon_Rainfall_mm': year_rainfall['JJAS'].values[0],
                        'Subdivisions_Used': ', '.join(subdivisions)
                    }
                    
                    # Add crop columns dynamically
                    for col in crop_row.index:
                        if col not in ['State', 'District', 'Year']:
                            record[f'Crop_{col}'] = crop_row[col]
                    
                    integrated_records.append(record)
        
        return pd.DataFrame(integrated_records)
    
    def suggest_similar_locations(self, query: str, all_locations: List[str]) -> List[str]:
        """Suggest locations when exact match not found"""
        query_lower = query.lower()
        
        suggestions = []
        for location in all_locations:
            location_lower = location.lower()
            
            # Check if query is substring of location
            if query_lower in location_lower:
                suggestions.append(location)
            
            # Check if they share significant words
            query_words = set(query_lower.split())
            location_words = set(location_lower.split())
            
            # Remove common words
            common_words = {'and', 'the', 'of', 'in', '&'}
            query_words -= common_words
            location_words -= common_words
            
            # If significant overlap, add as suggestion
            if query_words and location_words:
                overlap = len(query_words & location_words) / len(query_words | location_words)
                if overlap > 0.5:
                    suggestions.append(location)
        
        return list(set(suggestions))[:5]  # Return top 5 unique suggestions
    
    def validate_geographic_coverage(self, crop_states: List[str], rainfall_subdivisions: List[str]) -> Dict:
        """
        Validate how well crop and rainfall datasets cover each other
        Returns coverage statistics
        """
        coverage_info = {
            'total_crop_states': len(crop_states),
            'total_rainfall_subdivisions': len(rainfall_subdivisions),
            'states_with_rainfall_data': 0,
            'states_without_rainfall_data': [],
            'orphan_subdivisions': [],
            'mapping_details': []
        }
        
        for state in crop_states:
            expected_subdivisions = self.get_subdivisions_for_state(state)
            available_subdivisions = [s for s in expected_subdivisions if s in rainfall_subdivisions]
            
            if available_subdivisions:
                coverage_info['states_with_rainfall_data'] += 1
            else:
                coverage_info['states_without_rainfall_data'].append(state)
            
            coverage_info['mapping_details'].append({
                'state': state,
                'expected_subdivisions': expected_subdivisions,
                'available_subdivisions': available_subdivisions,
                'coverage': len(available_subdivisions) / len(expected_subdivisions) if expected_subdivisions else 0
            })
        
        # Find subdivisions not mapped to any crop state
        all_mapped_subdivisions = set()
        for subdivisions in self.subdivision_to_state.keys():
            all_mapped_subdivisions.add(subdivisions)
        
        coverage_info['orphan_subdivisions'] = [
            s for s in rainfall_subdivisions if s not in all_mapped_subdivisions
        ]
        
        return coverage_info