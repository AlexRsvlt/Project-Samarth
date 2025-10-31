"""
Smart Data Aggregator for Project Samarth
Handles intelligent aggregation across different administrative levels
Provides cross-domain insights despite data structure differences
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from geo_mapper import GeographicMapper


class SmartDataAggregator:
    """
    Intelligently aggregates and joins data from different sources.
    Handles mismatched administrative boundaries and supports
    cross-domain correlation and anomaly detection.
    """

    def __init__(self, geo_mapper: GeographicMapper):
        self.geo_mapper = geo_mapper

    # ===============================================================
    # RAINFALL AGGREGATION
    # ===============================================================

    def aggregate_rainfall_to_state(self, rainfall_df: pd.DataFrame,
                                    state_name: str,
                                    year_range: Optional[Tuple[int, int]] = None) -> Dict:
        """
        Aggregate rainfall data from subdivisions to state level.
        Handles multiple subdivisions per state.
        """
        result = self.geo_mapper.aggregate_subdivision_data(rainfall_df, state_name)

        if 'error' in result:
            return result

        data = result['data']

        # ✅ Apply year filter if provided
        if year_range:
            start, end = year_range
            data = data[(data['YEAR'] >= start) & (data['YEAR'] <= end)]

        if data.empty:
            return {'error': f'No data found for {state_name} in specified years'}

        return {
            'state': state_name,
            'subdivisions': result['subdivisions_used'],
            'years': data['YEAR'].tolist(),
            'annual_rainfall': data['ANNUAL'].tolist(),
            'monsoon_rainfall': data['JJAS'].tolist(),
            'stats': {
                'mean_annual': data['ANNUAL'].mean(),
                'std_annual': data['ANNUAL'].std(),
                'min_annual': data['ANNUAL'].min(),
                'max_annual': data['ANNUAL'].max(),
                'mean_monsoon': data['JJAS'].mean(),
                'monsoon_contribution': (
                    data['JJAS'].mean() / data['ANNUAL'].mean() * 100
                )
            },
            'note': f'Aggregated from {len(result["subdivisions_used"])} subdivision(s)'
        }

    # ===============================================================
    # CROP AGGREGATION
    # ===============================================================

    def aggregate_crop_to_state(self, crop_df: pd.DataFrame,
                                state_name: str,
                                crop_name: Optional[str] = None,
                                year_range: Optional[Tuple[int, int]] = None) -> Dict:
        """
        Aggregate crop data from districts to state level.
        """
        # Filter by state
        state_data = crop_df[crop_df['State'].str.contains(state_name, case=False, na=False)].copy()

        if state_data.empty:
            return {'error': f'No crop data found for {state_name}'}

        # Apply year filter
        if year_range:
            start, end = year_range
            state_data = state_data[(state_data['Year'] >= start) & (state_data['Year'] <= end)]

        if state_data.empty:
            return {'error': f'No data for {state_name} in specified years'}

        # Identify numeric crop columns
        crop_columns = [
            col for col in state_data.columns
            if col not in ['State', 'District', 'Year']
            and state_data[col].dtype in [np.float64, np.int64]
        ]

        # Aggregate by year
        aggregated = state_data.groupby('Year')[crop_columns].sum().reset_index()

        return {
            'state': state_name,
            'years': aggregated['Year'].tolist(),
            'districts_count': state_data['District'].nunique(),
            'crops_available': crop_columns,
            'aggregated_data': aggregated,
            'note': f'Aggregated from {state_data["District"].nunique()} district(s)'
        }

    # ===============================================================
    # INTEGRATION
    # ===============================================================

    def integrate_crop_rainfall(self, crop_df: pd.DataFrame,
                                rainfall_df: pd.DataFrame,
                                state_name: str,
                                crop_name: str,
                                year_range: Optional[Tuple[int, int]] = None) -> Dict:
        """
        Create integrated dataset for correlation analysis.
        Handles geographic mapping and temporal alignment.
        """
        rainfall_result = self.aggregate_rainfall_to_state(rainfall_df, state_name, year_range)
        if 'error' in rainfall_result:
            return rainfall_result

        crop_result = self.aggregate_crop_to_state(crop_df, state_name, crop_name, year_range)
        if 'error' in crop_result:
            return crop_result

        # Find overlapping years
        rainfall_years = set(rainfall_result['years'])
        crop_years = set(crop_result['years'])
        common_years = sorted(rainfall_years & crop_years)

        if not common_years:
            return {
                'error': 'No overlapping years between crop and rainfall data',
                'rainfall_years': sorted(rainfall_years),
                'crop_years': sorted(crop_years)
            }

        # Build integrated dataset
        integrated_data = []
        rainfall_dict = dict(zip(rainfall_result['years'], rainfall_result['annual_rainfall']))
        monsoon_dict = dict(zip(rainfall_result['years'], rainfall_result['monsoon_rainfall']))
        crop_data_df = crop_result['aggregated_data']

        if crop_name not in crop_data_df.columns:
            return {
                'error': f'Crop {crop_name} not found',
                'available_crops': crop_result['crops_available']
            }

        for year in common_years:
            if year in rainfall_dict:
                crop_value = crop_data_df[crop_data_df['Year'] == year][crop_name].values
                if len(crop_value) > 0:
                    integrated_data.append({
                        'year': year,
                        'rainfall': rainfall_dict[year],
                        'monsoon': monsoon_dict[year],
                        'crop_production': crop_value[0]
                    })

        if not integrated_data:
            return {'error': 'Failed to align data'}

        return {
            'state': state_name,
            'crop': crop_name,
            'subdivisions': rainfall_result['subdivisions'],
            'districts_count': crop_result['districts_count'],
            'years': [d['year'] for d in integrated_data],
            'rainfall': [d['rainfall'] for d in integrated_data],
            'monsoon': [d['monsoon'] for d in integrated_data],
            'production': [d['crop_production'] for d in integrated_data],
            'data_points': len(integrated_data),
            'note': f'Integrated data from {len(rainfall_result["subdivisions"])} subdivision(s) and {crop_result["districts_count"]} district(s)'
        }

    # ===============================================================
    # CORRELATION ANALYSIS
    # ===============================================================

    def analyze_correlation(self, integrated_data: Dict) -> Dict:
        """
        Perform correlation and regression analysis on integrated data.
        """
        if 'error' in integrated_data:
            return integrated_data

        rainfall = np.array(integrated_data['rainfall'])
        production = np.array(integrated_data['production'])

        # Clean data
        valid_mask = ~(np.isnan(rainfall) | np.isnan(production))
        rainfall_clean = rainfall[valid_mask]
        production_clean = production[valid_mask]

        if len(rainfall_clean) < 3:
            return {'error': 'Insufficient valid data points for correlation'}

        # Correlation coefficients
        correlation, p_value = stats.pearsonr(rainfall_clean, production_clean)
        spearman_corr, spearman_p = stats.spearmanr(rainfall_clean, production_clean)

        # Interpret strength
        abs_corr = abs(correlation)
        if abs_corr > 0.7:
            strength = "strong"
        elif abs_corr > 0.4:
            strength = "moderate"
        elif abs_corr > 0.2:
            strength = "weak"
        else:
            strength = "negligible"

        direction = "positive" if correlation > 0 else "negative"

        # Regression
        slope, intercept, r_value, p_val, std_err = stats.linregress(rainfall_clean, production_clean)

        return {
            'state': integrated_data['state'],
            'crop': integrated_data['crop'],
            'correlation': {
                'pearson_r': round(correlation, 4),
                'pearson_p_value': round(p_value, 4),
                'spearman_r': round(spearman_corr, 4),
                'spearman_p_value': round(spearman_p, 4),
                'strength': strength,
                'direction': direction,
                'significant': p_value < 0.05
            },
            'regression': {
                'slope': round(slope, 4),
                'intercept': round(intercept, 2),
                'r_squared': round(r_value ** 2, 4),
                'interpretation': f"For every 1mm increase in rainfall, {integrated_data['crop']} production changes by {slope:.2f} units"
            },
            'data_quality': {
                'total_years': len(integrated_data['years']),
                'valid_data_points': len(rainfall_clean),
                'data_completeness': round(len(rainfall_clean) / len(rainfall) * 100, 1),
                'subdivisions_used': integrated_data['subdivisions'],
                'districts_count': integrated_data['districts_count']
            },
            'years_analyzed': integrated_data['years'],
            'note': integrated_data['note']
        }

    # ===============================================================
    # STATE COMPARISON
    # ===============================================================

    def compare_states_rainfall(self, rainfall_df: pd.DataFrame,
                                state_names: List[str],
                                year_range: Optional[Tuple[int, int]] = None,
                                metric: str = 'annual') -> Dict:
        """
        Compare rainfall statistics across multiple states.
        """
        comparisons = []
        chart_data = {'labels': [], 'datasets': []}

        for state in state_names:
            result = self.aggregate_rainfall_to_state(rainfall_df, state, year_range)

            if 'error' not in result:
                comparisons.append({
                    'state': state,
                    'mean_rainfall': result['stats']['mean_annual'],
                    'std_rainfall': result['stats']['std_annual'],
                    'subdivisions': result['subdivisions']
                })

                chart_data['datasets'].append({
                    'label': state,
                    'data': result['annual_rainfall']
                })

                # Use years from first state
                if not chart_data['labels']:
                    chart_data['labels'] = result['years']

        if not comparisons:
            return {'error': 'No valid data for comparison'}

        # Sort by mean rainfall
        comparisons.sort(key=lambda x: x['mean_rainfall'], reverse=True)

        return {
            'comparisons': comparisons,
            'chart_data': chart_data,
            'metric': metric,
            'year_range': year_range
        }

    # ===============================================================
    # IMPACT ANALYSIS
    # ===============================================================

    def identify_impact_years(self, integrated_data: Dict, threshold_std: float = 1.5) -> Dict:
        """
        Identify years where rainfall anomalies significantly impacted crop production.
        """
        if 'error' in integrated_data:
            return integrated_data

        rainfall = np.array(integrated_data['rainfall'])
        production = np.array(integrated_data['production'])
        years = integrated_data['years']

        # Z-scores
        rainfall_mean = rainfall.mean()
        rainfall_std = rainfall.std()
        production_mean = production.mean()
        production_std = production.std()

        rainfall_z = (rainfall - rainfall_mean) / rainfall_std if rainfall_std > 0 else np.zeros_like(rainfall)
        production_z = (production - production_mean) / production_std if production_std > 0 else np.zeros_like(production)

        impact_years = []
        for i, year in enumerate(years):
            if abs(rainfall_z[i]) > threshold_std:
                impact_type = "Drought" if rainfall_z[i] < 0 else "Excess Rainfall"
                production_impact = "Decreased" if production_z[i] < 0 else "Increased"

                impact_years.append({
                    'year': year,
                    'rainfall_mm': rainfall[i],
                    'rainfall_anomaly': f"{rainfall_z[i]:+.2f} σ",
                    'production': production[i],
                    'production_anomaly': f"{production_z[i]:+.2f} σ",
                    'event_type': impact_type,
                    'production_impact': production_impact,
                    'severity': 'High' if abs(rainfall_z[i]) > 2 else 'Moderate'
                })

        return {
            'state': integrated_data['state'],
            'crop': integrated_data['crop'],
            'impact_years': impact_years,
            'total_anomalous_years': len(impact_years),
            'analysis_period': f"{min(years)}-{max(years)}",
            'threshold': f"{threshold_std} standard deviations"
        }

    # ===============================================================
    # CROSS-DOMAIN INSIGHTS
    # ===============================================================

    def generate_cross_domain_insights(self, crop_df: pd.DataFrame,
                                       rainfall_df: pd.DataFrame,
                                       state_name: str) -> Dict:
        """
        Generate comprehensive insights combining crop and rainfall data.
        Key function for policymakers.
        """
        insights = {
            'state': state_name,
            'geographic_coverage': {},
            'temporal_coverage': {},
            'correlations': {},
            'trends': {},
            'anomalies': {},
            'recommendations': []
        }

        # 1. Geographic coverage
        subdivisions = self.geo_mapper.get_subdivisions_for_state(state_name)
        insights['geographic_coverage'] = {
            'rainfall_subdivisions': subdivisions,
            'subdivision_count': len(subdivisions),
            'note': 'Data aggregated from multiple meteorological subdivisions'
        }

        # 2. Temporal coverage
        rainfall_result = self.aggregate_rainfall_to_state(rainfall_df, state_name)
        crop_result = self.aggregate_crop_to_state(crop_df, state_name)

        if 'error' not in rainfall_result and 'error' not in crop_result:
            rainfall_years = set(rainfall_result['years'])
            crop_years = set(crop_result['years'])

            insights['temporal_coverage'] = {
                'rainfall_data_years': f"{min(rainfall_years)}-{max(rainfall_years)}",
                'crop_data_years': f"{min(crop_years)}-{max(crop_years)}",
                'overlapping_years': sorted(rainfall_years & crop_years),
                'overlap_count': len(rainfall_years & crop_years)
            }

            # 3. Analyze top crops
            available_crops = crop_result['crops_available'][:5]  # Top 5 crops
            for crop in available_crops:
                integrated = self.integrate_crop_rainfall(crop_df, rainfall_df, state_name, crop)
                if 'error' not in integrated:
                    corr_analysis = self.analyze_correlation(integrated)
                    if 'error' not in corr_analysis:
                        insights['correlations'][crop] = {
                            'correlation': corr_analysis['correlation']['pearson_r'],
                            'strength': corr_analysis['correlation']['strength'],
                            'significant': corr_analysis['correlation']['significant']
                        }

        # 4. Generate recommendations based on correlations
        if insights['correlations']:
            strong_correlations = [
                crop for crop, data in insights['correlations'].items()
                if abs(data['correlation']) > 0.6
            ]

            if strong_correlations:
                insights['recommendations'].append(
                    f"Strong rainfall dependency detected for: {', '.join(strong_correlations)}. "
                    f"Consider irrigation infrastructure and water management policies."
                )

            negative_correlations = [
                crop for crop, data in insights['correlations'].items()
                if data['correlation'] < -0.4
            ]

            if negative_correlations:
                insights['recommendations'].append(
                    f"Crops showing negative rainfall correlation: {', '.join(negative_correlations)}. "
                    f"Investigate drainage issues or excess water damage."
                )

        return insights
