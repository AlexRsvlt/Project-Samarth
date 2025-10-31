"""
Project Samarth - Enterprise Grade Application
Handles cross-ministry data integration with intelligent query routing
Built for policymakers and researchers
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import traceback

# Import our custom modules
from geo_mapper import GeographicMapper
from data_aggregator import SmartDataAggregator
from query_router import IntelligentQueryRouter
from data_loader import CropDataLoader, RainfallDataLoader

app = Flask(__name__)

# Global instances
crop_loader = None
rainfall_loader = None
geo_mapper = None
aggregator = None
query_router = None

def initialize_system():
    """Initialize all system components"""
    global crop_loader, rainfall_loader, geo_mapper, aggregator, query_router
    
    print("\n" + "="*80)
    print("🌾 PROJECT SAMARTH - ENTERPRISE SYSTEM INITIALIZATION")
    print("="*80)
    
    # Paths - update these to match your files
    CROP_PATH = "data/crop_production.csv"
    RAINFALL_PATH = "data/rainfall_india.csv"
    
    success = True
    
    # Load crop data
    print("\n📊 Loading Crop Production Data...")
    if os.path.exists(CROP_PATH):
        try:
            crop_loader = CropDataLoader(CROP_PATH)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            success = False
    else:
        print(f"   ⚠️  File not found: {CROP_PATH}")
        success = False
    
    # Load rainfall data
    print("\n🌧️  Loading Rainfall Data...")
    if os.path.exists(RAINFALL_PATH):
        try:
            rainfall_loader = RainfallDataLoader(RAINFALL_PATH)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            success = False
    else:
        print(f"   ⚠️  File not found: {RAINFALL_PATH}")
        success = False
    
    if not (crop_loader and rainfall_loader):
        print("\n❌ System initialization failed - missing data files")
        print("="*80 + "\n")
        return False
    
    # Initialize geographic mapper
    print("\n🗺️  Initializing Geographic Mapping System...")
    try:
        geo_mapper = GeographicMapper()
        
        # Validate mapping coverage
        coverage = geo_mapper.validate_geographic_coverage(
            crop_loader.states,
            rainfall_loader.subdivisions
        )
        
        print(f"   ✓ Mapped {coverage['states_with_rainfall_data']}/{coverage['total_crop_states']} states")
        
        if coverage['states_without_rainfall_data']:
            print(f"   ⚠️  States without rainfall mapping: {len(coverage['states_without_rainfall_data'])}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        success = False
    
    # Initialize data aggregator
    print("\n📈 Initializing Smart Data Aggregator...")
    try:
        aggregator = SmartDataAggregator(geo_mapper)
        print("   ✓ Aggregator ready")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        success = False
    
    # Initialize query router
    print("\n🧠 Initializing Intelligent Query Router...")
    try:
        query_router = IntelligentQueryRouter(
            geo_mapper=geo_mapper,
            available_crops=crop_loader.get_crop_list() if crop_loader else [],
            available_states=crop_loader.states if crop_loader else [],
            available_subdivisions=rainfall_loader.subdivisions if rainfall_loader else []
        )
        print("   ✓ Query router ready")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        success = False
    
    if success:
        print("\n" + "="*80)
        print("✅ SYSTEM READY - All components initialized successfully")
        print("="*80 + "\n")
    
    return success

@app.route("/")
def index():
    """Main interface"""
    context = {
        'system_ready': all([crop_loader, rainfall_loader, geo_mapper, aggregator, query_router]),
        'stats': {}
    }
    
    if context['system_ready']:
        context['stats'] = {
            'states': len(crop_loader.states),
            'subdivisions': len(rainfall_loader.subdivisions),
            'crops': len(crop_loader.get_crop_list()),
            'crop_years': f"{min(crop_loader.years)}-{max(crop_loader.years)}",
            'rainfall_years': f"{min(rainfall_loader.years)}-{max(rainfall_loader.years)}",
            'mapped_states': len([s for s in crop_loader.states 
                                 if geo_mapper.get_subdivisions_for_state(s)])
        }
    
    return render_template("index_enterprise.html", **context)

@app.route("/query", methods=["POST"])
def process_query():
    """Main query processing endpoint"""
    if not all([crop_loader, rainfall_loader, geo_mapper, aggregator, query_router]):
        return jsonify({
            "error": "System not properly initialized",
            "status": "error"
        })
    
    question = request.form.get("question", "").strip()
    
    if not question:
        return jsonify({"error": "Please enter a question"})
    
    try:
        # Parse the query
        parsed = query_router.parse_query(question)
        
        # Check if query is parseable
        if not parsed['parseable']:
            return jsonify({
                "answer": "I couldn't understand your question clearly.",
                "suggestions": parsed['suggestions'],
                "clarifying_questions": query_router.generate_clarifying_questions(parsed),
                "parsed_query": parsed
            })
        
        # Route to appropriate handler
        result = route_and_execute(parsed)
        
        # Add parsed query info for debugging (optional)
        result['query_info'] = {
            'intent': parsed['primary_intent'],
            'complexity': parsed['complexity'],
            'confidence': parsed['confidence']
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing query: {e}")
        print(traceback.format_exc())
        return jsonify({
            "error": f"An error occurred: {str(e)}",
            "status": "error"
        })

def route_and_execute(parsed_query: dict) -> dict:
    """Route query to appropriate handler and execute"""
    intent = parsed_query['primary_intent']
    geo = parsed_query['geographic_entities']
    temporal = parsed_query['temporal_info']
    crops = parsed_query['crops']
    
    # Extract year range
    year_range = None
    if temporal['type'] in ['range', 'last_n', 'decade']:
        year_range = (temporal['start_year'], temporal['end_year'])
    
    # Get primary state
    primary_state = geo['states'][0] if geo['states'] else None
    
    try:
        # Route based on intent
        if intent == 'correlation':
            return handle_correlation_intent(primary_state, crops, year_range)
        
        elif intent == 'trend':
            if parsed_query['data_sources']['rainfall']:
                return handle_rainfall_trend(primary_state, year_range)
            else:
                return handle_crop_trend(primary_state, crops, year_range)
        
        elif intent == 'comparison':
            if len(geo['states']) >= 2:
                return handle_state_comparison(geo['states'][:3], year_range)
            else:
                return {"error": "Please specify at least 2 states to compare"}
        
        elif intent in ['drought', 'excess', 'extreme']:
            return handle_extreme_events(primary_state, intent, year_range)
        
        elif intent == 'seasonal':
            return handle_seasonal_analysis(primary_state, year_range)
        
        elif intent == 'cross_domain':
            return handle_cross_domain_insights(primary_state)
        
        elif intent == 'anomaly':
            return handle_anomaly_detection(primary_state, crops, year_range)
        
        else:  # statistics or default
            return handle_statistics_query(primary_state, crops, year_range, parsed_query)
            
    except Exception as e:
        return {
            "error": f"Error executing query: {str(e)}",
            "details": traceback.format_exc()
        }

def handle_correlation_intent(state: str, crops: list, year_range: tuple) -> dict:
    """Handle correlation analysis queries"""
    if not state:
        return {"error": "Please specify a state for correlation analysis"}
    
    crop = crops[0] if crops else None
    if not crop:
        # Use first available crop
        available_crops = crop_loader.get_crop_list()
        crop = available_crops[0] if available_crops else None
    
    if not crop:
        return {"error": "No crop data available"}
    
    # Get integrated data
    integrated = aggregator.integrate_crop_rainfall(
        crop_loader.df,
        rainfall_loader.df,
        state,
        crop,
        year_range
    )
    
    if 'error' in integrated:
        return integrated
    
    # Perform correlation analysis
    correlation_result = aggregator.analyze_correlation(integrated)
    
    if 'error' in correlation_result:
        return correlation_result
    
    # Format answer
    corr_data = correlation_result['correlation']
    answer = (
        f"Analysis of {crop} production and rainfall in {state} shows a "
        f"{corr_data['strength']} {corr_data['direction']} correlation "
        f"(Pearson r = {corr_data['pearson_r']}, p-value = {corr_data['pearson_p_value']}). "
    )
    
    if corr_data['significant']:
        answer += "This relationship is statistically significant. "
    
    answer += correlation_result['regression']['interpretation'] + ". "
    answer += f"Analysis based on {correlation_result['data_quality']['valid_data_points']} years of data."
    
    # Prepare chart
    chart_data = {
        'type': 'scatter',
        'labels': integrated['years'],
        'datasets': [{
            'label': f'{crop} vs Rainfall',
            'data': [{'x': r, 'y': p} for r, p in zip(integrated['rainfall'], integrated['production'])]
        }]
    }
    
    return {
        "answer": answer,
        "chart": chart_data,
        "sources": ["IMD", "MoAFW"],
        "details": correlation_result,
        "note": integrated['note']
    }

def handle_rainfall_trend(state: str, year_range: tuple) -> dict:
    """Handle rainfall trend analysis"""
    if not state:
        return {"error": "Please specify a state"}
    
    result = aggregator.aggregate_rainfall_to_state(
        rainfall_loader.df,
        state,
        year_range
    )
    
    if 'error' in result:
        return result
    
    # Calculate trend
    from scipy import stats as scipy_stats
    years = result['years']
    rainfall = result['annual_rainfall']
    
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(years, rainfall)
    
    trend = "increasing" if slope > 0 else "decreasing"
    decadal_change = slope * 10
    
    answer = (
        f"Rainfall in {state} shows a {trend} trend at {abs(decadal_change):.2f} mm/decade. "
        f"Average annual rainfall: {result['stats']['mean_annual']:.1f} mm "
        f"(range: {result['stats']['min_annual']:.1f} - {result['stats']['max_annual']:.1f} mm). "
        f"Monsoon contributes {result['stats']['monsoon_contribution']:.1f}% of annual rainfall. "
        f"{result['note']}"
    )
    
    chart_data = {
        'labels': years,
        'datasets': [{
            'label': f'{state} Annual Rainfall',
            'data': rainfall
        }]
    }
    
    return {
        "answer": answer,
        "chart": chart_data,
        "sources": ["IMD"],
        "stats": result['stats']
    }

def handle_state_comparison(states: list, year_range: tuple) -> dict:
    """Handle state comparison"""
    result = aggregator.compare_states_rainfall(
        rainfall_loader.df,
        states,
        year_range
    )
    
    if 'error' in result:
        return result
    
    comparisons = result['comparisons']
    answer_parts = []
    
    for comp in comparisons:
        answer_parts.append(
            f"{comp['state']}: {comp['mean_rainfall']:.1f} mm "
            f"(±{comp['std_rainfall']:.1f} mm)"
        )
    
    answer = "Average annual rainfall comparison: " + ", ".join(answer_parts) + "."
    
    return {
        "answer": answer,
        "chart": result['chart_data'],
        "sources": ["IMD"],
        "comparisons": comparisons
    }

def handle_extreme_events(state: str, event_type: str, year_range: tuple) -> dict:
    """Handle extreme event queries"""
    if not state:
        return {"error": "Please specify a state"}
    
    result = aggregator.aggregate_rainfall_to_state(
        rainfall_loader.df,
        state,
        year_range
    )
    
    if 'error' in result:
        return result
    
    # Create dataframe for analysis
    df = pd.DataFrame({
        'Year': result['years'],
        'Rainfall': result['annual_rainfall']
    })
    
    if event_type == 'drought':
        threshold = df['Rainfall'].quantile(0.25)
        events = df[df['Rainfall'] < threshold]
        answer = f"Drought years in {state} (rainfall < {threshold:.1f} mm): "
    elif event_type == 'excess':
        threshold = df['Rainfall'].quantile(0.75)
        events = df[df['Rainfall'] > threshold]
        answer = f"Excess rainfall years in {state} (rainfall > {threshold:.1f} mm): "
    else:  # extreme
        wettest = df.nlargest(5, 'Rainfall')
        driest = df.nsmallest(5, 'Rainfall')
        answer = f"Wettest years in {state}: {', '.join(map(str, wettest['Year'].tolist()))}. "
        answer += f"Driest years: {', '.join(map(str, driest['Year'].tolist()))}."
        return {"answer": answer, "sources": ["IMD"]}
    
    event_years = ', '.join(map(str, events['Year'].head(10).tolist()))
    if len(events) > 10:
        event_years += f", and {len(events) - 10} more"
    
    answer += event_years + "."
    
    return {
        "answer": answer,
        "sources": ["IMD"],
        "events": events.to_dict('records')
    }

def handle_seasonal_analysis(state: str, year_range: tuple) -> dict:
    """Handle seasonal rainfall analysis"""
    if not state:
        return {"error": "Please specify a state"}
    
    result = aggregator.aggregate_rainfall_to_state(
        rainfall_loader.df,
        state,
        year_range
    )
    
    if 'error' in result:
        return result
    
    answer = (
        f"Seasonal rainfall analysis for {state}: "
        f"Average monsoon (JJAS) rainfall is {result['stats']['mean_monsoon']:.1f} mm, "
        f"contributing {result['stats']['monsoon_contribution']:.1f}% of total annual rainfall."
    )
    
    chart_data = {
        'labels': result['years'],
        'datasets': [{
            'label': 'Monsoon Rainfall',
            'data': result['monsoon_rainfall']
        }]
    }
    
    return {
        "answer": answer,
        "chart": chart_data,
        "sources": ["IMD"]
    }

def handle_cross_domain_insights(state: str) -> dict:
    """Generate comprehensive cross-domain insights"""
    if not state:
        return {"error": "Please specify a state"}
    
    insights = aggregator.generate_cross_domain_insights(
        crop_loader.df,
        rainfall_loader.df,
        state
    )
    
    # Format answer
    answer_parts = [f"Comprehensive analysis for {state}:"]
    
    if insights['temporal_coverage'].get('overlap_count', 0) > 0:
        answer_parts.append(
            f"Data available for {insights['temporal_coverage']['overlap_count']} overlapping years."
        )
    
    if insights['correlations']:
        strong_crops = [crop for crop, data in insights['correlations'].items() 
                       if abs(data['correlation']) > 0.6]
        if strong_crops:
            answer_parts.append(
                f"Strong rainfall-production correlation found for: {', '.join(strong_crops)}."
            )
    
    for rec in insights['recommendations']:
        answer_parts.append(rec)
    
    answer = " ".join(answer_parts)
    
    return {
        "answer": answer,
        "sources": ["IMD", "MoAFW"],
        "insights": insights
    }

def handle_anomaly_detection(state: str, crops: list, year_range: tuple) -> dict:
    """Detect anomalous years"""
    if not state or not crops:
        return {"error": "Please specify state and crop for anomaly detection"}
    
    crop = crops[0]
    
    integrated = aggregator.integrate_crop_rainfall(
        crop_loader.df,
        rainfall_loader.df,
        state,
        crop,
        year_range
    )
    
    if 'error' in integrated:
        return integrated
    
    anomalies = aggregator.identify_impact_years(integrated)
    
    if 'error' in anomalies:
        return anomalies
    
    if not anomalies['impact_years']:
        answer = f"No significant anomalies detected for {crop} in {state} during the analysis period."
    else:
        answer = (
            f"Detected {anomalies['total_anomalous_years']} anomalous years for {crop} in {state}. "
            f"Notable events: "
        )
        
        for event in anomalies['impact_years'][:3]:
            answer += f"{event['year']} ({event['event_type']}, {event['severity']} severity), "
        
        answer = answer.rstrip(', ') + "."
    
    return {
        "answer": answer,
        "sources": ["IMD", "MoAFW"],
        "anomalies": anomalies
    }

def handle_statistics_query(state: str, crops: list, year_range: tuple, parsed_query: dict) -> dict:
    """Handle general statistics queries"""
    if not state:
        return {"error": "Please specify a state or region"}
    
    # Determine if rainfall or crop query
    if parsed_query['data_sources']['rainfall']:
        result = aggregator.aggregate_rainfall_to_state(
            rainfall_loader.df,
            state,
            year_range
        )
        
        if 'error' in result:
            return result
        
        stats = result['stats']
        answer = (
            f"Rainfall statistics for {state}: "
            f"Mean = {stats['mean_annual']:.1f} mm, "
            f"Std Dev = {stats['std_annual']:.1f} mm, "
            f"Range = {stats['min_annual']:.1f} - {stats['max_annual']:.1f} mm. "
            f"Monsoon contributes {stats['monsoon_contribution']:.1f}% of annual rainfall. "
            f"{result['note']}"
        )
        
        return {
            "answer": answer,
            "sources": ["IMD"],
            "stats": stats
        }
    
    elif parsed_query['data_sources']['crop']:
        crop = crops[0] if crops else None
        result = aggregator.aggregate_crop_to_state(
            crop_loader.df,
            state,
            crop,
            year_range
        )
        
        if 'error' in result:
            return result
        
        answer = (
            f"Crop data for {state}: "
            f"{result['districts_count']} districts, "
            f"{len(result['crops_available'])} crops available. "
            f"{result['note']}"
        )
        
        return {
            "answer": answer,
            "sources": ["MoAFW"],
            "available_crops": result['crops_available']
        }
    
    return {"error": "Could not determine query type"}

# API endpoints
@app.route("/api/validate_location", methods=["POST"])
def validate_location():
    """Validate if a location exists and suggest alternatives"""
    location = request.json.get('location', '')
    
    suggestions = geo_mapper.suggest_similar_locations(
        location,
        crop_loader.states + rainfall_loader.subdivisions
    )
    
    return jsonify({
        "valid": len(suggestions) > 0,
        "suggestions": suggestions
    })

@app.route("/api/system_status")
def system_status():
    """Get system status"""
    return jsonify({
        "status": "operational" if all([crop_loader, rainfall_loader, geo_mapper, aggregator, query_router]) else "degraded",
        "components": {
            "crop_data": crop_loader is not None,
            "rainfall_data": rainfall_loader is not None,
            "geo_mapper": geo_mapper is not None,
            "aggregator": aggregator is not None,
            "query_router": query_router is not None
        }
    })

@app.route("/health")
def health_check():
    """Health check for monitoring"""
    return jsonify({
        "status": "healthy",
        "version": "2.0-enterprise"
    })

if __name__ == "__main__":
    # Initialize system
    if initialize_system():
        print("🚀 Starting Flask server on http://localhost:5000")
        print("   Press Ctrl+C to stop\n")
        app.run(debug=True, port=5000, threaded=True)
    else:
        print("\n❌ Failed to initialize system. Please check your data files and try again.\n")