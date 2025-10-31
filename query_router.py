"""
Intelligent Query Router for Project Samarth
Understands complex natural language queries and routes to appropriate handlers
Handles ambiguity and provides helpful suggestions
"""

import re
from typing import Dict, List, Optional, Tuple
from geo_mapper import GeographicMapper

class IntelligentQueryRouter:
    """
    Advanced NLP query parser that understands:
    - Geographic references (states, districts, subdivisions)
    - Temporal references (years, ranges, decades)
    - Agricultural concepts (crops, production, yield)
    - Climate concepts (rainfall, drought, monsoon)
    - Analytical intents (correlation, trend, comparison)
    """
    
    def __init__(self, geo_mapper: GeographicMapper, 
                 available_crops: List[str], 
                 available_states: List[str],
                 available_subdivisions: List[str]):
        self.geo_mapper = geo_mapper
        self.available_crops = [c.lower() for c in available_crops]
        self.available_states = [s.lower() for s in available_states]
        self.available_subdivisions = [s.lower() for s in available_subdivisions]
        
        # Intent patterns
        self.intent_keywords = {
            'correlation': ['correlation', 'relationship', 'impact', 'affect', 'influence', 
                          'depend', 'связь', 'effect of', 'how does', 'связано'],
            'trend': ['trend', 'change over time', 'increasing', 'decreasing', 
                     'over years', 'historical', 'evolution', 'pattern'],
            'comparison': ['compare', 'difference', 'vs', 'versus', 'between', 
                          'higher', 'lower', 'more than', 'less than'],
            'drought': ['drought', 'dry years', 'low rainfall', 'water scarcity', 
                       'below average'],
            'excess': ['flood', 'excess', 'high rainfall', 'wet years', 'above average'],
            'extreme': ['wettest', 'driest', 'highest', 'lowest', 'maximum', 'minimum', 
                       'extreme', 'record'],
            'seasonal': ['monsoon', 'winter', 'summer', 'season', 'jjas', 'seasonal'],
            'statistics': ['average', 'mean', 'median', 'statistics', 'summary', 
                          'typical', 'normal'],
            'forecast': ['predict', 'forecast', 'future', 'projection', 'estimate'],
            'anomaly': ['anomaly', 'unusual', 'abnormal', 'outlier', 'exceptional'],
            'cross_domain': ['insights', 'overview', 'comprehensive', 'holistic', 
                           'integrated', 'complete picture']
        }
        
        # Crop keywords
        self.crop_keywords = ['crop', 'production', 'yield', 'harvest', 'cultivation', 
                             'farming', 'agriculture', 'grown', 'produce']
        
        # Climate keywords
        self.climate_keywords = ['rainfall', 'rain', 'precipitation', 'monsoon', 
                                'weather', 'climate', 'water']
    
    def parse_query(self, question: str) -> Dict:
        """
        Parse natural language question into structured query
        Returns comprehensive query structure with confidence scores
        """
        question_lower = question.lower()
        
        # Identify primary intent
        primary_intent = self._identify_primary_intent(question_lower)
        
        # Extract geographic entities
        geographic_entities = self._extract_geographic_entities(question)
        
        # Extract temporal references
        temporal_info = self._extract_temporal_info(question_lower)
        
        # Extract crop references
        crops = self._extract_crops(question_lower)
        
        # Determine data sources needed
        needs_rainfall = any(kw in question_lower for kw in self.climate_keywords)
        needs_crop = any(kw in question_lower for kw in self.crop_keywords) or len(crops) > 0
        needs_integration = 'correlation' in primary_intent or 'cross_domain' in primary_intent
        
        # Extract numerical constraints
        constraints = self._extract_constraints(question_lower)
        
        # Determine query complexity
        complexity = self._assess_complexity(
            primary_intent, geographic_entities, temporal_info, crops
        )
        
        # Generate suggestions if query is ambiguous
        suggestions = []
        confidence = 1.0
        
        if not geographic_entities['states'] and not geographic_entities['subdivisions']:
            suggestions.append("Please specify a state or region (e.g., Kerala, West Bengal)")
            confidence *= 0.5
        
        if needs_crop and not crops:
            suggestions.append("Specify a crop for more detailed analysis (e.g., rice, wheat, coconut)")
            confidence *= 0.7
        
        return {
            'original_question': question,
            'primary_intent': primary_intent,
            'secondary_intents': self._identify_secondary_intents(question_lower),
            'geographic_entities': geographic_entities,
            'temporal_info': temporal_info,
            'crops': crops,
            'data_sources': {
                'rainfall': needs_rainfall,
                'crop': needs_crop,
                'integrated': needs_integration
            },
            'constraints': constraints,
            'complexity': complexity,
            'confidence': round(confidence, 2),
            'suggestions': suggestions,
            'parseable': confidence > 0.3
        }
    
    def _identify_primary_intent(self, question: str) -> str:
        """Identify the main intent of the query"""
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for kw in keywords if kw in question)
            if score > 0:
                # Weight by keyword specificity
                intent_scores[intent] = score * (1 + len(keywords) / 100)
        
        if not intent_scores:
            # Default based on question structure
            if '?' in question:
                if 'what' in question or 'show' in question:
                    return 'statistics'
                elif 'how' in question:
                    return 'correlation'
                elif 'which' in question or 'find' in question:
                    return 'extreme'
            return 'statistics'
        
        return max(intent_scores, key=intent_scores.get)
    
    def _identify_secondary_intents(self, question: str) -> List[str]:
        """Identify secondary intents"""
        secondary = []
        
        for intent, keywords in self.intent_keywords.items():
            if any(kw in question for kw in keywords):
                secondary.append(intent)
        
        return secondary[:3]  # Return top 3
    
    def _extract_geographic_entities(self, question: str) -> Dict:
        """Extract all geographic references"""
        entities = {
            'states': [],
            'subdivisions': [],
            'districts': [],
            'mapped_locations': []
        }
        
        question_lower = question.lower()
        
        # Extract states
        for state in self.available_states:
            state_normalized = self.geo_mapper._normalize_name(state)
            if state_normalized in question_lower:
                entities['states'].append(state)
                
                # Also get corresponding subdivisions
                subdivisions = self.geo_mapper.get_subdivisions_for_state(state)
                if subdivisions:
                    entities['mapped_locations'].append({
                        'state': state,
                        'subdivisions': subdivisions
                    })
        
        # Extract subdivisions directly mentioned
        for subdiv in self.available_subdivisions:
            subdiv_lower = subdiv.lower()
            if subdiv_lower in question_lower:
                entities['subdivisions'].append(subdiv)
                
                # Get corresponding states
                states = self.geo_mapper.get_states_for_subdivision(subdiv)
                if states and not any(s in entities['states'] for s in states):
                    entities['states'].extend(states)
        
        # Use geo_mapper for smart matching
        if not entities['states']:
            state_from_query = self.geo_mapper.find_state_from_query(question)
            if state_from_query:
                entities['states'].append(state_from_query)
        
        if not entities['subdivisions'] and entities['states']:
            # Get subdivisions for found states
            for state in entities['states']:
                subdivs = self.geo_mapper.get_subdivisions_for_state(state)
                entities['subdivisions'].extend(subdivs)
        
        # Extract districts (pattern: "District_N" or "district name")
        district_pattern = r'district[_ ](\w+)'
        districts = re.findall(district_pattern, question_lower)
        entities['districts'] = districts
        
        return entities
    
    def _extract_temporal_info(self, question: str) -> Dict:
        """Extract temporal references"""
        temporal = {
            'type': None,
            'start_year': None,
            'end_year': None,
            'specific_year': None,
            'period': None
        }
        
        # Pattern: "from YYYY to YYYY"
        range_match = re.search(r'from (\d{4}) to (\d{4})', question)
        if range_match:
            temporal['type'] = 'range'
            temporal['start_year'] = int(range_match.group(1))
            temporal['end_year'] = int(range_match.group(2))
            return temporal
        
        # Pattern: "between YYYY and YYYY"
        range_match = re.search(r'between (\d{4}) and (\d{4})', question)
        if range_match:
            temporal['type'] = 'range'
            temporal['start_year'] = int(range_match.group(1))
            temporal['end_year'] = int(range_match.group(2))
            return temporal
        
        # Pattern: "in YYYY"
        single_match = re.search(r'in (\d{4})', question)
        if single_match:
            temporal['type'] = 'specific'
            temporal['specific_year'] = int(single_match.group(1))
            return temporal
        
        # Pattern: "last N years"
        last_n_match = re.search(r'last (\d+) years?', question)
        if last_n_match:
            n = int(last_n_match.group(1))
            temporal['type'] = 'last_n'
            temporal['period'] = n
            temporal['end_year'] = 2017  # Assuming latest available
            temporal['start_year'] = 2017 - n
            return temporal
        
        # Pattern: "past N years"
        past_n_match = re.search(r'past (\d+) years?', question)
        if past_n_match:
            n = int(past_n_match.group(1))
            temporal['type'] = 'last_n'
            temporal['period'] = n
            temporal['end_year'] = 2017
            temporal['start_year'] = 2017 - n
            return temporal
        
        # Pattern: "YYYY to YYYY" (without "from")
        simple_range = re.search(r'(\d{4})\s*to\s*(\d{4})', question)
        if simple_range:
            temporal['type'] = 'range'
            temporal['start_year'] = int(simple_range.group(1))
            temporal['end_year'] = int(simple_range.group(2))
            return temporal
        
        # Pattern: decade (1990s, 2000s)
        decade_match = re.search(r'(\d{3})0s', question)
        if decade_match:
            start = int(decade_match.group(1) + '0')
            temporal['type'] = 'decade'
            temporal['start_year'] = start
            temporal['end_year'] = start + 9
            temporal['period'] = 'decade'
            return temporal
        
        return temporal
    
    def _extract_crops(self, question: str) -> List[str]:
        """Extract crop names from question"""
        found_crops = []
        
        for crop in self.available_crops:
            if crop in question:
                found_crops.append(crop)
        
        # Handle plurals and variations
        variations = {
            'coconuts': 'coconut',
            'cashews': 'cashewnut',
            'sugarcanes': 'sugarcane',
            'potatoes': 'sweet pot.',  # Adjust based on your data
        }
        
        for variant, standard in variations.items():
            if variant in question and standard.lower() in self.available_crops:
                if standard.lower() not in [c.lower() for c in found_crops]:
                    found_crops.append(standard)
        
        return found_crops
    
    def _extract_constraints(self, question: str) -> Dict:
        """Extract numerical and logical constraints"""
        constraints = {
            'top_n': None,
            'threshold': None,
            'comparison_operator': None,
            'aggregation': None
        }
        
        # Top N
        top_match = re.search(r'top (\d+)', question)
        if top_match:
            constraints['top_n'] = int(top_match.group(1))
        
        # Threshold values
        threshold_match = re.search(r'(?:above|below|over|under|more than|less than)\s*(\d+(?:\.\d+)?)', question)
        if threshold_match:
            constraints['threshold'] = float(threshold_match.group(1))
            
            if any(word in question for word in ['above', 'over', 'more than']):
                constraints['comparison_operator'] = '>'
            else:
                constraints['comparison_operator'] = '<'
        
        # Aggregation type
        if 'total' in question or 'sum' in question:
            constraints['aggregation'] = 'sum'
        elif 'average' in question or 'mean' in question:
            constraints['aggregation'] = 'mean'
        elif 'maximum' in question or 'highest' in question:
            constraints['aggregation'] = 'max'
        elif 'minimum' in question or 'lowest' in question:
            constraints['aggregation'] = 'min'
        
        return constraints
    
    def _assess_complexity(self, intent: str, geographic: Dict, 
                          temporal: Dict, crops: List[str]) -> str:
        """Assess query complexity"""
        complexity_score = 0
        
        # Multiple locations increase complexity
        total_locations = len(geographic['states']) + len(geographic['subdivisions'])
        if total_locations > 2:
            complexity_score += 2
        elif total_locations > 0:
            complexity_score += 1
        
        # Temporal complexity
        if temporal['type'] == 'range':
            year_span = temporal.get('end_year', 0) - temporal.get('start_year', 0)
            if year_span > 50:
                complexity_score += 2
            elif year_span > 10:
                complexity_score += 1
        
        # Multiple crops
        if len(crops) > 1:
            complexity_score += 1
        
        # Intent complexity
        complex_intents = ['correlation', 'cross_domain', 'anomaly', 'forecast']
        if intent in complex_intents:
            complexity_score += 2
        
        if complexity_score >= 5:
            return 'high'
        elif complexity_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def generate_clarifying_questions(self, parsed_query: Dict) -> List[str]:
        """Generate clarifying questions for ambiguous queries"""
        questions = []
        
        geo = parsed_query['geographic_entities']
        
        # Check for ambiguous locations
        if len(geo['states']) > 3:
            questions.append(
                f"You mentioned {len(geo['states'])} states. Would you like to focus on specific ones?"
            )
        
        # Check for missing temporal info
        if not parsed_query['temporal_info']['type']:
            questions.append(
                "What time period are you interested in? (e.g., last 10 years, 2000-2010)"
            )
        
        # Check for crop ambiguity
        if parsed_query['data_sources']['crop'] and not parsed_query['crops']:
            questions.append(
                "Which specific crop would you like to analyze?"
            )
        
        return questions
    
    def suggest_related_queries(self, parsed_query: Dict) -> List[str]:
        """Suggest related queries user might be interested in"""
        suggestions = []
        
        intent = parsed_query['primary_intent']
        states = parsed_query['geographic_entities']['states']
        crops = parsed_query['crops']
        
        if states and intent == 'trend':
            suggestions.append(
                f"Compare rainfall trends across {' and '.join(states[:2])}"
            )
        
        if crops and states:
            suggestions.append(
                f"Analyze correlation between rainfall and {crops[0]} in {states[0]}"
            )
        
        if intent == 'statistics':
            suggestions.append(
                "Identify drought years and their impact on production"
            )
        
        return suggestions[:3]  # Top 3 suggestions