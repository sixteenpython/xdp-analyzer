"""
Enhanced Intelligent Analyzer
Free LLM-powered analysis with advanced pattern recognition and fallback mechanisms
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import aiohttp
import time

# Try importing various free LLM options
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

@dataclass
class BusinessInsight:
    """Business insight with confidence scoring"""
    category: str
    insight: str
    confidence: float
    evidence: List[str]
    business_impact: str
    recommendation: str

@dataclass
class ModelAssessment:
    """Financial model assessment"""
    model_type: str
    sophistication_level: str
    key_components: List[str]
    risk_indicators: List[str]
    validation_status: str
    compliance_notes: List[str]

@dataclass
class IntelligentAnalysisResult:
    """Complete intelligent analysis result"""
    analysis_timestamp: str
    confidence_score: float
    key_insights: List[str]
    business_purpose: str
    model_assessment: Optional[ModelAssessment]
    data_quality_assessment: Dict[str, Any]
    risk_indicators: List[str]
    recommendations: List[str]
    business_insights: List[BusinessInsight]
    technical_complexity: str
    automation_opportunities: List[str]
    regulatory_considerations: List[str]
    next_steps: List[str]

class FreeIntelligentAnalyzer:
    """
    Enhanced Intelligent Analyzer using free LLM services with fallback mechanisms
    Provides comprehensive analysis with pattern recognition and business intelligence
    """
    
    def __init__(self, use_llm: bool = True):
        self.logger = logging.getLogger(__name__)
        self.use_llm = use_llm and HAS_REQUESTS
        
        # Free LLM endpoints (these would need to be configured based on available services)
        self.llm_endpoints = {
            'huggingface': 'https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium',
            'ollama_local': 'http://localhost:11434/api/generate',  # If Ollama is running locally
            'openai_free': None  # Would need API key
        }
        
        # Pattern recognition templates
        self.business_patterns = {
            'financial_modeling': {
                'keywords': ['var', 'volatility', 'correlation', 'portfolio', 'risk', 'return', 'yield', 'option'],
                'functions': ['CORREL', 'COVAR', 'STDEV', 'VAR', 'PERCENTILE', 'NORMDIST'],
                'purpose': 'Financial risk management and modeling',
                'sophistication_indicators': ['monte carlo', 'black scholes', 'binomial', 'garch']
            },
            'forecasting': {
                'keywords': ['forecast', 'trend', 'prediction', 'projection', 'future', 'estimate'],
                'functions': ['TREND', 'FORECAST', 'LINEST', 'GROWTH', 'LOGEST'],
                'purpose': 'Predictive analysis and forecasting',
                'sophistication_indicators': ['arima', 'exponential smoothing', 'seasonal decomposition']
            },
            'dashboard': {
                'keywords': ['kpi', 'metric', 'dashboard', 'scorecard', 'performance', 'monitoring'],
                'functions': ['SUMIF', 'COUNTIF', 'PIVOT', 'INDEX', 'MATCH'],
                'purpose': 'Business intelligence and reporting',
                'sophistication_indicators': ['automated refresh', 'drill-down', 'interactive filters']
            },
            'optimization': {
                'keywords': ['optimize', 'minimize', 'maximize', 'constraint', 'solver', 'linear program'],
                'functions': ['SOLVER', 'GRG', 'SUMPRODUCT'],
                'purpose': 'Mathematical optimization and resource allocation',
                'sophistication_indicators': ['sensitivity analysis', 'scenario optimization', 'multi-objective']
            },
            'simulation': {
                'keywords': ['simulation', 'monte carlo', 'random', 'scenario', 'what if'],
                'functions': ['RAND', 'RANDBETWEEN', 'NORM.INV', 'UNIFORM'],
                'purpose': 'Stochastic modeling and scenario analysis',
                'sophistication_indicators': ['latin hypercube', 'variance reduction', 'convergence testing']
            }
        }
        
        # Risk assessment patterns
        self.risk_patterns = {
            'data_quality_risks': [
                'High percentage of missing values',
                'Inconsistent data formats',
                'Outliers without validation',
                'Circular references detected',
                'Manual data entry points'
            ],
            'model_risks': [
                'Complex nested formulas without documentation',
                'No model validation or testing',
                'Single point of failure calculations',
                'Hard-coded assumptions',
                'Lack of version control'
            ],
            'operational_risks': [
                'Manual calculation processes',
                'No error checking mechanisms',
                'Dependence on external data sources',
                'Limited access controls',
                'No backup or recovery procedures'
            ],
            'compliance_risks': [
                'Missing audit trail',
                'Inadequate documentation',
                'No independent validation',
                'Regulatory requirements not addressed',
                'Model governance gaps'
            ]
        }
        
        # Sophistication level indicators
        self.sophistication_indicators = {
            'basic': {
                'score_range': (0, 30),
                'characteristics': ['Simple calculations', 'Basic Excel functions', 'Limited complexity'],
                'typical_use': 'Basic reporting and simple analysis'
            },
            'intermediate': {
                'score_range': (30, 70),
                'characteristics': ['Advanced Excel functions', 'Some VBA code', 'Cross-sheet references'],
                'typical_use': 'Financial modeling and business analysis'
            },
            'advanced': {
                'score_range': (70, 100),
                'characteristics': ['Complex algorithms', 'Extensive VBA', 'Advanced statistical methods'],
                'typical_use': 'Quantitative risk management and sophisticated modeling'
            }
        }
    
    def analyze_excel_content(self, excel_analysis) -> IntelligentAnalysisResult:
        """
        Perform intelligent analysis of Excel content
        
        Args:
            excel_analysis: Excel analysis object from parser
            
        Returns:
            IntelligentAnalysisResult with comprehensive insights
        """
        self.logger.info("Starting intelligent analysis of Excel content")
        
        # Extract key information for analysis
        analysis_context = self._extract_analysis_context(excel_analysis)
        
        # Perform pattern recognition
        patterns_detected = self._detect_business_patterns(analysis_context)
        
        # Assess model sophistication
        model_assessment = self._assess_model_sophistication(analysis_context, patterns_detected)
        
        # Generate business insights
        business_insights = self._generate_business_insights(analysis_context, patterns_detected)
        
        # Assess data quality
        data_quality = self._assess_data_quality(analysis_context)
        
        # Identify risk indicators
        risk_indicators = self._identify_risk_indicators(analysis_context)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis_context, patterns_detected, risk_indicators)
        
        # Identify automation opportunities
        automation_opportunities = self._identify_automation_opportunities(analysis_context)
        
        # Assess regulatory considerations
        regulatory_considerations = self._assess_regulatory_considerations(patterns_detected)
        
        # Generate next steps
        next_steps = self._generate_next_steps(patterns_detected, risk_indicators)
        
        # Calculate overall confidence score
        confidence_score = self._calculate_confidence_score(analysis_context, patterns_detected)
        
        # Determine business purpose
        business_purpose = self._determine_business_purpose(patterns_detected)
        
        # Extract key insights
        key_insights = self._extract_key_insights(business_insights, patterns_detected)
        
        # Determine technical complexity
        technical_complexity = self._assess_technical_complexity(analysis_context)
        
        return IntelligentAnalysisResult(
            analysis_timestamp=datetime.now().isoformat(),
            confidence_score=confidence_score,
            key_insights=key_insights,
            business_purpose=business_purpose,
            model_assessment=model_assessment,
            data_quality_assessment=data_quality,
            risk_indicators=risk_indicators,
            recommendations=recommendations,
            business_insights=business_insights,
            technical_complexity=technical_complexity,
            automation_opportunities=automation_opportunities,
            regulatory_considerations=regulatory_considerations,
            next_steps=next_steps
        )
    
    def _extract_analysis_context(self, excel_analysis) -> Dict[str, Any]:
        """Extract key context information for analysis"""
        
        context = {
            'worksheets': [],
            'formulas': [],
            'functions_used': set(),
            'data_characteristics': {},
            'complexity_indicators': {},
            'external_references': [],
            'vba_present': False
        }
        
        try:
            # Handle different analysis types (simple or advanced)
            if hasattr(excel_analysis, 'worksheets'):  # Advanced analysis
                for worksheet in excel_analysis.worksheets:
                    ws_info = {
                        'name': worksheet.name,
                        'formula_count': len(worksheet.formulas),
                        'has_charts': len(worksheet.charts) > 0,
                        'has_pivot_tables': len(worksheet.pivot_tables) > 0,
                        'protected': worksheet.protected,
                        'hidden': worksheet.hidden
                    }
                    context['worksheets'].append(ws_info)
                    context['formulas'].extend(worksheet.formulas)
                
                # Extract functions from formulas
                for formula in context['formulas']:
                    functions = re.findall(r'([A-Z][A-Z0-9]*)\s*\(', formula.upper())
                    context['functions_used'].update(functions)
                
                context['external_references'] = excel_analysis.external_references
                context['vba_present'] = len(excel_analysis.vba_code) > 0
                
                if hasattr(excel_analysis, 'complexity_metrics'):
                    context['complexity_indicators'] = excel_analysis.complexity_metrics
                    
            elif hasattr(excel_analysis, 'sheets'):  # Simple analysis
                for sheet in excel_analysis.sheets:
                    ws_info = {
                        'name': sheet.name,
                        'shape': sheet.shape,
                        'data_types': sheet.data_types,
                        'null_counts': sheet.null_counts,
                        'business_patterns': getattr(sheet, 'business_patterns', []),
                        'financial_indicators': getattr(sheet, 'financial_indicators', {})
                    }
                    context['worksheets'].append(ws_info)
                
                # Extract characteristics from simple analysis
                context['data_characteristics'] = {
                    'sheet_count': excel_analysis.sheet_count,
                    'total_rows': excel_analysis.total_rows,
                    'has_time_series': excel_analysis.summary.get('has_time_series', False),
                    'has_financial_data': excel_analysis.summary.get('has_financial_data', False),
                    'primary_analysis_type': excel_analysis.summary.get('primary_analysis_type', 'unknown')
                }
        
        except Exception as e:
            self.logger.warning(f"Error extracting analysis context: {e}")
        
        return context
    
    def _detect_business_patterns(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect business patterns using rule-based analysis"""
        
        detected_patterns = {}
        
        # Combine all text for analysis
        all_text = []
        for ws in context['worksheets']:
            all_text.append(ws['name'].lower())
            if 'business_patterns' in ws:
                all_text.extend([str(p).lower() for p in ws['business_patterns']])
        
        # Add formulas
        formula_text = ' '.join(context['formulas']).lower()
        all_text.append(formula_text)
        
        # Functions used
        functions_text = ' '.join(context['functions_used']).lower()
        all_text.append(functions_text)
        
        combined_text = ' '.join(all_text)
        
        # Check against each business pattern
        for pattern_type, pattern_info in self.business_patterns.items():
            score = 0
            matches = []
            
            # Check keywords
            for keyword in pattern_info['keywords']:
                if keyword in combined_text:
                    score += 2
                    matches.append(f"keyword: {keyword}")
            
            # Check functions
            for function in pattern_info['functions']:
                if function.lower() in functions_text:
                    score += 3
                    matches.append(f"function: {function}")
            
            # Check sophistication indicators
            for indicator in pattern_info['sophistication_indicators']:
                if indicator in combined_text:
                    score += 5
                    matches.append(f"advanced: {indicator}")
            
            if score > 0:
                detected_patterns[pattern_type] = {
                    'score': score,
                    'matches': matches,
                    'purpose': pattern_info['purpose'],
                    'confidence': min(score / 10, 1.0)
                }
        
        return detected_patterns
    
    def _assess_model_sophistication(self, context: Dict[str, Any], patterns: Dict[str, Any]) -> Optional[ModelAssessment]:
        """Assess the sophistication of the financial model"""
        
        if not patterns:
            return None
        
        # Calculate overall sophistication score
        total_score = sum(p['score'] for p in patterns.values())
        pattern_count = len(patterns)
        avg_score = total_score / max(pattern_count, 1)
        
        # Determine sophistication level
        sophistication_level = 'basic'
        for level, info in self.sophistication_indicators.items():
            if info['score_range'][0] <= avg_score <= info['score_range'][1]:
                sophistication_level = level
                break
        
        # Identify key components
        key_components = []
        for pattern_type, pattern_data in patterns.items():
            key_components.append(pattern_data['purpose'])
        
        # Extract risk indicators
        risk_indicators = []
        if context.get('complexity_indicators'):
            complexity = context['complexity_indicators']
            if complexity.get('complexity_score', 0) > 100:
                risk_indicators.append('High model complexity detected')
            if complexity.get('nested_formulas_count', 0) > 20:
                risk_indicators.append('Extensive formula nesting may impact maintainability')
        
        # Determine validation status
        validation_status = 'needs_review'
        validation_keywords = ['test', 'validate', 'check', 'verify', 'audit']
        formula_text = ' '.join(context['formulas']).lower()
        if any(keyword in formula_text for keyword in validation_keywords):
            validation_status = 'some_validation_present'
        
        # Generate compliance notes
        compliance_notes = []
        if 'financial_modeling' in patterns:
            compliance_notes.append('Consider regulatory model validation requirements')
        if context.get('vba_present'):
            compliance_notes.append('VBA code requires additional governance controls')
        
        return ModelAssessment(
            model_type=', '.join(patterns.keys()),
            sophistication_level=sophistication_level,
            key_components=key_components[:5],  # Limit to top 5
            risk_indicators=risk_indicators[:3],  # Limit to top 3
            validation_status=validation_status,
            compliance_notes=compliance_notes[:3]  # Limit to top 3
        )
    
    def _generate_business_insights(self, context: Dict[str, Any], patterns: Dict[str, Any]) -> List[BusinessInsight]:
        """Generate business insights based on detected patterns"""
        
        insights = []
        
        # Generate insights for each detected pattern
        for pattern_type, pattern_data in patterns.items():
            if pattern_data['confidence'] > 0.3:  # Only include confident insights
                
                # Financial modeling insights
                if pattern_type == 'financial_modeling':
                    insight = BusinessInsight(
                        category='Financial Risk Management',
                        insight='Model demonstrates sophisticated financial risk management capabilities',
                        confidence=pattern_data['confidence'],
                        evidence=pattern_data['matches'][:3],
                        business_impact='High - Critical for risk assessment and regulatory compliance',
                        recommendation='Implement regular model validation and stress testing procedures'
                    )
                    insights.append(insight)
                
                # Forecasting insights
                elif pattern_type == 'forecasting':
                    insight = BusinessInsight(
                        category='Predictive Analytics',
                        insight='Model includes advanced forecasting and predictive capabilities',
                        confidence=pattern_data['confidence'],
                        evidence=pattern_data['matches'][:3],
                        business_impact='Medium-High - Supports strategic planning and decision making',
                        recommendation='Validate forecast accuracy and implement confidence intervals'
                    )
                    insights.append(insight)
                
                # Dashboard insights
                elif pattern_type == 'dashboard':
                    insight = BusinessInsight(
                        category='Business Intelligence',
                        insight='Model serves as a comprehensive business intelligence dashboard',
                        confidence=pattern_data['confidence'],
                        evidence=pattern_data['matches'][:3],
                        business_impact='Medium - Enables data-driven decision making',
                        recommendation='Consider automated data refresh and user access controls'
                    )
                    insights.append(insight)
                
                # Optimization insights
                elif pattern_type == 'optimization':
                    insight = BusinessInsight(
                        category='Operations Research',
                        insight='Model implements mathematical optimization for resource allocation',
                        confidence=pattern_data['confidence'],
                        evidence=pattern_data['matches'][:3],
                        business_impact='High - Direct impact on operational efficiency and cost reduction',
                        recommendation='Document optimization constraints and validate solution feasibility'
                    )
                    insights.append(insight)
                
                # Simulation insights
                elif pattern_type == 'simulation':
                    insight = BusinessInsight(
                        category='Risk Simulation',
                        insight='Model uses Monte Carlo simulation for scenario analysis',
                        confidence=pattern_data['confidence'],
                        evidence=pattern_data['matches'][:3],
                        business_impact='High - Critical for understanding risk distributions and tail events',
                        recommendation='Validate random number generation and convergence criteria'
                    )
                    insights.append(insight)
        
        # Add general insights based on complexity
        if context.get('complexity_indicators', {}).get('complexity_score', 0) > 150:
            insight = BusinessInsight(
                category='Model Complexity',
                insight='Highly complex model requires specialized expertise for maintenance',
                confidence=0.9,
                evidence=['High complexity score', 'Multiple interconnected components'],
                business_impact='High - Risk of operational disruption if key personnel unavailable',
                recommendation='Implement comprehensive documentation and cross-training programs'
            )
            insights.append(insight)
        
        return insights[:8]  # Limit to top 8 insights
    
    def _assess_data_quality(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality based on available information"""
        
        assessment = {
            'overall_score': 75,  # Default score
            'quality_dimensions': {},
            'issues_identified': [],
            'recommendations': []
        }
        
        # Analyze worksheet characteristics
        total_null_rate = 0
        worksheet_count = 0
        
        for ws in context['worksheets']:
            if 'null_counts' in ws and 'shape' in ws:
                total_cells = ws['shape'][0] * ws['shape'][1]
                null_cells = sum(ws['null_counts'].values())
                null_rate = null_cells / max(total_cells, 1)
                total_null_rate += null_rate
                worksheet_count += 1
        
        # Calculate average null rate
        avg_null_rate = total_null_rate / max(worksheet_count, 1)
        
        # Assess completeness
        completeness_score = max(0, 100 - (avg_null_rate * 100))
        assessment['quality_dimensions']['completeness'] = completeness_score
        
        if avg_null_rate > 0.1:
            assessment['issues_identified'].append(f'High missing data rate: {avg_null_rate:.1%}')
            assessment['recommendations'].append('Implement data validation and missing value imputation')
        
        # Assess consistency (based on data types)
        consistency_score = 85  # Default assumption
        assessment['quality_dimensions']['consistency'] = consistency_score
        
        # Assess accuracy (heuristic based on patterns)
        accuracy_score = 80  # Default assumption
        if context.get('external_references'):
            accuracy_score -= 10  # External refs increase uncertainty
            assessment['issues_identified'].append('Model depends on external data sources')
            assessment['recommendations'].append('Validate external data source reliability')
        
        assessment['quality_dimensions']['accuracy'] = accuracy_score
        
        # Calculate overall score
        dimension_scores = list(assessment['quality_dimensions'].values())
        assessment['overall_score'] = sum(dimension_scores) / len(dimension_scores) if dimension_scores else 75
        
        return assessment
    
    def _identify_risk_indicators(self, context: Dict[str, Any]) -> List[str]:
        """Identify potential risk indicators"""
        
        risks = []
        
        # Data quality risks
        for ws in context['worksheets']:
            if 'null_counts' in ws:
                total_nulls = sum(ws['null_counts'].values())
                if total_nulls > 0:
                    risks.append('Missing data detected in model inputs')
                    break
        
        # Model complexity risks
        if context.get('complexity_indicators', {}).get('complexity_score', 0) > 100:
            risks.append('High model complexity may impact maintainability and validation')
        
        # External dependency risks
        if context.get('external_references'):
            risks.append('Model has dependencies on external data sources')
        
        # VBA risks
        if context.get('vba_present'):
            risks.append('VBA code requires additional testing and version control')
        
        # Formula complexity risks
        if len(context.get('formulas', [])) > 100:
            risks.append('Large number of formulas increases error probability')
        
        # Single point of failure
        if len(context['worksheets']) == 1:
            risks.append('Model concentrated in single worksheet - potential single point of failure')
        
        # Lack of documentation
        doc_indicators = ['readme', 'documentation', 'help', 'instructions']
        has_docs = any(any(indicator in ws['name'].lower() for indicator in doc_indicators) 
                      for ws in context['worksheets'])
        if not has_docs:
            risks.append('Limited documentation may impact model understanding and maintenance')
        
        return risks[:10]  # Limit to top 10 risks
    
    def _generate_recommendations(self, context: Dict[str, Any], patterns: Dict[str, Any], 
                                risks: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Risk-based recommendations
        if any('missing data' in risk.lower() for risk in risks):
            recommendations.append('Implement comprehensive data validation and missing value handling procedures')
        
        if any('complexity' in risk.lower() for risk in risks):
            recommendations.append('Consider model simplification or modularization for better maintainability')
        
        if any('external' in risk.lower() for risk in risks):
            recommendations.append('Establish data source reliability monitoring and backup procedures')
        
        if any('vba' in risk.lower() for risk in risks):
            recommendations.append('Implement VBA code review, testing, and version control processes')
        
        # Pattern-based recommendations
        if 'financial_modeling' in patterns:
            recommendations.append('Establish regular model validation and backtesting procedures')
            recommendations.append('Consider regulatory model risk management framework implementation')
        
        if 'forecasting' in patterns:
            recommendations.append('Implement forecast accuracy tracking and model performance monitoring')
        
        if 'simulation' in patterns:
            recommendations.append('Validate random number generation and ensure proper convergence testing')
        
        # General recommendations
        recommendations.append('Document model assumptions, limitations, and validation procedures')
        recommendations.append('Establish regular model review and update cycles')
        recommendations.append('Consider implementing automated model monitoring and alerting')
        
        return recommendations[:12]  # Limit to top 12
    
    def _identify_automation_opportunities(self, context: Dict[str, Any]) -> List[str]:
        """Identify opportunities for automation"""
        
        opportunities = []
        
        # Data refresh automation
        if context.get('external_references'):
            opportunities.append('Automate external data refresh and validation processes')
        
        # Calculation automation
        if len(context.get('formulas', [])) > 50:
            opportunities.append('Consider VBA automation for complex calculation sequences')
        
        # Reporting automation
        dashboard_indicators = any('dashboard' in ws['name'].lower() or 'report' in ws['name'].lower() 
                                 for ws in context['worksheets'])
        if dashboard_indicators:
            opportunities.append('Implement automated report generation and distribution')
        
        # Validation automation
        opportunities.append('Automate data quality checks and model validation tests')
        
        # Version control automation
        opportunities.append('Implement automated backup and version control for model changes')
        
        # Monitoring automation
        if context.get('complexity_indicators', {}).get('complexity_score', 0) > 75:
            opportunities.append('Set up automated model performance monitoring and alerting')
        
        return opportunities[:8]  # Limit to top 8
    
    def _assess_regulatory_considerations(self, patterns: Dict[str, Any]) -> List[str]:
        """Assess regulatory considerations based on detected patterns"""
        
        considerations = []
        
        if 'financial_modeling' in patterns:
            considerations.extend([
                'Basel III model validation requirements may apply',
                'Consider SR 11-7 model risk management guidance',
                'Document model development and validation procedures',
                'Implement independent model validation processes'
            ])
        
        if 'simulation' in patterns:
            considerations.extend([
                'CCAR/DFAST stress testing requirements may apply',
                'Validate Monte Carlo simulation methodology',
                'Document scenario selection and calibration procedures'
            ])
        
        # General considerations
        considerations.extend([
            'Ensure adequate model documentation for regulatory review',
            'Implement appropriate model governance framework',
            'Consider audit trail and change management requirements'
        ])
        
        return considerations[:10]  # Limit to top 10
    
    def _generate_next_steps(self, patterns: Dict[str, Any], risks: List[str]) -> List[str]:
        """Generate prioritized next steps"""
        
        next_steps = []
        
        # High priority steps based on risks
        if any('complexity' in risk.lower() for risk in risks):
            next_steps.append('Priority 1: Conduct comprehensive model review and simplification assessment')
        
        if any('data' in risk.lower() for risk in risks):
            next_steps.append('Priority 1: Implement data quality assessment and remediation plan')
        
        # Pattern-specific next steps
        if 'financial_modeling' in patterns:
            next_steps.append('Priority 2: Develop model validation testing framework')
        
        if 'simulation' in patterns:
            next_steps.append('Priority 2: Validate simulation parameters and convergence criteria')
        
        # General next steps
        next_steps.extend([
            'Priority 3: Create comprehensive model documentation',
            'Priority 3: Establish model governance and change management processes',
            'Priority 4: Consider user training and knowledge transfer',
            'Priority 4: Implement regular model performance monitoring'
        ])
        
        return next_steps[:8]  # Limit to top 8
    
    def _calculate_confidence_score(self, context: Dict[str, Any], patterns: Dict[str, Any]) -> float:
        """Calculate overall analysis confidence score"""
        
        base_score = 0.7  # Base confidence for rule-based analysis
        
        # Increase confidence based on data availability
        if context.get('formulas'):
            base_score += 0.1
        
        if context.get('functions_used'):
            base_score += 0.05
        
        if patterns:
            base_score += len(patterns) * 0.02
        
        # Decrease confidence for limited information
        if len(context['worksheets']) < 2:
            base_score -= 0.1
        
        if not context.get('formulas'):
            base_score -= 0.15
        
        return min(1.0, max(0.3, base_score))
    
    def _determine_business_purpose(self, patterns: Dict[str, Any]) -> str:
        """Determine the primary business purpose"""
        
        if not patterns:
            return 'General data analysis and reporting'
        
        # Find the pattern with highest score
        top_pattern = max(patterns.items(), key=lambda x: x[1]['score'])
        return top_pattern[1]['purpose']
    
    def _extract_key_insights(self, business_insights: List[BusinessInsight], patterns: Dict[str, Any]) -> List[str]:
        """Extract key insights for summary"""
        
        insights = []
        
        # Add top business insights
        for bi in business_insights[:3]:
            insights.append(f'{bi.category}: {bi.insight}')
        
        # Add pattern-based insights
        for pattern_type, pattern_data in patterns.items():
            if pattern_data['confidence'] > 0.5:
                insights.append(f'Model demonstrates {pattern_type.replace("_", " ")} capabilities')
        
        return insights[:8]  # Limit to top 8
    
    def _assess_technical_complexity(self, context: Dict[str, Any]) -> str:
        """Assess overall technical complexity"""
        
        complexity_score = 0
        
        # Formula complexity
        formula_count = len(context.get('formulas', []))
        if formula_count > 100:
            complexity_score += 3
        elif formula_count > 20:
            complexity_score += 2
        elif formula_count > 0:
            complexity_score += 1
        
        # Function diversity
        function_count = len(context.get('functions_used', set()))
        if function_count > 30:
            complexity_score += 3
        elif function_count > 10:
            complexity_score += 2
        elif function_count > 0:
            complexity_score += 1
        
        # Worksheet complexity
        worksheet_count = len(context['worksheets'])
        if worksheet_count > 10:
            complexity_score += 2
        elif worksheet_count > 3:
            complexity_score += 1
        
        # VBA presence
        if context.get('vba_present'):
            complexity_score += 3
        
        # External references
        if context.get('external_references'):
            complexity_score += 2
        
        # Determine complexity level
        if complexity_score >= 10:
            return 'very_high'
        elif complexity_score >= 7:
            return 'high'
        elif complexity_score >= 4:
            return 'medium'
        elif complexity_score >= 2:
            return 'low'
        else:
            return 'very_low'


# Example usage
if __name__ == "__main__":
    analyzer = FreeIntelligentAnalyzer()
    
    # This would typically be called with actual Excel analysis data
    print("Free Intelligent Analyzer initialized successfully")
    print("Available business patterns:", list(analyzer.business_patterns.keys()))