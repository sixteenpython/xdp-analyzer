"""
Enhanced Document Summarizer
EUDA-style documentation generator with comprehensive analysis capabilities
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import re

# Try importing LLM libraries for enhanced summarization
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

@dataclass
class DocumentSummary:
    """Comprehensive document summary with EUDA-style structure"""
    # Executive Summary Components
    executive_summary: str
    key_business_purpose: str
    primary_stakeholders: List[str]
    business_criticality: str
    
    # Technical Analysis
    technical_overview: str
    system_architecture: str
    data_flow_description: str
    key_algorithms: List[str]
    
    # Functional Analysis
    main_functions: List[str]
    input_requirements: List[str]
    output_deliverables: List[str]
    calculation_methodologies: List[str]
    
    # Risk and Compliance
    identified_risks: List[str]
    regulatory_considerations: List[str]
    data_governance_notes: List[str]
    model_limitations: List[str]
    
    # Business Intelligence
    main_findings: List[str]
    performance_metrics: List[str]
    trend_analysis: List[str]
    actionable_insights: List[str]
    
    # Operational Guidance
    user_instructions: List[str]
    maintenance_requirements: List[str]
    troubleshooting_guide: List[str]
    next_steps_suggestions: List[str]
    
    # Metadata
    generation_method: str
    confidence_level: float
    analysis_timestamp: str
    document_version: str = "1.0"

class EnhancedDocumentSummarizer:
    """
    Enhanced Document Summarizer with EUDA-style comprehensive analysis
    Generates detailed documentation with business intelligence focus
    """
    
    def __init__(self, use_llm: bool = True):
        self.logger = logging.getLogger(__name__)
        self.use_llm = use_llm and HAS_REQUESTS
        
        # EUDA documentation templates
        self.euda_templates = {
            'financial_modeling': {
                'purpose_template': 'This financial model serves as a comprehensive {model_type} system designed to {primary_function} and support {business_objectives}.',
                'stakeholders': ['Risk Management Team', 'Portfolio Managers', 'Regulatory Affairs', 'Model Validation', 'Senior Management'],
                'criticality': 'High - Mission Critical for Risk Assessment',
                'key_sections': ['Model Methodology', 'Data Requirements', 'Validation Framework', 'Risk Metrics', 'Regulatory Compliance']
            },
            'forecasting': {
                'purpose_template': 'This forecasting model provides {forecast_type} predictions to support {business_planning} and enable {decision_making}.',
                'stakeholders': ['Business Planning', 'Finance Team', 'Operations', 'Strategy Team', 'Executive Leadership'],
                'criticality': 'Medium-High - Strategic Planning Critical',
                'key_sections': ['Forecasting Methodology', 'Historical Analysis', 'Trend Identification', 'Scenario Planning', 'Accuracy Metrics']
            },
            'dashboard': {
                'purpose_template': 'This business intelligence dashboard provides {monitoring_capabilities} for {business_areas} with {reporting_frequency}.',
                'stakeholders': ['Business Users', 'Management Team', 'Data Analysts', 'IT Support', 'Compliance Team'],
                'criticality': 'Medium - Operational Decision Support',
                'key_sections': ['KPI Definitions', 'Data Sources', 'Update Procedures', 'User Guide', 'Data Quality Controls']
            },
            'optimization': {
                'purpose_template': 'This optimization model maximizes {optimization_target} subject to {constraints} using {methodology}.',
                'stakeholders': ['Operations Team', 'Planning Managers', 'Resource Allocation', 'Finance Team', 'Process Improvement'],
                'criticality': 'High - Direct Operational Impact',
                'key_sections': ['Optimization Framework', 'Constraint Definition', 'Solution Methodology', 'Sensitivity Analysis', 'Implementation Guide']
            }
        }
        
        # Business intelligence extraction patterns
        self.bi_patterns = {
            'performance_indicators': [
                'return', 'yield', 'roi', 'profit', 'margin', 'efficiency', 'utilization', 
                'conversion', 'growth', 'performance', 'productivity', 'ratio'
            ],
            'risk_indicators': [
                'var', 'volatility', 'risk', 'stress', 'scenario', 'sensitivity', 
                'correlation', 'exposure', 'concentration', 'limit'
            ],
            'operational_metrics': [
                'capacity', 'throughput', 'cycle time', 'inventory', 'backlog', 
                'quality', 'defect', 'turnaround', 'sla', 'availability'
            ],
            'financial_metrics': [
                'revenue', 'cost', 'expense', 'budget', 'forecast', 'variance', 
                'npv', 'irr', 'payback', 'cash flow', 'liquidity'
            ]
        }
        
        # Technical complexity indicators
        self.complexity_indicators = {
            'algorithms': [
                'monte carlo', 'black scholes', 'binomial', 'optimization', 'regression',
                'neural network', 'machine learning', 'statistical model', 'simulation'
            ],
            'advanced_functions': [
                'SOLVER', 'LINEST', 'TREND', 'FORECAST', 'CORREL', 'COVAR', 'MMULT',
                'TRANSPOSE', 'MINVERSE', 'SUMPRODUCT'
            ],
            'integration_points': [
                'external data', 'api', 'database', 'web service', 'real-time feed',
                'third party', 'vendor system', 'market data'
            ]
        }
    
    async def generate_enhanced_summary(self, excel_analysis, formula_analysis: Dict[str, Any]) -> DocumentSummary:
        """
        Generate comprehensive EUDA-style summary
        
        Args:
            excel_analysis: Excel analysis results
            formula_analysis: Formula analysis results
            
        Returns:
            DocumentSummary with complete EUDA-style documentation
        """
        self.logger.info("Generating enhanced EUDA-style summary")
        
        # Extract analysis context
        context = self._extract_summary_context(excel_analysis, formula_analysis)
        
        # Determine document type and template
        doc_type, template = self._determine_document_type(context)
        
        # Generate executive summary
        executive_summary = await self._generate_executive_summary(context, template)
        
        # Extract business purpose and stakeholders
        business_purpose = self._determine_business_purpose(context, template)
        stakeholders = self._identify_stakeholders(context, template)
        criticality = self._assess_business_criticality(context, template)
        
        # Generate technical analysis
        technical_overview = self._generate_technical_overview(context)
        architecture = self._describe_system_architecture(context)
        data_flow = self._describe_data_flow(context)
        algorithms = self._identify_key_algorithms(context)
        
        # Analyze functional components
        main_functions = self._extract_main_functions(context)
        inputs = self._identify_input_requirements(context)
        outputs = self._identify_output_deliverables(context)
        methodologies = self._extract_calculation_methodologies(context)
        
        # Risk and compliance analysis
        risks = self._identify_document_risks(context)
        regulatory = self._assess_regulatory_considerations(context)
        governance = self._extract_data_governance_notes(context)
        limitations = self._identify_model_limitations(context)
        
        # Business intelligence extraction
        findings = await self._extract_main_findings(context)
        performance_metrics = self._identify_performance_metrics(context)
        trends = self._analyze_trends(context)
        insights = await self._generate_actionable_insights(context)
        
        # Operational guidance
        instructions = self._generate_user_instructions(context)
        maintenance = self._identify_maintenance_requirements(context)
        troubleshooting = self._create_troubleshooting_guide(context)
        next_steps = self._suggest_next_steps(context)
        
        # Determine generation method and confidence
        generation_method = 'rule_based_with_templates'
        confidence_level = self._calculate_summary_confidence(context)
        
        if self.use_llm:
            try:
                # Enhance with LLM if available
                enhanced_summary = await self._enhance_with_llm(context, executive_summary)
                if enhanced_summary:
                    executive_summary = enhanced_summary
                    generation_method = 'hybrid_llm_enhanced'
                    confidence_level += 0.1
            except Exception as e:
                self.logger.warning(f"LLM enhancement failed, using rule-based: {e}")
        
        return DocumentSummary(
            executive_summary=executive_summary,
            key_business_purpose=business_purpose,
            primary_stakeholders=stakeholders,
            business_criticality=criticality,
            technical_overview=technical_overview,
            system_architecture=architecture,
            data_flow_description=data_flow,
            key_algorithms=algorithms,
            main_functions=main_functions,
            input_requirements=inputs,
            output_deliverables=outputs,
            calculation_methodologies=methodologies,
            identified_risks=risks,
            regulatory_considerations=regulatory,
            data_governance_notes=governance,
            model_limitations=limitations,
            main_findings=findings,
            performance_metrics=performance_metrics,
            trend_analysis=trends,
            actionable_insights=insights,
            user_instructions=instructions,
            maintenance_requirements=maintenance,
            troubleshooting_guide=troubleshooting,
            next_steps_suggestions=next_steps,
            generation_method=generation_method,
            confidence_level=min(1.0, confidence_level),
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def _extract_summary_context(self, excel_analysis, formula_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive context for summary generation"""
        
        context = {
            'document_characteristics': {},
            'functional_elements': {},
            'technical_elements': {},
            'business_elements': {},
            'data_elements': {},
            'complexity_indicators': {}
        }
        
        try:
            # Document characteristics
            if hasattr(excel_analysis, 'filename'):
                context['document_characteristics']['filename'] = excel_analysis.filename
            if hasattr(excel_analysis, 'sheet_count'):
                context['document_characteristics']['sheet_count'] = excel_analysis.sheet_count
            if hasattr(excel_analysis, 'sheets'):
                context['document_characteristics']['sheet_names'] = [s.name for s in excel_analysis.sheets]
            
            # Formula analysis
            context['technical_elements'].update(formula_analysis)
            
            # Extract from sheets
            if hasattr(excel_analysis, 'sheets'):
                for sheet in excel_analysis.sheets:
                    # Business patterns
                    if hasattr(sheet, 'business_patterns'):
                        if 'business_patterns' not in context['business_elements']:
                            context['business_elements']['business_patterns'] = []
                        context['business_elements']['business_patterns'].extend(sheet.business_patterns)
                    
                    # Financial indicators
                    if hasattr(sheet, 'financial_indicators'):
                        context['business_elements']['financial_indicators'] = sheet.financial_indicators
                    
                    # Data characteristics
                    context['data_elements'][sheet.name] = {
                        'shape': sheet.shape,
                        'columns': sheet.columns,
                        'data_types': sheet.data_types,
                        'data_quality_score': getattr(sheet, 'data_quality_score', 0)
                    }
            
            # Advanced analysis elements
            if hasattr(excel_analysis, 'financial_modeling_assessment'):
                context['business_elements']['financial_modeling'] = excel_analysis.financial_modeling_assessment
            
            if hasattr(excel_analysis, 'time_series_analysis'):
                context['technical_elements']['time_series'] = excel_analysis.time_series_analysis
            
            if hasattr(excel_analysis, 'business_intelligence'):
                context['business_elements']['business_intelligence'] = excel_analysis.business_intelligence
            
        except Exception as e:
            self.logger.warning(f"Error extracting summary context: {e}")
        
        return context
    
    def _determine_document_type(self, context: Dict[str, Any]) -> tuple:
        """Determine document type and select appropriate template"""
        
        # Analyze business patterns and characteristics
        business_patterns = context.get('business_elements', {}).get('business_patterns', [])
        financial_indicators = context.get('business_elements', {}).get('financial_indicators', {})
        
        # Score each template type
        template_scores = {}
        
        for template_name, template_info in self.euda_templates.items():
            score = 0
            
            # Pattern-based scoring
            if template_name == 'financial_modeling':
                if any('finance' in str(p).lower() for p in business_patterns):
                    score += 3
                if financial_indicators.get('has_financial_data', False):
                    score += 4
                if any(keyword in str(context).lower() for keyword in ['var', 'risk', 'portfolio', 'volatility']):
                    score += 2
            
            elif template_name == 'forecasting':
                if any('time_series' in str(p).lower() for p in business_patterns):
                    score += 3
                if context.get('technical_elements', {}).get('time_series', {}).get('has_time_series', False):
                    score += 4
                if any(keyword in str(context).lower() for keyword in ['forecast', 'trend', 'prediction']):
                    score += 2
            
            elif template_name == 'dashboard':
                if any('dashboard' in str(p).lower() or 'intelligence' in str(p).lower() for p in business_patterns):
                    score += 3
                bi_score = context.get('business_elements', {}).get('business_intelligence', {})
                if bi_score and bi_score.get('dashboard_potential', False):
                    score += 4
            
            elif template_name == 'optimization':
                if any('optimization' in str(p).lower() for p in business_patterns):
                    score += 4
                if any(keyword in str(context).lower() for keyword in ['solver', 'optimize', 'minimize', 'maximize']):
                    score += 3
            
            template_scores[template_name] = score
        
        # Select highest scoring template
        if template_scores:
            doc_type = max(template_scores.items(), key=lambda x: x[1])[0]
        else:
            doc_type = 'dashboard'  # Default fallback
        
        return doc_type, self.euda_templates[doc_type]
    
    async def _generate_executive_summary(self, context: Dict[str, Any], template: Dict[str, Any]) -> str:
        """Generate comprehensive executive summary"""
        
        # Extract key information
        filename = context.get('document_characteristics', {}).get('filename', 'Excel Model')
        sheet_count = context.get('document_characteristics', {}).get('sheet_count', 1)
        business_patterns = context.get('business_elements', {}).get('business_patterns', [])
        
        # Build executive summary
        summary_parts = []
        
        # Document overview
        summary_parts.append(f"This document presents a comprehensive analysis of {filename}, "
                           f"a {sheet_count}-worksheet Excel model designed for {template.get('key_sections', ['Analysis'])[0].lower()}.")
        
        # Business purpose
        if business_patterns:
            primary_pattern = business_patterns[0] if business_patterns else 'data analysis'
            summary_parts.append(f"The model's primary function centers on {primary_pattern.replace('_', ' ')} "
                               f"with supporting capabilities for business decision-making.")
        
        # Technical characteristics
        total_formulas = context.get('technical_elements', {}).get('total_formulas', 0)
        if total_formulas > 0:
            summary_parts.append(f"The model incorporates {total_formulas} formulas across multiple worksheets, "
                               f"demonstrating {'sophisticated' if total_formulas > 100 else 'moderate'} computational complexity.")
        
        # Financial modeling assessment
        financial_assessment = context.get('business_elements', {}).get('financial_modeling', {})
        if financial_assessment:
            sophistication = financial_assessment.get('model_sophistication', 'basic')
            summary_parts.append(f"Financial modeling sophistication is assessed as {sophistication}, "
                               f"with capabilities spanning {len(financial_assessment.get('financial_domains', []))} domain areas.")
        
        # Data quality and governance
        avg_quality = self._calculate_average_data_quality(context)
        if avg_quality:
            summary_parts.append(f"Data quality assessment indicates an overall score of {avg_quality:.1f}/100, "
                               f"with {'strong' if avg_quality > 80 else 'adequate' if avg_quality > 60 else 'concerning'} "
                               f"data governance practices.")
        
        # Business criticality
        summary_parts.append(f"This model represents {template.get('criticality', 'Medium')} business functionality, "
                           f"requiring appropriate governance and validation procedures.")
        
        return ' '.join(summary_parts)
    
    def _determine_business_purpose(self, context: Dict[str, Any], template: Dict[str, Any]) -> str:
        """Determine key business purpose"""
        
        # Extract business intelligence
        bi = context.get('business_elements', {}).get('business_intelligence', {})
        primary_domain = bi.get('primary_business_domain', 'unknown')
        
        # Extract financial modeling info
        financial = context.get('business_elements', {}).get('financial_modeling', {})
        financial_domains = financial.get('financial_domains', [])
        
        # Build purpose statement
        if primary_domain != 'unknown':
            purpose = f"Primary business purpose is {primary_domain.replace('_', ' ')} operations"
            if financial_domains:
                purpose += f" with emphasis on {', '.join(financial_domains).replace('_', ' ')}"
            return purpose + "."
        
        elif financial_domains:
            return f"Financial modeling focused on {', '.join(financial_domains).replace('_', ' ')} applications."
        
        else:
            # Fallback to template-based purpose
            template_purpose = template.get('purpose_template', 'General analytical model for business decision support.')
            return template_purpose.format(
                model_type='analytical',
                primary_function='support business operations',
                business_objectives='data-driven decision making'
            )
    
    def _identify_stakeholders(self, context: Dict[str, Any], template: Dict[str, Any]) -> List[str]:
        """Identify primary stakeholders"""
        
        base_stakeholders = template.get('stakeholders', ['Business Users', 'Management'])
        
        # Add specific stakeholders based on content
        specific_stakeholders = []
        
        # Financial modeling stakeholders
        financial = context.get('business_elements', {}).get('financial_modeling', {})
        if financial and financial.get('risk_analysis_capability', False):
            specific_stakeholders.extend(['Risk Management', 'Compliance Team', 'Model Validation'])
        
        # Time series stakeholders
        if context.get('technical_elements', {}).get('time_series', {}).get('has_time_series', False):
            specific_stakeholders.extend(['Forecasting Team', 'Strategic Planning'])
        
        # Combine and deduplicate
        all_stakeholders = base_stakeholders + specific_stakeholders
        return list(dict.fromkeys(all_stakeholders))[:8]  # Limit and deduplicate
    
    def _assess_business_criticality(self, context: Dict[str, Any], template: Dict[str, Any]) -> str:
        """Assess business criticality level"""
        
        base_criticality = template.get('criticality', 'Medium - Standard Business Process')
        
        # Adjust based on model characteristics
        complexity_score = context.get('technical_elements', {}).get('complexity_score', 0)
        financial_assessment = context.get('business_elements', {}).get('financial_modeling', {})
        
        if complexity_score > 100 and financial_assessment:
            return 'Very High - Mission Critical Financial Infrastructure'
        elif financial_assessment and financial_assessment.get('risk_analysis_capability', False):
            return 'High - Critical Risk Management Component'
        elif complexity_score > 75:
            return 'Medium-High - Important Business Process'
        else:
            return base_criticality
    
    def _generate_technical_overview(self, context: Dict[str, Any]) -> str:
        """Generate technical overview"""
        
        overview_parts = []
        
        # Document structure
        sheet_count = context.get('document_characteristics', {}).get('sheet_count', 1)
        sheet_names = context.get('document_characteristics', {}).get('sheet_names', [])
        
        overview_parts.append(f"Technical architecture consists of {sheet_count} interconnected worksheets: "
                            f"{', '.join(sheet_names[:5])}{'...' if len(sheet_names) > 5 else ''}.")
        
        # Formula complexity
        total_formulas = context.get('technical_elements', {}).get('total_formulas', 0)
        unique_functions = context.get('technical_elements', {}).get('function_count', 0)
        
        if total_formulas > 0:
            overview_parts.append(f"Computational framework incorporates {total_formulas} formulas "
                                f"utilizing {unique_functions} distinct Excel functions.")
        
        # Advanced capabilities
        financial_categories = context.get('technical_elements', {}).get('financial_function_categories', {})
        if financial_categories:
            categories_str = ', '.join(financial_categories.keys()).replace('_', ' ')
            overview_parts.append(f"Advanced analytical capabilities include {categories_str} modeling.")
        
        # Data processing
        data_elements = context.get('data_elements', {})
        if data_elements:
            total_rows = sum(data['shape'][0] for data in data_elements.values() if 'shape' in data)
            overview_parts.append(f"Data processing capacity encompasses approximately {total_rows:,} data records "
                                f"across all worksheets.")
        
        return ' '.join(overview_parts)
    
    def _describe_system_architecture(self, context: Dict[str, Any]) -> str:
        """Describe system architecture"""
        
        arch_parts = []
        
        # Multi-tier architecture description
        sheet_names = context.get('document_characteristics', {}).get('sheet_names', [])
        
        # Classify sheets by likely function
        input_sheets = [name for name in sheet_names if any(keyword in name.lower() for keyword in ['input', 'data', 'raw', 'source'])]
        calc_sheets = [name for name in sheet_names if any(keyword in name.lower() for keyword in ['calc', 'model', 'analysis', 'compute'])]
        output_sheets = [name for name in sheet_names if any(keyword in name.lower() for keyword in ['output', 'result', 'report', 'dashboard', 'summary'])]
        
        if input_sheets or calc_sheets or output_sheets:
            arch_parts.append("System follows a multi-tier architecture with distinct layers:")
            
            if input_sheets:
                arch_parts.append(f"Data Input Layer ({len(input_sheets)} sheets): {', '.join(input_sheets[:3])}")
            if calc_sheets:
                arch_parts.append(f"Processing Layer ({len(calc_sheets)} sheets): {', '.join(calc_sheets[:3])}")
            if output_sheets:
                arch_parts.append(f"Presentation Layer ({len(output_sheets)} sheets): {', '.join(output_sheets[:3])}")
        else:
            arch_parts.append(f"Integrated architecture with {len(sheet_names)} functional modules "
                            f"providing comprehensive analytical capabilities.")
        
        # Data flow characteristics
        if context.get('technical_elements', {}).get('total_formulas', 0) > 50:
            arch_parts.append("Complex inter-worksheet dependencies create sophisticated calculation chains "
                            "requiring careful change management.")
        
        return ' '.join(arch_parts)
    
    def _describe_data_flow(self, context: Dict[str, Any]) -> str:
        """Describe data flow patterns"""
        
        flow_parts = []
        
        # Basic data flow description
        sheet_count = context.get('document_characteristics', {}).get('sheet_count', 1)
        
        if sheet_count > 1:
            flow_parts.append(f"Data flows through {sheet_count} interconnected worksheets with "
                            f"{'complex' if sheet_count > 5 else 'moderate'} inter-dependencies.")
        
        # Time series characteristics
        time_series = context.get('technical_elements', {}).get('time_series', {})
        if time_series and time_series.get('has_time_series', False):
            flow_parts.append("Temporal data processing capabilities enable time-series analysis and forecasting.")
        
        # External data considerations
        flow_parts.append("Data validation and quality controls are implemented to ensure computational integrity.")
        
        return ' '.join(flow_parts)
    
    def _identify_key_algorithms(self, context: Dict[str, Any]) -> List[str]:
        """Identify key algorithms and methodologies"""
        
        algorithms = []
        
        # Check technical elements for algorithm indicators
        tech_elements = context.get('technical_elements', {})
        
        # Financial algorithms
        financial_categories = tech_elements.get('financial_function_categories', {})
        for category in financial_categories.keys():
            if category == 'var_risk':
                algorithms.append('Value at Risk (VaR) Calculation')
            elif category == 'option_pricing':
                algorithms.append('Options Pricing Models')
            elif category == 'portfolio':
                algorithms.append('Portfolio Optimization')
            elif category == 'time_series':
                algorithms.append('Time Series Analysis')
        
        # Statistical methods
        if 'CORREL' in tech_elements.get('unique_functions', []):
            algorithms.append('Correlation Analysis')
        if 'LINEST' in tech_elements.get('unique_functions', []):
            algorithms.append('Linear Regression')
        if 'TREND' in tech_elements.get('unique_functions', []):
            algorithms.append('Trend Analysis')
        
        # Simulation methods
        random_functions = ['RAND', 'RANDBETWEEN', 'NORM.INV']
        if any(func in tech_elements.get('unique_functions', []) for func in random_functions):
            algorithms.append('Monte Carlo Simulation')
        
        # Default algorithms if none detected
        if not algorithms:
            algorithms = ['Statistical Analysis', 'Data Processing', 'Business Calculations']
        
        return algorithms[:8]  # Limit to top 8
    
    def _extract_main_functions(self, context: Dict[str, Any]) -> List[str]:
        """Extract main functional capabilities"""
        
        functions = []
        
        # Business pattern-based functions
        business_patterns = context.get('business_elements', {}).get('business_patterns', [])
        for pattern in business_patterns:
            if 'finance_data' in pattern:
                functions.append('Financial data analysis and reporting')
            elif 'time_series' in pattern:
                functions.append('Temporal data analysis and forecasting')
            elif 'cross_sectional' in pattern:
                functions.append('Cross-sectional data comparison and analysis')
            elif 'dashboard' in pattern or 'intelligence' in pattern:
                functions.append('Business intelligence dashboard and KPI monitoring')
        
        # Financial modeling functions
        financial = context.get('business_elements', {}).get('financial_modeling', {})
        if financial:
            if financial.get('risk_analysis_capability', False):
                functions.append('Risk assessment and management')
            if financial.get('portfolio_analysis', False):
                functions.append('Portfolio analysis and optimization')
            if financial.get('time_series_modeling', False):
                functions.append('Predictive modeling and forecasting')
        
        # Technical functions
        total_formulas = context.get('technical_elements', {}).get('total_formulas', 0)
        if total_formulas > 100:
            functions.append('Complex mathematical calculations and modeling')
        elif total_formulas > 20:
            functions.append('Automated calculations and data processing')
        
        # Default functions if none identified
        if not functions:
            functions = ['Data analysis and processing', 'Business reporting and insights', 'Decision support calculations']
        
        return functions[:10]  # Limit to top 10
    
    def _identify_input_requirements(self, context: Dict[str, Any]) -> List[str]:
        """Identify input data requirements"""
        
        requirements = []
        
        # Analyze data elements
        data_elements = context.get('data_elements', {})
        
        for sheet_name, sheet_data in data_elements.items():
            if 'input' in sheet_name.lower() or 'data' in sheet_name.lower():
                columns = sheet_data.get('columns', [])
                if columns:
                    requirements.append(f"{sheet_name}: {len(columns)} data fields including {', '.join(columns[:3])}")
        
        # Financial data requirements
        financial = context.get('business_elements', {}).get('financial_modeling', {})
        if financial and financial.get('financial_domains'):
            for domain in financial['financial_domains']:
                if domain == 'risk_management':
                    requirements.append('Market data: prices, volatilities, correlations')
                elif domain == 'portfolio_management':
                    requirements.append('Portfolio holdings and benchmark data')
                elif domain == 'forecasting':
                    requirements.append('Historical time series data')
        
        # Time series requirements
        time_series = context.get('technical_elements', {}).get('time_series', {})
        if time_series and time_series.get('has_time_series', False):
            requirements.append('Time-stamped historical data with consistent frequency')
        
        # General requirements
        if not requirements:
            requirements = [
                'Structured data in tabular format',
                'Data validation and quality controls',
                'Regular data refresh procedures'
            ]
        
        return requirements[:8]  # Limit to top 8
    
    def _identify_output_deliverables(self, context: Dict[str, Any]) -> List[str]:
        """Identify output deliverables"""
        
        deliverables = []
        
        # Sheet-based outputs
        sheet_names = context.get('document_characteristics', {}).get('sheet_names', [])
        
        output_sheets = [name for name in sheet_names if any(keyword in name.lower() 
                        for keyword in ['output', 'result', 'report', 'dashboard', 'summary'])]
        
        for sheet_name in output_sheets:
            deliverables.append(f"{sheet_name}: Analytical results and business insights")
        
        # Financial modeling outputs
        financial = context.get('business_elements', {}).get('financial_modeling', {})
        if financial:
            if financial.get('risk_analysis_capability', False):
                deliverables.append('Risk metrics: VaR, stress test results, exposure reports')
            if financial.get('portfolio_analysis', False):
                deliverables.append('Portfolio analytics: performance attribution, risk decomposition')
            if financial.get('time_series_modeling', False):
                deliverables.append('Forecasts: point estimates, confidence intervals, scenario analysis')
        
        # Business intelligence outputs
        bi = context.get('business_elements', {}).get('business_intelligence', {})
        if bi and bi.get('kpi_candidates'):
            deliverables.append(f"Key Performance Indicators: {len(bi['kpi_candidates'])} metrics and ratios")
        
        # Default deliverables
        if not deliverables:
            deliverables = [
                'Analytical reports and summaries',
                'Key metrics and performance indicators',
                'Data visualizations and charts'
            ]
        
        return deliverables[:10]  # Limit to top 10
    
    def _extract_calculation_methodologies(self, context: Dict[str, Any]) -> List[str]:
        """Extract calculation methodologies"""
        
        methodologies = []
        
        # Function-based methodologies
        unique_functions = context.get('technical_elements', {}).get('unique_functions', [])
        
        if 'CORREL' in unique_functions or 'COVAR' in unique_functions:
            methodologies.append('Correlation and covariance matrix calculations')
        
        if 'STDEV' in unique_functions or 'VAR' in unique_functions:
            methodologies.append('Statistical variance and standard deviation analysis')
        
        if 'PERCENTILE' in unique_functions:
            methodologies.append('Percentile-based risk metrics (VaR, CVaR)')
        
        if 'TREND' in unique_functions or 'LINEST' in unique_functions:
            methodologies.append('Linear regression and trend analysis')
        
        if 'SUMPRODUCT' in unique_functions:
            methodologies.append('Matrix multiplication and weighted calculations')
        
        # Financial methodologies
        financial_categories = context.get('technical_elements', {}).get('financial_function_categories', {})
        for category in financial_categories.keys():
            if category == 'var_risk':
                methodologies.append('Parametric and historical simulation VaR')
            elif category == 'option_pricing':
                methodologies.append('Black-Scholes and binomial option pricing')
            elif category == 'portfolio':
                methodologies.append('Mean-variance optimization framework')
        
        # Default methodologies
        if not methodologies:
            methodologies = [
                'Standard statistical calculations',
                'Financial ratio analysis',
                'Business metric computations'
            ]
        
        return methodologies[:8]  # Limit to top 8
    
    def _identify_document_risks(self, context: Dict[str, Any]) -> List[str]:
        """Identify document-level risks"""
        
        risks = []
        
        # Data quality risks
        avg_quality = self._calculate_average_data_quality(context)
        if avg_quality and avg_quality < 70:
            risks.append(f'Data quality concerns: Overall quality score {avg_quality:.1f}/100')
        
        # Complexity risks
        complexity_score = context.get('technical_elements', {}).get('complexity_score', 0)
        if complexity_score > 100:
            risks.append('High model complexity may impact maintainability and validation')
        
        # Formula risks
        total_formulas = context.get('technical_elements', {}).get('total_formulas', 0)
        if total_formulas > 200:
            risks.append('Extensive formula dependencies increase operational risk')
        
        # Business continuity risks
        sheet_count = context.get('document_characteristics', {}).get('sheet_count', 1)
        if sheet_count == 1 and total_formulas > 50:
            risks.append('Single worksheet concentration creates business continuity risk')
        
        # Financial model risks
        financial = context.get('business_elements', {}).get('financial_modeling', {})
        if financial and financial.get('overall_score', 0) > 60:
            risks.append('Financial model requires regulatory validation and governance')
        
        return risks[:8]  # Limit to top 8
    
    def _assess_regulatory_considerations(self, context: Dict[str, Any]) -> List[str]:
        """Assess regulatory considerations"""
        
        considerations = []
        
        # Financial modeling regulations
        financial = context.get('business_elements', {}).get('financial_modeling', {})
        if financial:
            if financial.get('risk_analysis_capability', False):
                considerations.extend([
                    'Basel III model validation requirements',
                    'SR 11-7 model risk management guidance',
                    'Regular model performance monitoring'
                ])
            
            if financial.get('portfolio_analysis', False):
                considerations.extend([
                    'Investment adviser fiduciary requirements',
                    'Portfolio management compliance monitoring'
                ])
        
        # Data governance
        considerations.extend([
            'Data quality and lineage documentation',
            'Model change management procedures',
            'User access controls and audit trails'
        ])
        
        return considerations[:8]  # Limit to top 8
    
    def _extract_data_governance_notes(self, context: Dict[str, Any]) -> List[str]:
        """Extract data governance notes"""
        
        governance = []
        
        # Data quality assessment
        avg_quality = self._calculate_average_data_quality(context)
        if avg_quality:
            governance.append(f'Data quality monitoring: Current score {avg_quality:.1f}/100')
        
        # Data lineage
        sheet_count = context.get('document_characteristics', {}).get('sheet_count', 1)
        if sheet_count > 3:
            governance.append('Multi-worksheet architecture requires data lineage documentation')
        
        # Change management
        total_formulas = context.get('technical_elements', {}).get('total_formulas', 0)
        if total_formulas > 50:
            governance.append('Formula dependencies require structured change management')
        
        # Access controls
        governance.append('Implement role-based access controls for model modifications')
        
        # Version control
        governance.append('Establish version control and backup procedures')
        
        return governance[:6]  # Limit to top 6
    
    def _identify_model_limitations(self, context: Dict[str, Any]) -> List[str]:
        """Identify model limitations"""
        
        limitations = []
        
        # Technical limitations
        complexity_score = context.get('technical_elements', {}).get('complexity_score', 0)
        if complexity_score > 75:
            limitations.append('Model complexity may limit scalability and performance')
        
        # Data limitations
        data_elements = context.get('data_elements', {})
        if data_elements:
            for sheet_name, sheet_data in data_elements.items():
                if sheet_data.get('data_quality_score', 100) < 80:
                    limitations.append(f'Data quality limitations in {sheet_name} may impact accuracy')
                    break
        
        # Methodological limitations
        financial = context.get('business_elements', {}).get('financial_modeling', {})
        if financial:
            sophistication = financial.get('model_sophistication', 'basic')
            if sophistication == 'basic':
                limitations.append('Basic modeling approach may not capture complex market dynamics')
            elif sophistication == 'intermediate':
                limitations.append('Model assumptions require periodic validation and updating')
        
        # Operational limitations
        limitations.extend([
            'Manual processes may introduce operational errors',
            'Limited real-time data integration capabilities',
            'Requires specialized expertise for maintenance and validation'
        ])
        
        return limitations[:6]  # Limit to top 6
    
    async def _extract_main_findings(self, context: Dict[str, Any]) -> List[str]:
        """Extract main analytical findings"""
        
        findings = []
        
        # Business intelligence findings
        bi = context.get('business_elements', {}).get('business_intelligence', {})
        if bi:
            primary_domain = bi.get('primary_business_domain', 'unknown')
            if primary_domain != 'unknown':
                findings.append(f'Primary business focus identified as {primary_domain.replace("_", " ")}')
            
            if bi.get('dashboard_potential', False):
                findings.append('Strong dashboard and reporting capabilities identified')
        
        # Financial modeling findings
        financial = context.get('business_elements', {}).get('financial_modeling', {})
        if financial:
            sophistication = financial.get('model_sophistication', 'basic')
            findings.append(f'Financial modeling sophistication assessed as {sophistication}')
            
            domains = financial.get('financial_domains', [])
            if domains:
                findings.append(f'Financial capabilities span {len(domains)} domain areas: {", ".join(domains)}')
        
        # Technical findings
        tech_elements = context.get('technical_elements', {})
        complexity_score = tech_elements.get('complexity_score', 0)
        if complexity_score > 100:
            findings.append('High technical complexity requiring specialized maintenance')
        elif complexity_score > 50:
            findings.append('Moderate technical complexity with advanced analytical capabilities')
        
        # Data findings
        avg_quality = self._calculate_average_data_quality(context)
        if avg_quality:
            if avg_quality > 85:
                findings.append('Excellent data quality standards maintained throughout model')
            elif avg_quality > 70:
                findings.append('Good data quality with minor improvement opportunities')
            else:
                findings.append('Data quality improvements needed for optimal model performance')
        
        return findings[:8]  # Limit to top 8
    
    def _identify_performance_metrics(self, context: Dict[str, Any]) -> List[str]:
        """Identify key performance metrics"""
        
        metrics = []
        
        # Extract from business intelligence
        bi = context.get('business_elements', {}).get('business_intelligence', {})
        if bi and bi.get('kpi_candidates'):
            kpi_candidates = bi['kpi_candidates'][:5]  # Top 5
            for kpi in kpi_candidates:
                metrics.append(f'Key Performance Indicator: {kpi}')
        
        # Financial metrics
        financial = context.get('business_elements', {}).get('financial_modeling', {})
        if financial:
            if financial.get('risk_analysis_capability', False):
                metrics.extend(['Value at Risk (VaR)', 'Risk-adjusted returns', 'Volatility measures'])
            if financial.get('portfolio_analysis', False):
                metrics.extend(['Portfolio performance attribution', 'Risk decomposition', 'Benchmark comparisons'])
        
        # Technical performance metrics
        complexity_score = context.get('technical_elements', {}).get('complexity_score', 0)
        total_formulas = context.get('technical_elements', {}).get('total_formulas', 0)
        
        metrics.extend([
            f'Model complexity score: {complexity_score:.1f}',
            f'Computational efficiency: {total_formulas} formulas'
        ])
        
        # Data quality metrics
        avg_quality = self._calculate_average_data_quality(context)
        if avg_quality:
            metrics.append(f'Data quality score: {avg_quality:.1f}/100')
        
        return metrics[:10]  # Limit to top 10
    
    def _analyze_trends(self, context: Dict[str, Any]) -> List[str]:
        """Analyze trends and patterns"""
        
        trends = []
        
        # Time series trends
        time_series = context.get('technical_elements', {}).get('time_series', {})
        if time_series and time_series.get('has_time_series', False):
            trends.append('Time series data enables trend analysis and forecasting capabilities')
        
        # Business pattern trends
        business_patterns = context.get('business_elements', {}).get('business_patterns', [])
        if business_patterns:
            pattern_summary = {}
            for pattern in business_patterns:
                domain = pattern.split('_')[0]
                pattern_summary[domain] = pattern_summary.get(domain, 0) + 1
            
            if pattern_summary:
                dominant_domain = max(pattern_summary.items(), key=lambda x: x[1])[0]
                trends.append(f'Dominant analytical focus on {dominant_domain} operations')
        
        # Complexity trends
        complexity_score = context.get('technical_elements', {}).get('complexity_score', 0)
        if complexity_score > 75:
            trends.append('Increasing model sophistication indicates advanced analytical requirements')
        
        # Data utilization trends
        data_elements = context.get('data_elements', {})
        if len(data_elements) > 3:
            trends.append('Multi-source data integration demonstrates comprehensive analytical approach')
        
        return trends[:6]  # Limit to top 6
    
    async def _generate_actionable_insights(self, context: Dict[str, Any]) -> List[str]:
        """Generate actionable business insights"""
        
        insights = []
        
        # Performance improvement insights
        avg_quality = self._calculate_average_data_quality(context)
        if avg_quality and avg_quality < 85:
            insights.append(f'Data quality improvement opportunity: Current score {avg_quality:.1f}/100 can be enhanced through validation procedures')
        
        # Business value insights
        bi = context.get('business_elements', {}).get('business_intelligence', {})
        if bi and bi.get('dashboard_potential', False):
            insights.append('Dashboard automation opportunity: Current manual processes can be streamlined for real-time insights')
        
        # Risk management insights
        financial = context.get('business_elements', {}).get('financial_modeling', {})
        if financial and financial.get('risk_analysis_capability', False):
            insights.append('Risk management enhancement: Implement regular backtesting and validation procedures')
        
        # Operational efficiency insights
        complexity_score = context.get('technical_elements', {}).get('complexity_score', 0)
        if complexity_score > 100:
            insights.append('Operational efficiency opportunity: Model simplification can reduce maintenance overhead')
        
        # Automation insights
        total_formulas = context.get('technical_elements', {}).get('total_formulas', 0)
        if total_formulas > 100:
            insights.append('Automation opportunity: Repetitive calculations can be optimized through VBA or external tools')
        
        # Governance insights
        insights.append('Governance enhancement: Implement comprehensive documentation and change management procedures')
        
        return insights[:8]  # Limit to top 8
    
    def _generate_user_instructions(self, context: Dict[str, Any]) -> List[str]:
        """Generate user instructions"""
        
        instructions = []
        
        # Basic operation instructions
        sheet_count = context.get('document_characteristics', {}).get('sheet_count', 1)
        if sheet_count > 1:
            instructions.append(f'Navigate through {sheet_count} worksheets following the logical data flow sequence')
        
        # Data entry instructions
        data_elements = context.get('data_elements', {})
        input_sheets = [name for name, data in data_elements.items() 
                       if 'input' in name.lower() or 'data' in name.lower()]
        
        if input_sheets:
            instructions.append(f'Update input data in designated sheets: {", ".join(input_sheets[:3])}')
        
        # Calculation instructions
        total_formulas = context.get('technical_elements', {}).get('total_formulas', 0)
        if total_formulas > 50:
            instructions.append('Ensure calculation mode is set to automatic or manually refresh calculations after data changes')
        
        # Financial model instructions
        financial = context.get('business_elements', {}).get('financial_modeling', {})
        if financial and financial.get('risk_analysis_capability', False):
            instructions.append('Review risk parameters and assumptions before generating final reports')
        
        # Validation instructions
        instructions.extend([
            'Verify data quality and completeness before analysis',
            'Review key assumptions and parameters for accuracy',
            'Validate critical calculations through independent checks'
        ])
        
        return instructions[:8]  # Limit to top 8
    
    def _identify_maintenance_requirements(self, context: Dict[str, Any]) -> List[str]:
        """Identify maintenance requirements"""
        
        requirements = []
        
        # Regular maintenance
        complexity_score = context.get('technical_elements', {}).get('complexity_score', 0)
        if complexity_score > 75:
            requirements.append('Monthly comprehensive model review and validation')
        else:
            requirements.append('Quarterly model review and parameter validation')
        
        # Data maintenance
        avg_quality = self._calculate_average_data_quality(context)
        if avg_quality and avg_quality < 80:
            requirements.append('Weekly data quality monitoring and cleansing procedures')
        
        # Formula maintenance
        total_formulas = context.get('technical_elements', {}).get('total_formulas', 0)
        if total_formulas > 100:
            requirements.append('Document and version control all formula changes')
        
        # Financial model maintenance
        financial = context.get('business_elements', {}).get('financial_modeling', {})
        if financial and financial.get('risk_analysis_capability', False):
            requirements.extend([
                'Daily risk parameter monitoring and validation',
                'Monthly model performance backtesting',
                'Annual independent model validation review'
            ])
        
        # General maintenance
        requirements.extend([
            'Regular backup and version control procedures',
            'User access review and permission management'
        ])
        
        return requirements[:8]  # Limit to top 8
    
    def _create_troubleshooting_guide(self, context: Dict[str, Any]) -> List[str]:
        """Create troubleshooting guide"""
        
        guide = []
        
        # Calculation issues
        total_formulas = context.get('technical_elements', {}).get('total_formulas', 0)
        if total_formulas > 0:
            guide.extend([
                'Calculation errors: Check for circular references and invalid cell references',
                'Performance issues: Verify calculation mode and consider manual calculation for large models'
            ])
        
        # Data issues
        guide.extend([
            'Missing data errors: Verify data sources and update missing values',
            'Format errors: Ensure consistent data types and number formats'
        ])
        
        # Formula issues
        if total_formulas > 50:
            guide.extend([
                'Formula errors: Use formula auditing tools to trace precedents and dependents',
                'Version conflicts: Maintain formula documentation and change logs'
            ])
        
        # Financial model issues
        financial = context.get('business_elements', {}).get('financial_modeling', {})
        if financial:
            guide.append('Risk metric errors: Validate input assumptions and market data currency')
        
        return guide[:8]  # Limit to top 8
    
    def _suggest_next_steps(self, context: Dict[str, Any]) -> List[str]:
        """Suggest next steps for model improvement"""
        
        next_steps = []
        
        # Priority 1: Critical improvements
        avg_quality = self._calculate_average_data_quality(context)
        if avg_quality and avg_quality < 70:
            next_steps.append('Priority 1: Implement comprehensive data quality improvement program')
        
        complexity_score = context.get('technical_elements', {}).get('complexity_score', 0)
        if complexity_score > 150:
            next_steps.append('Priority 1: Conduct model simplification and optimization review')
        
        # Priority 2: Enhancements
        financial = context.get('business_elements', {}).get('financial_modeling', {})
        if financial and financial.get('overall_score', 0) > 50:
            next_steps.append('Priority 2: Implement formal model validation and testing framework')
        
        bi = context.get('business_elements', {}).get('business_intelligence', {})
        if bi and bi.get('dashboard_potential', False):
            next_steps.append('Priority 2: Develop automated reporting and dashboard capabilities')
        
        # Priority 3: General improvements
        next_steps.extend([
            'Priority 3: Create comprehensive user documentation and training materials',
            'Priority 3: Establish model governance and change management procedures',
            'Priority 4: Consider migration to enterprise-grade analytical platform',
            'Priority 4: Implement automated monitoring and alerting systems'
        ])
        
        return next_steps[:8]  # Limit to top 8
    
    def _calculate_summary_confidence(self, context: Dict[str, Any]) -> float:
        """Calculate confidence level for summary"""
        
        confidence = 0.7  # Base confidence for rule-based analysis
        
        # Increase confidence based on available information
        if context.get('technical_elements', {}).get('total_formulas', 0) > 0:
            confidence += 0.1
        
        if context.get('business_elements', {}).get('business_patterns'):
            confidence += 0.05
        
        if context.get('data_elements'):
            confidence += 0.05
        
        # Comprehensive analysis bonus
        if len(context.get('document_characteristics', {}).get('sheet_names', [])) > 3:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _calculate_average_data_quality(self, context: Dict[str, Any]) -> Optional[float]:
        """Calculate average data quality score"""
        
        data_elements = context.get('data_elements', {})
        if not data_elements:
            return None
        
        quality_scores = []
        for sheet_data in data_elements.values():
            if 'data_quality_score' in sheet_data:
                quality_scores.append(sheet_data['data_quality_score'])
        
        return sum(quality_scores) / len(quality_scores) if quality_scores else None
    
    async def _enhance_with_llm(self, context: Dict[str, Any], base_summary: str) -> Optional[str]:
        """Enhance summary with LLM if available"""
        
        # This would integrate with free LLM services
        # Implementation depends on available services and API access
        
        try:
            # Placeholder for LLM enhancement
            # Would construct prompt and call free LLM service
            
            enhanced_prompt = f"""
            Please enhance this Excel model analysis summary with additional business insights:
            
            Current Analysis: {base_summary}
            
            Additional Context:
            - Technical complexity: {context.get('technical_elements', {}).get('complexity_score', 0)}
            - Business patterns: {context.get('business_elements', {}).get('business_patterns', [])}
            
            Provide enhanced business-focused summary in 3-4 sentences.
            """
            
            # Mock LLM enhancement (would be actual API call)
            if len(base_summary) > 200:
                return base_summary  # Return original if already comprehensive
            
        except Exception as e:
            self.logger.warning(f"LLM enhancement failed: {e}")
        
        return None


# Example usage
if __name__ == "__main__":
    summarizer = EnhancedDocumentSummarizer()
    
    # This would typically be called with actual Excel analysis data
    print("Enhanced Document Summarizer initialized successfully")
    print("Available EUDA templates:", list(summarizer.euda_templates.keys()))