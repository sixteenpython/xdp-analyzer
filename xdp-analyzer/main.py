"""
Enhanced XDP Analyzer - Main Entry Point
Comprehensive Excel workbook analysis with advanced financial modeling capabilities
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
import json
from datetime import datetime

# Add modules to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# Import enhanced parsers and analyzers
from parsers.simple_excel_parser import SimpleExcelParser
from parsers.excel_parser import AdvancedExcelParser
from analyzers.intelligent_analyzer import FreeIntelligentAnalyzer
from analyzers.enhanced_summarizer import EnhancedDocumentSummarizer
from analyzers.statistical_analyzer import AdvancedStatisticalAnalyzer
from analyzers.time_series_analyzer import AdvancedTimeSeriesAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xdp_analyzer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class EnhancedXDPAnalyzer:
    """
    Enhanced XDP Analyzer with comprehensive Excel analysis capabilities
    Integrates multiple analysis engines for deep insights
    """
    
    def __init__(self, analysis_mode: str = 'comprehensive'):
        """
        Initialize the enhanced analyzer
        
        Args:
            analysis_mode: 'simple', 'advanced', or 'comprehensive'
        """
        self.logger = logging.getLogger(__name__)
        self.analysis_mode = analysis_mode
        
        # Initialize parsers
        self.simple_parser = SimpleExcelParser()
        self.advanced_parser = AdvancedExcelParser()
        
        # Initialize analyzers
        self.intelligent_analyzer = FreeIntelligentAnalyzer()
        self.summarizer = EnhancedDocumentSummarizer(use_llm=True)
        self.statistical_analyzer = AdvancedStatisticalAnalyzer()
        self.time_series_analyzer = AdvancedTimeSeriesAnalyzer()
        
        self.logger.info(f"Enhanced XDP Analyzer initialized in {analysis_mode} mode")
    
    def analyze_excel_file(self, file_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of Excel file
        
        Args:
            file_path: Path to Excel file
            output_dir: Optional output directory for results
            
        Returns:
            Complete analysis results
        """
        
        file_path = Path(file_path)
        self.logger.info(f"Starting comprehensive analysis of {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Initialize results container
        analysis_results = {
            'file_info': {
                'filename': file_path.name,
                'file_size': file_path.stat().st_size,
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_mode': self.analysis_mode
            },
            'parsing_results': {},
            'statistical_analysis': {},
            'time_series_analysis': {},
            'intelligent_analysis': {},
            'enhanced_summary': {},
            'business_insights': [],
            'recommendations': [],
            'risk_assessment': {},
            'model_validation': {}
        }
        
        try:
            # Stage 1: Excel Parsing
            self.logger.info("Stage 1: Parsing Excel file...")
            parsing_results = self._perform_excel_parsing(file_path)
            analysis_results['parsing_results'] = parsing_results
            
            # Stage 2: Statistical Analysis
            self.logger.info("Stage 2: Performing statistical analysis...")
            statistical_results = self._perform_statistical_analysis(parsing_results)
            analysis_results['statistical_analysis'] = statistical_results
            
            # Stage 3: Time Series Analysis
            self.logger.info("Stage 3: Analyzing time series data...")
            time_series_results = self._perform_time_series_analysis(parsing_results)
            analysis_results['time_series_analysis'] = time_series_results
            
            # Stage 4: Intelligent Analysis
            self.logger.info("Stage 4: Performing intelligent analysis...")
            intelligent_results = self._perform_intelligent_analysis(parsing_results)
            analysis_results['intelligent_analysis'] = intelligent_results
            
            # Stage 5: Enhanced Summary Generation
            self.logger.info("Stage 5: Generating enhanced summary...")
            enhanced_summary = self._generate_enhanced_summary(parsing_results, statistical_results)
            analysis_results['enhanced_summary'] = enhanced_summary
            
            # Stage 6: Business Insights Integration
            self.logger.info("Stage 6: Integrating business insights...")
            business_insights = self._integrate_business_insights(analysis_results)
            analysis_results['business_insights'] = business_insights
            
            # Stage 7: Risk Assessment
            self.logger.info("Stage 7: Performing risk assessment...")
            risk_assessment = self._perform_risk_assessment(analysis_results)
            analysis_results['risk_assessment'] = risk_assessment
            
            # Stage 8: Model Validation
            self.logger.info("Stage 8: Validating financial models...")
            model_validation = self._validate_financial_models(analysis_results)
            analysis_results['model_validation'] = model_validation
            
            # Stage 9: Generate Comprehensive Recommendations
            self.logger.info("Stage 9: Generating recommendations...")
            recommendations = self._generate_comprehensive_recommendations(analysis_results)
            analysis_results['recommendations'] = recommendations
            
            # Save results if output directory specified
            if output_dir:
                self._save_results(analysis_results, output_dir, file_path.stem)
            
            self.logger.info("Comprehensive analysis completed successfully!")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            analysis_results['error'] = str(e)
            return analysis_results
    
    def _perform_excel_parsing(self, file_path: Path) -> Dict[str, Any]:
        """Perform Excel parsing using both simple and advanced parsers"""
        
        parsing_results = {
            'simple_parsing': None,
            'advanced_parsing': None,
            'parsing_comparison': {}
        }
        
        try:
            # Simple parsing (always works)
            self.logger.info("Performing simple Excel parsing...")
            simple_analysis = self.simple_parser.parse(file_path)
            parsing_results['simple_parsing'] = {
                'success': True,
                'analysis': simple_analysis,
                'sheet_count': simple_analysis.sheet_count,
                'total_rows': simple_analysis.total_rows,
                'data_quality_score': simple_analysis.summary.get('data_quality_score', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Simple parsing failed: {e}")
            parsing_results['simple_parsing'] = {'success': False, 'error': str(e)}
        
        try:
            # Advanced parsing (formula-aware)
            if self.analysis_mode in ['advanced', 'comprehensive']:
                self.logger.info("Performing advanced Excel parsing...")
                advanced_analysis = self.advanced_parser.parse(file_path)
                parsing_results['advanced_parsing'] = {
                    'success': True,
                    'analysis': advanced_analysis,
                    'formula_count': advanced_analysis.formulas_summary.get('total_formulas', 0),
                    'vba_modules': len(advanced_analysis.vba_code),
                    'complexity_score': advanced_analysis.complexity_metrics.get('complexity_score', 0)
                }
                
        except Exception as e:
            self.logger.warning(f"Advanced parsing failed, using simple parsing: {e}")
            parsing_results['advanced_parsing'] = {'success': False, 'error': str(e)}
        
        # Compare parsing results
        if parsing_results['simple_parsing']['success'] and parsing_results['advanced_parsing']['success']:
            parsing_results['parsing_comparison'] = self._compare_parsing_results(
                parsing_results['simple_parsing'], parsing_results['advanced_parsing']
            )
        
        return parsing_results
    
    def _perform_statistical_analysis(self, parsing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        
        statistical_results = {
            'sheet_analyses': {},
            'cross_sheet_analysis': {},
            'financial_modeling_assessment': {},
            'data_quality_assessment': {}
        }
        
        # Get the best available parsing result
        analysis_data = self._get_best_parsing_result(parsing_results)
        
        if not analysis_data:
            return statistical_results
        
        try:
            # Convert analysis to DataFrames and analyze each sheet
            for sheet in analysis_data.sheets:
                if hasattr(sheet, 'sample_data') and sheet.sample_data:
                    # Convert sample data to DataFrame
                    import pandas as pd
                    df = pd.DataFrame(sheet.sample_data)
                    
                    if not df.empty:
                        # Perform statistical analysis
                        sheet_stats = self.statistical_analyzer.analyze_dataset(df, sheet.name)
                        statistical_results['sheet_analyses'][sheet.name] = sheet_stats
            
            # Cross-sheet analysis
            if len(statistical_results['sheet_analyses']) > 1:
                statistical_results['cross_sheet_analysis'] = self._perform_cross_sheet_statistical_analysis(
                    statistical_results['sheet_analyses']
                )
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {e}")
            statistical_results['error'] = str(e)
        
        return statistical_results
    
    def _perform_time_series_analysis(self, parsing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform time series analysis on temporal data"""
        
        time_series_results = {
            'time_series_detected': False,
            'series_analyses': {},
            'cross_series_analysis': {},
            'forecasting_assessment': {}
        }
        
        analysis_data = self._get_best_parsing_result(parsing_results)
        
        if not analysis_data:
            return time_series_results
        
        try:
            # Look for time series data in sheets
            import pandas as pd
            
            for sheet in analysis_data.sheets:
                if hasattr(sheet, 'sample_data') and sheet.sample_data:
                    df = pd.DataFrame(sheet.sample_data)
                    
                    # Identify time columns
                    time_columns = []
                    value_columns = []
                    
                    for col in df.columns:
                        col_lower = str(col).lower()
                        # Check for time/date columns
                        if any(term in col_lower for term in ['date', 'time', 'period', 'year', 'month']):
                            time_columns.append(col)
                        # Check for numeric value columns
                        elif pd.api.types.is_numeric_dtype(df[col]):
                            value_columns.append(col)
                    
                    # Perform time series analysis if temporal data found
                    if time_columns and value_columns:
                        time_series_results['time_series_detected'] = True
                        
                        for time_col in time_columns[:2]:  # Limit to first 2 time columns
                            try:
                                ts_analysis = self.time_series_analyzer.analyze_time_series(
                                    df, time_col, value_columns
                                )
                                if ts_analysis:
                                    time_series_results['series_analyses'][f"{sheet.name}_{time_col}"] = ts_analysis
                            except Exception as e:
                                self.logger.warning(f"Time series analysis failed for {sheet.name}: {e}")
            
            # Cross-series analysis if multiple time series found
            if len(time_series_results['series_analyses']) > 1:
                time_series_results['cross_series_analysis'] = self._perform_cross_series_analysis(
                    time_series_results['series_analyses']
                )
            
        except Exception as e:
            self.logger.error(f"Time series analysis failed: {e}")
            time_series_results['error'] = str(e)
        
        return time_series_results
    
    def _perform_intelligent_analysis(self, parsing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform intelligent analysis using the enhanced analyzer"""
        
        intelligent_results = {
            'analysis_available': False,
            'analysis': None,
            'confidence_score': 0
        }
        
        try:
            # Use advanced parsing if available, otherwise simple
            if parsing_results.get('advanced_parsing', {}).get('success'):
                analysis_data = parsing_results['advanced_parsing']['analysis']
                
                # Create mock formula analysis for intelligent analyzer
                formula_analysis = {
                    'total_formulas': analysis_data.formulas_summary.get('total_formulas', 0),
                    'function_usage': analysis_data.formulas_summary.get('unique_functions', {}),
                    'business_categories': {},
                    'average_complexity': analysis_data.complexity_metrics.get('complexity_score', 0) / 10
                }
                
                # Perform intelligent analysis
                intelligent_analysis = self.intelligent_analyzer.analyze_excel_content(analysis_data)
                
                intelligent_results = {
                    'analysis_available': True,
                    'analysis': intelligent_analysis,
                    'confidence_score': intelligent_analysis.confidence_score
                }
                
            elif parsing_results.get('simple_parsing', {}).get('success'):
                # Use simple parsing result
                analysis_data = parsing_results['simple_parsing']['analysis']
                
                # Create adapted analysis for intelligent analyzer
                adapted_analysis = self._adapt_simple_analysis_for_intelligent_analyzer(analysis_data)
                
                intelligent_analysis = self.intelligent_analyzer.analyze_excel_content(adapted_analysis)
                
                intelligent_results = {
                    'analysis_available': True,
                    'analysis': intelligent_analysis,
                    'confidence_score': intelligent_analysis.confidence_score
                }
        
        except Exception as e:
            self.logger.error(f"Intelligent analysis failed: {e}")
            intelligent_results['error'] = str(e)
        
        return intelligent_results
    
    def _generate_enhanced_summary(self, parsing_results: Dict[str, Any], statistical_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced summary using the enhanced summarizer"""
        
        summary_results = {
            'summary_available': False,
            'summary': None,
            'generation_method': 'none'
        }
        
        try:
            analysis_data = self._get_best_parsing_result(parsing_results)
            
            if analysis_data:
                # Create formula analysis summary
                formula_analysis = {
                    'total_formulas': 0,
                    'function_usage': {},
                    'business_categories': {},
                    'average_complexity': 0
                }
                
                # Extract from advanced parsing if available
                if parsing_results.get('advanced_parsing', {}).get('success'):
                    adv_analysis = parsing_results['advanced_parsing']['analysis']
                    formula_analysis = {
                        'total_formulas': adv_analysis.formulas_summary.get('total_formulas', 0),
                        'function_usage': dict(adv_analysis.formulas_summary.get('unique_functions', [])),
                        'business_categories': adv_analysis.business_logic,
                        'average_complexity': adv_analysis.complexity_metrics.get('complexity_score', 0) / 10
                    }
                
                # Generate enhanced summary
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    enhanced_summary = loop.run_until_complete(
                        self.summarizer.generate_enhanced_summary(analysis_data, formula_analysis)
                    )
                    
                    summary_results = {
                        'summary_available': True,
                        'summary': enhanced_summary,
                        'generation_method': enhanced_summary.generation_method
                    }
                finally:
                    loop.close()
        
        except Exception as e:
            self.logger.error(f"Enhanced summary generation failed: {e}")
            summary_results['error'] = str(e)
        
        return summary_results
    
    def _integrate_business_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Integrate business insights from all analysis stages"""
        
        insights = []
        
        try:
            # From statistical analysis
            if analysis_results.get('statistical_analysis', {}).get('sheet_analyses'):
                for sheet_name, sheet_stats in analysis_results['statistical_analysis']['sheet_analyses'].items():
                    insights.extend(sheet_stats.get('business_insights', [])[:2])
            
            # From time series analysis
            if analysis_results.get('time_series_analysis', {}).get('series_analyses'):
                for series_name, ts_analysis in analysis_results['time_series_analysis']['series_analyses'].items():
                    for series_result in ts_analysis.values():
                        insights.extend(series_result.business_insights[:2])
            
            # From intelligent analysis
            if analysis_results.get('intelligent_analysis', {}).get('analysis'):
                intelligent = analysis_results['intelligent_analysis']['analysis']
                insights.extend(intelligent.key_insights[:3])
            
            # From enhanced summary
            if analysis_results.get('enhanced_summary', {}).get('summary'):
                summary = analysis_results['enhanced_summary']['summary']
                insights.extend(summary.actionable_insights[:3])
            
            # Deduplicate and rank insights
            unique_insights = list(set(insights))
            return unique_insights[:10]  # Return top 10 insights
            
        except Exception as e:
            self.logger.error(f"Business insights integration failed: {e}")
            return [f"Analysis completed with {len(insights)} insights available"]
    
    def _perform_risk_assessment(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        
        risk_assessment = {
            'overall_risk_level': 'medium',
            'data_quality_risks': [],
            'model_risks': [],
            'operational_risks': [],
            'financial_model_risks': [],
            'mitigation_strategies': []
        }
        
        try:
            # Data quality risks
            if analysis_results.get('statistical_analysis', {}).get('sheet_analyses'):
                for sheet_stats in analysis_results['statistical_analysis']['sheet_analyses'].values():
                    if sheet_stats.get('data_quality_metrics', {}).get('overall_score', 100) < 70:
                        risk_assessment['data_quality_risks'].append(
                            f"Poor data quality detected in sheet (score: {sheet_stats['data_quality_metrics']['overall_score']:.1f})"
                        )
            
            # Model complexity risks
            if analysis_results.get('parsing_results', {}).get('advanced_parsing', {}).get('success'):
                complexity = analysis_results['parsing_results']['advanced_parsing'].get('complexity_score', 0)
                if complexity > 100:
                    risk_assessment['model_risks'].append(
                        f"High model complexity (score: {complexity}) may impact maintainability"
                    )
            
            # Financial model risks
            if analysis_results.get('intelligent_analysis', {}).get('analysis'):
                intelligent = analysis_results['intelligent_analysis']['analysis']
                risk_assessment['financial_model_risks'].extend(intelligent.risk_indicators[:3])
            
            # Determine overall risk level
            total_risks = (len(risk_assessment['data_quality_risks']) + 
                          len(risk_assessment['model_risks']) + 
                          len(risk_assessment['operational_risks']) + 
                          len(risk_assessment['financial_model_risks']))
            
            if total_risks > 8:
                risk_assessment['overall_risk_level'] = 'high'
            elif total_risks < 3:
                risk_assessment['overall_risk_level'] = 'low'
            
            # Generate mitigation strategies
            risk_assessment['mitigation_strategies'] = self._generate_risk_mitigation_strategies(risk_assessment)
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            risk_assessment['error'] = str(e)
        
        return risk_assessment
    
    def _validate_financial_models(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate detected financial models"""
        
        validation_results = {
            'models_detected': [],
            'validation_results': {},
            'compliance_assessment': {},
            'recommendations': []
        }
        
        try:
            # Detect financial models from analysis
            if analysis_results.get('statistical_analysis', {}).get('sheet_analyses'):
                for sheet_name, sheet_stats in analysis_results['statistical_analysis']['sheet_analyses'].items():
                    financial_assessment = sheet_stats.get('financial_modeling_assessment', {})
                    
                    if financial_assessment.get('suitable_for_var_modeling'):
                        validation_results['models_detected'].append('VaR Model')
                    if financial_assessment.get('suitable_for_portfolio_optimization'):
                        validation_results['models_detected'].append('Portfolio Optimization')
                    if financial_assessment.get('suitable_for_derivatives_pricing'):
                        validation_results['models_detected'].append('Derivatives Pricing')
            
            # Validate each detected model
            for model in validation_results['models_detected']:
                validation_results['validation_results'][model] = self._validate_specific_model(model, analysis_results)
            
            # Assess regulatory compliance
            validation_results['compliance_assessment'] = self._assess_regulatory_compliance(validation_results['models_detected'])
            
            # Generate model-specific recommendations
            validation_results['recommendations'] = self._generate_model_validation_recommendations(validation_results)
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    def _generate_comprehensive_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations based on all analyses"""
        
        recommendations = []
        
        try:
            # Data quality recommendations
            if analysis_results.get('risk_assessment', {}).get('data_quality_risks'):
                recommendations.append("Implement data quality controls and validation procedures")
            
            # Model complexity recommendations
            if analysis_results.get('parsing_results', {}).get('advanced_parsing', {}).get('complexity_score', 0) > 100:
                recommendations.append("Consider model simplification or modularization for better maintainability")
            
            # Statistical analysis recommendations
            if analysis_results.get('statistical_analysis', {}).get('sheet_analyses'):
                for sheet_stats in analysis_results['statistical_analysis']['sheet_analyses'].values():
                    recommendations.extend(sheet_stats.get('financial_modeling_assessment', {}).get('recommendations', [])[:2])
            
            # Time series recommendations
            if analysis_results.get('time_series_analysis', {}).get('time_series_detected'):
                recommendations.append("Leverage time series analysis for forecasting and trend analysis")
            
            # Intelligent analysis recommendations
            if analysis_results.get('intelligent_analysis', {}).get('analysis'):
                intelligent = analysis_results['intelligent_analysis']['analysis']
                recommendations.extend(intelligent.recommendations[:3])
            
            # Enhanced summary recommendations
            if analysis_results.get('enhanced_summary', {}).get('summary'):
                summary = analysis_results['enhanced_summary']['summary']
                recommendations.extend(summary.next_steps_suggestions[:2])
            
            # Model validation recommendations
            if analysis_results.get('model_validation', {}).get('recommendations'):
                recommendations.extend(analysis_results['model_validation']['recommendations'][:2])
            
            # Deduplicate and prioritize
            unique_recommendations = list(set(recommendations))
            return unique_recommendations[:12]  # Return top 12 recommendations
            
        except Exception as e:
            self.logger.error(f"Comprehensive recommendations generation failed: {e}")
            return ["Complete detailed review of analysis results and implement best practices"]
    
    def _compare_parsing_results(self, simple_result: Dict, advanced_result: Dict) -> Dict[str, Any]:
        """Compare simple and advanced parsing results"""
        
        comparison = {
            'parsing_agreement': 'high',
            'differences': [],
            'recommended_parser': 'advanced'
        }
        
        try:
            simple_sheets = simple_result.get('sheet_count', 0)
            advanced_sheets = len(advanced_result['analysis'].worksheets)
            
            if simple_sheets == advanced_sheets:
                comparison['differences'].append(f"Both parsers detected {simple_sheets} sheets")
            else:
                comparison['differences'].append(f"Sheet count difference: Simple={simple_sheets}, Advanced={advanced_sheets}")
                comparison['parsing_agreement'] = 'medium'
            
            # Formula detection
            advanced_formulas = advanced_result.get('formula_count', 0)
            if advanced_formulas > 0:
                comparison['differences'].append(f"Advanced parser detected {advanced_formulas} formulas")
                comparison['recommended_parser'] = 'advanced'
            else:
                comparison['recommended_parser'] = 'simple'
                
        except Exception as e:
            comparison['error'] = str(e)
        
        return comparison
    
    def _get_best_parsing_result(self, parsing_results: Dict[str, Any]):
        """Get the best available parsing result"""
        
        if parsing_results.get('advanced_parsing', {}).get('success'):
            return parsing_results['advanced_parsing']['analysis']
        elif parsing_results.get('simple_parsing', {}).get('success'):
            return parsing_results['simple_parsing']['analysis']
        else:
            return None
    
    def _adapt_simple_analysis_for_intelligent_analyzer(self, simple_analysis):
        """Adapt simple analysis to work with intelligent analyzer"""
        
        # Create a mock object that mimics expected structure
        class AdaptedAnalysis:
            def __init__(self, simple_analysis):
                self.worksheets = []
                self.vba_code = {}
                self.external_references = []
                self.defined_names = []
                self.custom_properties = {}
                self.formulas_summary = {'total_formulas': 0, 'unique_functions': []}
                self.business_logic = {}
                self.data_flow = {}
                self.complexity_metrics = {'complexity_score': 0}
                
                # Convert sheets
                for sheet in simple_analysis.sheets:
                    worksheet = AdaptedWorksheet(sheet)
                    self.worksheets.append(worksheet)
        
        class AdaptedWorksheet:
            def __init__(self, sheet):
                self.name = sheet.name
                self.cells = []
                self.formulas = []
                self.charts = []
                self.pivot_tables = []
                self.data_validation = []
                self.conditional_formatting = []
                self.named_ranges = []
                self.vba_references = []
                self.hidden = False
                self.protected = False
                self.dimensions = {'max_row': sheet.shape[0], 'max_column': sheet.shape[1]}
        
        return AdaptedAnalysis(simple_analysis)
    
    def _perform_cross_sheet_statistical_analysis(self, sheet_analyses: Dict) -> Dict[str, Any]:
        """Perform statistical analysis across multiple sheets"""
        
        cross_analysis = {
            'correlation_across_sheets': {},
            'common_patterns': [],
            'data_consistency': {}
        }
        
        try:
            # Find common patterns across sheets
            all_patterns = []
            for sheet_stats in sheet_analyses.values():
                financial_assessment = sheet_stats.get('financial_modeling_assessment', {})
                if financial_assessment.get('suitable_for_var_modeling'):
                    all_patterns.append('var_modeling')
                if financial_assessment.get('suitable_for_portfolio_optimization'):
                    all_patterns.append('portfolio_optimization')
            
            cross_analysis['common_patterns'] = list(set(all_patterns))
            
        except Exception as e:
            cross_analysis['error'] = str(e)
        
        return cross_analysis
    
    def _perform_cross_series_analysis(self, series_analyses: Dict) -> Dict[str, Any]:
        """Perform analysis across multiple time series"""
        
        cross_analysis = {
            'correlation_patterns': {},
            'common_characteristics': [],
            'lead_lag_relationships': {}
        }
        
        try:
            # Analyze common characteristics
            all_characteristics = []
            for ts_analysis in series_analyses.values():
                for series_result in ts_analysis.values():
                    if series_result.volatility_analysis and series_result.volatility_analysis.volatility_clustering:
                        all_characteristics.append('volatility_clustering')
                    if series_result.forecasting_assessment.get('predictability') == 'high':
                        all_characteristics.append('high_predictability')
            
            cross_analysis['common_characteristics'] = list(set(all_characteristics))
            
        except Exception as e:
            cross_analysis['error'] = str(e)
        
        return cross_analysis
    
    def _generate_risk_mitigation_strategies(self, risk_assessment: Dict) -> List[str]:
        """Generate risk mitigation strategies"""
        
        strategies = []
        
        if risk_assessment['data_quality_risks']:
            strategies.append("Implement automated data validation and cleansing procedures")
        
        if risk_assessment['model_risks']:
            strategies.append("Establish model governance framework with regular validation cycles")
        
        if risk_assessment['financial_model_risks']:
            strategies.append("Implement model risk management procedures per regulatory guidelines")
        
        if risk_assessment['overall_risk_level'] == 'high':
            strategies.append("Consider independent model validation and audit procedures")
        
        return strategies
    
    def _validate_specific_model(self, model_type: str, analysis_results: Dict) -> Dict[str, Any]:
        """Validate a specific financial model"""
        
        validation = {
            'model_type': model_type,
            'validation_status': 'needs_review',
            'validation_tests': [],
            'recommendations': []
        }
        
        if model_type == 'VaR Model':
            validation['validation_tests'] = [
                'Back-testing against historical data',
                'Stress testing under extreme scenarios',
                'Model parameter stability assessment'
            ]
            validation['recommendations'] = [
                'Implement daily back-testing procedures',
                'Validate distributional assumptions quarterly'
            ]
        elif model_type == 'Portfolio Optimization':
            validation['validation_tests'] = [
                'Out-of-sample performance testing',
                'Correlation matrix stability analysis',
                'Transaction cost impact assessment'
            ]
            validation['recommendations'] = [
                'Monitor correlation breakdown during stress periods',
                'Implement robust optimization techniques'
            ]
        
        return validation
    
    def _assess_regulatory_compliance(self, detected_models: List[str]) -> Dict[str, Any]:
        """Assess regulatory compliance requirements"""
        
        compliance = {
            'applicable_regulations': [],
            'compliance_requirements': [],
            'documentation_needs': []
        }
        
        if 'VaR Model' in detected_models:
            compliance['applicable_regulations'].extend(['Basel III', 'Market Risk Capital Rules'])
            compliance['compliance_requirements'].append('Model validation and back-testing')
            compliance['documentation_needs'].append('Model methodology documentation')
        
        if 'Derivatives Pricing' in detected_models:
            compliance['applicable_regulations'].extend(['IFRS 13', 'ASC 820'])
            compliance['compliance_requirements'].append('Fair value measurement validation')
        
        return compliance
    
    def _generate_model_validation_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate model validation recommendations"""
        
        recommendations = []
        
        for model in validation_results['models_detected']:
            if model == 'VaR Model':
                recommendations.extend([
                    'Establish daily VaR back-testing procedures',
                    'Document model assumptions and limitations',
                    'Implement stress testing framework'
                ])
            elif model == 'Portfolio Optimization':
                recommendations.extend([
                    'Monitor correlation matrix stability',
                    'Validate optimization constraints',
                    'Assess transaction cost impact'
                ])
        
        return recommendations[:5]
    
    def _save_results(self, analysis_results: Dict[str, Any], output_dir: str, filename_stem: str):
        """Save analysis results to files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results as JSON
        results_file = output_path / f"{filename_stem}_comprehensive_analysis.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, default=str, ensure_ascii=False)
        
        # Save executive summary
        if analysis_results.get('enhanced_summary', {}).get('summary'):
            summary = analysis_results['enhanced_summary']['summary']
            summary_file = output_path / f"{filename_stem}_executive_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Executive Summary: {filename_stem}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"{summary.executive_summary}\n\n")
                f.write(f"Key Business Purpose: {summary.key_business_purpose}\n\n")
                f.write("Main Findings:\n")
                for i, finding in enumerate(summary.main_findings, 1):
                    f.write(f"{i}. {finding}\n")
                f.write("\nRecommendations:\n")
                for i, rec in enumerate(analysis_results.get('recommendations', []), 1):
                    f.write(f"{i}. {rec}\n")
        
        self.logger.info(f"Analysis results saved to {output_path}")


def main():
    """Main entry point for command line usage"""
    
    parser = argparse.ArgumentParser(description='Enhanced XDP Analyzer - Comprehensive Excel Analysis')
    parser.add_argument('file_path', help='Path to Excel file to analyze')
    parser.add_argument('--mode', choices=['simple', 'advanced', 'comprehensive'], 
                       default='comprehensive', help='Analysis mode (default: comprehensive)')
    parser.add_argument('--output-dir', help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize analyzer
        analyzer = EnhancedXDPAnalyzer(analysis_mode=args.mode)
        
        # Perform analysis
        results = analyzer.analyze_excel_file(args.file_path, args.output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("ENHANCED XDP ANALYZER - ANALYSIS COMPLETE")
        print("="*60)
        
        file_info = results.get('file_info', {})
        print(f"File: {file_info.get('filename', 'Unknown')}")
        print(f"Analysis Mode: {file_info.get('analysis_mode', 'Unknown')}")
        print(f"Timestamp: {file_info.get('analysis_timestamp', 'Unknown')}")
        
        # Results summary
        if results.get('parsing_results', {}).get('simple_parsing', {}).get('success'):
            simple_result = results['parsing_results']['simple_parsing']
            print(f"\nSheets Analyzed: {simple_result.get('sheet_count', 0)}")
            print(f"Data Quality Score: {simple_result.get('data_quality_score', 0)}/100")
        
        if results.get('statistical_analysis', {}).get('sheet_analyses'):
            print(f"Statistical Analyses: {len(results['statistical_analysis']['sheet_analyses'])}")
        
        if results.get('time_series_analysis', {}).get('time_series_detected'):
            print("Time Series Data: Detected")
        
        # Business insights
        insights = results.get('business_insights', [])
        if insights:
            print(f"\nTop Business Insights:")
            for i, insight in enumerate(insights[:3], 1):
                print(f"  {i}. {insight}")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec}")
        
        # Risk assessment
        risk_assessment = results.get('risk_assessment', {})
        if risk_assessment:
            print(f"\nRisk Level: {risk_assessment.get('overall_risk_level', 'Unknown').upper()}")
        
        print("\n" + "="*60)
        
        if 'error' in results:
            print(f"Analysis completed with errors: {results['error']}")
            return 1
        else:
            print("Analysis completed successfully!")
            return 0
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())