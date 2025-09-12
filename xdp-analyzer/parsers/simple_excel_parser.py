"""
Enhanced Simple Excel Parser
Reliable pandas-based Excel parsing with comprehensive analysis capabilities
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SheetAnalysis:
    """Comprehensive analysis of an Excel worksheet"""
    name: str
    shape: tuple
    columns: List[str]
    data_types: Dict[str, str]
    null_counts: Dict[str, int]
    non_null_counts: Dict[str, int]
    unique_values: Dict[str, Any]
    sample_data: List[Dict]
    formula_count: int
    has_formulas: bool
    # Enhanced statistical analysis
    statistical_summary: Dict[str, Any] = field(default_factory=dict)
    correlation_matrix: Optional[np.ndarray] = None
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    time_series_columns: List[str] = field(default_factory=list)
    # Financial modeling indicators
    financial_indicators: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    volatility_analysis: Dict[str, Any] = field(default_factory=dict)
    # Business patterns
    business_patterns: List[str] = field(default_factory=list)
    data_quality_score: float = 0.0

@dataclass
class SimpleExcelAnalysis:
    """Complete simple Excel analysis results"""
    filename: str
    file_type: str
    sheet_count: int
    sheet_names: List[str]
    sheets: List[SheetAnalysis]
    total_rows: int
    total_columns: int
    summary: Dict[str, Any]
    analysis_timestamp: str
    # Enhanced analysis fields
    cross_sheet_analysis: Dict[str, Any] = field(default_factory=dict)
    financial_modeling_assessment: Dict[str, Any] = field(default_factory=dict)
    time_series_analysis: Dict[str, Any] = field(default_factory=dict)
    business_intelligence: Dict[str, Any] = field(default_factory=dict)

class SimpleExcelParser:
    """
    Enhanced Simple Excel parser with comprehensive analysis capabilities
    Uses pandas for reliability with advanced statistical and financial analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['.xlsx', '.xlsm', '.xls', '.xlsb']
        
        # Financial modeling patterns
        self.financial_patterns = {
            'var_risk': ['var', 'value at risk', 'volatility', 'correlation', 'portfolio'],
            'option_pricing': ['black scholes', 'binomial', 'option', 'strike', 'expiry'],
            'fixed_income': ['yield', 'duration', 'convexity', 'bond', 'coupon'],
            'derivatives': ['swap', 'forward', 'future', 'derivative', 'hedge'],
            'performance': ['return', 'alpha', 'beta', 'sharpe', 'information ratio']
        }
        
        # Time series indicators
        self.time_series_patterns = [
            'date', 'time', 'period', 'month', 'year', 'quarter', 'day',
            'timestamp', 'datetime'
        ]
        
        # Business intelligence keywords
        self.business_keywords = {
            'finance': ['revenue', 'profit', 'cost', 'budget', 'forecast', 'cash flow'],
            'operations': ['inventory', 'production', 'efficiency', 'capacity', 'throughput'],
            'sales': ['pipeline', 'conversion', 'quota', 'territory', 'customer'],
            'marketing': ['campaign', 'roi', 'ctr', 'impressions', 'engagement'],
            'hr': ['employee', 'payroll', 'performance', 'benefits', 'training']
        }
    
    def parse(self, file_path: Union[str, Path]) -> SimpleExcelAnalysis:
        """
        Parse Excel file with comprehensive analysis
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            SimpleExcelAnalysis object with complete analysis
        """
        file_path = Path(file_path)
        self.logger.info(f"Starting enhanced simple parse of {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        try:
            # Read Excel file with all sheets
            excel_file = pd.ExcelFile(file_path, engine=self._get_engine(file_ext))
            sheet_names = excel_file.sheet_names
            
            # Analyze each sheet
            sheet_analyses = []
            total_rows = 0
            max_columns = 0
            
            for sheet_name in sheet_names:
                try:
                    sheet_analysis = self._analyze_sheet(excel_file, sheet_name)
                    sheet_analyses.append(sheet_analysis)
                    total_rows += sheet_analysis.shape[0]
                    max_columns = max(max_columns, sheet_analysis.shape[1])
                except Exception as e:
                    self.logger.warning(f"Error analyzing sheet {sheet_name}: {e}")
                    # Create minimal analysis for failed sheets
                    sheet_analyses.append(SheetAnalysis(
                        name=sheet_name,
                        shape=(0, 0),
                        columns=[],
                        data_types={},
                        null_counts={},
                        non_null_counts={},
                        unique_values={},
                        sample_data=[],
                        formula_count=0,
                        has_formulas=False
                    ))
            
            # Generate summary
            summary = self._generate_summary(sheet_analyses)
            
            # Perform cross-sheet analysis
            cross_sheet = self._perform_cross_sheet_analysis(sheet_analyses, excel_file)
            
            # Assess financial modeling capabilities
            financial_assessment = self._assess_financial_modeling(sheet_analyses)
            
            # Analyze time series data
            time_series = self._analyze_time_series(sheet_analyses)
            
            # Extract business intelligence
            business_intel = self._extract_business_intelligence(sheet_analyses)
            
            analysis = SimpleExcelAnalysis(
                filename=file_path.name,
                file_type=file_ext,
                sheet_count=len(sheet_names),
                sheet_names=sheet_names,
                sheets=sheet_analyses,
                total_rows=total_rows,
                total_columns=max_columns,
                summary=summary,
                analysis_timestamp=datetime.now().isoformat(),
                cross_sheet_analysis=cross_sheet,
                financial_modeling_assessment=financial_assessment,
                time_series_analysis=time_series,
                business_intelligence=business_intel
            )
            
            excel_file.close()
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {str(e)}")
            raise
    
    def _get_engine(self, file_ext: str) -> str:
        """Get appropriate pandas engine for file type"""
        engine_map = {
            '.xlsx': 'openpyxl',
            '.xlsm': 'openpyxl',
            '.xls': 'xlrd',
            '.xlsb': 'pyxlsb'
        }
        return engine_map.get(file_ext, 'openpyxl')
    
    def _analyze_sheet(self, excel_file: pd.ExcelFile, sheet_name: str) -> SheetAnalysis:
        """Analyze individual sheet with enhanced capabilities"""
        
        # Read sheet data
        df = excel_file.parse(sheet_name, header=0)
        
        # Basic information
        shape = df.shape
        columns = list(df.columns)
        
        # Data type analysis
        data_types = {}
        null_counts = {}
        non_null_counts = {}
        unique_values = {}
        
        for col in columns:
            dtype_str = str(df[col].dtype)
            data_types[col] = dtype_str
            null_counts[col] = df[col].isnull().sum()
            non_null_counts[col] = df[col].notnull().sum()
            
            # Unique values (limited to avoid memory issues)
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 20:  # Only store if manageable
                unique_values[col] = unique_vals.tolist()
        
        # Sample data (first few rows)
        sample_data = []
        if not df.empty:
            sample_rows = min(3, len(df))
            for i in range(sample_rows):
                row_dict = {}
                for col in columns:
                    value = df.iloc[i][col]
                    if pd.isna(value):
                        row_dict[col] = None
                    else:
                        row_dict[col] = str(value)[:100]  # Limit string length
                sample_data.append(row_dict)
        
        # Enhanced statistical analysis
        statistical_summary = self._calculate_statistics(df)
        
        # Identify column types
        numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
        categorical_columns = list(df.select_dtypes(include=['object', 'category']).columns)
        time_series_columns = self._identify_time_series_columns(df)
        
        # Correlation analysis for numeric columns
        correlation_matrix = None
        if len(numeric_columns) > 1:
            correlation_matrix = df[numeric_columns].corr().values
        
        # Financial indicators
        financial_indicators = self._analyze_financial_indicators(df, sheet_name)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(df, numeric_columns)
        
        # Volatility analysis
        volatility_analysis = self._analyze_volatility(df, numeric_columns)
        
        # Business patterns
        business_patterns = self._identify_business_patterns(df, sheet_name)
        
        # Data quality score
        data_quality_score = self._calculate_data_quality_score(df)
        
        return SheetAnalysis(
            name=sheet_name,
            shape=shape,
            columns=columns,
            data_types=data_types,
            null_counts=null_counts,
            non_null_counts=non_null_counts,
            unique_values=unique_values,
            sample_data=sample_data,
            formula_count=0,  # Simple parser doesn't extract formulas
            has_formulas=False,
            statistical_summary=statistical_summary,
            correlation_matrix=correlation_matrix,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            time_series_columns=time_series_columns,
            financial_indicators=financial_indicators,
            risk_metrics=risk_metrics,
            volatility_analysis=volatility_analysis,
            business_patterns=business_patterns,
            data_quality_score=data_quality_score
        )
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistical summary"""
        stats = {}
        
        # Numeric column statistics
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats['numeric_summary'] = {
                'mean': numeric_df.mean().to_dict(),
                'median': numeric_df.median().to_dict(),
                'std': numeric_df.std().to_dict(),
                'min': numeric_df.min().to_dict(),
                'max': numeric_df.max().to_dict(),
                'skewness': numeric_df.skew().to_dict(),
                'kurtosis': numeric_df.kurtosis().to_dict()
            }
            
            # Percentiles
            percentiles = [0.05, 0.25, 0.75, 0.95]
            stats['percentiles'] = {}
            for p in percentiles:
                stats['percentiles'][f'p{int(p*100)}'] = numeric_df.quantile(p).to_dict()
        
        # Categorical statistics
        categorical_df = df.select_dtypes(include=['object', 'category'])
        if not categorical_df.empty:
            stats['categorical_summary'] = {}
            for col in categorical_df.columns:
                value_counts = categorical_df[col].value_counts()
                stats['categorical_summary'][col] = {
                    'unique_count': len(value_counts),
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'top_5_values': value_counts.head().to_dict()
                }
        
        return stats
    
    def _identify_time_series_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that contain time series data"""
        time_columns = []
        
        for col in df.columns:
            col_lower = col.lower()
            # Check column name
            if any(pattern in col_lower for pattern in self.time_series_patterns):
                time_columns.append(col)
                continue
            
            # Check data type
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                time_columns.append(col)
                continue
            
            # Check if values look like dates
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().astype(str).head(10)
                date_like = 0
                for value in sample_values:
                    try:
                        pd.to_datetime(value, infer_datetime_format=True)
                        date_like += 1
                    except:
                        pass
                
                if date_like >= 5:  # At least half look like dates
                    time_columns.append(col)
        
        return time_columns
    
    def _analyze_financial_indicators(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """Analyze financial modeling indicators"""
        indicators = {
            'has_financial_data': False,
            'financial_categories': [],
            'risk_indicators': [],
            'performance_metrics': [],
            'model_complexity': 'basic'
        }
        
        # Check sheet name for financial terms
        sheet_lower = sheet_name.lower()
        text_to_analyze = sheet_lower + ' ' + ' '.join(df.columns).lower()
        
        # Analyze against financial patterns
        for category, patterns in self.financial_patterns.items():
            if any(pattern in text_to_analyze for pattern in patterns):
                indicators['financial_categories'].append(category)
                indicators['has_financial_data'] = True
        
        # Check for specific financial indicators in data
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Look for returns, volatility patterns
        for col in numeric_columns:
            col_data = df[col].dropna()
            if len(col_data) > 10:
                # Check if data looks like returns (centered around 0, small values)
                if col_data.mean() < 0.1 and col_data.std() > 0:
                    if 'return' in col.lower() or abs(col_data.mean()) < col_data.std():
                        indicators['risk_indicators'].append(f'{col}: potential returns data')
                
                # Check for price-like data (positive, trending)
                elif col_data.min() > 0 and len(col_data) > 20:
                    if col_data.std() / col_data.mean() > 0.1:  # Some volatility
                        indicators['performance_metrics'].append(f'{col}: potential price series')
        
        # Assess model complexity
        if len(indicators['financial_categories']) > 2:
            indicators['model_complexity'] = 'advanced'
        elif len(indicators['financial_categories']) > 0:
            indicators['model_complexity'] = 'intermediate'
        
        return indicators
    
    def _calculate_risk_metrics(self, df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
        """Calculate risk-related metrics"""
        risk_metrics = {}
        
        for col in numeric_columns:
            col_data = df[col].dropna()
            if len(col_data) > 10:
                # Basic risk metrics
                mean_val = col_data.mean()
                std_val = col_data.std()
                
                risk_metrics[col] = {
                    'volatility': std_val,
                    'coefficient_of_variation': std_val / abs(mean_val) if mean_val != 0 else np.inf,
                    'value_at_risk_95': col_data.quantile(0.05) if mean_val < std_val else None,
                    'max_drawdown': self._calculate_max_drawdown(col_data),
                    'tail_risk': self._calculate_tail_risk(col_data)
                }
        
        return risk_metrics
    
    def _calculate_max_drawdown(self, series: pd.Series) -> float:
        """Calculate maximum drawdown for a series"""
        try:
            cumulative = (1 + series).cumprod() if series.mean() < 0.1 else series.cumsum()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except:
            return 0.0
    
    def _calculate_tail_risk(self, series: pd.Series) -> Dict[str, float]:
        """Calculate tail risk metrics"""
        try:
            return {
                'left_tail_5': series.quantile(0.05),
                'right_tail_95': series.quantile(0.95),
                'expected_shortfall_5': series[series <= series.quantile(0.05)].mean()
            }
        except:
            return {}
    
    def _analyze_volatility(self, df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
        """Analyze volatility patterns in numeric data"""
        volatility_analysis = {}
        
        for col in numeric_columns:
            col_data = df[col].dropna()
            if len(col_data) > 20:
                # Calculate rolling volatility if enough data
                returns = col_data.pct_change().dropna()
                
                if len(returns) > 10:
                    volatility_analysis[col] = {
                        'historical_volatility': returns.std(),
                        'volatility_of_volatility': returns.rolling(10).std().std() if len(returns) > 20 else None,
                        'volatility_clustering': self._test_volatility_clustering(returns),
                        'garch_indicators': self._detect_garch_patterns(returns)
                    }
        
        return volatility_analysis
    
    def _test_volatility_clustering(self, returns: pd.Series) -> bool:
        """Simple test for volatility clustering"""
        if len(returns) < 20:
            return False
        
        # Test if high/low volatility periods cluster
        abs_returns = returns.abs()
        rolling_vol = abs_returns.rolling(5).mean()
        
        # Count regime changes
        high_vol = rolling_vol > rolling_vol.median()
        changes = (high_vol != high_vol.shift()).sum()
        
        # If fewer changes than expected, indicates clustering
        expected_changes = len(returns) * 0.4
        return changes < expected_changes
    
    def _detect_garch_patterns(self, returns: pd.Series) -> Dict[str, Any]:
        """Detect GARCH-like patterns in returns"""
        if len(returns) < 30:
            return {}
        
        # Simple GARCH indicators
        squared_returns = returns ** 2
        lag1_corr = squared_returns.autocorr(lag=1) if len(squared_returns) > 30 else 0
        
        return {
            'squared_returns_autocorr': lag1_corr,
            'potential_garch': lag1_corr > 0.2,
            'volatility_persistence': lag1_corr
        }
    
    def _identify_business_patterns(self, df: pd.DataFrame, sheet_name: str) -> List[str]:
        """Identify business patterns in the data"""
        patterns = []
        
        # Analyze sheet name and column names
        text_to_analyze = (sheet_name + ' ' + ' '.join(df.columns)).lower()
        
        for category, keywords in self.business_keywords.items():
            if any(keyword in text_to_analyze for keyword in keywords):
                patterns.append(f'{category}_data')
        
        # Analyze data patterns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Time series patterns
        if len(self._identify_time_series_columns(df)) > 0:
            patterns.append('time_series_analysis')
        
        # High-frequency data patterns
        if len(df) > 1000:
            patterns.append('high_frequency_data')
        
        # Cross-sectional data
        if len(numeric_columns) > 10:
            patterns.append('cross_sectional_analysis')
        
        # Panel data indicators
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 2 and len(numeric_columns) > 3:
            patterns.append('panel_data_structure')
        
        return list(set(patterns))
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score"""
        if df.empty:
            return 0.0
        
        score = 100.0
        
        # Penalize for missing data
        missing_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        score -= missing_rate * 30
        
        # Reward for data variety
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        if numeric_cols > 0:
            score += min(numeric_cols * 2, 10)
        
        # Penalize for duplicate rows
        duplicate_rate = df.duplicated().sum() / len(df)
        score -= duplicate_rate * 20
        
        # Reward for reasonable data distribution
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            # Check for outliers (simple method)
            outlier_rate = 0
            for col in numeric_df.columns:
                q1, q3 = numeric_df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = ((numeric_df[col] < q1 - 1.5*iqr) | (numeric_df[col] > q3 + 1.5*iqr)).sum()
                outlier_rate += outliers / len(numeric_df[col])
            
            outlier_rate /= len(numeric_df.columns)
            if outlier_rate < 0.05:  # Less than 5% outliers is good
                score += 5
            elif outlier_rate > 0.2:  # More than 20% outliers is concerning
                score -= 10
        
        return max(0, min(100, score))
    
    def _perform_cross_sheet_analysis(self, sheets: List[SheetAnalysis], excel_file: pd.ExcelFile) -> Dict[str, Any]:
        """Perform analysis across multiple sheets"""
        cross_analysis = {
            'sheet_relationships': [],
            'common_columns': [],
            'data_consistency': {},
            'aggregation_opportunities': []
        }
        
        if len(sheets) < 2:
            return cross_analysis
        
        # Find common columns across sheets
        all_columns = set()
        sheet_columns = {}
        
        for sheet in sheets:
            sheet_columns[sheet.name] = set(sheet.columns)
            all_columns.update(sheet.columns)
        
        # Identify common columns
        common_cols = set.intersection(*sheet_columns.values()) if sheet_columns else set()
        cross_analysis['common_columns'] = list(common_cols)
        
        # Analyze relationships between sheets
        for i, sheet1 in enumerate(sheets):
            for sheet2 in sheets[i+1:]:
                overlap = len(set(sheet1.columns) & set(sheet2.columns))
                if overlap > 0:
                    cross_analysis['sheet_relationships'].append({
                        'sheet1': sheet1.name,
                        'sheet2': sheet2.name,
                        'common_columns': overlap,
                        'relationship_strength': overlap / max(len(sheet1.columns), len(sheet2.columns))
                    })
        
        # Data consistency analysis
        if common_cols and len(sheets) > 1:
            cross_analysis['data_consistency'] = self._analyze_data_consistency(sheets, common_cols, excel_file)
        
        return cross_analysis
    
    def _analyze_data_consistency(self, sheets: List[SheetAnalysis], common_columns: List[str], excel_file: pd.ExcelFile) -> Dict[str, Any]:
        """Analyze data consistency across sheets"""
        consistency = {}
        
        for col in common_columns[:5]:  # Limit to avoid performance issues
            col_analysis = {
                'data_types_consistent': True,
                'value_ranges': {},
                'unique_values_overlap': 0
            }
            
            data_types = []
            all_values = set()
            
            for sheet in sheets:
                if col in sheet.data_types:
                    data_types.append(sheet.data_types[col])
                    
                    # Get unique values if available
                    if col in sheet.unique_values:
                        sheet_values = set(str(v) for v in sheet.unique_values[col])
                        if not all_values:
                            all_values = sheet_values
                        else:
                            all_values = all_values & sheet_values
            
            # Check data type consistency
            col_analysis['data_types_consistent'] = len(set(data_types)) <= 1
            col_analysis['unique_values_overlap'] = len(all_values)
            
            consistency[col] = col_analysis
        
        return consistency
    
    def _assess_financial_modeling(self, sheets: List[SheetAnalysis]) -> Dict[str, Any]:
        """Assess financial modeling capabilities"""
        assessment = {
            'overall_score': 0,
            'model_sophistication': 'basic',
            'financial_domains': set(),
            'risk_analysis_capability': False,
            'time_series_modeling': False,
            'portfolio_analysis': False,
            'derivative_pricing': False
        }
        
        total_financial_indicators = 0
        
        for sheet in sheets:
            financial_indicators = sheet.financial_indicators
            total_financial_indicators += len(financial_indicators.get('financial_categories', []))
            
            # Update domains
            assessment['financial_domains'].update(financial_indicators.get('financial_categories', []))
            
            # Check for specific capabilities
            if 'var_risk' in financial_indicators.get('financial_categories', []):
                assessment['risk_analysis_capability'] = True
                assessment['overall_score'] += 25
            
            if sheet.time_series_columns:
                assessment['time_series_modeling'] = True
                assessment['overall_score'] += 20
            
            if 'portfolio' in str(financial_indicators).lower():
                assessment['portfolio_analysis'] = True
                assessment['overall_score'] += 20
            
            if 'option_pricing' in financial_indicators.get('financial_categories', []):
                assessment['derivative_pricing'] = True
                assessment['overall_score'] += 30
        
        # Determine sophistication
        if assessment['overall_score'] > 60:
            assessment['model_sophistication'] = 'advanced'
        elif assessment['overall_score'] > 30:
            assessment['model_sophistication'] = 'intermediate'
        
        assessment['financial_domains'] = list(assessment['financial_domains'])
        return assessment
    
    def _analyze_time_series(self, sheets: List[SheetAnalysis]) -> Dict[str, Any]:
        """Analyze time series characteristics"""
        time_analysis = {
            'has_time_series': False,
            'time_series_sheets': [],
            'frequency_analysis': {},
            'trend_analysis': {},
            'seasonality_indicators': {}
        }
        
        for sheet in sheets:
            if sheet.time_series_columns:
                time_analysis['has_time_series'] = True
                time_analysis['time_series_sheets'].append({
                    'sheet': sheet.name,
                    'time_columns': sheet.time_series_columns,
                    'data_points': sheet.shape[0]
                })
        
        return time_analysis
    
    def _extract_business_intelligence(self, sheets: List[SheetAnalysis]) -> Dict[str, Any]:
        """Extract business intelligence insights"""
        business_intel = {
            'primary_business_domain': 'unknown',
            'key_metrics_identified': [],
            'dashboard_potential': False,
            'kpi_candidates': [],
            'business_processes': []
        }
        
        all_patterns = []
        all_columns = []
        
        for sheet in sheets:
            all_patterns.extend(sheet.business_patterns)
            all_columns.extend(sheet.columns)
        
        # Identify primary domain
        pattern_counts = {}
        for pattern in all_patterns:
            domain = pattern.split('_')[0]
            pattern_counts[domain] = pattern_counts.get(domain, 0) + 1
        
        if pattern_counts:
            business_intel['primary_business_domain'] = max(pattern_counts, key=pattern_counts.get)
        
        # Identify KPI candidates
        kpi_keywords = ['total', 'sum', 'count', 'average', 'rate', 'ratio', 'percent', 'score']
        for col in all_columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in kpi_keywords):
                business_intel['kpi_candidates'].append(col)
        
        # Dashboard potential
        if len(sheets) > 2 and len(business_intel['kpi_candidates']) > 3:
            business_intel['dashboard_potential'] = True
        
        return business_intel
    
    def _generate_summary(self, sheets: List[SheetAnalysis]) -> Dict[str, Any]:
        """Generate comprehensive summary"""
        
        total_cells = sum(sheet.shape[0] * sheet.shape[1] for sheet in sheets)
        total_formulas = sum(sheet.formula_count for sheet in sheets)
        sheets_with_formulas = sum(1 for sheet in sheets if sheet.has_formulas)
        
        # Data quality analysis
        avg_quality_score = np.mean([sheet.data_quality_score for sheet in sheets]) if sheets else 0
        
        # Find most complex sheet
        most_complex_sheet = max(sheets, key=lambda x: len(x.numeric_columns) + len(x.categorical_columns)).name if sheets else "None"
        
        # Calculate average sheet size
        avg_sheet_size = total_cells / len(sheets) if sheets else 0
        
        return {
            'total_cells': total_cells,
            'total_formulas': total_formulas,
            'sheets_with_formulas': sheets_with_formulas,
            'data_quality_score': round(avg_quality_score, 1),
            'most_complex_sheet': most_complex_sheet,
            'average_sheet_size': round(avg_sheet_size, 0),
            'has_time_series': any(sheet.time_series_columns for sheet in sheets),
            'has_financial_data': any(sheet.financial_indicators.get('has_financial_data', False) for sheet in sheets),
            'primary_analysis_type': self._determine_primary_analysis_type(sheets)
        }
    
    def _determine_primary_analysis_type(self, sheets: List[SheetAnalysis]) -> str:
        """Determine the primary type of analysis this workbook supports"""
        
        # Count different analysis indicators
        analysis_scores = {
            'financial_modeling': 0,
            'time_series_analysis': 0,
            'cross_sectional_analysis': 0,
            'business_intelligence': 0,
            'data_repository': 0
        }
        
        for sheet in sheets:
            # Financial modeling score
            if sheet.financial_indicators.get('has_financial_data', False):
                analysis_scores['financial_modeling'] += 2
            if sheet.risk_metrics:
                analysis_scores['financial_modeling'] += 1
            
            # Time series score
            if sheet.time_series_columns:
                analysis_scores['time_series_analysis'] += 2
            if sheet.shape[0] > 100:  # Long time series
                analysis_scores['time_series_analysis'] += 1
            
            # Cross-sectional score
            if len(sheet.numeric_columns) > 5:
                analysis_scores['cross_sectional_analysis'] += 1
            if sheet.correlation_matrix is not None:
                analysis_scores['cross_sectional_analysis'] += 1
            
            # Business intelligence score
            if sheet.business_patterns:
                analysis_scores['business_intelligence'] += len(sheet.business_patterns)
            
            # Data repository score (baseline)
            analysis_scores['data_repository'] += 1
        
        # Return the analysis type with highest score
        return max(analysis_scores, key=analysis_scores.get)

# Example usage
if __name__ == "__main__":
    parser = SimpleExcelParser()
    
    # Test with a sample file
    try:
        analysis = parser.parse("sample.xlsx")
        print(f"Analysis complete: {analysis.filename}")
        print(f"Sheets: {analysis.sheet_count}")
        print(f"Primary Analysis Type: {analysis.summary['primary_analysis_type']}")
        print(f"Financial Modeling Score: {analysis.financial_modeling_assessment['overall_score']}")
    except Exception as e:
        print(f"Error: {e}")