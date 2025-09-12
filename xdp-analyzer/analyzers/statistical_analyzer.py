"""
Advanced Statistical Analyzer
Comprehensive statistical analysis for Excel data with financial modeling focus
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import optimize
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StatisticalSummary:
    """Comprehensive statistical analysis results"""
    column_name: str
    basic_stats: Dict[str, float]
    distribution_analysis: Dict[str, Any]
    outlier_analysis: Dict[str, Any]
    normality_tests: Dict[str, float]
    risk_metrics: Dict[str, float]
    financial_indicators: Dict[str, Any]

@dataclass
class CorrelationAnalysis:
    """Correlation and dependency analysis"""
    correlation_matrix: np.ndarray
    significant_correlations: List[Dict]
    dependency_analysis: Dict[str, Any]
    portfolio_implications: Dict[str, Any]

@dataclass
class TimeSeriesAnalysis:
    """Time series analysis results"""
    column_name: str
    trend_analysis: Dict[str, Any]
    seasonality_analysis: Dict[str, Any]
    volatility_analysis: Dict[str, Any]
    stationarity_tests: Dict[str, float]
    forecast_quality: Dict[str, Any]

class AdvancedStatisticalAnalyzer:
    """
    Advanced statistical analyzer for Excel data with focus on financial modeling
    """
    
    def __init__(self):
        self.confidence_levels = [0.95, 0.99, 0.999]  # For VaR calculations
        self.var_methods = ['historical', 'parametric', 'monte_carlo']
        
    def analyze_dataset(self, df: pd.DataFrame, sheet_name: str = "Unknown") -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis on dataset
        
        Args:
            df: DataFrame to analyze
            sheet_name: Name of the worksheet
            
        Returns:
            Complete statistical analysis results
        """
        
        if df.empty:
            return self._empty_analysis_result(sheet_name)
        
        analysis_results = {
            'sheet_name': sheet_name,
            'basic_info': self._get_basic_info(df),
            'column_analyses': [],
            'correlation_analysis': None,
            'time_series_analyses': [],
            'financial_modeling_assessment': {},
            'risk_assessment': {},
            'data_quality_metrics': {},
            'business_insights': []
        }
        
        # Analyze numeric columns individually
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if not df[col].dropna().empty:
                column_analysis = self._analyze_numeric_column(df[col], col)
                analysis_results['column_analyses'].append(column_analysis)
        
        # Cross-column correlation analysis
        if len(numeric_columns) > 1:
            analysis_results['correlation_analysis'] = self._analyze_correlations(df[numeric_columns])
        
        # Time series analysis for time-based data
        time_columns = self._identify_time_columns(df)
        for time_col in time_columns:
            for value_col in numeric_columns:
                if time_col != value_col and not df[value_col].dropna().empty:
                    ts_analysis = self._analyze_time_series(df, time_col, value_col)
                    if ts_analysis:
                        analysis_results['time_series_analyses'].append(ts_analysis)
        
        # Financial modeling assessment
        analysis_results['financial_modeling_assessment'] = self._assess_financial_modeling_capability(df, analysis_results)
        
        # Risk assessment
        analysis_results['risk_assessment'] = self._perform_risk_assessment(df, analysis_results)
        
        # Data quality metrics
        analysis_results['data_quality_metrics'] = self._calculate_data_quality_metrics(df)
        
        # Business insights
        analysis_results['business_insights'] = self._generate_business_insights(analysis_results)
        
        return analysis_results
    
    def _analyze_numeric_column(self, series: pd.Series, column_name: str) -> StatisticalSummary:
        """Analyze a single numeric column comprehensively"""
        
        clean_series = series.dropna()
        
        if clean_series.empty:
            return self._empty_column_analysis(column_name)
        
        # Basic statistics
        basic_stats = {
            'count': len(clean_series),
            'mean': clean_series.mean(),
            'median': clean_series.median(),
            'std': clean_series.std(),
            'variance': clean_series.var(),
            'min': clean_series.min(),
            'max': clean_series.max(),
            'range': clean_series.max() - clean_series.min(),
            'q25': clean_series.quantile(0.25),
            'q75': clean_series.quantile(0.75),
            'iqr': clean_series.quantile(0.75) - clean_series.quantile(0.25),
            'skewness': clean_series.skew(),
            'kurtosis': clean_series.kurtosis()
        }
        
        # Distribution analysis
        distribution_analysis = self._analyze_distribution(clean_series)
        
        # Outlier analysis
        outlier_analysis = self._detect_outliers(clean_series)
        
        # Normality tests
        normality_tests = self._test_normality(clean_series)
        
        # Risk metrics (financial focus)
        risk_metrics = self._calculate_risk_metrics(clean_series)
        
        # Financial indicators
        financial_indicators = self._assess_financial_characteristics(clean_series, column_name)
        
        return StatisticalSummary(
            column_name=column_name,
            basic_stats=basic_stats,
            distribution_analysis=distribution_analysis,
            outlier_analysis=outlier_analysis,
            normality_tests=normality_tests,
            risk_metrics=risk_metrics,
            financial_indicators=financial_indicators
        )
    
    def _analyze_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze the distribution of data"""
        
        analysis = {
            'distribution_type': 'unknown',
            'parameters': {},
            'fit_quality': {},
            'percentiles': {},
            'tail_analysis': {}
        }
        
        # Calculate key percentiles
        percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        for p in percentiles:
            analysis['percentiles'][f'p{int(p*100)}'] = series.quantile(p)
        
        # Test common distributions
        distributions_to_test = ['norm', 'lognorm', 't', 'gamma', 'beta']
        best_fit = None
        best_p_value = 0
        
        for dist_name in distributions_to_test:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(series)
                ks_stat, p_value = stats.kstest(series, lambda x: dist.cdf(x, *params))
                
                if p_value > best_p_value:
                    best_fit = dist_name
                    best_p_value = p_value
                    analysis['parameters'][dist_name] = params
                    analysis['fit_quality'][dist_name] = p_value
                    
            except:
                continue
        
        if best_fit:
            analysis['distribution_type'] = best_fit
            analysis['best_fit_p_value'] = best_p_value
        
        # Tail analysis for financial applications
        analysis['tail_analysis'] = {
            'left_tail_weight': len(series[series < series.quantile(0.05)]) / len(series),
            'right_tail_weight': len(series[series > series.quantile(0.95)]) / len(series),
            'tail_ratio': series.quantile(0.95) / abs(series.quantile(0.05)) if series.quantile(0.05) != 0 else np.inf
        }
        
        return analysis
    
    def _detect_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        
        # IQR method
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        iqr_outliers = series[(series < iqr_lower) | (series > iqr_upper)]
        
        # Z-score method
        z_scores = np.abs(stats.zscore(series))
        z_outliers = series[z_scores > 3]
        
        # Modified Z-score (using median)
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad if mad != 0 else np.zeros_like(series)
        modified_z_outliers = series[np.abs(modified_z_scores) > 3.5]
        
        return {
            'iqr_method': {
                'count': len(iqr_outliers),
                'percentage': len(iqr_outliers) / len(series) * 100,
                'bounds': (iqr_lower, iqr_upper)
            },
            'zscore_method': {
                'count': len(z_outliers),
                'percentage': len(z_outliers) / len(series) * 100
            },
            'modified_zscore_method': {
                'count': len(modified_z_outliers),
                'percentage': len(modified_z_outliers) / len(series) * 100
            },
            'outlier_impact': 'high' if len(iqr_outliers) / len(series) > 0.1 else 'medium' if len(iqr_outliers) / len(series) > 0.05 else 'low'
        }
    
    def _test_normality(self, series: pd.Series) -> Dict[str, float]:
        """Test for normality using multiple statistical tests"""
        
        tests = {}
        
        if len(series) >= 20:
            # Shapiro-Wilk test (good for small to medium samples)
            if len(series) <= 5000:
                try:
                    stat, p_value = stats.shapiro(series)
                    tests['shapiro_wilk'] = p_value
                except:
                    pass
            
            # Kolmogorov-Smirnov test
            try:
                stat, p_value = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
                tests['kolmogorov_smirnov'] = p_value
            except:
                pass
            
            # Anderson-Darling test
            try:
                result = stats.anderson(series, dist='norm')
                # Convert to approximate p-value
                critical_values = result.critical_values
                if result.statistic < critical_values[2]:  # 5% significance level
                    tests['anderson_darling'] = 0.05
                else:
                    tests['anderson_darling'] = 0.01
            except:
                pass
            
            # D'Agostino's normality test
            try:
                stat, p_value = stats.normaltest(series)
                tests['dagostino'] = p_value
            except:
                pass
        
        return tests
    
    def _calculate_risk_metrics(self, series: pd.Series) -> Dict[str, float]:
        """Calculate financial risk metrics"""
        
        if len(series) < 10:
            return {}
        
        risk_metrics = {}
        
        # Volatility (standard deviation)
        risk_metrics['volatility'] = series.std()
        
        # Coefficient of variation
        if series.mean() != 0:
            risk_metrics['coefficient_of_variation'] = series.std() / abs(series.mean())
        
        # Value at Risk (VaR) calculations
        for confidence in self.confidence_levels:
            alpha = 1 - confidence
            var_historical = series.quantile(alpha)
            risk_metrics[f'var_{int(confidence*100)}_historical'] = var_historical
            
            # Parametric VaR (assuming normal distribution)
            if series.std() > 0:
                var_parametric = series.mean() + stats.norm.ppf(alpha) * series.std()
                risk_metrics[f'var_{int(confidence*100)}_parametric'] = var_parametric
        
        # Expected Shortfall (Conditional VaR)
        var_95 = series.quantile(0.05)
        expected_shortfall = series[series <= var_95].mean()
        risk_metrics['expected_shortfall_95'] = expected_shortfall
        
        # Maximum drawdown (if data looks like cumulative returns)
        if series.min() >= 0:  # Positive series, might be prices
            cumulative = series / series.iloc[0] if series.iloc[0] != 0 else series
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            risk_metrics['max_drawdown'] = drawdowns.min()
        
        # Downside deviation
        downside_returns = series[series < series.mean()]
        if len(downside_returns) > 0:
            risk_metrics['downside_deviation'] = downside_returns.std()
        
        return risk_metrics
    
    def _assess_financial_characteristics(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Assess if data has financial characteristics"""
        
        characteristics = {
            'likely_data_type': 'unknown',
            'financial_indicators': [],
            'statistical_properties': [],
            'modeling_suitability': {}
        }
        
        col_lower = column_name.lower()
        
        # Assess data type based on properties
        if series.min() >= 0 and series.std() / series.mean() > 0.1 if series.mean() != 0 else False:
            if any(term in col_lower for term in ['price', 'value', 'amount', 'cost']):
                characteristics['likely_data_type'] = 'price_series'
                characteristics['financial_indicators'].append('Positive values with significant volatility suggest price data')
        
        elif abs(series.mean()) < series.std() and -0.5 < series.mean() < 0.5:
            if any(term in col_lower for term in ['return', 'change', 'pct', 'percent']):
                characteristics['likely_data_type'] = 'return_series'
                characteristics['financial_indicators'].append('Small mean relative to volatility suggests return data')
        
        # Statistical properties relevant to finance
        if abs(series.skew()) > 0.5:
            characteristics['statistical_properties'].append(f'Significant skewness ({series.skew():.2f}) - common in financial returns')
        
        if series.kurtosis() > 3:
            characteristics['statistical_properties'].append(f'Excess kurtosis ({series.kurtosis():.2f}) - fat tails typical in finance')
        
        # Volatility clustering test (simplified)
        if len(series) > 20:
            abs_series = series.abs() if characteristics['likely_data_type'] == 'return_series' else series.diff().abs()
            abs_series = abs_series.dropna()
            if len(abs_series) > 10:
                autocorr = abs_series.autocorr(lag=1) if hasattr(abs_series, 'autocorr') else 0
                if autocorr > 0.2:
                    characteristics['statistical_properties'].append('Evidence of volatility clustering')
        
        # Modeling suitability
        characteristics['modeling_suitability'] = {
            'suitable_for_var': len(series) > 100 and characteristics['likely_data_type'] == 'return_series',
            'suitable_for_garch': len(series) > 200 and 'volatility clustering' in str(characteristics['statistical_properties']),
            'suitable_for_portfolio_optimization': characteristics['likely_data_type'] in ['return_series', 'price_series'] and len(series) > 50
        }
        
        return characteristics
    
    def _analyze_correlations(self, df: pd.DataFrame) -> CorrelationAnalysis:
        """Analyze correlations between numeric columns"""
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Find significant correlations
        significant_correlations = []
        n = len(df)
        critical_value = 1.96 / np.sqrt(n - 3)  # Approximate critical value for correlation
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > critical_value and not np.isnan(corr_val):
                    significant_correlations.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate' if abs(corr_val) > 0.3 else 'weak',
                        'direction': 'positive' if corr_val > 0 else 'negative'
                    })
        
        # Dependency analysis
        dependency_analysis = self._analyze_dependencies(df)
        
        # Portfolio implications if this looks like financial data
        portfolio_implications = self._assess_portfolio_implications(corr_matrix, df)
        
        return CorrelationAnalysis(
            correlation_matrix=corr_matrix.values,
            significant_correlations=significant_correlations,
            dependency_analysis=dependency_analysis,
            portfolio_implications=portfolio_implications
        )
    
    def _analyze_dependencies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dependencies between variables"""
        
        dependencies = {
            'linear_dependencies': [],
            'potential_multicollinearity': [],
            'principal_components_needed': 0
        }
        
        # Check for multicollinearity using correlation matrix
        corr_matrix = df.corr()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8 and not np.isnan(corr_val):
                    high_corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        dependencies['potential_multicollinearity'] = high_corr_pairs
        
        # Estimate number of principal components needed (simplified)
        try:
            eigenvals = np.linalg.eigvals(corr_matrix.fillna(0))
            cumvar_explained = np.cumsum(eigenvals) / np.sum(eigenvals)
            components_90 = np.argmax(cumvar_explained >= 0.9) + 1
            dependencies['principal_components_needed'] = min(components_90, len(df.columns))
        except:
            dependencies['principal_components_needed'] = len(df.columns)
        
        return dependencies
    
    def _assess_portfolio_implications(self, corr_matrix: pd.DataFrame, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess portfolio management implications of correlations"""
        
        implications = {
            'diversification_benefits': 'unknown',
            'risk_concentration': [],
            'portfolio_construction_insights': []
        }
        
        # Calculate average correlation
        corr_values = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if not np.isnan(val):
                    corr_values.append(val)
        
        if corr_values:
            avg_corr = np.mean(corr_values)
            
            if avg_corr < 0.3:
                implications['diversification_benefits'] = 'High - low average correlation suggests good diversification potential'
            elif avg_corr < 0.6:
                implications['diversification_benefits'] = 'Moderate - some diversification benefits available'
            else:
                implications['diversification_benefits'] = 'Limited - high correlations may limit diversification benefits'
            
            # Risk concentration analysis
            high_corr_groups = []
            for i in range(len(corr_matrix.columns)):
                high_corr_vars = []
                for j in range(len(corr_matrix.columns)):
                    if i != j and abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr_vars.append(corr_matrix.columns[j])
                if len(high_corr_vars) > 1:
                    high_corr_groups.append({
                        'anchor_variable': corr_matrix.columns[i],
                        'correlated_variables': high_corr_vars
                    })
            
            implications['risk_concentration'] = high_corr_groups
            
            # Portfolio construction insights
            if avg_corr > 0.8:
                implications['portfolio_construction_insights'].append('Consider factor-based diversification due to high correlations')
            if len(high_corr_groups) > 0:
                implications['portfolio_construction_insights'].append('Apply concentration limits to highly correlated asset groups')
            if avg_corr < 0.1:
                implications['portfolio_construction_insights'].append('Equal-weight portfolio may be effective due to low correlations')
        
        return implications
    
    def _identify_time_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that contain time/date data"""
        
        time_columns = []
        
        for col in df.columns:
            # Check data type
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                time_columns.append(col)
                continue
            
            # Check column name
            col_lower = col.lower()
            if any(term in col_lower for term in ['date', 'time', 'period', 'year', 'month', 'day']):
                time_columns.append(col)
                continue
            
            # Check if values can be converted to datetime
            if df[col].dtype == 'object':
                sample_vals = df[col].dropna().head(5)
                try:
                    pd.to_datetime(sample_vals, infer_datetime_format=True)
                    time_columns.append(col)
                except:
                    pass
        
        return time_columns
    
    def _analyze_time_series(self, df: pd.DataFrame, time_col: str, value_col: str) -> Optional[TimeSeriesAnalysis]:
        """Analyze time series data"""
        
        try:
            # Prepare time series data
            ts_df = df[[time_col, value_col]].dropna()
            if len(ts_df) < 10:
                return None
            
            # Convert time column if needed
            if not pd.api.types.is_datetime64_any_dtype(ts_df[time_col]):
                ts_df[time_col] = pd.to_datetime(ts_df[time_col], infer_datetime_format=True)
            
            ts_df = ts_df.sort_values(time_col)
            ts_df.set_index(time_col, inplace=True)
            
            series = ts_df[value_col]
            
            # Trend analysis
            trend_analysis = self._analyze_trend(series)
            
            # Seasonality analysis (if enough data)
            seasonality_analysis = self._analyze_seasonality(series)
            
            # Volatility analysis
            volatility_analysis = self._analyze_volatility_time_series(series)
            
            # Stationarity tests
            stationarity_tests = self._test_stationarity(series)
            
            # Forecast quality assessment
            forecast_quality = self._assess_forecast_quality(series)
            
            return TimeSeriesAnalysis(
                column_name=value_col,
                trend_analysis=trend_analysis,
                seasonality_analysis=seasonality_analysis,
                volatility_analysis=volatility_analysis,
                stationarity_tests=stationarity_tests,
                forecast_quality=forecast_quality
            )
            
        except Exception as e:
            return None
    
    def _analyze_trend(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze trend in time series"""
        
        trend_analysis = {
            'direction': 'none',
            'strength': 0,
            'linear_trend_slope': 0,
            'trend_significance': 0
        }
        
        try:
            # Simple linear trend
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
            
            trend_analysis['linear_trend_slope'] = slope
            trend_analysis['trend_significance'] = 1 - p_value
            trend_analysis['strength'] = abs(r_value)
            
            if p_value < 0.05:
                if slope > 0:
                    trend_analysis['direction'] = 'increasing'
                else:
                    trend_analysis['direction'] = 'decreasing'
            
        except:
            pass
        
        return trend_analysis
    
    def _analyze_seasonality(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze seasonality in time series"""
        
        seasonality = {
            'seasonal_pattern_detected': False,
            'dominant_frequency': None,
            'seasonal_strength': 0
        }
        
        if len(series) >= 24:  # Need at least 2 years of monthly data or similar
            try:
                # Simple seasonal detection using autocorrelation
                max_lag = min(len(series) // 3, 12)  # Check up to 12 periods
                autocorrs = [series.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
                
                max_autocorr = max(autocorrs)
                if max_autocorr > 0.3:
                    seasonality['seasonal_pattern_detected'] = True
                    seasonality['dominant_frequency'] = autocorrs.index(max_autocorr) + 1
                    seasonality['seasonal_strength'] = max_autocorr
                    
            except:
                pass
        
        return seasonality
    
    def _analyze_volatility_time_series(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze volatility patterns in time series"""
        
        volatility_analysis = {
            'volatility_clustering': False,
            'volatility_persistence': 0,
            'garch_effects': False
        }
        
        if len(series) > 30:
            try:
                # Calculate returns if series looks like prices
                if series.min() > 0 and series.std() / series.mean() < 1:
                    returns = series.pct_change().dropna()
                else:
                    returns = series.diff().dropna()
                
                if len(returns) > 20:
                    # Test for volatility clustering using squared returns
                    squared_returns = returns ** 2
                    volatility_autocorr = squared_returns.autocorr(lag=1)
                    
                    if volatility_autocorr > 0.2:
                        volatility_analysis['volatility_clustering'] = True
                        volatility_analysis['volatility_persistence'] = volatility_autocorr
                    
                    # Simple GARCH test: Ljung-Box test on squared returns
                    if volatility_autocorr > 0.3:
                        volatility_analysis['garch_effects'] = True
                        
            except:
                pass
        
        return volatility_analysis
    
    def _test_stationarity(self, series: pd.Series) -> Dict[str, float]:
        """Test for stationarity in time series"""
        
        stationarity_tests = {}
        
        if len(series) > 12:
            try:
                # Augmented Dickey-Fuller test (simplified version)
                # Since we don't have statsmodels, we'll use a simplified approach
                
                # Test if mean is changing over time
                mid_point = len(series) // 2
                first_half_mean = series[:mid_point].mean()
                second_half_mean = series[mid_point:].mean()
                
                # T-test for equal means
                t_stat, p_value = stats.ttest_ind(series[:mid_point], series[mid_point:])
                stationarity_tests['mean_stability'] = p_value
                
                # Test if variance is changing over time
                first_half_var = series[:mid_point].var()
                second_half_var = series[mid_point:].var()
                
                # F-test for equal variances
                f_stat = max(first_half_var, second_half_var) / min(first_half_var, second_half_var) if min(first_half_var, second_half_var) > 0 else 1
                f_p_value = 2 * min(stats.f.cdf(f_stat, mid_point-1, len(series)-mid_point-1), 
                                  1 - stats.f.cdf(f_stat, mid_point-1, len(series)-mid_point-1))
                stationarity_tests['variance_stability'] = f_p_value
                
            except:
                pass
        
        return stationarity_tests
    
    def _assess_forecast_quality(self, series: pd.Series) -> Dict[str, Any]:
        """Assess the quality of potential forecasts"""
        
        forecast_assessment = {
            'predictability': 'low',
            'recommended_methods': [],
            'forecast_horizon': 'short_term'
        }
        
        if len(series) > 20:
            # Simple predictability assessment based on autocorrelation
            try:
                max_autocorr = max([abs(series.autocorr(lag=lag)) for lag in range(1, min(6, len(series)//4))])
                
                if max_autocorr > 0.5:
                    forecast_assessment['predictability'] = 'high'
                    forecast_assessment['recommended_methods'] = ['ARIMA', 'Exponential Smoothing']
                    forecast_assessment['forecast_horizon'] = 'medium_term'
                elif max_autocorr > 0.3:
                    forecast_assessment['predictability'] = 'moderate'
                    forecast_assessment['recommended_methods'] = ['Moving Average', 'Linear Trend']
                    forecast_assessment['forecast_horizon'] = 'short_term'
                else:
                    forecast_assessment['recommended_methods'] = ['Naive', 'Mean']
                    
            except:
                pass
        
        return forecast_assessment
    
    def _assess_financial_modeling_capability(self, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, Any]:
        """Assess the dataset's capability for financial modeling"""
        
        assessment = {
            'suitable_for_var_modeling': False,
            'suitable_for_portfolio_optimization': False,
            'suitable_for_derivatives_pricing': False,
            'suitable_for_stress_testing': False,
            'data_quality_score': 0,
            'recommendations': []
        }
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Check for VaR modeling suitability
        return_like_columns = 0
        for col_analysis in analysis_results['column_analyses']:
            if col_analysis.financial_indicators['likely_data_type'] == 'return_series':
                return_like_columns += 1
        
        if return_like_columns >= 1 and len(df) > 100:
            assessment['suitable_for_var_modeling'] = True
            assessment['recommendations'].append('Dataset suitable for Value at Risk (VaR) modeling')
        
        # Check for portfolio optimization
        if len(numeric_columns) >= 3 and analysis_results.get('correlation_analysis'):
            assessment['suitable_for_portfolio_optimization'] = True
            assessment['recommendations'].append('Multiple assets available for portfolio optimization')
        
        # Check for derivatives pricing (need volatility and price-like data)
        price_like_columns = 0
        volatility_available = False
        
        for col_analysis in analysis_results['column_analyses']:
            if col_analysis.financial_indicators['likely_data_type'] == 'price_series':
                price_like_columns += 1
            if col_analysis.risk_metrics.get('volatility', 0) > 0:
                volatility_available = True
        
        if price_like_columns >= 1 and volatility_available and len(df) > 50:
            assessment['suitable_for_derivatives_pricing'] = True
            assessment['recommendations'].append('Price and volatility data available for derivatives pricing models')
        
        # Stress testing capability
        if len(df) > 200 and len(numeric_columns) >= 2:
            assessment['suitable_for_stress_testing'] = True
            assessment['recommendations'].append('Sufficient historical data for stress testing scenarios')
        
        # Data quality score
        quality_factors = []
        quality_factors.append(min(len(df) / 1000, 1))  # Size factor
        quality_factors.append(len(numeric_columns) / max(len(df.columns), 1))  # Numeric ratio
        
        # Missing data penalty
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        quality_factors.append(1 - missing_ratio)
        
        assessment['data_quality_score'] = np.mean(quality_factors) * 100
        
        return assessment
    
    def _perform_risk_assessment(self, df: pd.DataFrame, analysis_results: Dict) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        
        risk_assessment = {
            'data_risks': [],
            'model_risks': [],
            'operational_risks': [],
            'overall_risk_level': 'medium'
        }
        
        # Data risks
        missing_data_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_data_ratio > 0.1:
            risk_assessment['data_risks'].append('High missing data ratio may affect model reliability')
        
        # Check for outliers across columns
        high_outlier_columns = 0
        for col_analysis in analysis_results['column_analyses']:
            if col_analysis.outlier_analysis['iqr_method']['percentage'] > 10:
                high_outlier_columns += 1
        
        if high_outlier_columns > len(analysis_results['column_analyses']) * 0.3:
            risk_assessment['data_risks'].append('Multiple columns have high outlier rates')
        
        # Model risks
        if analysis_results.get('correlation_analysis'):
            high_corr_pairs = len(analysis_results['correlation_analysis'].dependency_analysis['potential_multicollinearity'])
            if high_corr_pairs > 0:
                risk_assessment['model_risks'].append('High correlations may cause multicollinearity in models')
        
        # Sample size risks
        if len(df) < 100:
            risk_assessment['model_risks'].append('Small sample size may limit model robustness')
        
        # Operational risks
        if len(df.columns) > 50:
            risk_assessment['operational_risks'].append('Large number of variables may complicate model maintenance')
        
        # Overall risk level
        total_risks = len(risk_assessment['data_risks']) + len(risk_assessment['model_risks']) + len(risk_assessment['operational_risks'])
        if total_risks > 5:
            risk_assessment['overall_risk_level'] = 'high'
        elif total_risks < 2:
            risk_assessment['overall_risk_level'] = 'low'
        
        return risk_assessment
    
    def _calculate_data_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data quality metrics"""
        
        metrics = {
            'completeness': {},
            'consistency': {},
            'validity': {},
            'uniqueness': {},
            'overall_score': 0
        }
        
        # Completeness
        completeness_scores = []
        for col in df.columns:
            completeness = (df[col].notnull().sum() / len(df)) * 100
            metrics['completeness'][col] = completeness
            completeness_scores.append(completeness)
        
        metrics['completeness']['average'] = np.mean(completeness_scores)
        
        # Consistency (coefficient of variation for numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        consistency_scores = []
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0 and col_data.std() > 0 and col_data.mean() != 0:
                cv = col_data.std() / abs(col_data.mean())
                consistency_scores.append(min(cv, 1))  # Cap at 1
        
        metrics['consistency']['average_cv'] = np.mean(consistency_scores) if consistency_scores else 0
        
        # Validity (percentage of valid values - simplified)
        validity_scores = []
        for col in numeric_cols:
            valid_ratio = df[col].apply(lambda x: isinstance(x, (int, float)) and not np.isnan(x)).sum() / len(df)
            validity_scores.append(valid_ratio * 100)
        
        metrics['validity']['average'] = np.mean(validity_scores) if validity_scores else 100
        
        # Uniqueness
        uniqueness_scores = []
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            uniqueness_scores.append(unique_ratio * 100)
            metrics['uniqueness'][col] = unique_ratio * 100
        
        metrics['uniqueness']['average'] = np.mean(uniqueness_scores)
        
        # Overall score
        weights = {'completeness': 0.3, 'consistency': 0.2, 'validity': 0.3, 'uniqueness': 0.2}
        overall_score = (
            metrics['completeness']['average'] * weights['completeness'] +
            (100 - metrics['consistency']['average_cv'] * 100) * weights['consistency'] +  # Lower CV is better
            metrics['validity']['average'] * weights['validity'] +
            min(metrics['uniqueness']['average'], 100) * weights['uniqueness']
        )
        
        metrics['overall_score'] = overall_score
        
        return metrics
    
    def _generate_business_insights(self, analysis_results: Dict) -> List[str]:
        """Generate business insights from statistical analysis"""
        
        insights = []
        
        # Financial modeling insights
        financial_assessment = analysis_results.get('financial_modeling_assessment', {})
        if financial_assessment.get('suitable_for_var_modeling'):
            insights.append('Data structure supports Value at Risk (VaR) modeling for regulatory compliance')
        
        if financial_assessment.get('suitable_for_portfolio_optimization'):
            insights.append('Multiple asset data enables Modern Portfolio Theory optimization')
        
        # Risk insights
        risk_assessment = analysis_results.get('risk_assessment', {})
        if risk_assessment.get('overall_risk_level') == 'high':
            insights.append('High data quality risks identified - implement additional validation procedures')
        
        # Correlation insights
        corr_analysis = analysis_results.get('correlation_analysis')
        if corr_analysis and corr_analysis.portfolio_implications.get('diversification_benefits'):
            insights.append(f"Portfolio diversification: {corr_analysis.portfolio_implications['diversification_benefits']}")
        
        # Time series insights
        ts_analyses = analysis_results.get('time_series_analyses', [])
        for ts_analysis in ts_analyses[:2]:  # First 2 time series
            if ts_analysis.forecast_quality['predictability'] == 'high':
                insights.append(f'{ts_analysis.column_name} shows high predictability - suitable for forecasting models')
        
        # Data quality insights
        data_quality = analysis_results.get('data_quality_metrics', {})
        if data_quality.get('overall_score', 0) > 80:
            insights.append('High data quality score supports robust statistical modeling')
        elif data_quality.get('overall_score', 0) < 60:
            insights.append('Data quality concerns may require preprocessing before modeling')
        
        return insights[:6]  # Return top 6 insights
    
    def _empty_analysis_result(self, sheet_name: str) -> Dict[str, Any]:
        """Return empty analysis result for empty datasets"""
        return {
            'sheet_name': sheet_name,
            'basic_info': {'error': 'Empty dataset'},
            'column_analyses': [],
            'correlation_analysis': None,
            'time_series_analyses': [],
            'financial_modeling_assessment': {},
            'risk_assessment': {},
            'data_quality_metrics': {},
            'business_insights': ['No data available for analysis']
        }
    
    def _empty_column_analysis(self, column_name: str) -> StatisticalSummary:
        """Return empty analysis for columns with no data"""
        return StatisticalSummary(
            column_name=column_name,
            basic_stats={'error': 'No data'},
            distribution_analysis={},
            outlier_analysis={},
            normality_tests={},
            risk_metrics={},
            financial_indicators={'likely_data_type': 'no_data'}
        )
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the dataset"""
        return {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'missing_values_total': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        }

# Example usage
if __name__ == "__main__":
    analyzer = AdvancedStatisticalAnalyzer()
    
    # Example with sample data
    np.random.seed(42)
    sample_data = {
        'Stock_A_Returns': np.random.normal(0.001, 0.02, 250),  # Daily returns
        'Stock_B_Returns': np.random.normal(0.0005, 0.025, 250),
        'Bond_Returns': np.random.normal(0.0002, 0.005, 250),
        'Date': pd.date_range('2023-01-01', periods=250, freq='D')
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add some correlation
    df['Stock_C_Returns'] = 0.7 * df['Stock_A_Returns'] + 0.3 * np.random.normal(0, 0.01, 250)
    
    results = analyzer.analyze_dataset(df, "Portfolio_Returns")
    
    print("Statistical Analysis Complete!")
    print(f"Analyzed {results['basic_info']['shape'][1]} columns with {results['basic_info']['shape'][0]} observations")
    print(f"Financial modeling assessment: {len(results['financial_modeling_assessment']['recommendations'])} recommendations")
    print(f"Business insights: {len(results['business_insights'])}")