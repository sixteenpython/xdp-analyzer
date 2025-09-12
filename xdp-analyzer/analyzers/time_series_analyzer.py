"""
Advanced Time Series Analyzer
Comprehensive time series analysis for financial and economic data
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TimeSeriesComponents:
    """Decomposed time series components"""
    trend: pd.Series
    seasonal: Optional[pd.Series]
    residual: pd.Series
    seasonal_period: Optional[int]

@dataclass
class ForecastResult:
    """Time series forecast results"""
    method: str
    forecast_values: np.ndarray
    confidence_intervals: Dict[str, np.ndarray]
    forecast_index: pd.DatetimeIndex
    accuracy_metrics: Dict[str, float]

@dataclass
class VolatilityAnalysis:
    """Volatility analysis results"""
    realized_volatility: pd.Series
    volatility_clustering: bool
    garch_effects: bool
    volatility_regimes: List[Dict]
    risk_metrics: Dict[str, float]

@dataclass
class TimeSeriesAnalysisResult:
    """Complete time series analysis results"""
    series_name: str
    basic_properties: Dict[str, Any]
    stationarity_analysis: Dict[str, Any]
    decomposition: Optional[TimeSeriesComponents]
    volatility_analysis: Optional[VolatilityAnalysis]
    correlation_structure: Dict[str, Any]
    forecasting_assessment: Dict[str, Any]
    financial_metrics: Dict[str, Any]
    business_insights: List[str]

class AdvancedTimeSeriesAnalyzer:
    """
    Advanced time series analyzer with focus on financial applications
    """
    
    def __init__(self):
        self.min_observations = 20
        self.volatility_window = 20
        self.confidence_levels = [0.90, 0.95, 0.99]
        
    def analyze_time_series(self, data: pd.DataFrame, time_col: str, value_cols: List[str]) -> Dict[str, TimeSeriesAnalysisResult]:
        """
        Analyze multiple time series
        
        Args:
            data: DataFrame containing time series data
            time_col: Name of the time column
            value_cols: List of value column names to analyze
            
        Returns:
            Dictionary of analysis results for each time series
        """
        
        results = {}
        
        # Prepare time series data
        ts_data = self._prepare_time_series_data(data, time_col, value_cols)
        
        if ts_data is None:
            return {}
        
        # Analyze each time series
        for col in value_cols:
            if col in ts_data.columns:
                series = ts_data[col].dropna()
                if len(series) >= self.min_observations:
                    analysis = self._analyze_single_series(series, col)
                    results[col] = analysis
        
        return results
    
    def _prepare_time_series_data(self, data: pd.DataFrame, time_col: str, value_cols: List[str]) -> Optional[pd.DataFrame]:
        """Prepare and clean time series data"""
        
        try:
            # Create working copy
            ts_data = data[[time_col] + value_cols].copy()
            
            # Convert time column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(ts_data[time_col]):
                ts_data[time_col] = pd.to_datetime(ts_data[time_col], infer_datetime_format=True)
            
            # Set time column as index
            ts_data.set_index(time_col, inplace=True)
            ts_data.sort_index(inplace=True)
            
            # Remove duplicate timestamps
            ts_data = ts_data[~ts_data.index.duplicated(keep='first')]
            
            return ts_data
            
        except Exception as e:
            return None
    
    def _analyze_single_series(self, series: pd.Series, series_name: str) -> TimeSeriesAnalysisResult:
        """Analyze a single time series comprehensively"""
        
        # Basic properties
        basic_properties = self._analyze_basic_properties(series)
        
        # Stationarity analysis
        stationarity_analysis = self._test_stationarity(series)
        
        # Time series decomposition
        decomposition = self._decompose_series(series)
        
        # Volatility analysis (if appropriate)
        volatility_analysis = self._analyze_volatility(series, series_name)
        
        # Correlation structure (autocorrelations)
        correlation_structure = self._analyze_correlation_structure(series)
        
        # Forecasting assessment
        forecasting_assessment = self._assess_forecasting_capability(series)
        
        # Financial metrics
        financial_metrics = self._calculate_financial_metrics(series, series_name)
        
        # Business insights
        business_insights = self._generate_business_insights(
            series_name, basic_properties, stationarity_analysis, 
            volatility_analysis, financial_metrics
        )
        
        return TimeSeriesAnalysisResult(
            series_name=series_name,
            basic_properties=basic_properties,
            stationarity_analysis=stationarity_analysis,
            decomposition=decomposition,
            volatility_analysis=volatility_analysis,
            correlation_structure=correlation_structure,
            forecasting_assessment=forecasting_assessment,
            financial_metrics=financial_metrics,
            business_insights=business_insights
        )
    
    def _analyze_basic_properties(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze basic time series properties"""
        
        properties = {
            'length': len(series),
            'start_date': series.index[0],
            'end_date': series.index[-1],
            'frequency': self._infer_frequency(series),
            'missing_values': series.isnull().sum(),
            'missing_percentage': (series.isnull().sum() / len(series)) * 100,
            'data_type_detected': self._detect_data_type(series),
            'summary_statistics': {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'median': series.median(),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis()
            }
        }
        
        return properties
    
    def _infer_frequency(self, series: pd.Series) -> Dict[str, Any]:
        """Infer the frequency of the time series"""
        
        freq_info = {
            'inferred_freq': None,
            'average_interval': None,
            'irregular_intervals': False
        }
        
        try:
            # Calculate time differences
            time_diffs = pd.Series(series.index).diff().dropna()
            
            if len(time_diffs) > 0:
                mode_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                freq_info['average_interval'] = str(mode_diff)
                
                # Check if intervals are regular
                diff_std = time_diffs.std()
                diff_mean = time_diffs.mean()
                
                if diff_std / diff_mean < 0.1:  # Less than 10% variation
                    freq_info['irregular_intervals'] = False
                    
                    # Try to map to standard frequencies
                    avg_seconds = mode_diff.total_seconds()
                    if abs(avg_seconds - 86400) < 3600:  # ~1 day
                        freq_info['inferred_freq'] = 'Daily'
                    elif abs(avg_seconds - 604800) < 86400:  # ~1 week
                        freq_info['inferred_freq'] = 'Weekly'
                    elif abs(avg_seconds - 2629746) < 86400:  # ~1 month
                        freq_info['inferred_freq'] = 'Monthly'
                    elif abs(avg_seconds - 7889238) < 604800:  # ~3 months
                        freq_info['inferred_freq'] = 'Quarterly'
                    elif abs(avg_seconds - 31556952) < 2629746:  # ~1 year
                        freq_info['inferred_freq'] = 'Annual'
                    else:
                        freq_info['inferred_freq'] = f'{avg_seconds}s'
                else:
                    freq_info['irregular_intervals'] = True
                    
        except:
            pass
        
        return freq_info
    
    def _detect_data_type(self, series: pd.Series) -> str:
        """Detect the type of time series data"""
        
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return 'empty'
        
        # Check for price-like data (positive, trending)
        if series_clean.min() > 0:
            cv = series_clean.std() / series_clean.mean()
            if 0.05 < cv < 0.5:  # Moderate volatility
                return 'price_series'
        
        # Check for return-like data (small mean, higher volatility)
        if abs(series_clean.mean()) < series_clean.std():
            if -0.5 < series_clean.mean() < 0.5:
                return 'return_series'
        
        # Check for rate/percentage data
        if 0 <= series_clean.min() and series_clean.max() <= 1:
            return 'rate_or_percentage'
        
        # Check for count data
        if all(series_clean == series_clean.astype(int)) and series_clean.min() >= 0:
            return 'count_data'
        
        return 'general_numeric'
    
    def _test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Test for stationarity using multiple methods"""
        
        stationarity = {
            'likely_stationary': False,
            'tests': {},
            'differencing_needed': 0,
            'transformation_suggested': None
        }
        
        series_clean = series.dropna()
        
        if len(series_clean) < 20:
            return stationarity
        
        # Visual stationarity checks
        stationarity['tests'].update(self._visual_stationarity_tests(series_clean))
        
        # Statistical stationarity tests (simplified versions)
        stationarity['tests'].update(self._statistical_stationarity_tests(series_clean))
        
        # Determine if differencing is needed
        non_stationary_indicators = 0
        
        if stationarity['tests'].get('mean_stability', 1) < 0.05:
            non_stationary_indicators += 1
        if stationarity['tests'].get('variance_stability', 1) < 0.05:
            non_stationary_indicators += 1
        if stationarity['tests'].get('trend_test', 1) < 0.05:
            non_stationary_indicators += 1
        
        if non_stationary_indicators >= 2:
            stationarity['differencing_needed'] = 1
            
            # Test if first difference is stationary
            diff_series = series_clean.diff().dropna()
            if len(diff_series) > 10:
                diff_tests = self._visual_stationarity_tests(diff_series)
                if diff_tests.get('mean_stability', 0) > 0.1:
                    stationarity['likely_stationary'] = True
                    stationarity['transformation_suggested'] = 'first_difference'
        else:
            stationarity['likely_stationary'] = True
            stationarity['transformation_suggested'] = 'none'
        
        return stationarity
    
    def _visual_stationarity_tests(self, series: pd.Series) -> Dict[str, float]:
        """Perform visual stationarity tests"""
        
        tests = {}
        
        try:
            # Split series in half and compare means and variances
            mid_point = len(series) // 2
            first_half = series[:mid_point]
            second_half = series[mid_point:]
            
            # Test for constant mean
            t_stat, p_value = stats.ttest_ind(first_half, second_half)
            tests['mean_stability'] = p_value
            
            # Test for constant variance
            f_stat = max(first_half.var(), second_half.var()) / max(min(first_half.var(), second_half.var()), 1e-10)
            # Approximate p-value for F-test
            if f_stat > 1:
                tests['variance_stability'] = 2 * (1 - stats.f.cdf(f_stat, len(first_half)-1, len(second_half)-1))
            else:
                tests['variance_stability'] = 1.0
            
            # Simple trend test
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
            tests['trend_test'] = p_value
            
        except:
            tests = {'mean_stability': 1.0, 'variance_stability': 1.0, 'trend_test': 1.0}
        
        return tests
    
    def _statistical_stationarity_tests(self, series: pd.Series) -> Dict[str, float]:
        """Perform statistical tests for stationarity"""
        
        tests = {}
        
        # Unit root test (simplified version)
        try:
            # Lag-1 regression: y_t = alpha + beta * y_{t-1} + error
            y = series[1:].values
            y_lag = series[:-1].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(y_lag, y)
            
            # If slope is significantly less than 1, series might be stationary
            # This is a very simplified version of ADF test
            if slope < 0.95 and p_value < 0.05:
                tests['unit_root_test'] = 0.05  # Reject unit root
            else:
                tests['unit_root_test'] = 0.95  # Fail to reject unit root
                
        except:
            tests['unit_root_test'] = 0.5
        
        return tests
    
    def _decompose_series(self, series: pd.Series) -> Optional[TimeSeriesComponents]:
        """Decompose time series into trend, seasonal, and residual components"""
        
        if len(series) < 24:  # Need at least 2 cycles for meaningful decomposition
            return None
        
        try:
            # Simple moving average decomposition
            
            # Detect seasonality period
            seasonal_period = self._detect_seasonal_period(series)
            
            if seasonal_period and seasonal_period > 1 and len(series) > 2 * seasonal_period:
                # Calculate trend using moving average
                trend = series.rolling(window=seasonal_period, center=True).mean()
                
                # Calculate seasonal component
                detrended = series - trend
                seasonal_means = detrended.groupby(detrended.index % seasonal_period).mean()
                
                # Create seasonal series
                seasonal = pd.Series(index=series.index, dtype=float)
                for i in range(len(series)):
                    seasonal.iloc[i] = seasonal_means.iloc[i % seasonal_period]
                
                # Calculate residual
                residual = series - trend - seasonal
                
                return TimeSeriesComponents(
                    trend=trend,
                    seasonal=seasonal,
                    residual=residual,
                    seasonal_period=seasonal_period
                )
            else:
                # No seasonality detected, just trend and residual
                window_size = max(5, len(series) // 10)
                trend = series.rolling(window=window_size, center=True).mean()
                residual = series - trend
                
                return TimeSeriesComponents(
                    trend=trend,
                    seasonal=None,
                    residual=residual,
                    seasonal_period=None
                )
                
        except:
            return None
    
    def _detect_seasonal_period(self, series: pd.Series) -> Optional[int]:
        """Detect seasonal period using autocorrelation"""
        
        if len(series) < 24:
            return None
        
        try:
            # Test common seasonal periods
            max_period = min(len(series) // 3, 365)  # Don't test beyond 1/3 of data length
            
            periods_to_test = []
            
            # Add common business periods based on data length
            if max_period >= 12:
                periods_to_test.extend([12])  # Monthly data
            if max_period >= 4:
                periods_to_test.extend([4])   # Quarterly data
            if max_period >= 7:
                periods_to_test.extend([7])   # Weekly pattern in daily data
            if max_period >= 252:
                periods_to_test.extend([252]) # Business days in a year
            
            # Add other periods to test
            for p in [5, 10, 20, 30, 60, 90, 120, 180, 365]:
                if p <= max_period:
                    periods_to_test.append(p)
            
            best_period = None
            best_autocorr = 0
            
            for period in set(periods_to_test):
                if period < len(series):
                    try:
                        autocorr = self._calculate_autocorrelation(series, period)
                        if autocorr > best_autocorr and autocorr > 0.3:
                            best_autocorr = autocorr
                            best_period = period
                    except:
                        continue
            
            return best_period
            
        except:
            return None
    
    def _calculate_autocorrelation(self, series: pd.Series, lag: int) -> float:
        """Calculate autocorrelation at specified lag"""
        
        try:
            if hasattr(series, 'autocorr'):
                return abs(series.autocorr(lag=lag))
            else:
                # Manual calculation
                n = len(series)
                if lag >= n:
                    return 0
                
                y1 = series[:-lag]
                y2 = series[lag:]
                
                correlation = np.corrcoef(y1, y2)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0
                
        except:
            return 0
    
    def _analyze_volatility(self, series: pd.Series, series_name: str) -> Optional[VolatilityAnalysis]:
        """Analyze volatility patterns in the time series"""
        
        series_clean = series.dropna()
        
        if len(series_clean) < 30:
            return None
        
        # Determine if this looks like returns data
        data_type = self._detect_data_type(series_clean)
        
        if data_type == 'return_series':
            returns = series_clean
        elif data_type == 'price_series':
            # Calculate returns from prices
            returns = series_clean.pct_change().dropna()
        else:
            # Calculate simple differences
            returns = series_clean.diff().dropna()
        
        if len(returns) < 20:
            return None
        
        # Calculate realized volatility
        realized_vol = returns.rolling(window=self.volatility_window).std()
        
        # Test for volatility clustering
        volatility_clustering = self._test_volatility_clustering(returns)
        
        # Test for GARCH effects
        garch_effects = self._test_garch_effects(returns)
        
        # Identify volatility regimes
        volatility_regimes = self._identify_volatility_regimes(realized_vol)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_volatility_risk_metrics(returns, realized_vol)
        
        return VolatilityAnalysis(
            realized_volatility=realized_vol,
            volatility_clustering=volatility_clustering,
            garch_effects=garch_effects,
            volatility_regimes=volatility_regimes,
            risk_metrics=risk_metrics
        )
    
    def _test_volatility_clustering(self, returns: pd.Series) -> bool:
        """Test for volatility clustering"""
        
        try:
            # Test autocorrelation in squared returns
            squared_returns = returns ** 2
            
            # Calculate autocorrelation at lag 1
            autocorr_lag1 = self._calculate_autocorrelation(squared_returns, 1)
            
            # Also test absolute returns
            abs_returns = returns.abs()
            abs_autocorr_lag1 = self._calculate_autocorrelation(abs_returns, 1)
            
            # Volatility clustering if either squared or absolute returns show persistence
            return autocorr_lag1 > 0.1 or abs_autocorr_lag1 > 0.1
            
        except:
            return False
    
    def _test_garch_effects(self, returns: pd.Series) -> bool:
        """Test for GARCH effects (simplified)"""
        
        try:
            # Ljung-Box test on squared returns (simplified version)
            squared_returns = returns ** 2
            
            # Test multiple lags for autocorrelation
            autocorrs = []
            max_lags = min(10, len(returns) // 4)
            
            for lag in range(1, max_lags + 1):
                autocorr = self._calculate_autocorrelation(squared_returns, lag)
                autocorrs.append(autocorr)
            
            # If multiple lags show significant autocorrelation, likely GARCH effects
            significant_lags = sum(1 for ac in autocorrs if ac > 0.15)
            
            return significant_lags >= 3
            
        except:
            return False
    
    def _identify_volatility_regimes(self, realized_vol: pd.Series) -> List[Dict]:
        """Identify volatility regimes using simple threshold method"""
        
        regimes = []
        
        try:
            vol_clean = realized_vol.dropna()
            
            if len(vol_clean) < 20:
                return regimes
            
            # Define regimes based on percentiles
            low_threshold = vol_clean.quantile(0.33)
            high_threshold = vol_clean.quantile(0.67)
            
            # Identify regime periods
            current_regime = None
            regime_start = None
            
            for date, vol in vol_clean.items():
                if vol <= low_threshold:
                    regime = 'low_volatility'
                elif vol >= high_threshold:
                    regime = 'high_volatility'
                else:
                    regime = 'medium_volatility'
                
                if regime != current_regime:
                    if current_regime is not None:
                        regimes.append({
                            'regime': current_regime,
                            'start': regime_start,
                            'end': date,
                            'duration': (date - regime_start).days,
                            'avg_volatility': vol_clean[regime_start:date].mean()
                        })
                    current_regime = regime
                    regime_start = date
            
            # Add final regime
            if current_regime is not None:
                regimes.append({
                    'regime': current_regime,
                    'start': regime_start,
                    'end': vol_clean.index[-1],
                    'duration': (vol_clean.index[-1] - regime_start).days,
                    'avg_volatility': vol_clean[regime_start:].mean()
                })
                
        except:
            pass
        
        return regimes
    
    def _calculate_volatility_risk_metrics(self, returns: pd.Series, realized_vol: pd.Series) -> Dict[str, float]:
        """Calculate volatility-based risk metrics"""
        
        metrics = {}
        
        try:
            returns_clean = returns.dropna()
            vol_clean = realized_vol.dropna()
            
            # Volatility of volatility
            if len(vol_clean) > 1:
                metrics['volatility_of_volatility'] = vol_clean.std()
            
            # Volatility skewness
            metrics['volatility_skewness'] = vol_clean.skew()
            
            # Maximum volatility
            metrics['max_volatility'] = vol_clean.max()
            
            # Volatility percentiles
            for p in [0.95, 0.99]:
                metrics[f'volatility_percentile_{int(p*100)}'] = vol_clean.quantile(p)
            
            # Volatility-adjusted returns (Sharpe-like ratio)
            if vol_clean.mean() > 0:
                metrics['volatility_adjusted_return'] = returns_clean.mean() / vol_clean.mean()
            
            # Volatility clustering intensity
            vol_autocorr = self._calculate_autocorrelation(vol_clean, 1)
            metrics['volatility_clustering_intensity'] = vol_autocorr
            
        except:
            pass
        
        return metrics
    
    def _analyze_correlation_structure(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze autocorrelation structure of the time series"""
        
        structure = {
            'autocorrelations': {},
            'partial_autocorrelations': {},  # Simplified version
            'ljung_box_test': {},
            'optimal_ar_order': 0,
            'optimal_ma_order': 0
        }
        
        series_clean = series.dropna()
        
        if len(series_clean) < 20:
            return structure
        
        # Calculate autocorrelations
        max_lags = min(20, len(series_clean) // 4)
        
        for lag in range(1, max_lags + 1):
            autocorr = self._calculate_autocorrelation(series_clean, lag)
            structure['autocorrelations'][lag] = autocorr
        
        # Simplified partial autocorrelations (just use autocorr as approximation)
        structure['partial_autocorrelations'] = structure['autocorrelations'].copy()
        
        # Simple model order selection based on autocorrelation decay
        significant_lags = [lag for lag, autocorr in structure['autocorrelations'].items() if autocorr > 0.1]
        
        structure['optimal_ar_order'] = len(significant_lags) if len(significant_lags) <= 3 else 3
        structure['optimal_ma_order'] = min(2, structure['optimal_ar_order'])
        
        # Ljung-Box test (simplified)
        try:
            # Test if autocorrelations are jointly significant
            autocorr_values = list(structure['autocorrelations'].values())
            if autocorr_values:
                # Simple chi-square approximation
                test_stat = len(series_clean) * sum(ac**2 for ac in autocorr_values)
                degrees_freedom = len(autocorr_values)
                p_value = 1 - stats.chi2.cdf(test_stat, degrees_freedom)
                structure['ljung_box_test'] = {
                    'statistic': test_stat,
                    'p_value': p_value,
                    'significant_autocorrelation': p_value < 0.05
                }
        except:
            pass
        
        return structure
    
    def _assess_forecasting_capability(self, series: pd.Series) -> Dict[str, Any]:
        """Assess the forecasting capability of the time series"""
        
        assessment = {
            'predictability': 'low',
            'recommended_models': [],
            'forecast_horizon': 'short',
            'seasonal_forecasting': False,
            'trend_forecasting': False,
            'volatility_forecasting': False
        }
        
        series_clean = series.dropna()
        
        if len(series_clean) < 20:
            return assessment
        
        # Assess predictability based on autocorrelation
        max_autocorr = 0
        if len(series_clean) > 5:
            for lag in range(1, min(6, len(series_clean)//4)):
                autocorr = abs(self._calculate_autocorrelation(series_clean, lag))
                max_autocorr = max(max_autocorr, autocorr)
        
        if max_autocorr > 0.5:
            assessment['predictability'] = 'high'
            assessment['forecast_horizon'] = 'medium'
        elif max_autocorr > 0.3:
            assessment['predictability'] = 'moderate'
            assessment['forecast_horizon'] = 'short'
        
        # Recommend models based on characteristics
        if max_autocorr > 0.3:
            assessment['recommended_models'].append('ARIMA')
        
        # Check for trend
        x = np.arange(len(series_clean))
        slope, _, r_value, p_value, _ = stats.linregress(x, series_clean.values)
        
        if p_value < 0.05 and abs(r_value) > 0.3:
            assessment['trend_forecasting'] = True
            assessment['recommended_models'].append('Linear Trend')
        
        # Check for seasonality
        seasonal_period = self._detect_seasonal_period(series_clean)
        if seasonal_period:
            assessment['seasonal_forecasting'] = True
            assessment['recommended_models'].append('Seasonal Decomposition')
        
        # Check for volatility patterns (for financial series)
        data_type = self._detect_data_type(series_clean)
        if data_type in ['return_series', 'price_series']:
            returns = series_clean.pct_change().dropna() if data_type == 'price_series' else series_clean
            if len(returns) > 20 and self._test_volatility_clustering(returns):
                assessment['volatility_forecasting'] = True
                assessment['recommended_models'].append('GARCH')
        
        # Default models
        if not assessment['recommended_models']:
            assessment['recommended_models'] = ['Simple Moving Average', 'Exponential Smoothing']
        
        return assessment
    
    def _calculate_financial_metrics(self, series: pd.Series, series_name: str) -> Dict[str, Any]:
        """Calculate financial-specific metrics for time series"""
        
        metrics = {
            'returns_analysis': {},
            'risk_metrics': {},
            'performance_metrics': {},
            'regime_analysis': {}
        }
        
        series_clean = series.dropna()
        data_type = self._detect_data_type(series_clean)
        
        # Calculate returns based on data type
        if data_type == 'price_series':
            returns = series_clean.pct_change().dropna()
            prices = series_clean
        elif data_type == 'return_series':
            returns = series_clean
            prices = (1 + returns).cumprod()
        else:
            returns = series_clean.diff().dropna()
            prices = series_clean
        
        if len(returns) > 10:
            # Returns analysis
            metrics['returns_analysis'] = {
                'annualized_return': returns.mean() * 252 if len(returns) > 252 else returns.mean() * len(returns),
                'annualized_volatility': returns.std() * np.sqrt(252) if len(returns) > 252 else returns.std() * np.sqrt(len(returns)),
                'skewness': returns.skew(),
                'excess_kurtosis': returns.kurtosis(),
                'positive_returns_ratio': (returns > 0).sum() / len(returns)
            }
            
            # Risk metrics
            if len(returns) > 20:
                var_95 = returns.quantile(0.05)
                var_99 = returns.quantile(0.01)
                
                metrics['risk_metrics'] = {
                    'value_at_risk_95': var_95,
                    'value_at_risk_99': var_99,
                    'expected_shortfall_95': returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95,
                    'maximum_drawdown': self._calculate_maximum_drawdown(prices),
                    'downside_deviation': returns[returns < 0].std() if (returns < 0).any() else 0
                }
            
            # Performance metrics
            if metrics['returns_analysis']['annualized_volatility'] > 0:
                metrics['performance_metrics'] = {
                    'sharpe_ratio': metrics['returns_analysis']['annualized_return'] / metrics['returns_analysis']['annualized_volatility'],
                    'sortino_ratio': metrics['returns_analysis']['annualized_return'] / metrics['risk_metrics'].get('downside_deviation', 1) if metrics['risk_metrics'].get('downside_deviation', 0) > 0 else 0,
                    'calmar_ratio': metrics['returns_analysis']['annualized_return'] / abs(metrics['risk_metrics'].get('maximum_drawdown', 1)) if metrics['risk_metrics'].get('maximum_drawdown', 0) != 0 else 0
                }
            
            # Regime analysis
            if len(returns) > 50:
                regimes = self._analyze_return_regimes(returns)
                metrics['regime_analysis'] = regimes
        
        return metrics
    
    def _calculate_maximum_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        
        try:
            if prices.min() <= 0:
                # Handle negative prices by using cumulative sum
                cumulative = prices.cumsum()
            else:
                # Use cumulative product for price series
                cumulative = prices / prices.iloc[0]
            
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            
            return drawdowns.min()
            
        except:
            return 0.0
    
    def _analyze_return_regimes(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze return regimes (bull/bear markets, etc.)"""
        
        regime_analysis = {
            'bull_market_periods': [],
            'bear_market_periods': [],
            'average_regime_duration': 0,
            'regime_transition_frequency': 0
        }
        
        try:
            # Simple regime identification based on rolling returns
            window = min(20, len(returns) // 5)
            rolling_returns = returns.rolling(window=window).sum()
            
            # Define bull/bear based on rolling returns
            bull_threshold = rolling_returns.quantile(0.6)
            bear_threshold = rolling_returns.quantile(0.4)
            
            current_regime = None
            regime_start = None
            regime_changes = 0
            
            for date, roll_ret in rolling_returns.dropna().items():
                if roll_ret >= bull_threshold:
                    regime = 'bull'
                elif roll_ret <= bear_threshold:
                    regime = 'bear'
                else:
                    regime = 'neutral'
                
                if regime != current_regime and current_regime is not None:
                    regime_changes += 1
                    
                    # Record previous regime
                    if current_regime == 'bull':
                        regime_analysis['bull_market_periods'].append({
                            'start': regime_start,
                            'end': date,
                            'duration': (date - regime_start).days,
                            'cumulative_return': returns[regime_start:date].sum()
                        })
                    elif current_regime == 'bear':
                        regime_analysis['bear_market_periods'].append({
                            'start': regime_start,
                            'end': date,
                            'duration': (date - regime_start).days,
                            'cumulative_return': returns[regime_start:date].sum()
                        })
                
                current_regime = regime
                if regime_start is None:
                    regime_start = date
            
            # Calculate average regime duration
            all_durations = []
            for bull_period in regime_analysis['bull_market_periods']:
                all_durations.append(bull_period['duration'])
            for bear_period in regime_analysis['bear_market_periods']:
                all_durations.append(bear_period['duration'])
            
            if all_durations:
                regime_analysis['average_regime_duration'] = np.mean(all_durations)
            
            # Regime transition frequency (changes per year)
            total_days = (returns.index[-1] - returns.index[0]).days
            regime_analysis['regime_transition_frequency'] = (regime_changes / max(total_days, 1)) * 365
            
        except:
            pass
        
        return regime_analysis
    
    def _generate_business_insights(self, series_name: str, basic_properties: Dict, 
                                  stationarity_analysis: Dict, volatility_analysis: Optional[VolatilityAnalysis],
                                  financial_metrics: Dict) -> List[str]:
        """Generate business insights from time series analysis"""
        
        insights = []
        
        # Data type insights
        data_type = basic_properties.get('data_type_detected', 'unknown')
        if data_type == 'return_series':
            insights.append(f'{series_name} exhibits return-like characteristics suitable for risk modeling')
        elif data_type == 'price_series':
            insights.append(f'{series_name} shows price-like behavior suitable for trend analysis and forecasting')
        
        # Stationarity insights
        if stationarity_analysis.get('likely_stationary', False):
            insights.append(f'{series_name} appears stationary - suitable for ARIMA modeling and statistical analysis')
        else:
            diff_needed = stationarity_analysis.get('differencing_needed', 0)
            if diff_needed > 0:
                insights.append(f'{series_name} requires {diff_needed} order of differencing for stationarity')
        
        # Volatility insights
        if volatility_analysis:
            if volatility_analysis.volatility_clustering:
                insights.append(f'{series_name} exhibits volatility clustering - consider GARCH modeling for risk management')
            
            if volatility_analysis.garch_effects:
                insights.append(f'{series_name} shows GARCH effects - volatility forecasting is feasible')
            
            regime_count = len(volatility_analysis.volatility_regimes)
            if regime_count > 2:
                insights.append(f'{series_name} shows {regime_count} distinct volatility regimes')
        
        # Financial insights
        if financial_metrics.get('performance_metrics'):
            sharpe_ratio = financial_metrics['performance_metrics'].get('sharpe_ratio', 0)
            if sharpe_ratio > 1:
                insights.append(f'{series_name} shows strong risk-adjusted returns (Sharpe ratio: {sharpe_ratio:.2f})')
            elif sharpe_ratio < 0:
                insights.append(f'{series_name} shows negative risk-adjusted returns - risk management needed')
        
        # Risk insights
        if financial_metrics.get('risk_metrics'):
            max_dd = financial_metrics['risk_metrics'].get('maximum_drawdown', 0)
            if max_dd < -0.2:
                insights.append(f'{series_name} experienced significant drawdown ({max_dd:.1%}) - high risk asset')
        
        # Regime insights
        regime_analysis = financial_metrics.get('regime_analysis', {})
        if regime_analysis:
            transition_freq = regime_analysis.get('regime_transition_frequency', 0)
            if transition_freq > 2:
                insights.append(f'{series_name} shows frequent regime changes - consider regime-switching models')
        
        return insights[:8]  # Return top 8 insights

# Example usage
if __name__ == "__main__":
    analyzer = AdvancedTimeSeriesAnalyzer()
    
    # Create sample financial time series data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Simulate stock price with volatility clustering
    returns = []
    vol = 0.02
    
    for i in range(500):
        # GARCH-like volatility
        vol = 0.8 * vol + 0.2 * 0.02 + 0.1 * abs(returns[-1] if returns else 0)
        ret = np.random.normal(0.0005, vol)
        returns.append(ret)
    
    # Create price series
    prices = 100 * np.cumprod(1 + np.array(returns))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Date': dates,
        'Stock_Price': prices,
        'Stock_Returns': returns
    })
    
    # Analyze time series
    results = analyzer.analyze_time_series(data, 'Date', ['Stock_Price', 'Stock_Returns'])
    
    print("Time Series Analysis Complete!")
    for series_name, result in results.items():
        print(f"\n{series_name}:")
        print(f"  Data Type: {result.basic_properties['data_type_detected']}")
        print(f"  Predictability: {result.forecasting_assessment['predictability']}")
        print(f"  Business Insights: {len(result.business_insights)}")
        if result.volatility_analysis:
            print(f"  Volatility Clustering: {result.volatility_analysis.volatility_clustering}")
            print(f"  GARCH Effects: {result.volatility_analysis.garch_effects}")