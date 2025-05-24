# test_Test.py
import pytest
import numpy as np
from datetime import datetime, timedelta
from Test import (
    BlackScholesOptionPricer,
    calculate_days_to_expiry,
    greet
)

# Test the greet function (keeping existing tests)
def test_greet_default():
    assert greet() == "Hello, World!"

def test_greet_name():
    assert greet("GitLab CI/CD") == "Hello, GitLab CI/CD!"

# Test the BlackScholesOptionPricer class
class TestBlackScholesOptionPricer:
    @pytest.fixture
    def pricer(self):
        # Create a standard pricer for testing
        return BlackScholesOptionPricer(
            spot_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            dividend_yield=0.01
        )
    
    def test_initialization(self, pricer):
        assert pricer.S == 100.0
        assert pricer.K == 100.0
        assert pricer.T == 1.0
        assert pricer.r == 0.05
        assert pricer.sigma == 0.2
        assert pricer.q == 0.01
        
    def test_d1_d2_calculation(self, pricer):
        # Test that d1 and d2 are calculated correctly
        expected_d1 = (np.log(100.0 / 100.0) + (0.05 - 0.01 + 0.5 * 0.2**2) * 1.0) / (0.2 * np.sqrt(1.0))
        expected_d2 = expected_d1 - 0.2 * np.sqrt(1.0)
        
        assert abs(pricer.d1 - expected_d1) < 1e-10
        assert abs(pricer.d2 - expected_d2) < 1e-10
    
    def test_call_price(self, pricer):
        # For at-the-money options with 1 year to expiry, call price should be around 10-11
        call_price = pricer.call_price()
        assert 9.0 < call_price < 12.0
    
    def test_put_price(self, pricer):
        # For at-the-money options with 1 year to expiry, put price should be around 5-6
        put_price = pricer.put_price()
        assert 4.0 < put_price < 7.0
    
    def test_put_call_parity(self, pricer):
        # Test put-call parity: C - P = S*e^(-q*T) - K*e^(-r*T)
        call_price = pricer.call_price()
        put_price = pricer.put_price()
        expected_diff = pricer.S * np.exp(-pricer.q * pricer.T) - pricer.K * np.exp(-pricer.r * pricer.T)
        
        assert abs((call_price - put_price) - expected_diff) < 1e-10
    
    def test_call_delta(self, pricer):
        # For at-the-money options, call delta should be around 0.5-0.6
        call_delta = pricer.call_delta()
        assert 0.5 < call_delta < 0.6
    
    def test_put_delta(self, pricer):
        # For at-the-money options, put delta should be around -0.5 to -0.4
        put_delta = pricer.put_delta()
        assert -0.5 < put_delta < -0.4
    
    def test_gamma(self, pricer):
        # Gamma should be positive
        gamma = pricer.gamma()
        assert gamma > 0
    
    def test_vega(self, pricer):
        # Vega should be positive
        vega = pricer.vega()
        assert vega > 0
    
    def test_call_theta(self, pricer):
        # Call theta should be negative for most cases
        call_theta = pricer.call_theta()
        assert call_theta < 0
    
    def test_put_theta(self, pricer):
        # Put theta can be positive or negative
        put_theta = pricer.put_theta()
        # Just check it's a reasonable value
        assert -0.1 < put_theta < 0.1
    
    def test_call_rho(self, pricer):
        # Call rho should be positive
        call_rho = pricer.call_rho()
        assert call_rho > 0
    
    def test_put_rho(self, pricer):
        # Put rho should be negative
        put_rho = pricer.put_rho()
        assert put_rho < 0
    
    def test_implied_volatility(self, pricer):
        # Calculate option price with known volatility
        original_price = pricer.call_price()
        
        # Now try to recover the volatility
        implied_vol = pricer.implied_volatility(original_price, 'call')
        
        # Should be close to the original volatility
        assert abs(implied_vol - 0.2) < 1e-4
    
    def test_parameter_changes(self):
        # Test that changing parameters affects prices correctly
        base_pricer = BlackScholesOptionPricer(100, 100, 1, 0.05, 0.2)
        base_call = base_pricer.call_price()
        
        # Higher spot price should increase call price
        high_spot_pricer = BlackScholesOptionPricer(110, 100, 1, 0.05, 0.2)
        assert high_spot_pricer.call_price() > base_call
        
        # Higher volatility should increase call price
        high_vol_pricer = BlackScholesOptionPricer(100, 100, 1, 0.05, 0.3)
        assert high_vol_pricer.call_price() > base_call
        
        # Higher interest rate should increase call price
        high_rate_pricer = BlackScholesOptionPricer(100, 100, 1, 0.06, 0.2)
        assert high_rate_pricer.call_price() > base_call
        
        # Higher dividend yield should decrease call price
        dividend_pricer = BlackScholesOptionPricer(100, 100, 1, 0.05, 0.2, 0.02)
        assert dividend_pricer.call_price() < base_call

# Test the calculate_days_to_expiry function
def test_calculate_days_to_expiry():
    # Create a date 30 days in the future
    future_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Calculate days to expiry
    years_to_expiry = calculate_days_to_expiry(future_date)
    
    # Should be close to 30/365
    assert abs(years_to_expiry - 30/365) < 0.01
    
    # Test with a past date (should return minimum of 1 day)
    past_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    min_years = calculate_days_to_expiry(past_date)
    assert min_years == 1/365

# Test that the main function runs without errors
def test_main_runs(monkeypatch):
    from Test import main
    
    # Mock the input function to return empty strings (use defaults)
    monkeypatch.setattr('builtins.input', lambda _: '')
    
    # Mock plt.show to avoid displaying plots during tests
    import matplotlib.pyplot as plt
    original_show = plt.show
    plt.show = lambda: None
    
    try:
        # This should run without errors
        main()
    finally:
        # Restore plt.show
        plt.show = original_show
