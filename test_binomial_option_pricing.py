import unittest
import numpy as np
from binomial_option_pricing import binomial_option_pricing # Assuming your main script is named binomial_option_pricing.py

class TestBinomialOptionPricing(unittest.TestCase):

    # Standard parameters for common test cases
    S_std = 100.0   # Current stock price
    K_std = 100.0   # Strike price
    T_std = 1.0     # Time to maturity in years (1 year)
    r_std = 0.05    # Risk-free rate (5%)
    sigma_std = 0.20 # Volatility (20%)
    N_std = 1000    # Number of steps (higher N for better convergence to known values)

    # Note: Binomial model converges to Black-Scholes for European options as N -> infinity.
    # The expected values below are based on Black-Scholes for European options for high N,
    # and common references for American options. Minor deviations are expected due to the
    # discrete nature of the binomial model, especially for lower N.

    def test_european_call_standard(self):
        """Test a standard European Call option price."""
        # Expected value from Black-Scholes for these parameters (~10.450558)
        # Binomial with high N should approximate this.
        expected_price = 10.450558
        calculated_price = binomial_option_pricing(
            self.S_std, self.K_std, self.T_std, self.r_std, self.sigma_std, self.N_std,
            option_type='call', exercise_type='european'
        )
        self.assertAlmostEqual(calculated_price, expected_price, places=4) # Using 4 places for convergence

    def test_european_put_standard(self):
        """Test a standard European Put option price."""
        # Expected value from Black-Scholes for these parameters (~5.573526)
        expected_price = 5.573526
        calculated_price = binomial_option_pricing(
            self.S_std, self.K_std, self.T_std, self.r_std, self.sigma_std, self.N_std,
            option_type='put', exercise_type='european'
        )
        self.assertAlmostEqual(calculated_price, expected_price, places=4) # Using 4 places for convergence

    def test_american_call_standard(self):
        """
        Test a standard American Call option price.
        For non-dividend paying stocks, American Call = European Call.
        """
        expected_price = 10.450558 # Same as European Call
        calculated_price = binomial_option_pricing(
            self.S_std, self.K_std, self.T_std, self.r_std, self.sigma_std, self.N_std,
            option_type='call', exercise_type='american'
        )
        self.assertAlmostEqual(calculated_price, expected_price, places=4)

    def test_american_put_standard(self):
        """
        Test a standard American Put option price.
        American Put is typically slightly higher than European Put due to early exercise possibility.
        """
        # Expected value from common binomial calculators/references for N=1000 (~5.61)
        expected_price = 5.610
        calculated_price = binomial_option_pricing(
            self.S_std, self.K_std, self.T_std, self.r_std, self.sigma_std, self.N_std,
            option_type='put', exercise_type='american'
        )
        self.assertAlmostEqual(calculated_price, expected_price, places=3) # Adjust precision for American option

    def test_european_call_itm(self):
        """Test an In-The-Money European Call."""
        S = 110; K = 100; T = 0.5; r = 0.05; sigma = 0.20; N = 500
        # Expected value from BS for these params (~13.28)
        expected_price = 13.28
        calculated_price = binomial_option_pricing(S, K, T, r, sigma, N, 'call', 'european')
        self.assertAlmostEqual(calculated_price, expected_price, places=3)

    def test_european_put_otm(self):
        """Test an Out-The-Money European Put."""
        S = 110; K = 100; T = 0.5; r = 0.05; sigma = 0.20; N = 500
        # Expected value from BS for these params (~2.06)
        expected_price = 2.06
        calculated_price = binomial_option_pricing(S, K, T, r, sigma, N, 'put', 'european')
        self.assertAlmostEqual(calculated_price, expected_price, places=2)

    def test_zero_time_to_maturity(self):
        """Test when T (time to maturity) is zero."""
        S = 100; K = 90; T = 0; r = 0.05; sigma = 0.20; N = 100
        # For T=0, call price should be max(0, S-K)
        self.assertAlmostEqual(binomial_option_pricing(S, K, T, r, sigma, N, 'call', 'european'), 10.0)
        # For T=0, put price should be max(0, K-S)
        self.assertAlmostEqual(binomial_option_pricing(S, 90, T, r, sigma, N, 'put', 'european'), 0.0)
        self.assertAlmostEqual(binomial_option_pricing(S, 110, T, r, sigma, N, 'put', 'european'), 10.0)


    def test_zero_volatility(self):
        """Test when sigma (volatility) is zero."""
        S = 100; K = 100; T = 1; r = 0.05; sigma = 0; N = 100
        # With zero volatility, stock price moves deterministically to S * exp(r*T)
        # Call: max(0, S*exp(r*T) - K)
        expected_call = max(0, S * np.exp(r * T) - K)
        # Put: max(0, K - S*exp(r*T))
        expected_put = max(0, K - S * np.exp(r * T))

        self.assertAlmostEqual(binomial_option_pricing(S, K, T, r, sigma, N, 'call', 'european'), expected_call)
        self.assertAlmostEqual(binomial_option_pricing(S, K, T, r, sigma, N, 'put', 'european'), expected_put)

    def test_invalid_n(self):
        """Test for ValueError when N is non-positive."""
        with self.assertRaises(ValueError):
            binomial_option_pricing(100, 100, 1, 0.05, 0.20, 0)
        with self.assertRaises(ValueError):
            binomial_option_pricing(100, 100, 1, 0.05, 0.20, -10)

    def test_invalid_t(self):
        """Test for ValueError when T is non-positive."""
        with self.assertRaises(ValueError):
            binomial_option_pricing(100, 100, 0, 0.05, 0.20, 100)
        with self.assertRaises(ValueError):
            binomial_option_pricing(100, 100, -1, 0.05, 0.20, 100)

    def test_invalid_sigma(self):
        """Test for ValueError when sigma is non-positive."""
        with self.assertRaises(ValueError):
            binomial_option_pricing(100, 100, 1, 0.05, 0, 100)
        with self.assertRaises(ValueError):
            binomial_option_pricing(100, 100, 1, 0.05, -0.1, 100)

    def test_invalid_option_type(self):
        """Test for ValueError with invalid option_type."""
        with self.assertRaises(ValueError):
            binomial_option_pricing(100, 100, 1, 0.05, 0.20, 100, option_type='bond')

    def test_invalid_exercise_type(self):
        """Test for ValueError with invalid exercise_type."""
        with self.assertRaises(ValueError):
            binomial_option_pricing(100, 100, 1, 0.05, 0.20, 100, exercise_type='bermudian')

# This allows you to run the tests directly from the command line
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)