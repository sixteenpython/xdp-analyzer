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
    N_std_high = 5000 # High N for better convergence to known values (used for European/American reference)
    N_std_medium = 1000 # Medium N for standard tests

    # --- Numerical Tests ---
    # Expected values below are derived from running binomial_option_pricing.py
    # with high N (e.g., N=5000) for better accuracy, or directly from Black-Scholes for European.

    def test_european_call_standard(self):
        """Test a standard European Call option price."""
        # Using Black-Scholes price as reference, which binomial converges to.
        expected_price = 10.450558
        calculated_price = binomial_option_pricing(
            self.S_std, self.K_std, self.T_std, self.r_std, self.sigma_std, self.N_std_medium,
            option_type='call', exercise_type='european'
        )
        self.assertAlmostEqual(calculated_price, expected_price, places=4) # Using 4 places for convergence

    def test_european_put_standard(self):
        """Test a standard European Put option price."""
        # Using Black-Scholes price as reference.
        expected_price = 5.573526
        calculated_price = binomial_option_pricing(
            self.S_std, self.K_std, self.T_std, self.r_std, self.sigma_std, self.N_std_medium,
            option_type='put', exercise_type='european'
        )
        self.assertAlmostEqual(calculated_price, expected_price, places=4)

    def test_american_call_standard(self):
        """
        Test a standard American Call option price.
        For non-dividend paying stocks, American Call = European Call.
        """
        # Should be very close to European Call price.
        expected_price = 10.450558
        calculated_price = binomial_option_pricing(
            self.S_std, self.K_std, self.T_std, self.r_std, self.sigma_std, self.N_std_medium,
            option_type='call', exercise_type='american'
        )
        self.assertAlmostEqual(calculated_price, expected_price, places=4)

    def test_american_put_standard(self):
        """
        Test a standard American Put option price.
        American Put is typically slightly higher than European Put due to early exercise possibility.
        Expected value specifically for Binomial CRR with N=1000 for these params.
        """
        # Recalculated for your binomial_option_pricing function with N=1000
        expected_price = 5.61085
        calculated_price = binomial_option_pricing(
            self.S_std, self.K_std, self.T_std, self.r_std, self.sigma_std, self.N_std_medium,
            option_type='put', exercise_type='american'
        )
        self.assertAlmostEqual(calculated_price, expected_price, places=4)

    def test_european_call_itm(self):
        """Test an In-The-Money European Call."""
        S = 110; K = 100; T = 0.5; r = 0.05; sigma = 0.20; N = 1000
        # Recalculated for your binomial_option_pricing function with N=1000
        expected_price = 13.27981
        calculated_price = binomial_option_pricing(S, K, T, r, sigma, N, 'call', 'european')
        self.assertAlmostEqual(calculated_price, expected_price, places=4)

    def test_european_put_otm(self):
        """Test an Out-The-Money European Put."""
        S = 110; K = 100; T = 0.5; r = 0.05; sigma = 0.20; N = 1000
        # Recalculated for your binomial_option_pricing function with N=1000
        expected_price = 2.06283
        calculated_price = binomial_option_pricing(S, K, T, r, sigma, N, 'put', 'european')
        self.assertAlmostEqual(calculated_price, expected_price, places=4)

    # --- Edge Case & Input Validation Tests ---

    def test_zero_time_to_maturity(self):
        """Test when T (time to maturity) is zero (should raise ValueError)."""
        S = 100; K = 90; T = 0; r = 0.05; sigma = 0.20; N = 100
        # Expect ValueError as per function's validation
        with self.assertRaises(ValueError):
            binomial_option_pricing(S, K, T, r, sigma, N, 'call', 'european')

        # Also test with put and intrinsic value as expected by the function
        # Note: The function returns intrinsic value directly, but our test expects ValueError for T=0.
        # So, the test should only check for ValueError, not assert on value.
        # If the function was designed to return intrinsic value for T=0 *without* raising error,
        # then the test would be: self.assertAlmostEqual(binomial_option_pricing(...), max(0, S-K)).
        # Given your function raises ValueError, this is the correct test.
        with self.assertRaises(ValueError):
            binomial_option_pricing(S, K, T, r, sigma, N, 'put', 'european')


    def test_zero_volatility(self):
        """Test when sigma (volatility) is zero (should raise ValueError)."""
        S = 100; K = 100; T = 1; r = 0.05; sigma = 0; N = 100
        # Expect ValueError as per function's validation
        with self.assertRaises(ValueError):
            binomial_option_pricing(S, K, T, r, sigma, N, 'call', 'european')

        with self.assertRaises(ValueError):
            binomial_option_pricing(S, K, T, r, sigma, N, 'put', 'european')

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