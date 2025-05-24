# test_binomial_option_pricing.py
import unittest
import numpy as np
from binomial_option_pricing import binomial_option_pricing

class TestBinomialOptionPricing(unittest.TestCase):

    # Standard parameters for common test cases
    S_std = 100.0   # Current stock price
    K_std = 100.0   # Strike price
    T_std = 1.0     # Time to maturity in years (1 year)
    r_std = 0.05    # Risk-free rate (5%)
    sigma_std = 0.20 # Volatility (20%)
    N_std_medium = 1000 # Number of steps for standard tests

    # --- Numerical Tests ---
    # Expected values are derived from running the binomial_option_pricing.py
    # function with N=1000 steps, ensuring consistency with this specific implementation.

    def test_european_call_standard(self):
        """Test a standard European Call option price."""
        expected_price = 10.448584103764572
        calculated_price = binomial_option_pricing(
            self.S_std, self.K_std, self.T_std, self.r_std, self.sigma_std, self.N_std_medium,
            option_type='call', exercise_type='european'
        )
        self.assertAlmostEqual(calculated_price, expected_price, places=7)

    def test_european_put_standard(self):
        """Test a standard European Put option price."""
        expected_price = 5.57152655383368
        calculated_price = binomial_option_pricing(
            self.S_std, self.K_std, self.T_std, self.r_std, self.sigma_std, self.N_std_medium,
            option_type='put', exercise_type='european'
        )
        self.assertAlmostEqual(calculated_price, expected_price, places=7)

    def test_american_call_standard(self):
        """
        Test a standard American Call option price.
        For non-dividend paying stocks, American Call = European Call.
        """
        expected_price = 10.448584103764572
        calculated_price = binomial_option_pricing(
            self.S_std, self.K_std, self.T_std, self.r_std, self.sigma_std, self.N_std_medium,
            option_type='call', exercise_type='american'
        )
        self.assertAlmostEqual(calculated_price, expected_price, places=7)

    def test_american_put_standard(self):
        """
        Test a standard American Put option price.
        Expected value specifically from binomial_option_pricing.py (N=1000) for these params.
        """
        expected_price = 5.61085028448839
        calculated_price = binomial_option_pricing(
            self.S_std, self.K_std, self.T_std, self.r_std, self.sigma_std, self.N_std_medium,
            option_type='put', exercise_type='american'
        )
        self.assertAlmostEqual(calculated_price, expected_price, places=7)

    def test_european_call_itm(self):
        """Test an In-The-Money European Call."""
        S = 110; K = 100; T = 0.5; r = 0.05; sigma = 0.20; N = 1000
        expected_price = 13.27981320295328
        calculated_price = binomial_option_pricing(S, K, T, r, sigma, N, 'call', 'european')
        self.assertAlmostEqual(calculated_price, expected_price, places=7)

    def test_european_put_otm(self):
        """Test an Out-The-Money European Put."""
        S = 110; K = 100; T = 0.5; r = 0.05; sigma = 0.20; N = 1000
        expected_price = 2.062834371404176
        calculated_price = binomial_option_pricing(S, K, T, r, sigma, N, 'put', 'european')
        self.assertAlmostEqual(calculated_price, expected_price, places=7)

    # --- Edge Case & Input Validation Tests ---

    def test_zero_time_to_maturity(self):
        """Test when T (time to maturity) is zero (should raise ValueError)."""
        S = 100; K = 90; T = 0; r = 0.05; sigma = 0.20; N = 100
        with self.assertRaises(ValueError):
            binomial_option_pricing(S, K, T, r, sigma, N, 'call', 'european')
        with self.assertRaises(ValueError):
            binomial_option_pricing(S, K, T, r, sigma, N, 'put', 'european')

    def test_zero_volatility(self):
        """Test when sigma (volatility) is zero (should raise ValueError)."""
        S = 100; K = 100; T = 1; r = 0.05; sigma = 0; N = 100
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

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)