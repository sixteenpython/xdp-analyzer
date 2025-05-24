import unittest
import math
from black_scholes import black_scholes_option_price

class TestBlackScholes(unittest.TestCase):
    def test_call_option_price(self):
        # Test case with known values
        S = 100.0    # Stock price
        K = 100.0    # Strike price
        T = 1.0      # Time to expiration in years
        r = 0.05     # Risk-free rate (5%)
        sigma = 0.2  # Volatility (20%)
        
        # Expected value (calculated with these parameters)
        expected_call_price = 10.45
        
        # Calculate with our function
        calculated_price = black_scholes_option_price(S, K, T, r, sigma, "call")
        
        # Assert that the calculated price is close to the expected price
        self.assertAlmostEqual(calculated_price, expected_call_price, delta=0.01)
    
    def test_put_option_price(self):
        # Test case with known values
        S = 100.0    # Stock price
        K = 100.0    # Strike price
        T = 1.0      # Time to expiration in years
        r = 0.05     # Risk-free rate (5%)
        sigma = 0.2  # Volatility (20%)
        
        # Expected value (calculated with these parameters)
        expected_put_price = 5.57
        
        # Calculate with our function
        calculated_price = black_scholes_option_price(S, K, T, r, sigma, "put")
        
        # Assert that the calculated price is close to the expected price
        self.assertAlmostEqual(calculated_price, expected_put_price, delta=0.01)
    
    def test_put_call_parity(self):
        # Test put-call parity: C - P = S - K*e^(-rT)
        S = 100.0    # Stock price
        K = 100.0    # Strike price
        T = 1.0      # Time to expiration in years
        r = 0.05     # Risk-free rate (5%)
        sigma = 0.2  # Volatility (20%)
        
        call_price = black_scholes_option_price(S, K, T, r, sigma, "call")
        put_price = black_scholes_option_price(S, K, T, r, sigma, "put")
        
        # Calculate the right side of the put-call parity equation
        right_side = S - K * math.exp(-r * T)
        
        # Assert that put-call parity holds
        self.assertAlmostEqual(call_price - put_price, right_side, delta=0.01)
    
    def test_invalid_option_type(self):
        # Test that an invalid option type raises a ValueError
        S = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        with self.assertRaises(ValueError):
            black_scholes_option_price(S, K, T, r, sigma, "invalid_type")

if __name__ == "__main__":
    unittest.main()
