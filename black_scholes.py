# black_scholes.py

import math
from scipy.stats import norm

def black_scholes_option_price(S, K, T, r, sigma, option_type="call"):
    """
    Calculate option price using Black-Scholes model
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annual)
    sigma (float): Volatility of the underlying asset (annual)
    option_type (str): Type of option - "call" or "put"
    
    Returns:
    float: Option price
    """
    # Calculate d1 and d2 parameters
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    # Calculate option price based on type
    if option_type.lower() == "call":
        option_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        option_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    
    return option_price

def main():
    # Example values
    S = 100.0    # Stock price
    K = 100.0    # Strike price
    T = 1.0      # Time to expiration in years
    r = 0.05     # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    # Calculate call and put option prices
    call_price = black_scholes_option_price(S, K, T, r, sigma, "call")
    put_price = black_scholes_option_price(S, K, T, r, sigma, "put")
    
    print(f"Stock Price: ${S}")
    print(f"Strike Price: ${K}")
    print(f"Time to Expiration: {T} years")
    print(f"Risk-free Rate: {r*100}%")
    print(f"Volatility: {sigma*100}%")
    print(f"Call Option Price: ${call_price:.2f}")
    print(f"Put Option Price: ${put_price:.2f}")

if __name__ == "__main__":
    main()
