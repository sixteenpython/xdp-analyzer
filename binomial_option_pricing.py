# binomial_option_pricing.py
import numpy as np

def binomial_option_pricing(S, K, T, r, sigma, N, option_type='call', exercise_type='european'):
    """
    Calculates the price of a European or American option using the
    Cox-Ross-Rubinstein (CRR) Binomial Option Pricing Model.

    Parameters:
    S (float): Current underlying asset price
    K (float): Option strike price
    T (float): Time to maturity in years
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying asset (annualized)
    N (int): Number of time steps (binomial periods)
    option_type (str): 'call' or 'put'
    exercise_type (str): 'european' or 'american'

    Returns:
    float: The calculated option price
    """

    if N <= 0:
        raise ValueError("Number of steps (N) must be a positive integer.")
    if T <= 0:
        raise ValueError("Time to maturity (T) must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility (sigma) must be positive.")

    # Calculate time step duration
    dt = T / N

    # Calculate up (u) and down (d) factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    # Calculate risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize asset price tree (for clarity in structure)
    asset_prices = np.zeros((N + 1, N + 1))
    for i in range(N + 1):  # iterate through time steps
        for j in range(i + 1): # iterate through possible states at each time step
            asset_prices[j, i] = S * (u**j) * (d**(i - j))

    # Initialize option value tree (working backwards from maturity)
    option_values = np.zeros((N + 1, N + 1))

    # Calculate option values at maturity (last column of the tree)
    for j in range(N + 1):
        if option_type == 'call':
            option_values[j, N] = max(0, asset_prices[j, N] - K)
        elif option_type == 'put':
            option_values[j, N] = max(0, K - asset_prices[j, N])
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

    # Work backwards through the tree to find option values at earlier nodes
    for i in range(N - 1, -1, -1):  # Iterate from second-to-last step back to time 0
        for j in range(i + 1):
            # Calculate option value by discounting the expected future value (continuation value)
            continuation_value = np.exp(-r * dt) * (p * option_values[j + 1, i + 1] + (1 - p) * option_values[j, i + 1])

            if exercise_type == 'american':
                # For American options, compare continuation value with intrinsic value
                if option_type == 'call':
                    intrinsic_value = max(0, asset_prices[j, i] - K)
                else: # put option
                    intrinsic_value = max(0, K - asset_prices[j, i])
                option_values[j, i] = max(continuation_value, intrinsic_value)
            elif exercise_type == 'european':
                # For European options, the value is just the continuation value
                option_values[j, i] = continuation_value
            else:
                raise ValueError("exercise_type must be 'european' or 'american'.")

    # The option price at time 0 is the value at the first node (top-left)
    return option_values[0, 0]
    