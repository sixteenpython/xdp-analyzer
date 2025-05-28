# binomial_option_pricing.py
"""
This module provides functionality for pricing options using the Cox-Ross-Rubinstein (CRR) 
Binomial Option Pricing Model. It supports both European and American options,
and handles both call and put option types.

The implementation follows a modular approach with separate functions for:
- Input validation
- Tree parameter calculation
- Asset price tree construction
- Option value initialization
- Backward induction for option pricing
"""

import numpy as np
from typing import Literal, Union, Optional, Tuple, Dict
from numpy.typing import NDArray

# Type aliases for better readability
OptionType = Literal['call', 'put']
ExerciseType = Literal['european', 'american']


def validate_inputs(S: float, K: float, T: float, r: float, sigma: float, 
                  N: int, option_type: OptionType, exercise_type: ExerciseType) -> None:
    """
    Validates the inputs for the binomial option pricing model.
    
    This function checks all input parameters to ensure they are valid for the
    binomial option pricing calculation. It validates numerical constraints
    and allowed string values.
    
    Parameters
    ----------
    S : float
        Current underlying asset price
    K : float
        Option strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility of the underlying asset (annualized)
    N : int
        Number of time steps (binomial periods)
    option_type : OptionType
        Type of option, must be 'call' or 'put'
    exercise_type : ExerciseType
        Exercise style, must be 'european' or 'american'
    
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If any of the inputs are invalid:
        - N <= 0: Number of steps must be positive
        - T <= 0: Time to maturity must be positive
        - sigma <= 0: Volatility must be positive
        - Invalid option_type: Must be 'call' or 'put'
        - Invalid exercise_type: Must be 'european' or 'american'
    """
    if N <= 0:
        raise ValueError("Number of steps (N) must be a positive integer.")
    if T <= 0:
        raise ValueError("Time to maturity (T) must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility (sigma) must be positive.")
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'.")
    if exercise_type not in ('european', 'american'):
        raise ValueError("exercise_type must be 'european' or 'american'.")


def calculate_tree_parameters(T: float, r: float, sigma: float, N: int) -> Tuple[float, float, float, float]:
    """
    Calculates the parameters needed for the binomial tree.
    
    This function computes the essential parameters for the Cox-Ross-Rubinstein (CRR)
    binomial model: the time step size, up and down movement factors, and the
    risk-neutral probability.
    
    Parameters
    ----------
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility of the underlying asset (annualized)
    N : int
        Number of time steps (binomial periods)
    
    Returns
    -------
    Tuple[float, float, float, float]
        A tuple containing:
        - dt: Time step duration
        - u: Up factor (the factor by which the asset price increases in an up state)
        - d: Down factor (the factor by which the asset price decreases in a down state)
        - p: Risk-neutral probability of an up movement
    
    Notes
    -----
    The CRR model uses the following formulas:
    - u = exp(sigma * sqrt(dt))
    - d = 1/u
    - p = (exp(r * dt) - d) / (u - d)
    """
    # Calculate time step duration
    dt = T / N

    # Calculate up (u) and down (d) factors using Cox-Ross-Rubinstein model
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    # Calculate risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)
    
    return dt, u, d, p


def build_asset_price_tree(S: float, u: float, d: float, N: int) -> NDArray:
    """
    Builds the asset price tree for the binomial model.
    
    This function constructs a 2D array representing the possible asset price paths
    in the binomial lattice model. Each node in the tree represents a possible
    asset price at a given time step.
    
    Parameters
    ----------
    S : float
        Current underlying asset price
    u : float
        Up factor (the factor by which the asset price increases in an up state)
    d : float
        Down factor (the factor by which the asset price decreases in a down state)
    N : int
        Number of time steps (binomial periods)
    
    Returns
    -------
    NDArray
        A 2D array of shape (N+1, N+1) representing the asset price tree, where:
        - asset_prices[j, i] is the price after j up-moves and (i-j) down-moves
        - The first dimension represents the state (number of up moves)
        - The second dimension represents the time step
    
    Notes
    -----
    The asset price at each node (j, i) is calculated as:
    S * (u^j) * (d^(i-j)), where:
    - j is the number of up moves
    - i-j is the number of down moves
    - i is the current time step
    """
    # Initialize asset price tree (for clarity in structure)
    asset_prices = np.zeros((N + 1, N + 1))
    
    # Build the tree
    for i in range(N + 1):  # iterate through time steps
        for j in range(i + 1): # iterate through possible states at each time step
            # At each node, the asset price is S * u^j * d^(i-j)
            # where j is the number of up moves and (i-j) is the number of down moves
            asset_prices[j, i] = S * (u**j) * (d**(i - j))
            
    return asset_prices


def initialize_option_values_at_maturity(option_values: NDArray, asset_prices: NDArray, 
                                        K: float, N: int, option_type: OptionType) -> NDArray:
    """
    Initializes option values at maturity (the last column of the option values tree).
    
    This function calculates the payoff of the option at maturity for each possible
    final asset price, and sets these values in the last column of the option values tree.
    
    Parameters
    ----------
    option_values : NDArray
        The option values tree, a 2D array of shape (N+1, N+1)
    asset_prices : NDArray
        The asset price tree, a 2D array of shape (N+1, N+1)
    K : float
        Option strike price
    N : int
        Number of time steps (binomial periods)
    option_type : OptionType
        Type of option, must be 'call' or 'put'
    
    Returns
    -------
    NDArray
        The option values tree with initialized values at maturity
    
    Raises
    ------
    ValueError
        If option_type is not 'call' or 'put'
    
    Notes
    -----
    - For a call option, the payoff at maturity is max(0, S - K)
    - For a put option, the payoff at maturity is max(0, K - S)
    where S is the asset price at maturity and K is the strike price.
    """
    # Make a copy of the input array to avoid modifying the original
    result = option_values.copy()
    
    # Calculate option values at maturity (last column of the tree)
    for j in range(N + 1):
        if option_type == 'call':
            result[j, N] = max(0, asset_prices[j, N] - K)
        elif option_type == 'put':
            result[j, N] = max(0, K - asset_prices[j, N])
        else:
            raise ValueError("option_type must be 'call' or 'put'.")
            
    return result


def calculate_intrinsic_value(asset_price: float, K: float, option_type: OptionType) -> float:
    """
    Calculates the intrinsic value of an option at a given asset price.
    
    The intrinsic value represents the payoff that would be realized if the option
    were exercised immediately at the current asset price.
    
    Parameters
    ----------
    asset_price : float
        The current price of the underlying asset
    K : float
        Option strike price
    option_type : OptionType
        Type of option, must be 'call' or 'put'
    
    Returns
    -------
    float
        The intrinsic value of the option (max of 0 and the payoff)
    
    Raises
    ------
    ValueError
        If option_type is not 'call' or 'put'
    
    Notes
    -----
    - For a call option, the intrinsic value is max(0, asset_price - K)
    - For a put option, the intrinsic value is max(0, K - asset_price)
    
    The intrinsic value is never negative because an option would not be exercised
    if doing so would result in a loss.
    """
    if option_type == 'call':
        return max(0, asset_price - K)
    elif option_type == 'put':
        return max(0, K - asset_price)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")


def backward_induction(option_values: NDArray, asset_prices: NDArray, K: float, dt: float, 
                       r: float, p: float, N: int, option_type: OptionType, 
                       exercise_type: ExerciseType) -> NDArray:
    """
    Performs backward induction to calculate option values at each node in the tree.
    
    This function implements the backward induction algorithm that works recursively
    from the option values at maturity to calculate the option values at earlier nodes
    in the tree, all the way back to the root node at time 0.
    
    Parameters
    ----------
    option_values : NDArray
        The option values tree with values at maturity already set
    asset_prices : NDArray
        The asset price tree
    K : float
        Option strike price
    dt : float
        Time step duration
    r : float
        Risk-free interest rate (annualized)
    p : float
        Risk-neutral probability
    N : int
        Number of time steps (binomial periods)
    option_type : OptionType
        Type of option, must be 'call' or 'put'
    exercise_type : ExerciseType
        Exercise style, must be 'european' or 'american'
    
    Returns
    -------
    NDArray
        The option values tree with all values calculated
    
    Raises
    ------
    ValueError
        If exercise_type is not 'european' or 'american'
    
    Notes
    -----
    For each node in the tree, working backwards from maturity:
    
    1. The continuation value is calculated as the discounted expected future value:
       continuation_value = exp(-r * dt) * (p * option_values[j+1, i+1] + (1-p) * option_values[j, i+1])
    
    2. For European options, the node value is equal to the continuation value.
    
    3. For American options, the node value is the maximum of:
       - The continuation value
       - The intrinsic value (the payoff if exercised immediately)
    """
    # Make a copy to avoid modifying the original
    result = option_values.copy()
    
    # Work backwards through the tree to find option values at earlier nodes
    for i in range(N - 1, -1, -1):  # Iterate from second-to-last step back to time 0
        for j in range(i + 1):
            # Calculate option value by discounting the expected future value (continuation value)
            continuation_value = np.exp(-r * dt) * (p * result[j + 1, i + 1] + (1 - p) * result[j, i + 1])

            if exercise_type == 'american':
                # For American options, compare continuation value with intrinsic value
                intrinsic_value = calculate_intrinsic_value(asset_prices[j, i], K, option_type)
                result[j, i] = max(continuation_value, intrinsic_value)
            elif exercise_type == 'european':
                # For European options, the value is just the continuation value
                result[j, i] = continuation_value
            else:
                raise ValueError("exercise_type must be 'european' or 'american'.")
                
    return result


def binomial_option_pricing(S: float, K: float, T: float, r: float, sigma: float, 
                           N: int, option_type: OptionType = 'call', 
                           exercise_type: ExerciseType = 'european') -> float:
    """
    Calculates the price of a European or American option using the
    Cox-Ross-Rubinstein (CRR) Binomial Option Pricing Model.
    
    This is the main function of the module that orchestrates the entire option
    pricing process. It validates inputs, sets up the binomial tree, calculates
    option values at maturity, and then uses backward induction to find the
    current price of the option.

    Parameters
    ----------
    S : float
        Current underlying asset price
    K : float
        Option strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility of the underlying asset (annualized)
    N : int
        Number of time steps (binomial periods)
    option_type : OptionType, optional
        Type of option, must be 'call' or 'put', default is 'call'
    exercise_type : ExerciseType, optional
        Exercise style, must be 'european' or 'american', default is 'european'

    Returns
    -------
    float
        The calculated option price
    
    Raises
    ------
    ValueError
        If any of the inputs are invalid:
        - N <= 0: Number of steps must be positive
        - T <= 0: Time to maturity must be positive
        - sigma <= 0: Volatility must be positive
        - Invalid option_type: Must be 'call' or 'put'
        - Invalid exercise_type: Must be 'european' or 'american'
    
    Examples
    --------
    >>> binomial_option_pricing(100, 100, 1, 0.05, 0.2, 1000)
    10.448584103764572
    
    >>> binomial_option_pricing(100, 100, 1, 0.05, 0.2, 1000, 'put')
    5.57152655383368
    
    >>> binomial_option_pricing(100, 100, 1, 0.05, 0.2, 1000, 'put', 'american')
    6.089595282977953
    """
    # Validate inputs
    validate_inputs(S, K, T, r, sigma, N, option_type, exercise_type)
    
    # Calculate tree parameters
    dt, u, d, p = calculate_tree_parameters(T, r, sigma, N)
    
    # Build the asset price tree
    asset_prices = build_asset_price_tree(S, u, d, N)

    # Initialize option value tree (working backwards from maturity)
    option_values = np.zeros((N + 1, N + 1))
    
    # Calculate option values at maturity
    option_values = initialize_option_values_at_maturity(option_values, asset_prices, K, N, option_type)
    
    # Work backwards through the tree using backward induction
    option_values = backward_induction(option_values, asset_prices, K, dt, r, p, N, option_type, exercise_type)

    # The option price at time 0 is the value at the first node (top-left)
    return option_values[0, 0]