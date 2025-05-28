import numpy as np
from binomial_option_pricing import binomial_option_pricing

# Standard parameters
S_std = 100.0
K_std = 100.0
T_std = 1.0
r_std = 0.05
sigma_std = 0.20
N_std_medium = 1000

# European Call Standard
print("European Call Standard:")
calculated = binomial_option_pricing(S_std, K_std, T_std, r_std, sigma_std, N_std_medium, 'call', 'european')
print(f"expected_price = {calculated}")

# European Put Standard
print("\nEuropean Put Standard:")
calculated = binomial_option_pricing(S_std, K_std, T_std, r_std, sigma_std, N_std_medium, 'put', 'european')
print(f"expected_price = {calculated}")

# American Call Standard
print("\nAmerican Call Standard:")
calculated = binomial_option_pricing(S_std, K_std, T_std, r_std, sigma_std, N_std_medium, 'call', 'american')
print(f"expected_price = {calculated}")

# American Put Standard
print("\nAmerican Put Standard:")
calculated = binomial_option_pricing(S_std, K_std, T_std, r_std, sigma_std, N_std_medium, 'put', 'american')
print(f"expected_price = {calculated}")

# European Call ITM
print("\nEuropean Call ITM:")
S = 110; K = 100; T = 0.5; r = 0.05; sigma = 0.20; N = 1000
calculated = binomial_option_pricing(S, K, T, r, sigma, N, 'call', 'european')
print(f"expected_price = {calculated}")

# European Put OTM
print("\nEuropean Put OTM:")
S = 110; K = 100; T = 0.5; r = 0.05; sigma = 0.20; N = 1000
calculated = binomial_option_pricing(S, K, T, r, sigma, N, 'put', 'european')
print(f"expected_price = {calculated}")