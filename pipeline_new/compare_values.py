import numpy as np
from binomial_option_pricing import binomial_option_pricing

# Define the test parameters from test_binomial_option_pricing.py
S_std = 100.0
K_std = 100.0
T_std = 1.0
r_std = 0.05
sigma_std = 0.20
N_std_medium = 1000

# European Call Standard
print("European Call Standard:")
expected = 10.448584103764572
calculated = binomial_option_pricing(S_std, K_std, T_std, r_std, sigma_std, N_std_medium, 'call', 'european')
print(f"Expected: {expected}")
print(f"Calculated: {calculated}")
print(f"Difference: {abs(calculated - expected)}")
print()

# European Put Standard
print("European Put Standard:")
expected = 5.57152655383368
calculated = binomial_option_pricing(S_std, K_std, T_std, r_std, sigma_std, N_std_medium, 'put', 'european')
print(f"Expected: {expected}")
print(f"Calculated: {calculated}")
print(f"Difference: {abs(calculated - expected)}")
print()

# American Call Standard
print("American Call Standard:")
expected = 10.448584103764572
calculated = binomial_option_pricing(S_std, K_std, T_std, r_std, sigma_std, N_std_medium, 'call', 'american')
print(f"Expected: {expected}")
print(f"Calculated: {calculated}")
print(f"Difference: {abs(calculated - expected)}")
print()

# American Put Standard
print("American Put Standard:")
expected = 6.089595282977953
calculated = binomial_option_pricing(S_std, K_std, T_std, r_std, sigma_std, N_std_medium, 'put', 'american')
print(f"Expected: {expected}")
print(f"Calculated: {calculated}")
print(f"Difference: {abs(calculated - expected)}")
print()

# European Call ITM
print("European Call ITM:")
S = 110; K = 100; T = 0.5; r = 0.05; sigma = 0.20; N = 1000
expected = 14.076105196563155
calculated = binomial_option_pricing(S, K, T, r, sigma, N, 'call', 'european')
print(f"Expected: {expected}")
print(f"Calculated: {calculated}")
print(f"Difference: {abs(calculated - expected)}")
print()

# European Put OTM
print("European Put OTM:")
S = 110; K = 100; T = 0.5; r = 0.05; sigma = 0.20; N = 1000
expected = 1.6070963993929976
calculated = binomial_option_pricing(S, K, T, r, sigma, N, 'put', 'european')
print(f"Expected: {expected}")
print(f"Calculated: {calculated}")
print(f"Difference: {abs(calculated - expected)}")
print()