# get_golden_values.py
import numpy as np
from binomial_option_pricing import binomial_option_pricing

print("--- Generating Golden Values for Binomial Option Pricing Tests ---")
print("Copy these values into your test_binomial_option_pricing.py script.")
print("------------------------------------------------------------------")

# Standard parameters (matching test_binomial_option_pricing.py)
S_std = 100.0
K_std = 100.0
T_std = 1.0
r_std = 0.05
sigma_std = 0.20
N_std_medium = 1000 # Use the N from the test script

print("\n--- Standard Parameters (S=100, K=100, T=1, r=0.05, sigma=0.20, N=1000) ---")

calc_euro_call_std = binomial_option_pricing(S_std, K_std, T_std, r_std, sigma_std, N_std_medium, option_type='call', exercise_type='european')
print(f"European Call (Standard): {calc_euro_call_std:.15f}") # Print with high precision

calc_euro_put_std = binomial_option_pricing(S_std, K_std, T_std, r_std, sigma_std, N_std_medium, option_type='put', exercise_type='european')
print(f"European Put (Standard):  {calc_euro_put_std:.15f}")

calc_amer_call_std = binomial_option_pricing(S_std, K_std, T_std, r_std, sigma_std, N_std_medium, option_type='call', exercise_type='american')
print(f"American Call (Standard): {calc_amer_call_std:.15f}")

calc_amer_put_std = binomial_option_pricing(S_std, K_std, T_std, r_std, sigma_std, N_std_medium, option_type='put', exercise_type='american')
print(f"American Put (Standard):  {calc_amer_put_std:.15f}")


# Test Case: In-The-Money Call
S_itm = 110; K_itm = 100; T_itm = 0.5; r_itm = 0.05; sigma_itm = 0.20; N_itm = 1000
print(f"\n--- ITM Call (S=110, K=100, T=0.5, r=0.05, sigma=0.20, N=1000) ---")
calc_euro_call_itm = binomial_option_pricing(S_itm, K_itm, T_itm, r_itm, sigma_itm, N_itm, 'call', 'european')
print(f"European Call (ITM):      {calc_euro_call_itm:.15f}")

# Test Case: Out-The-Money Put
S_otm = 110; K_otm = 100; T_otm = 0.5; r_otm = 0.05; sigma_otm = 0.20; N_otm = 1000
print(f"\n--- OTM Put (S=110, K=100, T=0.5, r=0.05, sigma=0.20, N=1000) ---")
calc_euro_put_otm = binomial_option_pricing(S_otm, K_otm, T_otm, r_otm, sigma_otm, N_otm, 'put', 'european')
print(f"European Put (OTM):       {calc_euro_put_otm:.15f}")

print("------------------------------------------------------------------")
print("Golden values generated. Update your test_binomial_option_pricing.py with these values.")