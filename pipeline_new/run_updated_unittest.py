import unittest
import subprocess

print("Running the updated unit tests...")
try:
    result = subprocess.run(['python', '-m', 'unittest', 'updated_test_binomial_option_pricing.py'], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
except Exception as e:
    print(f"Error running tests: {e}")