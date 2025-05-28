import unittest
import subprocess

print("Running unittest...")
try:
    result = subprocess.run(['python', '-m', 'unittest', 'test_binomial_option_pricing.py'], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    if result.returncode == 0:
        print("All tests passed!")
    else:
        print(f"Tests failed with return code {result.returncode}")
except Exception as e:
    print(f"Error running tests: {e}")