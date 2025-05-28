import unittest
import subprocess
import sys

print("Running the unit tests locally...")
try:
    result = subprocess.run([sys.executable, '-m', 'unittest', 'test_binomial_option_pricing.py'], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    if result.returncode == 0:
        print("✅ All tests passed successfully!")
    else:
        print(f"❌ Tests failed with return code {result.returncode}")
        sys.exit(1)
except Exception as e:
    print(f"Error running tests: {e}")
    sys.exit(1)

print("\nTests completed. The updated test file should now pass in the CI/CD pipeline.")