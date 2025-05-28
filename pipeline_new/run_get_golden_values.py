import subprocess

print("Running get_golden_values.py...")
try:
    result = subprocess.run(['python', 'get_golden_values.py'], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
except Exception as e:
    print(f"Error running script: {e}")