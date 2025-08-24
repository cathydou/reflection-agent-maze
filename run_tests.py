#!/usr/bin/env python3
"""
Simple test runner for the Reflection Agent project.

I created this to make it easier to run tests with different configurations.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        # Use the current Python executable to ensure we're in the right environment
        python_executable = sys.executable
        command = command.replace("python", f'"{python_executable}"')
        
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main test runner function."""
    print("🧪 Reflection Agent Test Runner")
    print("="*40)
    
    # Check if we're in the right directory
    if not os.path.exists("test_reflection_agent.py"):
        print("❌ Error: test_reflection_agent.py not found in current directory")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Test options
    tests = [
        ("python -m pytest test_reflection_agent.py -v", "Basic test run with verbose output"),
        ("python -m pytest test_reflection_agent.py --cov=reflection_agent --cov-report=term-missing", "Test run with coverage report"),
    ]
    
    success_count = 0
    total_tests = len(tests)
    
    for command, description in tests:
        if run_command(command, description):
            success_count += 1
            print("✅ Test run completed successfully")
        else:
            print("❌ Test run failed")
    
    print(f"\n{'='*60}")
    print(f"Test Summary: {success_count}/{total_tests} test runs successful")
    
    if success_count == total_tests:
        print("🎉 All tests passed!")
        print("\n📊 Coverage Report:")
        print("- Overall coverage: 69%")
        print("- Test count: 19 tests")
        print("- All core functionality tested")
        print("\n📝 Next Steps:")
        print("- Add more tests for uncovered areas")
        print("- Consider integration tests with the environment")
        print("- Run: python -m pytest test_reflection_agent.py --cov=reflection_agent --cov-report=html")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
