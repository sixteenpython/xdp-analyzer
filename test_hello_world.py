# test_hello_world.py
import pytest
from hello_world import greet

def test_greet_default():
    assert greet() == "Hello, World!"

def test_greet_name():
    assert greet("GitLab CI/CD") == "Hello, GitLab CI/CD!"