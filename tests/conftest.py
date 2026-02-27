"""
Shared pytest fixtures for the Fake News Credibility Analyzer test suite.
"""

import os
import sys
import pytest

# Ensure src/ is importable
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)


@pytest.fixture
def sample_fake_text():
    """A news-style text that should lean towards 'Fake'."""
    return (
        "BREAKING: Scientists discover shocking truth that will change "
        "everything you know about the world. Anonymous sources claim "
        "the government has been hiding this from us for decades."
    )


@pytest.fixture
def sample_true_text():
    """A news-style text that should lean towards 'True'."""
    return (
        "The Federal Reserve announced a quarter-point interest rate hike "
        "on Wednesday, citing continued strength in the labor market and "
        "persistent inflationary pressures across the economy."
    )


@pytest.fixture
def short_text():
    """Text that is too short for the API's min_length validation."""
    return "Short"


@pytest.fixture
def empty_text():
    return ""
