"""
Unit tests for src/data_pipeline.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from data_pipeline import (
    remove_urls,
    remove_html,
    remove_special_chars,
    normalize_whitespace,
    preprocess_text,
)


class TestRemoveUrls:
    def test_removes_http_url(self):
        assert remove_urls("Visit http://example.com today") == "Visit  today"

    def test_removes_https_url(self):
        assert remove_urls("Go to https://example.com/page for info") == "Go to  for info"

    def test_removes_www_url(self):
        assert remove_urls("See www.example.com for details") == "See  for details"

    def test_no_url_unchanged(self):
        assert remove_urls("No links here") == "No links here"


class TestRemoveHtml:
    def test_removes_simple_tags(self):
        assert remove_html("<p>Hello</p>") == "Hello"

    def test_removes_nested_tags(self):
        assert remove_html("<div><b>Bold</b> text</div>") == "Bold text"

    def test_no_html_unchanged(self):
        assert remove_html("Just plain text") == "Just plain text"


class TestRemoveSpecialChars:
    def test_removes_punctuation(self):
        assert remove_special_chars("Hello, world!") == "Hello world"

    def test_removes_numbers(self):
        assert remove_special_chars("Test 123 here") == "Test  here"

    def test_keeps_letters_and_spaces(self):
        assert remove_special_chars("Hello World") == "Hello World"


class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("hello    world") == "hello world"

    def test_strips_leading_trailing(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_handles_tabs_newlines(self):
        assert normalize_whitespace("hello\t\nworld") == "hello world"


class TestPreprocessText:
    def test_full_pipeline(self):
        text = "Visit http://example.com <b>NOW</b> for amazing results!!!"
        result = preprocess_text(text)
        assert "http" not in result
        assert "<b>" not in result
        assert "!" not in result
        assert result == result.lower()

    def test_stopword_removal(self):
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))
        result = preprocess_text("this is a very good test", stop_words=stop_words)
        assert "this" not in result.split()
        assert "is" not in result.split()

    def test_lemmatization(self):
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        result = preprocess_text("the cats are running quickly", lemmatizer=lemmatizer)
        assert "cat" in result.split()

    def test_non_string_returns_empty(self):
        assert preprocess_text(None) == ""
        assert preprocess_text(123) == ""

    def test_empty_string_returns_empty(self):
        assert preprocess_text("") == ""
