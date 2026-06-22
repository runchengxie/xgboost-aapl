"""Tests for cache parameterisation."""

from __future__ import annotations

from pathlib import Path

from xgboost_aapl.config import Settings


def test_cache_file_path_is_parameterised() -> None:
    """Different symbols produce different cache paths."""
    s1 = Settings(symbol="AAPL", start_date="20210101", end_date="20220101")
    s2 = Settings(symbol="MSFT", start_date="20210101", end_date="20220101")
    assert s1.cache_file != s2.cache_file


def test_cache_file_path_contains_date_range() -> None:
    s = Settings(symbol="AAPL", start_date="20210101", end_date="20220101")
    filename = s.cache_file.name
    assert "AAPL" in filename
    assert "20210101" in filename
    assert "20220101" in filename


def test_cache_file_path_mkdir() -> None:
    """cache_file should create the cache directory if it doesn't exist."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        s = Settings(
            symbol="TEST",
            start_date="20200101",
            end_date="20200110",
            cache_dir=Path(tmp),
        )
        assert s.cache_file.parent.exists()
