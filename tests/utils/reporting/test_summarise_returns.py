import copy

import pytest

from utils.reporting.summarise_returns import summarise_returns

EXPECTED_KEYS = {"mean", "std", "min", "max", "num_episodes"}


def test_summarise_returns_three_values() -> None:
    summary = summarise_returns([1.0, 2.0, 3.0])
    assert set(summary) == EXPECTED_KEYS
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["std"] == pytest.approx(0.8165, abs=1e-4)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(3.0)
    assert summary["num_episodes"] == 3


def test_summarise_returns_single_value_has_zero_std() -> None:
    summary = summarise_returns([5.0])
    assert set(summary) == EXPECTED_KEYS
    assert summary["mean"] == pytest.approx(5.0)
    assert summary["std"] == pytest.approx(0.0)
    assert summary["min"] == pytest.approx(5.0)
    assert summary["max"] == pytest.approx(5.0)
    assert summary["num_episodes"] == 1


def test_summarise_returns_empty_raises_value_error() -> None:
    with pytest.raises(ValueError):
        summarise_returns([])


def test_summarise_returns_is_deterministic() -> None:
    episode_returns = [1.0, 2.0, 3.0]
    assert summarise_returns(episode_returns) == summarise_returns(episode_returns)


def test_summarise_returns_does_not_mutate_input() -> None:
    episode_returns = [1.0, 2.0, 3.0]
    original_episode_returns = copy.deepcopy(episode_returns)
    summarise_returns(episode_returns)
    assert episode_returns == original_episode_returns
