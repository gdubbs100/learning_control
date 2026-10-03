import copy

from utils.experiment.has_converged import has_converged


def test_has_converged_constant_returns() -> None:
    assert has_converged([0.0] * 6, window=3, tolerance=0.1) is True


def test_has_converged_large_change_between_windows_is_false() -> None:
    assert has_converged([0.0, 0.0, 0.0, 10.0, 10.0, 10.0], window=3, tolerance=0.1) is False


def test_has_converged_too_few_returns_is_false() -> None:
    assert has_converged([1.0, 2.0], window=3, tolerance=0.1) is False


def test_has_converged_small_change_within_tolerance() -> None:
    assert has_converged([0.0, 0.0, 0.0, 0.05, 0.05, 0.05], window=3, tolerance=0.1) is True


def test_has_converged_difference_exactly_tolerance_is_true() -> None:
    assert has_converged([0.0, 0.0, 0.0, 0.5, 0.5, 0.5], window=3, tolerance=0.5) is True


def test_has_converged_only_last_two_windows_matter() -> None:
    assert has_converged([100.0, -50.0, 1.0, 1.0, 1.0, 1.0], window=2, tolerance=0.0) is True


def test_has_converged_is_deterministic() -> None:
    episode_returns = [0.0, 1.0, 2.0, 3.0]
    assert has_converged(episode_returns, 2, 0.1) == has_converged(episode_returns, 2, 0.1)


def test_has_converged_does_not_mutate_input() -> None:
    episode_returns = [0.0, 1.0, 2.0, 3.0]
    original_episode_returns = copy.deepcopy(episode_returns)
    has_converged(episode_returns, 2, 0.1)
    assert episode_returns == original_episode_returns
