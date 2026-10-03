import pytest

from utils.reporting.format_episode_log import format_episode_log


@pytest.mark.parametrize(
    ("episode_index", "episode_return", "num_steps", "expected_line"),
    [
        (3, 22.0, 22, "episode=3 return=22.0 steps=22"),
        (0, 9.0, 9, "episode=0 return=9.0 steps=9"),
    ],
)
def test_format_episode_log_cases(
    episode_index: int, episode_return: float, num_steps: int, expected_line: str
) -> None:
    assert format_episode_log(episode_index, episode_return, num_steps) == expected_line


def test_format_episode_log_is_deterministic() -> None:
    assert format_episode_log(3, 22.0, 22) == format_episode_log(3, 22.0, 22)


def test_format_episode_log_does_not_mutate_inputs() -> None:
    episode_index, episode_return, num_steps = 3, 22.0, 22
    format_episode_log(episode_index, episode_return, num_steps)
    assert (episode_index, episode_return, num_steps) == (3, 22.0, 22)
