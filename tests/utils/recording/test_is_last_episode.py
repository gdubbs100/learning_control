import pytest

from utils.recording.is_last_episode import is_last_episode


@pytest.mark.parametrize(
    ("episode_index", "num_episodes", "expected_result"),
    [
        (99, 100, True),
        (0, 100, False),
        (98, 100, False),
        (0, 1, True),
    ],
)
def test_is_last_episode_cases(episode_index: int, num_episodes: int, expected_result: bool) -> None:
    assert is_last_episode(episode_index, num_episodes) is expected_result


def test_is_last_episode_is_deterministic() -> None:
    assert is_last_episode(99, 100) == is_last_episode(99, 100)


def test_is_last_episode_does_not_mutate_inputs() -> None:
    episode_index, num_episodes = 99, 100
    is_last_episode(episode_index, num_episodes)
    assert (episode_index, num_episodes) == (99, 100)
