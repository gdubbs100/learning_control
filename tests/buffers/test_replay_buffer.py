import inspect

import pytest

import buffers.replay_buffer as family_module
from buffers.replay_buffer import ReplayBuffer

BASE_CLASS = ReplayBuffer
# Arguments for the base class __init__ parameters, used to construct every class in the family.
BASE_INIT_ARGUMENTS = {"max_transitions": 5}
# Values for keyword-only parameters that child classes add, chosen by type annotation.
EXTRA_ARGUMENT_VALUES_BY_TYPE = {int: 0, float: 0.0, str: "", bool: False}


def concrete_family_classes() -> list[type]:
    """Every concrete class in the family file that inherits from the base class."""
    return [
        member
        for _, member in inspect.getmembers(family_module, inspect.isclass)
        if issubclass(member, BASE_CLASS)
        and member is not BASE_CLASS
        and member.__module__ == family_module.__name__
        and not inspect.isabstract(member)
    ]


def construction_arguments(cls: type) -> dict:
    """Base arguments plus a typed value for each required added keyword-only parameter."""
    arguments = dict(BASE_INIT_ARGUMENTS)
    for name, parameter in inspect.signature(cls.__init__).parameters.items():
        if name in arguments or name == "self":
            continue
        if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            continue
        if parameter.default is inspect.Parameter.empty:
            arguments[name] = EXTRA_ARGUMENT_VALUES_BY_TYPE[parameter.annotation]
    return arguments


def public_methods(cls: type) -> dict[str, object]:
    """Public methods defined on a class, including inherited ones."""
    return {
        name: member
        for name, member in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    }


def test_base_class_is_abstract() -> None:
    assert inspect.isabstract(BASE_CLASS)


@pytest.mark.parametrize("cls", concrete_family_classes(), ids=lambda cls: cls.__name__)
def test_family_class_follows_contract(cls: type) -> None:
    # Exactly one parent: the base class.
    assert cls.__bases__ == (BASE_CLASS,)

    # It instantiates and the base class attributes are set from the base arguments.
    instance = cls(**construction_arguments(cls))
    for name, value in BASE_INIT_ARGUMENTS.items():
        assert getattr(instance, name) == value

    # Every public method of the base class is present with a compatible signature.
    for name, base_method in public_methods(BASE_CLASS).items():
        child_method = getattr(cls, name, None)
        assert callable(child_method), f"{cls.__name__} is missing {name}"
        base_parameters = list(inspect.signature(base_method).parameters.values())
        child_parameters = list(inspect.signature(child_method).parameters.values())
        for base_parameter, child_parameter in zip(base_parameters, child_parameters):
            assert child_parameter.name == base_parameter.name
            assert child_parameter.kind == base_parameter.kind
        assert len(child_parameters) >= len(base_parameters)
        for extra_parameter in child_parameters[len(base_parameters):]:
            assert extra_parameter.default is not inspect.Parameter.empty, (
                f"{cls.__name__}.{name} adds required parameter {extra_parameter.name}"
            )

    # __init__ follows the extension rule.
    if "__init__" in vars(cls):
        base_init_parameters = inspect.signature(BASE_CLASS.__init__).parameters
        child_init_parameters = inspect.signature(cls.__init__).parameters
        for name, base_parameter in base_init_parameters.items():
            assert name in child_init_parameters, f"__init__ drops base parameter {name}"
            assert child_init_parameters[name].kind == base_parameter.kind
            assert child_init_parameters[name].default == base_parameter.default
        for name, child_parameter in child_init_parameters.items():
            if name not in base_init_parameters:
                assert child_parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
                    f"added __init__ parameter {name} must be keyword-only"
                )
        assert "super().__init__(" in inspect.getsource(cls.__init__)


# ListReplayBuffer

import numpy as np

from buffers.replay_buffer import ListReplayBuffer


def add_two_example_transitions(buffer: ReplayBuffer) -> None:
    """Add two fixed transitions with different values in every field."""
    buffer.add(np.array([1.0, 2.0, 3.0, 4.0]), 0, -1.5, np.array([1.5, 2.5, 3.5, 4.5]), False)
    buffer.add(np.array([5.0, 6.0, 7.0, 8.0]), 1, -2.5, np.array([5.5, 6.5, 7.5, 8.5]), True)


def test_list_replay_buffer_starts_empty() -> None:
    assert ListReplayBuffer(max_transitions=100).size() == 0


def test_list_replay_buffer_size_counts_added_transitions() -> None:
    buffer = ListReplayBuffer(max_transitions=100)
    add_two_example_transitions(buffer)
    assert buffer.size() == 2


def test_list_replay_buffer_as_arrays_has_expected_keys_and_shapes() -> None:
    buffer = ListReplayBuffer(max_transitions=100)
    add_two_example_transitions(buffer)
    arrays = buffer.as_arrays()
    assert set(arrays) == {"observations", "actions", "rewards", "next_observations", "terminated"}
    assert arrays["observations"].shape == (2, 4)
    assert arrays["actions"].shape == (2,)
    assert arrays["rewards"].shape == (2,)
    assert arrays["next_observations"].shape == (2, 4)
    assert arrays["terminated"].shape == (2,)
    assert arrays["terminated"].dtype == np.bool_


def test_list_replay_buffer_as_arrays_preserves_values_and_order() -> None:
    buffer = ListReplayBuffer(max_transitions=100)
    add_two_example_transitions(buffer)
    arrays = buffer.as_arrays()
    np.testing.assert_array_equal(arrays["observations"], [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    np.testing.assert_array_equal(arrays["actions"], [0, 1])
    np.testing.assert_array_equal(arrays["rewards"], [-1.5, -2.5])
    np.testing.assert_array_equal(
        arrays["next_observations"], [[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5]]
    )
    np.testing.assert_array_equal(arrays["terminated"], [False, True])


def test_list_replay_buffer_stores_copies_of_added_arrays() -> None:
    buffer = ListReplayBuffer(max_transitions=100)
    observation = np.array([1.0, 2.0, 3.0, 4.0])
    next_observation = np.array([1.5, 2.5, 3.5, 4.5])
    buffer.add(observation, 0, -1.0, next_observation, False)
    observation[:] = 99.0
    next_observation[:] = 99.0
    arrays = buffer.as_arrays()
    np.testing.assert_array_equal(arrays["observations"], [[1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(arrays["next_observations"], [[1.5, 2.5, 3.5, 4.5]])


def add_transitions_with_actions(buffer: ReplayBuffer, actions: list[int]) -> None:
    """Add one transition per action, each with a distinct observation and reward."""
    for action in actions:
        buffer.add(np.full(4, float(action)), action, -float(action), np.full(4, action + 0.5), False)


def test_list_replay_buffer_stores_max_transitions() -> None:
    assert ListReplayBuffer(max_transitions=7).max_transitions == 7


def test_list_replay_buffer_drops_oldest_transitions_when_full() -> None:
    buffer = ListReplayBuffer(max_transitions=3)
    add_transitions_with_actions(buffer, [0, 1, 2, 3, 4])
    arrays = buffer.as_arrays()
    assert buffer.size() == 3
    np.testing.assert_array_equal(arrays["actions"], [2, 3, 4])
    np.testing.assert_array_equal(arrays["rewards"], [-2.0, -3.0, -4.0])
    np.testing.assert_array_equal(arrays["observations"][:, 0], [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(arrays["next_observations"][:, 0], [2.5, 3.5, 4.5])


def test_list_replay_buffer_size_never_exceeds_max_transitions() -> None:
    buffer = ListReplayBuffer(max_transitions=3)
    for action in range(10):
        add_transitions_with_actions(buffer, [action % 2])
        assert buffer.size() <= 3


@pytest.mark.parametrize("invalid_max_transitions", [0, -1])
def test_list_replay_buffer_rejects_max_transitions_below_one(invalid_max_transitions: int) -> None:
    with pytest.raises(ValueError):
        ListReplayBuffer(max_transitions=invalid_max_transitions)


# ListReplayBuffer.sample


def buffer_with_actions(actions: list[int]) -> ListReplayBuffer:
    buffer = ListReplayBuffer(max_transitions=100)
    add_transitions_with_actions(buffer, actions)
    return buffer


def test_list_replay_buffer_sample_has_as_arrays_keys_and_requested_size() -> None:
    sample = buffer_with_actions(list(range(20))).sample(num_samples=5, seed=0)
    assert set(sample) == {"observations", "actions", "rewards", "next_observations", "terminated"}
    assert sample["observations"].shape == (5, 4)
    assert sample["next_observations"].shape == (5, 4)
    assert sample["actions"].shape == (5,)
    assert sample["rewards"].shape == (5,)
    assert sample["terminated"].shape == (5,)


def test_list_replay_buffer_sample_keeps_rows_aligned_across_keys() -> None:
    sample = buffer_with_actions(list(range(20))).sample(num_samples=8, seed=3)
    actions = sample["actions"]
    np.testing.assert_array_equal(sample["observations"][:, 0], actions.astype(float))
    np.testing.assert_array_equal(sample["rewards"], -actions.astype(float))
    np.testing.assert_array_equal(sample["next_observations"][:, 0], actions + 0.5)


def test_list_replay_buffer_sample_is_without_replacement() -> None:
    sample = buffer_with_actions(list(range(20))).sample(num_samples=20, seed=1)
    assert sorted(sample["actions"].tolist()) == list(range(20))


def test_list_replay_buffer_sample_is_capped_at_buffer_size() -> None:
    sample = buffer_with_actions([0, 1, 2]).sample(num_samples=10, seed=0)
    assert sorted(sample["actions"].tolist()) == [0, 1, 2]


def test_list_replay_buffer_sample_is_deterministic_for_a_seed() -> None:
    buffer = buffer_with_actions(list(range(20)))
    first = buffer.sample(num_samples=5, seed=7)
    second = buffer.sample(num_samples=5, seed=7)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_list_replay_buffer_sample_varies_with_seed() -> None:
    buffer = buffer_with_actions(list(range(20)))
    samples = {tuple(buffer.sample(num_samples=5, seed=seed)["actions"].tolist()) for seed in range(10)}
    assert len(samples) > 1


def test_list_replay_buffer_sample_does_not_change_the_buffer() -> None:
    buffer = buffer_with_actions(list(range(5)))
    buffer.sample(num_samples=3, seed=0)
    assert buffer.size() == 5
    np.testing.assert_array_equal(buffer.as_arrays()["actions"], [0, 1, 2, 3, 4])


def test_list_replay_buffer_sample_from_empty_buffer_raises() -> None:
    with pytest.raises(ValueError):
        ListReplayBuffer(max_transitions=10).sample(num_samples=1, seed=0)


def test_list_replay_buffer_sample_of_fewer_than_one_raises() -> None:
    with pytest.raises(ValueError):
        buffer_with_actions([0, 1]).sample(num_samples=0, seed=0)


# ListReplayBuffer.hyperparameters


def test_list_replay_buffer_hyperparameters() -> None:
    assert ListReplayBuffer(max_transitions=123).hyperparameters() == {
        "type": "ListReplayBuffer",
        "max_transitions": 123,
    }
