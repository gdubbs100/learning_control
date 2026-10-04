import inspect

import pytest

import models.dynamics_model as family_module
from models.dynamics_model import DynamicsModel

BASE_CLASS = DynamicsModel
# Arguments for the base class __init__ parameters, used to construct every class in the family.
BASE_INIT_ARGUMENTS = {"state_dim": 4}
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


# LinearDynamicsModel

import numpy as np
import torch

from models.dynamics_model import LinearDynamicsModel

TRUE_A = np.array([[1.0, 0.1], [0.0, 1.0]])
TRUE_B = np.array([0.0, 0.1])
TOLERANCE = 1e-8


def exact_transitions(
    a_matrix: np.ndarray, b_vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Noise-free transitions s' = A s + B u on a 5x5 state grid with u in {-1, 1}."""
    grid_values = np.linspace(-1.0, 1.0, 5)
    states = np.array([[x, v] for x in grid_values for v in grid_values for _ in (0, 1)])
    inputs = np.array([u for _ in grid_values for _ in grid_values for u in (-1.0, 1.0)])
    next_states = states @ a_matrix.T + inputs[:, None] * b_vector
    return states, inputs, next_states


def fitted_linear_model() -> LinearDynamicsModel:
    """A LinearDynamicsModel fitted on exact data from TRUE_A and TRUE_B."""
    model = LinearDynamicsModel(state_dim=2)
    model.fit(*exact_transitions(TRUE_A, TRUE_B))
    return model


def test_linear_model_is_not_fitted_before_fit() -> None:
    assert LinearDynamicsModel(state_dim=2).is_fitted() is False


def test_linear_model_is_fitted_after_fit() -> None:
    assert fitted_linear_model().is_fitted() is True


def test_linear_model_predict_before_fit_raises_runtime_error() -> None:
    model = LinearDynamicsModel(state_dim=2)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 2)), np.zeros(1))


def test_linear_model_fit_recovers_a_and_b() -> None:
    model = fitted_linear_model()
    np.testing.assert_allclose(model.A.numpy(), TRUE_A, atol=TOLERANCE)
    np.testing.assert_allclose(model.B.numpy(), TRUE_B, atol=TOLERANCE)


def test_linear_model_a_and_b_are_float64_torch_tensors() -> None:
    model = fitted_linear_model()
    assert isinstance(model.A, torch.Tensor) and model.A.dtype == torch.float64
    assert isinstance(model.B, torch.Tensor) and model.B.dtype == torch.float64
    assert model.A.shape == (2, 2)
    assert model.B.shape == (2,)


def test_linear_model_fit_returns_train_mse_near_zero_on_exact_data() -> None:
    model = LinearDynamicsModel(state_dim=2)
    metrics = model.fit(*exact_transitions(TRUE_A, TRUE_B))
    assert metrics["train_mse"] < 1e-12


def test_linear_model_predict_matches_a_s_plus_b_u() -> None:
    model = fitted_linear_model()
    prediction = model.predict(np.array([[0.3, -0.2]]), np.array([1.0]))
    np.testing.assert_allclose(prediction, np.array([[0.28, -0.1]]), atol=TOLERANCE)


def test_linear_model_predict_handles_a_batch_and_returns_numpy() -> None:
    model = fitted_linear_model()
    prediction = model.predict(np.array([[0.3, -0.2], [1.0, 1.0]]), np.array([1.0, -1.0]))
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (2, 2)
    np.testing.assert_allclose(prediction, np.array([[0.28, -0.1], [1.1, 0.9]]), atol=TOLERANCE)


def test_linear_model_refit_replaces_earlier_fit() -> None:
    model = fitted_linear_model()
    new_a = np.array([[0.5, 0.0], [0.0, 0.5]])
    new_b = np.array([1.0, 0.0])
    model.fit(*exact_transitions(new_a, new_b))
    np.testing.assert_allclose(model.A.numpy(), new_a, atol=TOLERANCE)
    np.testing.assert_allclose(model.B.numpy(), new_b, atol=TOLERANCE)


def test_linear_model_fit_is_deterministic() -> None:
    first_model = fitted_linear_model()
    second_model = fitted_linear_model()
    assert torch.equal(first_model.A, second_model.A)
    assert torch.equal(first_model.B, second_model.B)


def test_linear_model_fit_and_predict_do_not_mutate_inputs() -> None:
    states, inputs, next_states = exact_transitions(TRUE_A, TRUE_B)
    originals = (states.copy(), inputs.copy(), next_states.copy())
    model = LinearDynamicsModel(state_dim=2)
    model.fit(states, inputs, next_states)
    model.predict(states, inputs)
    np.testing.assert_array_equal(states, originals[0])
    np.testing.assert_array_equal(inputs, originals[1])
    np.testing.assert_array_equal(next_states, originals[2])


# LinearDynamicsModel.parameters


def test_linear_model_parameters_before_fit_raises_runtime_error() -> None:
    with pytest.raises(RuntimeError):
        LinearDynamicsModel(state_dim=2).parameters()


def test_linear_model_parameters_are_flat_named_floats() -> None:
    parameters = fitted_linear_model().parameters()
    assert list(parameters) == ["A_0_0", "A_0_1", "A_1_0", "A_1_1", "B_0", "B_1"]
    assert all(type(value) is float for value in parameters.values())


def test_linear_model_parameters_match_fitted_a_and_b() -> None:
    parameters = fitted_linear_model().parameters()
    for row in range(2):
        for column in range(2):
            assert parameters[f"A_{row}_{column}"] == pytest.approx(TRUE_A[row, column], abs=TOLERANCE)
        assert parameters[f"B_{row}"] == pytest.approx(TRUE_B[row], abs=TOLERANCE)


# LinearDynamicsModel.evaluate


def test_linear_model_evaluate_before_fit_raises_runtime_error() -> None:
    with pytest.raises(RuntimeError):
        LinearDynamicsModel(state_dim=2).evaluate(*exact_transitions(TRUE_A, TRUE_B))


def test_linear_model_evaluate_on_exact_data_gives_mse_near_zero() -> None:
    metrics = fitted_linear_model().evaluate(*exact_transitions(TRUE_A, TRUE_B))
    assert set(metrics) == {"mse"}
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-12)


def test_linear_model_evaluate_matches_hand_computed_mse() -> None:
    states, inputs, next_states = exact_transitions(TRUE_A, TRUE_B)
    # Error of 1.0 in the first state dimension only: squared errors are 1 and 0, mean 0.5.
    shifted_next_states = next_states + np.array([1.0, 0.0])
    metrics = fitted_linear_model().evaluate(states, inputs, shifted_next_states)
    assert metrics["mse"] == pytest.approx(0.5, abs=1e-6)


def test_linear_model_evaluate_uses_the_given_data_not_the_training_data() -> None:
    states, inputs, next_states = exact_transitions(TRUE_A, TRUE_B)
    metrics = fitted_linear_model().evaluate(states[:3], inputs[:3], next_states[:3] + 0.5)
    assert metrics["mse"] == pytest.approx(0.25, abs=1e-6)


def test_linear_model_evaluate_does_not_mutate_inputs_or_the_fit() -> None:
    model = fitted_linear_model()
    states, inputs, next_states = exact_transitions(TRUE_A, TRUE_B)
    originals = [array.copy() for array in (states, inputs, next_states)]
    parameters_before = model.parameters()
    model.evaluate(states, inputs, next_states)
    for array, original in zip((states, inputs, next_states), originals):
        np.testing.assert_array_equal(array, original)
    assert model.parameters() == parameters_before


# LinearDynamicsModel.hyperparameters


def test_linear_model_hyperparameters() -> None:
    assert LinearDynamicsModel(state_dim=4).hyperparameters() == {
        "type": "LinearDynamicsModel",
        "state_dim": 4,
    }
