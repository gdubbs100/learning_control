from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from buffers.replay_buffer import ListReplayBuffer, ReplayBuffer
from models.dynamics_model import DynamicsModel, LinearDynamicsModel
from planners.planner import Planner, RandomShootingPlanner
from utils.control.action_to_input import action_to_input
from utils.data.split_train_test import split_train_test


class Agent(ABC):
    """Base class for agents that choose discrete actions from environment observations.

    Every agent acts in an environment with a discrete action space of
    `action_space_size` actions, numbered 0 to `action_space_size - 1`.

    Attributes:
        action_space_size: the number of discrete actions the agent can choose from.
    """

    action_space_size: int

    def __init__(self, action_space_size: int) -> None:
        """Store the size of the discrete action space.

        Args:
            action_space_size: the number of discrete actions available.

        Returns:
            None.
        """
        self.action_space_size = action_space_size

    @abstractmethod
    def select_action(self, observation: np.ndarray) -> int:
        """Choose an action given the current observation.

        Args:
            observation: the current observation from the environment.

        Returns:
            The chosen action, an integer in [0, action_space_size).
        """

    def observe_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        """Receive one step of experience from the environment.

        Learning agents override this to record the transition. By default it does nothing.

        Args:
            observation: the observation the action was chosen from.
            action: the action taken.
            reward: the reward received for the step.
            next_observation: the observation after the step.
            terminated: whether the step ended the episode by termination.

        Returns:
            None.
        """

    def end_episode(self) -> None:
        """Tell the agent an episode has finished, so it can learn from it.

        Learning agents override this to update their policy or model. By default it does nothing.

        Args:
            None.

        Returns:
            None.
        """

    def diagnostics(self) -> dict[str, float]:
        """Report agent-specific numbers about learning, such as losses.

        Agents override this to log what matters for their algorithm. By default there is nothing to report.

        Args:
            None.

        Returns:
            A dict of diagnostic name to value, describing the most recent learning step.
            Empty by default.
        """
        return {}

    def hyperparameters(self) -> dict[str, Any]:
        """Report the settings of the agent that affect learning or performance, as JSON-safe values.

        Subclasses extend the returned dict with their own settings. Values are
        strings, numbers, booleans, None, lists, or dicts of these (for nested components).

        Args:
            None.

        Returns:
            A dict with "type" (the class name) and "action_space_size".
        """
        return {"type": type(self).__name__, "action_space_size": self.action_space_size}


class RandomAgent(Agent):
    """An agent that ignores observations and picks actions uniformly at random.

    Randomness comes from a torch.Generator seeded at construction, so two agents
    built with the same seed produce the same sequence of actions.

    Attributes:
        seed: the seed used for the agent's random number generator.
        random_generator: the seeded torch generator actions are drawn from.
    """

    seed: int
    random_generator: torch.Generator

    def __init__(self, action_space_size: int, *, seed: int) -> None:
        """Store the action space size and create a seeded random generator.

        Args:
            action_space_size: the number of discrete actions available.
            seed: the seed for the agent's random number generator.

        Returns:
            None.
        """
        super().__init__(action_space_size)
        self.seed = seed
        self.random_generator = torch.Generator()
        self.random_generator.manual_seed(seed)

    def select_action(self, observation: np.ndarray) -> int:
        """Choose an action uniformly at random, ignoring the observation.

        Differs from the base class by not using the observation at all.

        Args:
            observation: the current observation from the environment (ignored).

        Returns:
            A uniformly random action, an integer in [0, action_space_size).
        """
        random_action = torch.randint(
            low=0, high=self.action_space_size, size=(1,), generator=self.random_generator
        )
        return int(random_action.item())

    def hyperparameters(self) -> dict[str, Any]:
        """Report the action space size and the seed.

        Differs from the base class by adding "seed".

        Args:
            None.

        Returns:
            A dict with the base class entries and "seed".
        """
        return {**super().hyperparameters(), "seed": self.seed}


class ModelBasedMPCAgent(Agent):
    """An agent that learns a dynamics model and plans with it by model predictive control.

    Every transition is stored in a replay buffer. Every `retrain_every_k_episodes`
    episodes, if the buffer holds at least `min_buffer_size_for_update` transitions,
    a random sample of the buffer is split into a train and a test set, the dynamics
    model is re-fitted from scratch on the train set and scored on the test set. Until
    the first fit the agent acts uniformly at random; afterwards, at each step, the
    planner searches for the lowest-cost action sequence from the current observation
    and the first action of that sequence is taken (re-planning every step).

    Attributes:
        model: the dynamics model that is learned and used for planning.
        planner: the optimiser that picks action sequences using the model.
        buffer: the replay buffer holding past transitions.
        retrain_every_k_episodes: how many episodes pass between model fits.
        min_buffer_size_for_update: the fewest buffered transitions for which the model is fitted.
        test_fraction: the fraction of the sampled transitions held out to evaluate the model.
        fit_sample_size: how many transitions are sampled from the buffer per fit, or None for all.
        seed: the seed of the generator used for random actions before the first fit and for sampling.
        random_generator: the seeded generator for those random actions and sample seeds.
        episodes_seen: the number of episodes ended so far.
    """

    model: DynamicsModel
    planner: Planner
    buffer: ReplayBuffer
    retrain_every_k_episodes: int
    min_buffer_size_for_update: int
    test_fraction: float
    fit_sample_size: int | None
    seed: int
    random_generator: np.random.Generator
    episodes_seen: int

    def __init__(
        self,
        action_space_size: int,
        *,
        observation_size: int = 4,
        retrain_every_k_episodes: int = 5,
        min_buffer_size_for_update: int = 100,
        test_fraction: float = 0.2,
        fit_sample_size: int | None = None,
        horizon: int = 10,
        num_samples: int = 1000,
        max_transitions: int = 100_000,
        target_state: np.ndarray | None = None,
        model: DynamicsModel | None = None,
        planner: Planner | None = None,
        buffer: ReplayBuffer | None = None,
        seed: int = 0,
    ) -> None:
        """Store the components of the agent, building defaults for any that are not given.

        Args:
            action_space_size: the number of discrete actions available.
            observation_size: the number of dimensions of the observation (the state).
            retrain_every_k_episodes: how many episodes pass between model fits.
            min_buffer_size_for_update: the fewest buffered transitions for which the model
                is fitted. Must be at least 2, so the train/test split has a row on each side.
            test_fraction: the fraction of the sampled transitions held out to evaluate the model.
            fit_sample_size: how many transitions to sample from the buffer per fit.
                None samples the whole buffer (shuffled).
            horizon: the planning horizon, used when building the default planner.
            num_samples: the number of sequences per plan, used when building the default planner.
            max_transitions: the buffer capacity, used when building the default buffer.
            target_state: the state s* the planner steers towards, used when building the
                default planner. Defaults to zeros.
            model: the dynamics model. Defaults to a LinearDynamicsModel.
            planner: the planner. Defaults to a RandomShootingPlanner.
            buffer: the replay buffer. Defaults to a ListReplayBuffer.
            seed: the seed for random actions and for the default planner.

        Returns:
            None. Raises ValueError if `min_buffer_size_for_update` is below 2.
        """
        super().__init__(action_space_size)
        if min_buffer_size_for_update < 2:
            raise ValueError(
                f"min_buffer_size_for_update must be at least 2, got {min_buffer_size_for_update}"
            )
        if target_state is None:
            target_state = np.zeros(observation_size)
        if model is None:
            model = LinearDynamicsModel(state_dim=observation_size)
        if planner is None:
            planner = RandomShootingPlanner(
                horizon=horizon,
                action_space_size=action_space_size,
                target_state=target_state,
                num_samples=num_samples,
                seed=seed,
            )
        if buffer is None:
            buffer = ListReplayBuffer(max_transitions=max_transitions)
        self.model = model
        self.planner = planner
        self.buffer = buffer
        self.retrain_every_k_episodes = retrain_every_k_episodes
        self.min_buffer_size_for_update = min_buffer_size_for_update
        self.test_fraction = test_fraction
        self.fit_sample_size = fit_sample_size
        self.seed = seed
        self.random_generator = np.random.default_rng(seed)
        self.episodes_seen = 0
        self._latest_diagnostics: dict[str, float] = {}

    def select_action(self, observation: np.ndarray) -> int:
        """Choose a random action before the model is fitted, otherwise plan with the model.

        Differs from the base class by planning an action sequence with the learned
        model and returning its first action once the model has been fitted.

        Args:
            observation: the current observation from the environment.

        Returns:
            The chosen action, an integer in [0, action_space_size).
        """
        if not self.model.is_fitted():
            return int(self.random_generator.integers(0, self.action_space_size))
        planned_actions = self.planner.plan(self.model, np.asarray(observation, dtype=np.float64))
        return int(planned_actions[0])

    def observe_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store the transition in the replay buffer.

        Differs from the base class by recording the transition.

        Args:
            observation: the observation the action was chosen from.
            action: the action taken.
            reward: the reward received for the step.
            next_observation: the observation after the step.
            terminated: whether the step ended the episode by termination.

        Returns:
            None.
        """
        self.buffer.add(observation, action, reward, next_observation, terminated)

    def end_episode(self) -> None:
        """Count the episode and, every k episodes with enough data, re-fit and evaluate the model.

        Differs from the base class by training the model. On a retrain episode where the
        buffer holds at least `min_buffer_size_for_update` transitions, a random sample of
        the buffer (seeded from the agent's generator) is split into train and test sets, the
        model is fitted on the train set and evaluated on the test set. Otherwise the model
        is left alone.

        Args:
            None.

        Returns:
            None. Replaces the diagnostics with those of this episode.
        """
        self.episodes_seen += 1
        diagnostics: dict[str, float] = {"buffer_size": float(self.buffer.size())}
        retrain_due = self.episodes_seen % self.retrain_every_k_episodes == 0
        if retrain_due and self.buffer.size() >= self.min_buffer_size_for_update:
            sample_size = self.buffer.size() if self.fit_sample_size is None else self.fit_sample_size
            sample_seed = int(self.random_generator.integers(2**31 - 1))
            train, test = split_train_test(self.buffer.sample(sample_size, sample_seed), self.test_fraction)
            fit_metrics = self.model.fit(
                train["observations"], action_to_input(train["actions"]), train["next_observations"]
            )
            test_metrics = self.model.evaluate(
                test["observations"], action_to_input(test["actions"]), test["next_observations"]
            )
            diagnostics["model_train_mse"] = float(fit_metrics["train_mse"])
            diagnostics["model_test_mse"] = float(test_metrics["mse"])
            for name, value in self.model.parameters().items():
                diagnostics[f"model_param_{name}"] = float(value)
        self._latest_diagnostics = diagnostics

    def diagnostics(self) -> dict[str, float]:
        """Report the accuracy of the dynamics model and the buffer size from the latest episode end.

        Differs from the base class by reporting model diagnostics.

        Args:
            None.

        Returns:
            A dict with "buffer_size". When the model was re-fitted at the latest episode
            end it also has "model_train_mse", "model_test_mse" (on the held-out part of
            the sample) and one "model_param_<name>" entry per model parameter. Empty
            before the first episode ends.
        """
        return dict(self._latest_diagnostics)

    def hyperparameters(self) -> dict[str, Any]:
        """Report the agent's learning settings and those of its model, planner and buffer.

        Differs from the base class by adding the retraining settings and the seed, and
        the nested "model", "planner" and "buffer" hyperparameters.

        Args:
            None.

        Returns:
            A dict with the base class entries, "retrain_every_k_episodes",
            "min_buffer_size_for_update", "test_fraction", "fit_sample_size", "seed",
            and "model", "planner" and "buffer" dicts from those components.
        """
        return {
            **super().hyperparameters(),
            "retrain_every_k_episodes": self.retrain_every_k_episodes,
            "min_buffer_size_for_update": self.min_buffer_size_for_update,
            "test_fraction": self.test_fraction,
            "fit_sample_size": self.fit_sample_size,
            "seed": self.seed,
            "model": self.model.hyperparameters(),
            "planner": self.planner.hyperparameters(),
            "buffer": self.buffer.hyperparameters(),
        }


class ReinforceAgent(Agent):
    """A policy-gradient agent trained with REINFORCE, learning once at the end of each episode.

    The policy is a small neural network mapping an observation to action logits.
    Actions are sampled from the resulting categorical distribution. At the end of
    an episode the discounted returns-to-go are computed and normalised to zero mean
    and unit standard deviation, and one gradient step is taken on the loss
    -mean(log_prob(action) * normalised_return).

    Attributes:
        policy: the network from observation to action logits.
        optimizer: the Adam optimiser for the policy's parameters.
        discount: the discount factor applied to future rewards.
        seed: the seed for the network's initial weights and for action sampling.
        random_generator: the seeded torch generator actions are sampled from.
    """

    policy: torch.nn.Sequential
    optimizer: torch.optim.Optimizer
    discount: float
    seed: int
    random_generator: torch.Generator

    def __init__(
        self,
        action_space_size: int,
        *,
        observation_size: int = 4,
        hidden_size: int = 64,
        learning_rate: float = 1e-2,
        discount: float = 0.99,
        seed: int = 0,
    ) -> None:
        """Build the policy network, its optimiser and a seeded random generator.

        Args:
            action_space_size: the number of discrete actions available.
            observation_size: the number of dimensions of the observation.
            hidden_size: the number of units in the policy's hidden layer.
            learning_rate: the Adam learning rate.
            discount: the discount factor applied to future rewards.
            seed: the seed for the initial weights and for action sampling.

        Returns:
            None.
        """
        super().__init__(action_space_size)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.policy = torch.nn.Sequential(
                torch.nn.Linear(observation_size, hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_size, action_space_size),
            )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.discount = discount
        self.seed = seed
        self.random_generator = torch.Generator()
        self.random_generator.manual_seed(seed)
        self._episode_observations: list[np.ndarray] = []
        self._episode_actions: list[int] = []
        self._episode_rewards: list[float] = []
        self._latest_diagnostics: dict[str, float] = {}

    def select_action(self, observation: np.ndarray) -> int:
        """Sample an action from the policy's distribution for the observation.

        Differs from the base class by sampling from the learned policy.

        Args:
            observation: the current observation from the environment.

        Returns:
            The sampled action, an integer in [0, action_space_size).
        """
        with torch.no_grad():
            logits = self.policy(torch.as_tensor(observation, dtype=torch.float32))
            probabilities = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probabilities, num_samples=1, generator=self.random_generator)
        return int(action.item())

    def observe_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        """Record the step of the current episode.

        Differs from the base class by keeping the observation, action and reward
        until the episode ends.

        Args:
            observation: the observation the action was chosen from.
            action: the action taken.
            reward: the reward received for the step.
            next_observation: the observation after the step (not used).
            terminated: whether the step ended the episode by termination (not used).

        Returns:
            None.
        """
        self._episode_observations.append(np.array(observation, dtype=np.float32, copy=True))
        self._episode_actions.append(int(action))
        self._episode_rewards.append(float(reward))

    def end_episode(self) -> None:
        """Take one REINFORCE gradient step on the finished episode and clear it.

        Differs from the base class by updating the policy. Does nothing if no steps
        were recorded. Replaces the diagnostics with those of this update.

        Args:
            None.

        Returns:
            None.
        """
        if not self._episode_rewards:
            return
        returns_to_go = []
        running_return = 0.0
        for reward in reversed(self._episode_rewards):
            running_return = reward + self.discount * running_return
            returns_to_go.append(running_return)
        returns_to_go.reverse()
        returns = torch.tensor(returns_to_go, dtype=torch.float32)
        normalised_returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        observations = torch.as_tensor(np.stack(self._episode_observations), dtype=torch.float32)
        actions = torch.as_tensor(self._episode_actions, dtype=torch.int64)
        log_probabilities = torch.log_softmax(self.policy(observations), dim=-1)
        chosen_log_probabilities = log_probabilities.gather(1, actions[:, None]).squeeze(1)
        loss = -(chosen_log_probabilities * normalised_returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=float("inf"))
        self.optimizer.step()

        entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1).mean()
        self._latest_diagnostics = {
            "loss": float(loss.item()),
            "grad_norm": float(gradient_norm),
            "entropy": float(entropy.item()),
            "episode_return": float(sum(self._episode_rewards)),
        }
        self._episode_observations = []
        self._episode_actions = []
        self._episode_rewards = []

    def diagnostics(self) -> dict[str, float]:
        """Report the loss, gradient norm, policy entropy and return of the latest update.

        Differs from the base class by reporting policy-gradient diagnostics.

        Args:
            None.

        Returns:
            A dict with "loss", "grad_norm" (gradient norm before the step), "entropy"
            (mean policy entropy over the episode's observations, measured before the
            step) and "episode_return" (undiscounted sum of rewards). Empty before the
            first update.
        """
        return dict(self._latest_diagnostics)

    def hyperparameters(self) -> dict[str, Any]:
        """Report the policy network size, learning rate, discount and seed.

        Differs from the base class by adding "observation_size" and "hidden_size"
        (read from the policy's layers), "learning_rate" (read from the optimiser),
        "discount" and "seed".

        Args:
            None.

        Returns:
            A dict with the base class entries and the entries above.
        """
        return {
            **super().hyperparameters(),
            "observation_size": int(self.policy[0].in_features),
            "hidden_size": int(self.policy[0].out_features),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "discount": self.discount,
            "seed": self.seed,
        }
