import time
from abc import ABC, abstractmethod

import gymnasium as gym

from agents.agent import Agent
from utils.experiment.has_converged import has_converged


class Experiment(ABC):
    """Base class for experiments that run an agent in an environment and record results.

    An experiment holds an agent, an environment, how many episodes to run and a
    seed. Running it mutates the environment and the agent, and fills `results`
    with one entry per episode.

    Attributes:
        agent: the agent that chooses actions.
        env: the environment the agent acts in.
        num_episodes: the number of episodes to run.
        seed: the seed used to make the experiment reproducible.
        results: per-episode metrics, with keys "episode_returns" and
            "episode_lengths", each a list that starts empty.
    """

    agent: Agent
    env: gym.Env
    num_episodes: int
    seed: int
    results: dict[str, list[float]]

    def __init__(self, agent: Agent, env: gym.Env, num_episodes: int, seed: int) -> None:
        """Store the experiment's components and start with empty results.

        Args:
            agent: the agent that chooses actions.
            env: the environment the agent acts in.
            num_episodes: the number of episodes to run.
            seed: the seed used to make the experiment reproducible.

        Returns:
            None.
        """
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes
        self.seed = seed
        self.results = {"episode_returns": [], "episode_lengths": []}

    @abstractmethod
    def run(self) -> None:
        """Run the experiment, mutating the environment and agent, and fill `results`.

        Args:
            None.

        Returns:
            None. After the call, `results["episode_returns"]` and
            `results["episode_lengths"]` hold one value per episode run.
        """


class EpisodicExperiment(Experiment):
    """An experiment that plays a fixed number of episodes and records their returns and lengths.

    The agent only acts; it does not learn. Episode i is reset with seed `seed + i`,
    so the experiment is reproducible given a reproducible agent.
    """

    def run(self) -> None:
        """Play `num_episodes` episodes and record each episode's return and length.

        Differs from the base class by implementing the run as a plain loop over
        episodes: reset the environment with seed `seed + episode_index`, then ask
        the agent for an action and step the environment until the episode is
        terminated or truncated. No learning happens.

        Args:
            None.

        Returns:
            None. Appends one value per episode to `results["episode_returns"]`
            (sum of rewards) and `results["episode_lengths"]` (number of steps).
        """
        for episode_index in range(self.num_episodes):
            observation, _ = self.env.reset(seed=self.seed + episode_index)
            episode_return = 0.0
            episode_length = 0
            episode_finished = False
            while not episode_finished:
                action = self.agent.select_action(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                episode_return += float(reward)
                episode_length += 1
                episode_finished = terminated or truncated
            self.results["episode_returns"].append(episode_return)
            self.results["episode_lengths"].append(float(episode_length))


class LearningExperiment(Experiment):
    """An experiment for learning agents: it feeds experience to the agent and logs its diagnostics.

    Episode i is reset with seed `seed + i`. After every step the agent is told the
    transition, and after every episode the agent is told the episode ended (so it
    can learn) and its diagnostics are appended to `diagnostics_log`. The run stops
    after `num_episodes`, or earlier when `max_seconds` have elapsed or the episode
    returns have converged. At least one episode is always played.

    Attributes:
        max_seconds: the wall-clock time limit in seconds, checked after each episode, or None.
        convergence_window: the window size passed to `has_converged`, or None to never stop on convergence.
        convergence_tolerance: the tolerance passed to `has_converged`.
        diagnostics_log: one dict of the agent's diagnostics per episode played, in order.
    """

    max_seconds: float | None
    convergence_window: int | None
    convergence_tolerance: float
    diagnostics_log: list[dict[str, float]]

    def __init__(
        self,
        agent: Agent,
        env: gym.Env,
        num_episodes: int,
        seed: int,
        *,
        max_seconds: float | None = None,
        convergence_window: int | None = None,
        convergence_tolerance: float = 0.0,
    ) -> None:
        """Store the experiment's components, the stopping options and an empty diagnostics log.

        Args:
            agent: the agent that chooses actions and learns.
            env: the environment the agent acts in.
            num_episodes: the largest number of episodes to run.
            seed: the seed used to make the experiment reproducible.
            max_seconds: stop once this much wall-clock time has passed. None for no limit.
            convergence_window: stop once the mean return over the last this-many episodes
                is within `convergence_tolerance` of the mean over the window before it.
                None to never stop on convergence.
            convergence_tolerance: the largest change in mean return that counts as converged.

        Returns:
            None.
        """
        super().__init__(agent, env, num_episodes, seed)
        self.max_seconds = max_seconds
        self.convergence_window = convergence_window
        self.convergence_tolerance = convergence_tolerance
        self.diagnostics_log = []

    def run(self) -> None:
        """Play episodes, letting the agent learn, until the episodes, time or convergence run out.

        Differs from the base class by implementing the run as a loop over episodes that
        calls `agent.observe_transition` after every step and `agent.end_episode` after
        every episode, then appends a copy of `agent.diagnostics()` to `diagnostics_log`.
        The time limit and convergence are checked after each episode.

        Args:
            None.

        Returns:
            None. Appends one value per episode played to `results["episode_returns"]` and
            `results["episode_lengths"]`, and one dict per episode to `diagnostics_log`.
        """
        start_time = time.monotonic()
        for episode_index in range(self.num_episodes):
            observation, _ = self.env.reset(seed=self.seed + episode_index)
            episode_return = 0.0
            episode_length = 0
            episode_finished = False
            while not episode_finished:
                action = self.agent.select_action(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                self.agent.observe_transition(observation, action, float(reward), next_observation, terminated)
                observation = next_observation
                episode_return += float(reward)
                episode_length += 1
                episode_finished = terminated or truncated
            self.agent.end_episode()
            self.results["episode_returns"].append(episode_return)
            self.results["episode_lengths"].append(float(episode_length))
            self.diagnostics_log.append(dict(self.agent.diagnostics()))
            if self.max_seconds is not None and time.monotonic() - start_time >= self.max_seconds:
                break
            if self.convergence_window is not None and has_converged(
                self.results["episode_returns"], self.convergence_window, self.convergence_tolerance
            ):
                break
