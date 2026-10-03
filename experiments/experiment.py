from abc import ABC, abstractmethod

import gymnasium as gym

from agents.agent import Agent


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
