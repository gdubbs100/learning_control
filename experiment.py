from functools import partial
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from agents.agent import ModelBasedMPCAgent, RandomAgent, ReinforceAgent
from envs.cost_wrapper import QuadraticCostCartPole
from experiments.experiment import LearningExperiment
from utils.plotting.plot_episode_returns import plot_episode_returns
from utils.plotting.plot_returns_comparison import plot_returns_comparison
from utils.recording.is_last_episode import is_last_episode
from utils.reporting.diagnostics_to_dataframe import diagnostics_to_dataframe
from utils.reporting.format_episode_log import format_episode_log
from utils.reporting.results_to_dataframe import results_to_dataframe
from utils.reporting.summarise_returns import summarise_returns

ENV_NAME = "CartPole-v1"
NUM_EPISODES = 100
SEED = 0
OUTPUT_DIR = "outputs"
CARTPOLE_ACTION_SPACE_SIZE = 2
CARTPOLE_OBSERVATION_SIZE = 4
RESULTS_CSV_FILENAME = "results.csv"
DIAGNOSTICS_CSV_FILENAME = "diagnostics.csv"
RETURNS_PLOT_FILENAME = "returns.png"
COMPARISON_PLOT_FILENAME = "comparison.png"
RECORD_LAST_EPISODE_VIDEO = True
VIDEO_SUBDIRECTORY = "videos"
MAX_SECONDS_PER_AGENT = 600.0
CONVERGENCE_WINDOW = None
CONVERGENCE_TOLERANCE = 0.0
MPC_RETRAIN_EVERY_K_EPISODES = 5
MPC_PLANNING_HORIZON = 15
MPC_NUM_PLAN_SAMPLES = 500

if __name__ == "__main__":
    output_directory = Path(OUTPUT_DIR)
    agents = {
        "random": RandomAgent(action_space_size=CARTPOLE_ACTION_SPACE_SIZE, seed=SEED),
        "reinforce": ReinforceAgent(
            action_space_size=CARTPOLE_ACTION_SPACE_SIZE,
            observation_size=CARTPOLE_OBSERVATION_SIZE,
            seed=SEED,
        ),
        "model_based_mpc": ModelBasedMPCAgent(
            action_space_size=CARTPOLE_ACTION_SPACE_SIZE,
            observation_size=CARTPOLE_OBSERVATION_SIZE,
            retrain_every_k_episodes=MPC_RETRAIN_EVERY_K_EPISODES,
            horizon=MPC_PLANNING_HORIZON,
            num_samples=MPC_NUM_PLAN_SAMPLES,
            seed=SEED,
        ),
    }
    returns_by_agent = {}

    for agent_name, agent in agents.items():
        agent_output_directory = output_directory.joinpath(agent_name)
        if RECORD_LAST_EPISODE_VIDEO:
            environment = gym.make(ENV_NAME, render_mode="rgb_array")
        else:
            environment = gym.make(ENV_NAME)
        environment = QuadraticCostCartPole(environment)
        if RECORD_LAST_EPISODE_VIDEO:
            environment = RecordVideo(
                environment,
                video_folder=str(agent_output_directory.joinpath(VIDEO_SUBDIRECTORY)),
                episode_trigger=partial(is_last_episode, num_episodes=NUM_EPISODES),
            )
        experiment = LearningExperiment(
            agent,
            environment,
            NUM_EPISODES,
            SEED,
            max_seconds=MAX_SECONDS_PER_AGENT,
            convergence_window=CONVERGENCE_WINDOW,
            convergence_tolerance=CONVERGENCE_TOLERANCE,
        )
        experiment.run()
        environment.close()

        episode_returns = experiment.results["episode_returns"]
        episode_lengths = experiment.results["episode_lengths"]
        print(agent_name)
        for episode_index, (episode_return, episode_length) in enumerate(zip(episode_returns, episode_lengths)):
            print(format_episode_log(episode_index, episode_return, int(episode_length)))
        print(summarise_returns(episode_returns))

        agent_output_directory.mkdir(parents=True, exist_ok=True)
        results_table = results_to_dataframe(experiment.results)
        results_table.to_csv(agent_output_directory.joinpath(RESULTS_CSV_FILENAME), index=False)
        diagnostics_table = diagnostics_to_dataframe(experiment.diagnostics_log)
        diagnostics_table.to_csv(agent_output_directory.joinpath(DIAGNOSTICS_CSV_FILENAME), index=False)
        returns_figure = plot_episode_returns(episode_returns)
        returns_figure.savefig(agent_output_directory.joinpath(RETURNS_PLOT_FILENAME))
        returns_by_agent[agent_name] = episode_returns

    comparison_figure = plot_returns_comparison(returns_by_agent)
    comparison_figure.savefig(output_directory.joinpath(COMPARISON_PLOT_FILENAME))
