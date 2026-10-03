from functools import partial
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from agents.agent import RandomAgent
from experiments.experiment import EpisodicExperiment
from utils.plotting.plot_episode_returns import plot_episode_returns
from utils.recording.is_last_episode import is_last_episode
from utils.reporting.format_episode_log import format_episode_log
from utils.reporting.results_to_dataframe import results_to_dataframe
from utils.reporting.summarise_returns import summarise_returns

ENV_NAME = "CartPole-v1"
NUM_EPISODES = 100
SEED = 0
OUTPUT_DIR = "outputs"
CARTPOLE_ACTION_SPACE_SIZE = 2
RESULTS_CSV_FILENAME = "results.csv"
RETURNS_PLOT_FILENAME = "returns.png"
RECORD_LAST_EPISODE_VIDEO = True
VIDEO_SUBDIRECTORY = "videos"

if __name__ == "__main__":
    output_directory = Path(OUTPUT_DIR)
    if RECORD_LAST_EPISODE_VIDEO:
        environment = gym.make(ENV_NAME, render_mode="rgb_array")
        environment = RecordVideo(
            environment,
            video_folder=str(output_directory.joinpath(VIDEO_SUBDIRECTORY)),
            episode_trigger=partial(is_last_episode, num_episodes=NUM_EPISODES),
        )
    else:
        environment = gym.make(ENV_NAME)
    random_agent = RandomAgent(action_space_size=CARTPOLE_ACTION_SPACE_SIZE, seed=SEED)
    experiment = EpisodicExperiment(random_agent, environment, NUM_EPISODES, SEED)
    experiment.run()
    environment.close()

    episode_returns = experiment.results["episode_returns"]
    episode_lengths = experiment.results["episode_lengths"]
    for episode_index, (episode_return, episode_length) in enumerate(zip(episode_returns, episode_lengths)):
        print(format_episode_log(episode_index, episode_return, int(episode_length)))
    print(summarise_returns(episode_returns))

    output_directory.mkdir(parents=True, exist_ok=True)
    results_table = results_to_dataframe(experiment.results)
    results_table.to_csv(output_directory.joinpath(RESULTS_CSV_FILENAME), index=False)
    returns_figure = plot_episode_returns(episode_returns)
    returns_figure.savefig(output_directory.joinpath(RETURNS_PLOT_FILENAME))
