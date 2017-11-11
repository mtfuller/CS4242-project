"""

"""
# Uses OpenAI's gym CartPole environment to analyze policies
import gym

# Uses matplotlib to render bar chart
import matplotlib.pyplot as plt

# Uses stats module for some statistic calculations
from lib.stats import calculate_stats

# Uses numpy for a bit of math in the run() method
import numpy as np

class Policy:
    """Policy class used to store an agents policy function and its performance.

    The Policy class provides a way to store the policy function that the agent
    will use to decide the action it will perform in its environment. Policy
    instances will also store a list of all scores they accumulated using its
    defined policy. Once scores have been accumulated, one can use the analyze()
    method to calculate statistics on the list of scores and measure its
    performance.

    Attributes:
        policy: The function that takes in four parameters (position, velocity,
                angle, and angular velocity) and returns either a 0 (accelerate
                to the left) or a 1 (accelerate to the right).
        scores: A list of all the scores that the policy has been awarded.
        stats: A tuple that stores the mean, standard deviation, minimum, and
               maximum for the policy's scores in that order.
    """
    def __init__(self, func):
        """Initializes Policy instance with the given policy function."""
        self.policy = func
        self.scores = []
        self.stats = ()

    def add_score(self, score):
        """Adds a given score to the scores list.

        Args:
            score: A float of the given reward received for run of the game.
        """
        self.scores.append(score)

    def run(self, obs):
        """Returns the action that the policy has chosen.

        Args:
            obs: The 1D numpy array that is received from gym's environment
                 step() function

        Returns:
            An integer representing the action to be taken. Accelerate to the
            left (0) or accelerate to the right (1).
        """
        return self.policy(*obs)

    def analyze(self):
        """Calculates statistics for the current list of scores."""
        self.stats = calculate_stats(self.scores)

    def __str__(self):
        """Returns a string that contains the policy performance statistics."""
        policy_str = "Policy Performance:\n" + \
                     "\tMean: {}\n" + \
                     "\tSTD: {}\n" + \
                     "\tMin: {}\n" + \
                     "\tMax: {}\n"
        return policy_str.format(*self.stats)

class PolicyAnalyzer:
    """PolicyAnalyzer class that analyzes the performance of a given number of
    policies in a CartPole gym environment.

    The PolicyAnalyzer class provides a way for users to register their own
    defined policies and have the analyzer run statistics on them in batch.
    After all policies have been tried, a bar chart is rendered to the user that
    shows the overall performance of all policies.

    Args:
        episodes: (Default: 100) An integer of the number of games that should
                  be run for each
                  policy.
        steps: (Default: 1000) The number of steps (actions) that should be
               taken for each game.
        render: (Default: False) A boolean that represents whether or not the
                environment should be rendered to the screen.

    Attributes:
        episodes: An integer of the number of games that should be run for each
                  policy.
        steps: The number of steps (actions) that should be taken for each game.
        render: A boolean that represents whether or not the environment should
                be rendered to the screen during the analysis phase. (Basically,
                whether or not the env.render() method should be called or not).
        policies: A dictionary of all the Policy objects that have been
                  registered.
    """
    def __init__(self, **kwargs):
        """Inits the PolicyAnalyzer instance with default parameters if none
        given.
        """
        self.episodes = kwargs["episodes"] if "episodes" in kwargs else 100
        self.steps = kwargs["steps"] if "steps" in kwargs else 1000
        self.render = kwargs["render"] if "render" in kwargs else False
        self.graph = kwargs["graph"] if "graph" in kwargs else True
        self.policies = {}

    def register_policy(self, name, func):
        """Registers and creates a new Policy instance under a specified name.

        Args:
            name: A string of the name associated with the policy.
            func: The function that takes in four parameters (position,
                  velocity, angle, and angular velocity) and returns either a 0
                  (accelerate to the left) or a 1 (accelerate to the right).
        """
        self.policies[name] = Policy(func)

    def run(self):
        """Runs each registered policy for the given number of episodes and
        steps, analyzes each policy, and then renders a bar graph to help
        visuzlize the performance of each policy.
        """
        # Initialize some default values for the matplotlib Bar Graph
        fig, ax = plt.subplots()
        index = np.arange(4)
        n = len(self.policies) if len(self.policies) > 0 else 1
        bar_width = 0.35 / n
        bars = 0

        # Go through each registered policy
        for policy_name in self.policies:
            policy = self.policies[policy_name]

            env = gym.make("CartPole-v1")

            # Run the environment for the given number of episodes and steps
            for episode in range(self.episodes):
                # At the beginning of the episode reset the game
                episode_rewards = 0
                obs = env.reset()

                if self.render:
                    env.render()    # Launches game window to screen

                # Take 1000 steps at most
                for step in range(self.steps):
                    # Have the policy decide the action based off its
                    # observations
                    action = policy.run(obs)

                    # Perform the action
                    obs, reward, done, info = env.step(action)

                    if self.render:
                        env.render()

                    # Accumulate the rewards earned from the action
                    episode_rewards += reward

                    # If the pole fell more than 15 degrees, the cart has moved
                    # off screen, or the agent has won 200 points, stop the
                    # game.
                    if done:
                        break

                # Add the earned score to the Policy instance
                policy.add_score(episode_rewards)

            # After the policy has ran, calculate statistics on the scores
            policy.analyze()

            # Add a new bar to the bar graph, and print stats to console.
            plt.bar(index + (bar_width*bars), policy.stats, bar_width,
                label=policy_name)
            bars += 1
            print(policy_name+" - "+str(policy))

        # Setup and render matplotlib Bar Graph
        plt.xlabel('Statistics')
        plt.ylabel('Scores')
        plt.title('CartPole Policy Performance')
        plt.xticks(index + bar_width / 2, ('Mean', 'STD', 'Min', 'Max'))
        plt.legend()
        plt.tight_layout()
        if self.graph:
            plt.show()
