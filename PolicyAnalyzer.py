import gym
import numpy as np

class Policy:
    def __init__(self, func):
        self.policy = func
        self.scores = []
        self.stats = ()

    def add_score(self, score):
        self.scores.append(score)

    def run(self, obs):
        return self.policy(*obs)

    def analyze(self):
        self.stats = (
            np.mean(self.scores),
            np.std(self.scores),
            np.min(self.scores),
            np.max(self.scores)
        )

class PolicyAnalyzer:
    def __init__(self, **kwargs):
        self.episodes = kwargs["episodes"] if "episodes" in kwargs else 100
        self.steps = kwargs["steps"] if "steps" in kwargs else 1000
        self.render = kwargs["render"] if "render" in kwargs else False
        self.policies = {}

    def add_policy(self, name, func):
        self.policies[name] = Policy(func)

    def run(self):
        for policy_name in self.policies:
            policy = self.policies[policy_name]
            env = gym.make("CartPole-v0")
            for episode in range(self.episodes):
                episode_rewards = 0
                obs = env.reset()
                if self.render:
                    env.render()
                for step in range(self.steps):
                    action = policy.run(obs)
                    obs, reward, done, info = env.step(action)
                    if self.render:
                        env.render()
                    episode_rewards += reward
                    if done:
                        break
                    policy.add_score(episode_rewards)
            policy.analyze()
            print("POLICY: {}\n\tMean: {}\n\tSTD: {}\n\tMin: {}\n\tMax: {}".format(policy_name, *policy.stats))

def basic_policy(pos, vel, angle, angular_vel):
    return 0 if angle < 0 else 1

analyzer = PolicyAnalyzer(episodes=50, steps=1000, render=True)

analyzer.add_policy("Simple Policy", basic_policy)

analyzer.run()
