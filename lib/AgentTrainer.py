from multiprocessing import Process, Queue, Manager, Value, Pool
from itertools import product
from lib.stats import calculate_stats
import time
import gym
import numpy as np

class AgentTrainer:
    def __init__(self, AgentClass, **kwargs):
        self.AgentClass = AgentClass
        self.env = kwargs['env'] if 'env' in kwargs else 'CartPole-v1'
        self.episodes = kwargs['episodes'] if 'episodes' in kwargs else 1000
        self.steps = kwargs['steps'] if 'steps' in kwargs else 500
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 32
        self.save_interval = kwargs['save_interval'] if 'save_interval' in kwargs else 10
        self.agents = []

    def train(self, **kwargs):
        gamma = kwargs['gamma'] if 'gamma' in kwargs else [0.95]
        epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else [1.0]
        epsilon_min = kwargs['epsilon_min'] if 'epsilon_min' in kwargs else [0.01]
        epsilon_decay = kwargs['epsilon_decay'] if 'epsilon_decay' in kwargs else [0.995]
        learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else [0.001]

        agent_dict = {
            'gamma': gamma,
            'epsilon': epsilon,
            'epsilon_min': epsilon_min,
            'epsilon_decay': epsilon_decay,
            'learning_rate': learning_rate
        }

        perm = [dict(zip(agent_dict, v)) for v in product(*agent_dict.values())]

        print("Model training variations:",len(perm))

        subproc = []

        queue = Queue()

        manager = Manager()

        total = float(len(perm) * self.episodes)
        counter = manager.Value('i',0)

        n = 0
        args = []
        for p in perm:
            n += 1
            args.append(tuple([counter,total,n,p]))

        p = Pool(4)
        self.agents = p.map(self.train_agent, args)

    def train_agent(self, params):
        n = params[0]
        total = params[1]
        _id = params[2]
        config = params[3]
        env = gym.make(self.env)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = self.AgentClass(state_size, action_size, id=_id, **config)

        scores = []
        done = False

        for e in range(self.episodes):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            final_score = 0
            for time in range(self.steps):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                final_score += reward

                if done:
                    scores.append(final_score)
                    break

            n.value += 1
            print("Status: {0:.2f}%".format((100*n.value/total)))
            if len(agent.memory) > self.batch_size:
                agent.replay(self.batch_size)
            if e % self.save_interval == 0:
                agent.save()

        stats = calculate_stats(scores)

        agent.save()
        return {
            'id': agent.id,
            'state_size': agent.state_size,
            'action_size': agent.action_size,
            'gamma': agent.gamma,
            'epsilon': agent.epsilon,
            'epsilon_min': agent.epsilon_min,
            'epsilon_decay': agent.epsilon_decay,
            'learning_rate': agent.learning_rate,
            'scores': scores,
            'mean': stats[0],
            'std': stats[1],
            'min': stats[2],
            'max': stats[3],
            'file': agent.file
        }
