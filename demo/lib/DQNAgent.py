# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        mem_len = kwargs['memory_limit'] if 'memory_limit' in kwargs else 2000
        self.memory = deque(maxlen=mem_len)
        self.gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.95
        self.epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else 1.0
        self.epsilon_min = kwargs['epsilon_min'] if 'epsilon_min' in kwargs else 0.01
        self.epsilon_decay = kwargs['epsilon_decay'] if 'epsilon_decay' in kwargs else 0.995
        self.learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else 0.001
        self.hidden_nodes = kwargs['hidden_nodes'] if 'hidden_nodes' in kwargs else 24

        self.id = kwargs['id'] if 'id' in kwargs else 'Untitled'

        self.file = kwargs['file'] if 'file' in kwargs else DQNAgent.__name__+'_model'
        self.file = "./saves/"+self.file+'-'+str(self.id)+".h5"

        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.hidden_nodes, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.hidden_nodes, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, f):
        self.file = f
        self.model.load_weights(self.file)

    def save(self):
        self.model.save_weights(self.file)
