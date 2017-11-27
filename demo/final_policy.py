from lib.PolicyAnalyzer import PolicyAnalyzer
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np

model = Sequential()
model.add(Dense(24, input_dim=4, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mse',optimizer=Adam(lr=0.001))
model.load_weights('./models/FinalModel.h5')

def dql_optimal_policy(pos, vel, angle, angular_vel):
    state = np.reshape([pos, vel, angle, angular_vel], [1, 4])
    action_vals = model.predict(state)
    return np.argmax(action_vals[0])

if __name__ == '__main__':
    # If we want to see the policy's performance, we can use a PolicyAnalyzer
    # object that will run a certain number of CartPole games for a certain
    # number of steps each.
    analyzer = PolicyAnalyzer(episodes=10, steps=1000, render=True)

    # Simply add your policy and give it a name
    analyzer.register_policy("Final DQL Policy", dql_optimal_policy)

    # Run the policy analyzer and get stats on how your policy did.
    analyzer.run()
