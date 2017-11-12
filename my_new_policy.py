"""
This file shows how to create your own policy for the CartPole game.
"""
# Import the policy analyzer class that is used to
from lib.PolicyAnalyzer import PolicyAnalyzer
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np

model = Sequential()
model.add(Dense(24, input_dim=4, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mse',
              optimizer=Adam(lr=0.001))
model.load_weights('./saves/cartpole-dqn.h5')

# model2 = Sequential()
# model2.add(Dense(24, input_dim=4, activation='relu'))
# model2.add(Dense(24, activation='relu'))
# model2.add(Dense(24, activation='relu'))
# model2.add(Dense(2, activation='linear'))
# model2.compile(loss='mse',
#               optimizer=Adam(lr=0.001))
# model2.load_weights('./saves/cartpole-dqn-batch-percent.h5')


model3 = Sequential()
model3.add(Dense(24, input_dim=4, activation='relu'))
model3.add(Dense(24, activation='relu'))
model3.add(Dense(2, activation='linear'))
model3.compile(loss='mse',
              optimizer=Adam(lr=0.001))
model3.load_weights('./saves/DQNAgent_model_Optimal.h5')

# To create a new policy for the agent, you must implement a function that can
# use several observations about the environment and return an action.
#
# Environment Observations:
# There are four observations about the environment:
#   Position (float): This is the position of the cart. If the cart is in the
#                     center than it equals 0.0.
#   Velocity (float): This is the velocity of the cart alone the horizontal.
#   Angle (float): This is the angle of the pole from the vertical axis. If the
#                  pole is straight up in the air, then the angle is 0.0.
#   Angular Velocity (float): This is the angular velocity of the pole.
#
# Agent Actions:
# There are also two possible actions that the policy must decide on:
#   Accelerate to the Left (0): To choose this action, simply return 0.
#   Accelerate to the Right (1): To choose this action, simply return 1.
#
# Your Goal: Create a new policy that can use the position, velocity, etc. to
# decide either left (0) or right (1).

# Example Policies:
def basic_policy(pos, vel, angle, angular_vel):
    # In this example policty, we only factor in the angle. If the pole moves to
    # the left, then accelerate the cart to the left. If the pole moves to the
    # right, accelerate the cart to the right.
    return 0 if angle < 0 else 1

def bad_policy(pos, vel, angle, angular_vel):
    # In this example bad policty, we don't factor in anything. We just say to
    # go left.
    return 0

def dql_policy(pos, vel, angle, angular_vel):
    state = np.reshape([pos, vel, angle, angular_vel], [1, 4])
    action_vals = model.predict(state)
    return np.argmax(action_vals[0])

# def dql2_policy(pos, vel, angle, angular_vel):
#     state = np.reshape([pos, vel, angle, angular_vel], [1, 4])
#     action_vals = model2.predict(state)
#     return np.argmax(action_vals[0])

def dql_optimal_policy(pos, vel, angle, angular_vel):
    state = np.reshape([pos, vel, angle, angular_vel], [1, 4])
    action_vals = model3.predict(state)
    return np.argmax(action_vals[0])

if __name__ == '__main__':
    # If we want to see the policy's performance, we can use a PolicyAnalyzer
    # object that will run a certain number of CartPole games for a certain
    # number of steps each.
    analyzer = PolicyAnalyzer(episodes=10, steps=1000, render=True)

    # Simply add your policy and give it a name
    analyzer.register_policy("My Basic Policy", basic_policy)
    analyzer.register_policy("My Bad Policy", bad_policy)
    analyzer.register_policy("DQL Policy", dql_policy)
    # analyzer.register_policy("New DQL Policy", dql2_policy)
    analyzer.register_policy("Optimal DQL Policy", dql_optimal_policy)

    # Run the policy analyzer and get stats on how your policy did.
    analyzer.run()
