import numpy as np
from lib.DQNAgent import DQNAgent
from lib.PolicyAnalyzer import PolicyAnalyzer

model = input("What is the name of the model file: ")

agent = DQNAgent(4,2)

agent.load('./saves/'+model+'.h5')

def dql_policy(pos, vel, angle, angular_vel):
    state = np.reshape([pos, vel, angle, angular_vel], [1, 4])
    action_vals = agent.model.predict(state)
    return np.argmax(action_vals[0])

analyzer = PolicyAnalyzer(episodes=10, steps=1000, render=True)

# Simply add your policy and give it a name
analyzer.register_policy("DQN Policy", dql_policy)

# Run the policy analyzer and get stats on how your policy did.
analyzer.run()
