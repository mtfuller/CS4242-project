import numpy as np
import csv
from lib.DQNAgent import DQNAgent
from lib.AgentTrainer import AgentTrainer
from lib.PolicyAnalyzer import PolicyAnalyzer

def build_policy(params):
    _id = params['id']
    state_size = params['state_size']
    action_size = params['action_size']
    f = params['file']

    agent = DQNAgent(state_size, action_size, id=_id)
    agent.load(f)

    def dql_policy(pos, vel, angle, angular_vel):
        state = np.reshape([pos, vel, angle, angular_vel], [1, 4])
        action_vals = agent.model.predict(state)
        return np.argmax(action_vals[0])

    return dql_policy

if __name__ == '__main__':
    print("\nSeting up model trainer...")
    trainer = AgentTrainer(DQNAgent, episodes=100)
    print("Training multiple variations of the model...")
    trainer.train(gamma=[0.9, 0.95, 0.99])
    print("Training is complete.")


    print("\nCreating \"training_scores.csv\" and \"training_stats.csv\"...")
    scores_lists = []
    for agent in trainer.agents:
        scores_lists.append([agent['id']] + agent['scores'])
        del agent['scores']

    with open("./data/training_scores.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(scores_lists)

    keys = trainer.agents[0].keys()
    with open('./data/training_stats.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(trainer.agents)
    print("Finished.")

    print("\nSetting up policy analyzer...")
    analyzer = PolicyAnalyzer(episodes=100, steps=1000, render=False)

    for agent_params in trainer.agents:
        # Simply add your policy and give it a name
        analyzer.register_policy(str(agent_params['id']), build_policy(agent_params))

    print("Analyzing the performance of each agent...")
    # Run the policy analyzer and get stats on how your policy did.
    analyzer.run()

    print("Creating \"performance_stats.csv\"...")
    for agent in trainer.agents:
        stats = analyzer.policies[str(agent['id'])].stats
        agent['mean'] = stats[0]
        agent['std'] = stats[1]
        agent['min'] = stats[2]
        agent['max'] = stats[3]

    keys = trainer.agents[0].keys()
    with open('./data/performance_stats.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(trainer.agents)
    print("Finished.")

    print("\nModel tuning has completed.")
