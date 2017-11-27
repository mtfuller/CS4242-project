from lib.PolicyAnalyzer import PolicyAnalyzer

def basic_policy(pos, vel, angle, angular_vel):
    # In this example policty, we only factor in the angle. If the pole moves to
    # the left, then accelerate the cart to the left. If the pole moves to the
    # right, accelerate the cart to the right.
    return 0 if angle < 0 else 1

if __name__ == '__main__':
    # If we want to see the policy's performance, we can use a PolicyAnalyzer
    # object that will run a certain number of CartPole games for a certain
    # number of steps each.
    analyzer = PolicyAnalyzer(episodes=10, steps=1000, render=True)

    # Simply add your policy and give it a name
    analyzer.register_policy("My Basic Policy", basic_policy)

    # Run the policy analyzer and get stats on how your policy did.
    analyzer.run()
