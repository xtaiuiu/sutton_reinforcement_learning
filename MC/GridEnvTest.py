import gym
import time
from MC import MC_control_first_visit as mc

env = gym.make('GridWorld-v0')
actions = env.env.getAction()
policy = mc.mc_control()
print(policy)
for i in range(1000):
    o = env.reset()
    print (o)
    env.render()
    for t in range(100):
        #At = actions[np.random.randint(len(actions))]  # random policy
        At = mc.make_decision(policy, o)
        o, r, isd, info = env.step(At)
        print("move to {}, new state is {}".format(At, o))
        env.render()
        time.sleep(0.1)
        if isd:
            print("Episode finished after {} timesteps".format(t+1))
            break
    #time.sleep(5)
    print("Episode {} started".format(i+1))
env.env.close()