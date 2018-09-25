import gym
import time
import numpy as np
env = gym.make('MountainCar-v0')
S0 = env.reset()
print (env.action_space)
print (env.observation_space)
print (env.observation_space.high)
for t in range(10):
    print (env.action_space.sample())
print (env.observation_space.low)
isd = False
i = 0

while  isd:
    Sn, r, isd, info = env.step(env.action_space.sample())
    i += 1
    print ("Sn: {}, Reward: {}, isd: {}, info:{}".format(Sn, r, isd, info))
    env.render()

print(i)
a = input("")
env.env.close()