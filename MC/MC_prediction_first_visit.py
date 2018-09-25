import gym
import numpy as np
env = gym.make('GridWorld-v0')
#print (callable(env.env.getStates))
states = env.env.getStates()
actions = env.env.getAction()
terminal_states = env.env.getTerminate_states()
R = {s : [] for s in states} # returns
V = {s : 0 for s in states}

for i in range(1000): # totally 20 episodes
    G = 0
    S = env.reset()
    Sn = S
    visited_states = set()
    if not terminal_states.__contains__(S):
        visited_states.add(S)
    while not terminal_states.__contains__(Sn):
        At = actions[np.random.randint(len(actions))]  #random policy
        Sn, r, isd, info = env.step(At)
        if not terminal_states.__contains__(Sn):
            visited_states.add(Sn)
        G += r
    for s in visited_states:
        R[s].append(G)
for s in states:
    V[s] = round(sum(R[s])/len(R[s]), 2) if len(R[s]) != 0 else 0
print (V)

a = input("waiting...")
env.env.close()