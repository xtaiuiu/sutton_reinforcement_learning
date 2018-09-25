import gym
import numpy as np

def ep_greedy(Q, s, states, actions, ep):
    a_star = actions[0]
    for action in Q[s]:
        a_star = action if Q[s][action] > Q[s][a_star] else a_star
    pi_s = {}
    for action in actions:
        pi_s[action] = 1-ep+ep/len(actions) if action == a_star else ep/len(actions)
    return pi_s

def make_decision(policy, s):
    p_s = policy[s]
    rand_float = np.random.random()
    if rand_float < p_s['n']:
        return 'n'
    elif rand_float < p_s['n']+p_s['e']:
        return 'e'
    elif rand_float < p_s['n'] + p_s['e'] + p_s['s']:
        return 's'
    else:
        return 'w'

def mc_control():
    env = gym.make('GridWorld-v0')
    states = env.env.getStates()
    actions = env.env.getAction()
    terminal_states = env.env.getTerminate_states()
    R = {s : {} for s in states} # returns
    Q = {s : {} for s in states}
    policy = {s: {} for s in states}
    for s in states:
        R[s] = {a : [] for a in actions}
        Q[s] = {a : 0 for a in actions}
        policy[s] = {a: 1.0/len(actions) for a in actions}

    for i in range(10000): # totally 20 episodes
        G = 0
        S = env.reset()
        Sn = S
        first_s_q = set()
        while not terminal_states.__contains__(Sn):
            #At = actions[np.random.randint(len(actions))]  # random policy
            At = make_decision(policy, Sn)
            first_s_q.add((Sn, At))
            Sn, r, isd, info = env.step(At)
            G += r
        for key in first_s_q:
            R[key[0]][key[1]].append(G)
        for s in states:
            for a in actions:
                Q[s][a] = round(sum(R[s][a]) / len(R[s][a]), 2) if len(R[s][a]) != 0 else 0
        #policy iteration
        for s in states:
            policy[s] = ep_greedy(Q, s, states, actions, 0.05)
    env.env.close()
    return policy

