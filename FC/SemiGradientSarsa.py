import numpy as np
import gym
def sg_sarsa():
    env = gym.make("MountainCar-v0")
    alpha = 1/16.0
    ep = 0.1
    gama = 0.5
    actions = [0, 1, 2]
    l = 9
    w = [np.ones(l), np.ones(l), np.ones(l)]
    for i in range(10):
        [s1, s2] = env.reset()
        St = [s1, s2]
        x = ([1, s1, s2, s1*s2, s1*s1, s2*s2, s1*s2*s2, s1*s1*s2, s1*s1*s2*s2])
        Qt = q_function(St, w)
        At = ep_greedy(Qt, actions, ep)
        isd = False
        while not isd:
            Sn, r, isd, info = env.step(At)
            [s1, s2] = Sn
            x = ([1, s1, s2, s1*s2, s1*s1, s2*s2, s1*s2*s2, s1*s1*s2, s1*s1*s2*s2])
            if isd:
                w[At] += list_multiply(alpha * (r - Qt[At]), x)
                break
            Qn = q_function(Sn, w)
            An = ep_greedy(Qn, actions, ep)
            w[An] += list_multiply(alpha * (r + gama * Qn[An]- Qt[At]), x)
            At = An
    return w

def list_multiply(lam, a):
    res = []
    for i in a:
        res.append(i*lam)
    return res


def ep_greedy(Q, actions, ep):
    a_star = actions[0]
    for action in actions:
        a_star = action if Q[action] > Q[a_star] else a_star
    prob_s = {}
    for action in actions:
        prob_s[action] = 1-ep+ep/len(actions) if action == a_star else ep/len(actions)
    rand_float = np.random.random()
    acc = 0.0
    for action in actions:
        acc += prob_s[action]
        if rand_float < acc:
            return action

def q_function(s, w):
    [s1, s2] = s
    x = ([1, s1, s2, s1*s2, s1*s1, s2*s2, s1*s2*s2, s1*s1*s2, s1*s1*s2*s2])
    Q = [np.dot(x, w[0]), np.dot(x, w[1]), np.dot(x, w[2])]
    return Q
print(sg_sarsa())