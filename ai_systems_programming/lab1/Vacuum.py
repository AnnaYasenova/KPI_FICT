import numpy as np
import Environs
import Agency

def score_energy(scores, agents, state):
    for i, agent in enumerate(agents):
        if agent.action != 'move':
            scores[i] += 1
        if agent.action == 'suck':
            scores[i] += 2
    return scores

def score_suck(scores, agents, state):
    for i, agent in enumerate(agents):
        scores[i] += state.sum()
    return scores

def f_scoring(scores, agents, state):
    for i, agent in enumerate(agents):
        if agent.action != 'powerdown':
            scores[i] -= 1
        if agent.action == 'suck' and state[agent.loc[0], agent.loc[1]] == 1:
            scores[i] += 100
        if agent.action == 'powerdown' and (agent.loc-agent.home_loc).sum()!=0:
            scores[i] -= 1000
    return scores


def f_homeless(scores, agents, state):
    for i, agent in enumerate(agents):
        if agent.action != 'powerdown':
            scores[i] -= 1
        if agent.action == 'suck' and state[agent.loc[0], agent.loc[1]] == 1:
            scores[i] += 100
    return scores

def f_action(agents, state):
    for agent in agents:
        if agent.action == 'suck':
            state[agent.loc[0], agent.loc[1]] = 0
        elif (agent.action == 'move'
                and min(agent.loc + agent.bearing) >= 0
                and min(state.shape - agent.loc - agent.bearing) > 0
                and state[agent.loc[0] + agent.bearing[0],
                            agent.loc[1] + agent.bearing[1]] != 2):
            agent.loc += agent.bearing
        elif agent.action == 'rturn':
            agent.bearing = [agent.bearing[1], -agent.bearing[0]]
        elif agent.action == 'lturn':
            agent.bearing = [-agent.bearing[1], agent.bearing[0]]
    return state


def run_eval_environment(state, update, agents, performance, N):
    scores = [0 for _ in range(len(agents))]
    res = dict()
    for i in range(N[-1] + 1): #while i < N: #any([agent.action != 'powerdown' for agent in agents]):

        for agent in agents:
            #print(state.sum())
            agent.get_percept(state)
            agent.program()
        scores = performance(scores, agents, state)
        state = update(agents, state)
        if i in N:
            res[i] = scores[0]
            #res.append({i: scores[0]})
            #print(i, scores)
    return res#scores, res
N_arr = [100, 500, 1000, 2000, 5000, 10000]
'''
for S in Environs.MiniMax2Package():
    agents = [Agency.TrivialTableLookupAgent(np.array([0,0]), np.array([0,0]), 'E')]
    print run_eval_environment(S.grid, f_action, agents, f_scoring)
'''
import pandas as pd
agents = [
    Agency.EmptyRoomInternalStateReflexAgent(np.array([0,0]), np.array([0,0]), 'E'),
    Agency.TrivialTableLookupAgent(np.array([0,0]), np.array([0,0]), 'E'),
    Agency.BasicReflexAgent(np.array([0,0]), np.array([0,0]), 'E')
]
functions = [score_energy, score_suck]
amount_of_calcs = 100

df = pd.DataFrame()
for tmp in range(amount_of_calcs):
    # print()
    # print(tmp)
    for func in [functions[0]]:
        agents = [
            #Agency.EmptyRoomInternalStateReflexAgent(np.array([0, 0]), np.array([0, 0]), 'E'),
            #Agency.TrivialTableLookupAgent(np.array([0, 0]), np.array([0, 0]), 'E'),
            Agency.BasicReflexAgent(np.array([0, 0]), np.array([0, 0]), 'E')
        ]
        for agent in agents:
            # print(agent)
            # print(func)
            df = df.append(pd.DataFrame.from_records(
                run_eval_environment(Environs.LimitedRandom().grid, f_action, [agent], func, N_arr),
                index=[0]
            ))


print(df.mean(axis=0))