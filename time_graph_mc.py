import sys
import gym
import numpy as np
from collections import defaultdict


env = gym.make('Blackjack-v0')

def play_episode(env,Q):
    episode = []
    state = env.reset()
    while True:
        probs = [0.2, 0.8] if max(Q[state][0],Q[state][1]) == Q[state][1] else [0.8, 0.2]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

def update_Q_mc(episode, Q,returns_sum, N, gamma=1.0):
    for s, a, r in episode:
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == s)
            G = sum([x[2]*(gamma**i) for i,x in enumerate(episode[first_occurence_idx:])])
            returns_sum[s][a] += G
            N[s][a] += 1.0
            Q[s][a] = returns_sum[s][a] / N[s][a]



def rewards_predict_mc(env, num_episodes, gamma=1.0):
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    all_rewards = []
    for i_episode in range(1, num_episodes+1):

        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            update_Q_mc(episode, Q, returns_sum, N)
            
        episode = play_episode(env,Q)
        all_rewards.append(episode[len(episode)-1][2])

            
    return all_rewards






def update_Q_td(episode, Q,returns_sum, N, gamma=1.0,alpha = 1.0):
    for s, a, r in episode:
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == s)
            if first_occurence_idx + 2 < len(episode):
                N[s][a] += 1.0
                next_state = episode[first_occurence_idx +1][0]
                next_reward = episode[first_occurence_idx +1][2]
                returns_sum[s][a] += alpha*(r + gamma*max(Q[next_state][0],Q[next_state][1]) - Q[s][a])
                Q[s][a] += returns_sum[s][a]/N[s][a]
            else:
                N[s][a] += 1.0
                returns_sum[s][a] += r
                Q[s][a] = returns_sum[s][a]/N[s][a]  




def rewards_predict_td(env, num_episodes, gamma=1.0):
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    all_rewards = []
    for i_episode in range(1, num_episodes+1):

        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            update_Q_td(episode, Q, returns_sum, N)
            
        episode = play_episode(env,Q)
        all_rewards.append(episode[len(episode)-1][2])

            
    return all_rewards


td_rewards = rewards_predict_td(env,100000,1.0)
#need to update policy 

mc_rewards = rewards_predict_mc(env, 100000, 1.0)

def average(n_rewards):
    av_reward = []
    av_1000_terms = 0
    for i in range(len(n_rewards)):
        av_1000_terms += n_rewards[i]/1000
        if i%1000 == 0:
            av_reward.append(av_1000_terms)
            av_1000_terms = 0
    return av_reward



from matplotlib import pyplot as plt
plt.plot(average(mc_rewards), label = 'MC')
plt.plot(average(td_rewards), label = 'TD(1)')
plt.title('Learning rates: MC vs TD(1)')
plt.xlabel('Number of episodes')
plt.ylabel('Average reward')
plt.legend()
plt.show()



