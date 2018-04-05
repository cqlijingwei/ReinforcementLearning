#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:56:24 2018

@author: liuxiaoqin
"""

from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action, move):
        """Play a move, and then have a random agent play the next move."""
        if move==1:
            state, status, done = self.step(action)
            if not done and self.turn == 2:
                state, s2, done = self.random_step()
                if done:
                    if s2 == self.STATUS_WIN:
                        status = self.STATUS_LOSE
                    elif s2 == self.STATUS_TIE:
                        status = self.STATUS_TIE
                    else:
                        raise ValueError("???")
            return state, status, done
        else:
            state, status, done = self.random_step()
            if done:
                if status == self.STATUS_WIN:
                    status = self.STATUS_LOSE
            else:
                state, status, done = self.step(action)
            return state, status, done

    def play_against_itself(self, policy, action):
        """Play a move, and then play a move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            action, logprob = select_action(policy, state)
            state, s2, done = self.step(action)
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done


class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=64, output_size=9):
        super(Policy, self).__init__()
        self.neural_network = nn.Sequential(
                nn.Linear(input_size,hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
                )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        z = self.neural_network(x)
        return self.softmax(z)

def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr)
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ...

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    G = []
    G.append(0)

    for i in range(0,len(rewards)):
        G_t = rewards[len(rewards)-1-i]+gamma*G[0]
        G.insert(0,G_t)
    G.pop()

    return G

def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  : 1,
            Environment.STATUS_INVALID_MOVE: -10000,
            Environment.STATUS_WIN         : 10,
            Environment.STATUS_TIE         : 0,
            Environment.STATUS_LOSE        : -10
    }[status]



def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)


def train_against_random(policy, env, gamma=0.75, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0

    epi = []
    aveRet =[]
    winRateFirst = []
    winRateSecond =[]

    for i_episode in count(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        num = random.random()
        if (num<0.5):
            move=1
        else:
            move=2
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action,move)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            epi.extend([i_episode])
            aveRet.extend([running_reward / log_interval])
            win={}
            win[1]=0.0
            win[2]=0.0
            for i in range(100):
                for m in [1,2]:
                    state = env.reset()
                    done = False

                    while not done:
                        action, logprob = select_action(policy, state)
                        state, status, done = env.play_against_random(action, m)

                    if status == env.STATUS_WIN:
                        win[m] += 1
                    if i%20==0 and i_episode==160000:
                        env.render()

            winRateFirst.extend([win[1] / 100.0])
            winRateSecond.extend([win[2] / 100.0])

            print(
                'Episode {}\tAverage return: {:.2f}\nFirst move:\tGames Won: {}\tSecond move:\tGames Won: {}\t'.format(
                    i_episode, running_reward / log_interval, win[1], win[2]))
            running_reward = 0

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(), "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0:  # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if i_episode == 160000:
            plt.figure()
            plt.plot(epi, aveRet, label='average return')
            plt.xlabel("Episode")
            plt.ylabel("Average return")
            plt.title("Training curve of part1 A4 bonus")
            plt.savefig("part1A4bonusAverageReturn.png")

            plt.figure()
            plot2, = plt.plot(epi, winRateFirst, label='win rate going first')
            plot3, = plt.plot(epi, winRateSecond, label='win rate going second')

            plt.legend([plot2, plot3], ['win rate going first', 'win rate going second'])
            plt.title('part1A4bonus')
            plt.xlabel('Episode')
            plt.ylabel('Win rates going fisrt and second')
            plt.savefig('part1A4bonusWinRates.png')

            return







#part2
def generate_next_step_itself(policy, env, move="first", games=100):
    """Play games against random and return number of games won, lost or tied"""
    games_won, games_lost, games_tied, invalid_moves = 0, 0, 0, 0

    for i in range(games):
        state = env.reset()
        done = False

        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action, move)
            invalid_moves += (
                1 if status == env.STATUS_INVALID_MOVE else 0)  # env.render()

        if status == env.STATUS_WIN:
            games_won += 1
        elif status == env.STATUS_LOSE:
            games_lost += 1
        else:
            games_tied += 1

    return games_won, games_lost, games_tied, invalid_moves




def train_against_itself(policy, env, gamma=0.75, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000,
        gamma=0.9)
    running_reward = 0

    episodes = []
    returns = []

    win_rate_first = []
    loss_rate_first = []
    tie_rate_first = []

    win_rate_second = []
    loss_rate_second = []
    tie_rate_second = []

    for i_episode in count(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_itself(policy, action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            episodes.extend([i_episode])
            returns.extend([running_reward / log_interval])
            won_first, lost_first, tied_first, invalid_moves_first = generate_next_step_itself(
                policy, env, "first")
            won_second, lost_second, tied_second, invalid_moves_second = generate_next_step_itself(
                policy, env, "second")

            win_rate_first.extend([won_first / 100.0])
            loss_rate_first.extend([lost_first / 100.0])
            tie_rate_first.extend([tied_first / 100.0])

            win_rate_second.extend([won_second / 100.0])
            loss_rate_second.extend([lost_second / 100.0])
            tie_rate_second.extend([tied_second / 100.0])


            print('Episode {}\tAverage return: {:.2f}'.format(i_episode, running_reward / log_interval))
            running_reward = 0

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(), "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0:  # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if i_episode == 160000:

            fig = plt.figure()
            plt.plot(episodes, returns)
            plt.xlabel("episodes")
            plt.ylabel("average return")
            plt.title("Performance of self-play model")
            plt.savefig("part2/selfplay_model.png")

            plt.figure()
            plot2, = plt.plot(episodes, win_rate_first, label='win rate going first')
            plot3, = plt.plot(episodes, loss_rate_first, label='win rate going second')

            plt.legend([plot2, plot3],
                       ['win rate going first', 'win rate going second'])
            plt.title('self-play model win rate performance')
            plt.xlabel('Episode')
            plt.ylabel('Win rates going fisrt and second')
            plt.savefig('part2/part2performance.png')

            return


def game_with_random_adversary(policy, env, move="first", games = 50):
    games_won, games_lost, games_tied, invalid_moves = 0, 0, 0, 0

    for i in range(games):
        state = env.reset()
        done = False
        print("Game: %s"%i)

        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action, move)
            invalid_moves += (1 if status == env.STATUS_INVALID_MOVE else 0)
            env.render()

        if status == env.STATUS_WIN: games_won += 1
        elif status == env.STATUS_LOSE: games_lost += 1
        else: games_tied += 1

    return games_won, games_lost, games_tied, invalid_moves

def game_with_itself(policy, env, games = 50):
    games_won, games_lost, games_tied, invalid_moves = 0, 0, 0, 0

    for i in range(games):
        state = env.reset()
        done = False
        print("Game: %s"%i)

        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_itself(policy, action)
            invalid_moves += (1 if status == env.STATUS_INVALID_MOVE else 0)
            env.render()

        if status == env.STATUS_WIN: games_won += 1
        elif status == env.STATUS_LOSE: games_lost += 1
        else: games_tied += 1

    return games_won, games_lost, games_tied, invalid_moves


if __name__ == '__main__':
    import sys
    policy = Policy()
    env = Environment()

    if len(sys.argv) == 1:
        # `python tictactoe.py` to train the agent
        train_against_random(policy, env)
    else:
        # `python tictactoe.py <ep>` to print the first move distribution
        # using weightt checkpoint at episode int(<ep>)

        print ("haha")

        # to run part2, need to run 'python tictactoe.py 160000'
        ep = int(sys.argv[1])
        load_weights(policy, ep)

        train_against_itself(policy, env)

        # train_against_itself(policy, env)
        game_with_random_adversary(policy, env, "first", 2)
        game_with_random_adversary(policy, env, "second", 3)
        game_with_itself(policy, env, 5)