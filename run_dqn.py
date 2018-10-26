import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np

import chainer
from chainer import cuda
import chainerrl

from net import QFunction
from environment import TTFE, RandomActor


def train():
    parser = argparse.ArgumentParser(description='Solve 2048 by DQN')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--episode', '-e', type=int, default=40000,
                help='number of episodes to iterate')
    parser.add_argument('--panel', '-p', type=int, default=14,
                help='number of panel kinds')
    parser.add_argument('--target', '-t', type=int, default=2048,
                help='target panel')
    args = parser.parse_args()


    # Set up environment
    env = TTFE()
    ra = RandomActor(env)

    # Set up Q-function
    q_func = QFunction(ch_in=args.panel, ch_h=32, n_out=4)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        q_func.to_gpu()
        print('GPU {}\n'.format(args.gpu))
    xp = np if args.gpu < 0 else cuda.cupy

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(q_func)

    gamma = 0.95

    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.1, decay_steps=100000, random_action_func=ra.random_action_func)
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

    agent = chainerrl.agents.DoubleDQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=500, update_interval=1, target_update_interval=100)

    # Training loop
    n_episodes = args.episode
    max_steps = 1000
    score = 0
    win = 0
    for i in range(1, n_episodes + 1):
        env.__init__()
        reward = 0
        t = 0
        while t < max_steps:
            st = env.shape_state_for_train(args.panel)
            action = agent.act_and_train(st, reward)
            env.move(action)

            if args.target in env.state:
                win += 1
                reward = 1
                break

            if env.isGameOver():
                reward = -1
                break

            # env.show_CUI()

            t += 1

        st = env.shape_state_for_train(args.panel)
        agent.stop_episode_and_train(st.copy(), reward, True)

        score += env.score
        if i % 100 == 0:
            print("ep: {}, rnd: {}, stats: {}\neps: {}, win: {}, score: {}".format(
                i, ra.cnt, agent.get_statistics(), agent.explorer.epsilon, win, score / 10))
            score = 0
            win = 0

if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    train()