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
    parser.add_argument('--grid_dim', type=int, default=4,
                help='dimension of envioronment')
    parser.add_argument('--episode', '-e', type=int, default=40000,
                help='number of episodes to iterate')
    parser.add_argument('--target', '-t', type=int, default=2048,
                help='target panel')
    parser.add_argument('--max_step', type=int, default=500,
                help='max steps')
    parser.add_argument('--feature_type', type=str, default='normalized',
                help='shape kind of vectors to use')
    args = parser.parse_args()
    print(args)

    # Set up environment
    env = TTFE(args.grid_dim)
    ra = RandomActor(env)

    # Set up Q-function
    if args.feature_type == 'normalized':
        q_func = QFunction(ch_in=1, ch_h=128, n_out=4)
    elif args.feature_type == 'layered':
        q_func = QFunction(ch_in=14, ch_h=128, n_out=4)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        q_func.to_gpu()
        print('GPU: {}'.format(args.gpu))
    xp = np if args.gpu < 0 else cuda.cupy

    optimizer = chainer.optimizers.Adam(0.0005)
    optimizer.setup(q_func)

    s_eps = 1.0
    e_eps = 0.3
    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=s_eps, end_epsilon=e_eps, decay_steps=100000, random_action_func=ra.random_action_func)
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    print('epsilon: decays {} to {} in {} steps'.format(s_eps, e_eps, 100000))

    gamma = 0.99
    agent = chainerrl.agents.DoubleDQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=500, update_interval=1, target_update_interval=100)
    print('gamma: {}'.format(gamma))

    # Training loop
    n_episodes = args.episode
    max_steps = args.max_step
    display_interval = 100
    snapshot_interval = 5000
    # log
    score = 0
    win = 0
    actions = {0: 0, 1: 0, 2: 0, 3: 0}
    for i in range(1, n_episodes + 1):
        env.reset()
        score_bef = 0
        reward = 0
        t = 0
        while t < max_steps:
            st = env.shape_state_for_train(args.feature_type)
            action = agent.act_and_train(st, reward)
            actions[action] += 1
            env.move(action)

            # if env.score != score_bef:
            #     reward = 1
            # else:
            #     reward = 0
            # score_bef = env.score

            if args.target in env.state:
                reward = 1
                win += 1
                break

            if env.isGameOver():
                reward = -1
                break

            t += 1

        st = env.shape_state_for_train(args.feature_type)
        agent.stop_episode_and_train(st, reward, True)

        score += env.score

        if i % display_interval == 0:
            print("ep: {}, rnd: {}, stats: {} eps: {}\n win: {}, score: {}, action: {}".format(
                i, ra.cnt, agent.get_statistics(), agent.explorer.epsilon, win, score / display_interval, actions))
            ra.cnt = 0
            score = 0
            win = 0
            actions = {0: 0, 1: 0, 2: 0, 3: 0}

        if i % snapshot_interval == 0:
            agent.save('model/agent_' + str(i))

if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    train()