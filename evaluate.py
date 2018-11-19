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
from environment import TTFE


class MyAgent():
    def __init__(self, q_func):
        self.q_func = q_func

    def act(self, x):
        xp = self.q_func.xp
        x = x[np.newaxis, :, :, :]
        post = self.q_func(xp.asarray(x))

        actions = xp.argsort(post.q_values.data)[-1, :][::-1]
        return actions.tolist()

def evaluate():
    parser = argparse.ArgumentParser(description='Solve 2048 by DQN')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--episode', '-e', type=int, default=100,
                help='number of episodes to iterate')
    parser.add_argument('--feature_type', type=str, default='normalized',
                help='shape kind of vectors to use')
    parser.add_argument('--model_path', type=str,
                help='path to the model file')
    args = parser.parse_args()
    print(args)

    # Set up environment
    env = TTFE()

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

    chainer.serializers.load_npz(args.model_path, q_func)

    agent = MyAgent(q_func)

    # Training loop
    n_episodes = args.episode
    win = 0
    score = 0
    num = {'2': 0, '4': 0, '8': 0, '16': 0, '32': 0, '64': 0,
           '128': 0, '256': 0, '512': 0, '1024': 0, '2048': 0, '4096': 0}
    for i in range(1, n_episodes + 1):
        print('{:2d}/{:2d}'.format(i, n_episodes), end='\r')
        env.reset()

        while True:
            state = env.shape_state_for_train(args.feature_type)
            choices = agent.act(state)
            for action in choices:
                if env.isMovable(action):
                    reward = env.move(action)
                    break

            if env.isGameOver():
                break

        for i in range(1, int(np.log2(np.max(env.state))) + 1):
            num[str(2 ** i)] += 1
        score += env.score

    print('win: {}, score: {}\n{}'.format(win, score / n_episodes, num))


if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    evaluate()