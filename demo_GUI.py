import os
import sys
import time

import numpy as np
import argparse
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import chainer
import chainerrl

from net import QFunction
from environment import TTFE

ACTIONS = [Keys.UP, Keys.RIGHT, Keys.LEFT, Keys.DOWN]

class MyAgent():
    def __init__(self, q_func):
        self.q_func = q_func

    def act(self, x):
        xp = self.q_func.xp
        x = x[np.newaxis, :, :, :]
        post = self.q_func(xp.asarray(x))

        actions = xp.argsort(post.q_values.data)[-1, :][::-1]
        return actions.tolist()

def main():
    parser = argparse.ArgumentParser(description='Solve 2048 by DQN')
    parser.add_argument('--model_path', type=str,
                help='path to the model file')
    args = parser.parse_args()

    driver_path = './chromedriver'
    URL = 'https://play2048.co/'

    print('Initializing...')
    driver = webdriver.Chrome(driver_path)
    driver.get(URL)

    q_func = QFunction(ch_in=1, ch_h=128, n_out=4)

    chainer.serializers.load_npz(args.model_path, q_func)
    agent = MyAgent(q_func)

    env = TTFE(4)

    while True:
        print('>> ', end='')
        N = input()
        if N == 's':
            print('Start!')
            break

    driver.find_element_by_class_name('restart-button').click()
    driver.find_element_by_class_name('notice-close-button').click()

    def get_state():
        state = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        tiles = driver.find_elements_by_class_name('tile')

        for tile in tiles:
            attr = tile.get_attribute('class').split(' ')
            tile_num = int(attr[1].split('-')[1])
            tile_x = int(attr[2].split('-')[2]) - 1
            tile_y = int(attr[2].split('-')[3]) - 1

            state[tile_y][tile_x] = tile_num

        return np.array(state)

    while True:
        env.state = get_state()
        state = env.shape_state_for_train('normalized')
        choices = agent.act(state)
        env.show_CUI()
        for action in choices:
            if env.isMovable(action):
                driver.find_element_by_css_selector('body').send_keys(ACTIONS[action])
                break

        if env.isGameOver():
            break

        time.sleep(0.1)

    driver.quit()

if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    main()
