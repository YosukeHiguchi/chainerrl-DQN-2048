import random
import numpy as np


ACTION_MEANING = {
    0: "UP",
    1: "RIGHT",
    2: "LEFT",
    3: "DOWN"
}

class TTFE():
    def __init__(self, size):
        self.size = size
        self.state = np.array([0] * (size * size), dtype=np.uint16).reshape(size, size)
        self.score = 0

        self.spawn()
        self.spawn()

    def reset(self):
        self.__init__(self.size)

    def spawn(self):
        empty = np.where(self.state == 0)
        r = random.randint(0, len(empty[0]) - 1)
        self.state[empty[0][r]][empty[1][r]] = 2 if random.random() > 0.2 else 4

    def move(self, action):
        if not action in ACTION_MEANING.keys():
            return False

        state_old = np.array(self.state)

        if action == 0:
            self.state = self.state.transpose()
        elif action == 1:
            self.state = self.state[:, ::-1]
        elif action == 2:
            self.state = self.state
        elif action == 3:
            self.state = self.state[::-1].transpose()

        for i, col in enumerate(self.state):
            self.state[i] = col[np.append(np.where(col != 0), np.where(col == 0))]
            for j in range(self.size - 1):
                if self.state[i][j] == self.state[i][j + 1]:
                    self.state[i][j] *= 2
                    self.state[i][j + 1]  = 0
                    self.score += self.state[i][j]
            self.state[i] = col[np.append(np.where(col != 0), np.where(col == 0))]

        if action == 0:
            self.state = self.state.transpose()
        elif action == 1:
            self.state = self.state[:, ::-1]
        elif action == 2:
            self.state = self.state
        elif action == 3:
            self.state = self.state.transpose()[::-1]

        if np.all(self.state == state_old):
            return False
        else:
            self.spawn()
            return True

    def get_available_action(self):
        actions = []
        state_tmp = np.array(self.state)
        for action in ACTION_MEANING.keys():
            if self.move(action):
                actions.append(action)
            self.state = np.array(state_tmp)

        return random.choice(actions)

    def shape_state_for_train(self, panel=14):
        # st = np.empty((0, 4, 4), dtype=np.float32)
        # for i in range(panel):
        #     idx = np.where(self.state == 2 ** (i + 1))
        #     layer = np.array([0] * 16).reshape(4, 4)
        #     layer[idx[0], idx[1]] = 1
        #     st = np.append(st, layer[np.newaxis, :, :].astype(np.float32), axis=0)

        st = np.array(self.state, dtype=np.float32)
        st[np.where(st != 0)] = np.log2(st[np.where(st != 0)]) / 14
        return st[np.newaxis, :, :]

    def key_to_action(self, key):
        KEY_TO_ACTION = {
            'w': 0,
            'd': 1,
            'a': 2,
            's': 3
        }

        if not key in KEY_TO_ACTION.keys():
            return -1

        return KEY_TO_ACTION[key]

    def isGameOver(self):
        empty = np.where(self.state == 0)
        if len(empty[0]) != 0:
            return False

        state_tmp = np.array(self.state)
        for action in ACTION_MEANING.keys():
            if self.move(action):
                self.state = np.array(state_tmp)
                return False
            self.state = np.array(state_tmp)

        return True

    def show_CUI(self):
        print('\033[H\033[J') # clear
        import re
        print('SCORE: {:5d}'.format(self.score))

        b_col = ' '
        b_row = '|'
        for i in range(self.size):
            b_col += '----- '
            b_row += '     |'

        print(b_col)
        for col in self.state:
            print(b_row)
            num = '|'
            arr = [re.sub('^[0]', '', str(i)) for i in col]
            for a in arr:
                num += '{:^5s}|'.format(a)
            print('{}\n{}\n{}'.format(num, b_row, b_col))

class RandomActor():
    def __init__(self, env):
        self.env = env
        self.cnt = 0

    def random_action_func(self):
        self.cnt += 1
        return self.env.get_available_action()

if __name__ == '__main__':
    env = TTFE(4)
    env.show_CUI()
    while True:
        print('>>', end='')
        a = input()
        if env.move(env.key_to_action(a)):
            env.show_CUI()
        if env.isGameOver():
            print('You suck')
            break
