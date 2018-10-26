import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import chainerrl


class QFunction(chainer.Chain):
    def __init__(self, ch_in=14, ch_h=32, n_out=4):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(ch_in, ch_h, 2)
            self.conv2 = L.Convolution2D(ch_h, ch_h, 1)
            self.conv3 = L.Convolution2D(ch_h, ch_h, 1)
            self.fc1 = L.Linear(ch_h * 3 * 3, n_out)

    def __call__(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = self.fc1(h3)

        return chainerrl.action_value.DiscreteActionValue(h4)
