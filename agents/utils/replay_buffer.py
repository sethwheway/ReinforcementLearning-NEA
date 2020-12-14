from collections import deque
from random import sample


class ReplayBuffer(deque):
    def __init__(self, maxlen):
        super().__init__(maxlen=maxlen)

    def sample(self, batch_size):
        return sample(self, k=batch_size)
