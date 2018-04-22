import numpy as np

class Memory():
    """
    Memory class to hold past experience for experience replay
    """

    def __init__(self, state_size, memory_size, action_size, trace_length, data_type=np.float_):
        """
        :param state_size: Size of the current and input state
        :param memory_size:     Buffer size for experience. Default 500
        :param data_type:       Data type for state. Downgrade to save memory. Default float
        """
        self.memory_size = memory_size
        self.states = np.zeros(([memory_size] + [trace_length + 1] + state_size), dtype=data_type)
        self.actions = np.zeros([memory_size] + action_size)
        self.next_states = np.zeros(([memory_size] + state_size), dtype=data_type)
        self.rewards = np.zeros((memory_size, 1))
        self.terminals = np.full((memory_size, 1), True, dtype=bool)
        self.infos = np.full((memory_size, 1), True, dtype=bool)
        self.n_elements = 0
        self.index = 0

    def sample_batch(self, batch_size=32):
        """
        :param batch_size:      batch size
        :return:                states, actions, next_states, rewards,  terminal status
        """
        rand_ind = [np.random.randint(0, min(self.n_elements, self.memory_size)) for i in range(batch_size)]
        return self.states[rand_ind, :-1, :], \
               self.actions[rand_ind, :], \
               self.states[rand_ind, 1:, :], \
               self.rewards[rand_ind, :], \
               self.terminals[rand_ind, :], \
                self.infos[rand_ind, :]

    def append(self, state, action, reward, terminal, next_state, infos):
        """
        Adds to experience replay
        :param state:           current state
        :param action:          current action
        :param next_state:      next state
        :param reward:          reward for next state
        :param terminal:        Boolean not of whether next state is terminal
        :return:                None
        """
        self.states[self.index, :-1, :] = state
        self.states[self.index, -1, :] = next_state
        self.actions[self.index, :] = action
        self.rewards[self.index, :] = reward
        # self.next_states[self.index, :] = next_state
        self.terminals[self.index, :] = terminal
        self.infos[self.index, :] = infos
        self.index = (self.index + 1) % self.memory_size
        self.n_elements = min(self.n_elements + 1, self.memory_size)