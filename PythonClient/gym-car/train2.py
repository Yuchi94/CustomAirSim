import gym
import numpy as np
import gym_car
from network2 import *
from memory import *
from noise import *
from skimage.transform import pyramid_gaussian
from skimage.color import rgb2grey
import matplotlib.pyplot as plt
from time import sleep
from replay_buffer import ReplayBuffer

IMAGE_SIZE = [72, 128, 1]
MEMORY_SIZE = 250000
NUM_AGENTS = 4
MAX_EPISODES = 1000
TRACE_LENGTH = 5
BURN_IN = 100

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class Runner():
    def __init__(self,
                 env,
                 actor_layers,
                 critic_a_layers,
                 critic_s_layers,
                 lstm_layers,
                 trace_length,
                 actor_lr,
                 critic_lr,
                 tau,
                 batch_size,
                 gamma):

        self.env = gym.make(env)

        self.ddpg = DDPG(actor_layers,
                     critic_a_layers,
                     critic_s_layers,
                     lstm_layers,
                     trace_length,
                     actor_lr,
                     critic_lr,
                     IMAGE_SIZE,
                     [2],
                     tau,
                     batch_size)

        self.ddpg.buildNetwork()

        self.memory = Memory(IMAGE_SIZE, MEMORY_SIZE, [2], TRACE_LENGTH, data_type=np.uint8)
        self.noise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(2))
        self.gamma = gamma
        self.batch_size = batch_size

    def train(self, num_episodes, burn_in):
        self.ddpg.initTargetNetwork()
        self.burnIn(burn_in)

        for i in range(num_episodes):
            states = self.processStates(self.env.reset())
            state_history = [[state.copy() for i in range(TRACE_LENGTH)] for state in states]
            episode_reward = 0

            for x in range(MAX_EPISODES):
                actions = []

                for j in range(NUM_AGENTS):
                    action = self.ddpg.getOnlineActionProb(np.stack(state_history[j], axis=1))[0]
                    actions.append(action + self.noise())
                # print(actions)
                next_states, rewards, terminal, infos = self.env.step(actions)
                next_states = self.processStates(next_states)

                for j in range(NUM_AGENTS):
                    self.memory.append(np.stack(state_history[j], axis = 1), actions[j], rewards[j], terminal, next_states[j], infos[j])

                train_states, train_actions, train_next_states, train_rewards, train_terminal, train_infos = self.memory.sample_batch(self.batch_size)
                # plt.imshow(train_states[0][0,:,:,0], cmap='gray')
                # plt.show()

                #train critic
                target_q = self.ddpg.getTargetStateValues(train_next_states, self.ddpg.getTargetActionProb(train_next_states, i), i)
                y_i = train_rewards + np.invert(train_infos) * self.gamma * target_q
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag="Target predict values", simple_value=np.average(y_i)), ])
                self.ddpg.writer.add_summary(summary, i)

                self.ddpg.trainCritic(train_states, train_actions, y_i)

                #train actor
                action_prob = self.ddpg.getOnlineActionProb(train_states, i)
                grads = self.ddpg.getActionGradients(train_states, action_prob)
                self.ddpg.trainActor(train_states, grads[0])

                #update target network
                self.ddpg.updateTargetNetwork()

                for j in range(NUM_AGENTS):
                    state_history[j][:-1] = state_history[j][1:]
                    state_history[j][-1] = next_states[j]
                #print(rewards)
                episode_reward += np.average(rewards)
                if terminal:
                    print(episode_reward / x)
                    summary = tf.Summary(value=[tf.Summary.Value(tag="Average episode reward", simple_value=episode_reward),])
                    self.ddpg.writer.add_summary(summary, i)
                    break


    def burnIn(self, burn_in):

        states = self.processStates(self.env.reset())
        state_history = [[state.copy() for i in range(TRACE_LENGTH)] for state in states]

        for i in range(burn_in):
            actions = []

            for j in range(NUM_AGENTS):
                actions.append(self.ddpg.getOnlineActionProb(np.stack(state_history[j], axis=1))[0] + self.noise())

            next_states, rewards, terminal, infos = self.env.step(actions)
            next_states = self.processStates(next_states)

            for j in range(NUM_AGENTS):
                self.memory.append(np.stack(state_history[j], axis = 1), actions[j], rewards[j], terminal, next_states[j], infos[j])

            if terminal:
                states = self.processStates(self.env.reset())
                state_history = [[state.copy() for i in range(TRACE_LENGTH)] for state in states]

            else:
                for j in range(NUM_AGENTS):
                    state_history[j][:-1] = state_history[j][1:]
                    state_history[j][-1] = next_states[j]



    def processStates(self, states):
        # return states
        return [self.processState(s) for s in states]


    def processState(self, state):
        img = rgb2grey(state) * 255
        return tuple(pyramid_gaussian(img, downscale=2, max_layer = 1))[1][None,:,:,None]

if __name__ == '__main__':
    runner = Runner(env = 'carsim-v0',
                 actor_layers = [400, 300],
                 critic_a_layers = [400, 300],
                 critic_s_layers = [300],
                 lstm_layers = [10],
                 trace_length= TRACE_LENGTH,
                 actor_lr = 0.0001,
                 critic_lr = 0.001,
                 tau = 0.001,
                 batch_size = 64,
                 gamma = 0.99)
    runner.train(50000, 64)
    #runner.train2()
