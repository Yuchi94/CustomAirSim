import gym
import numpy as np
import gym_car
from network import *
from memory import *
from noise import *
from skimage.transform import pyramid_gaussian
from skimage.color import rgb2grey
import matplotlib.pyplot as plt
from time import sleep
IMAGE_SIZE = [72, 128, 1]
MEMORY_SIZE = 250000
NUM_AGENTS = 4
MAX_EPISODES = 1000
TRACE_LENGTH = 5
BURN_IN = 100

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
        self.noise = [OrnsteinUhlenbeckActionNoise(mu = np.zeros(2), sigma=0.1)] * NUM_AGENTS
        self.gamma = gamma
        self.batch_size = batch_size

    def train(self, num_episodes, burn_in):
        self.ddpg.initTargetNetwork()
        self.burnIn(burn_in)

        for i in range(num_episodes):
            states = self.processStates(self.env.reset())
            state_history = [[s.copy() for i in range(TRACE_LENGTH)] for s in states]
            collision_history = [False] * 4
            episode_reward = 0

            for x in range(MAX_EPISODES):
                actions = []

                for j in range(NUM_AGENTS):
                    if not collision_history[j]:
                        actions.append(self.ddpg.getOnlineActionProb(np.stack(state_history[j], axis = 1))[0] + self.noise[j]())
                    else:
                        actions.append([0,0])

                next_states, rewards, terminal, infos = self.env.step(actions)
                next_states = self.processStates(next_states)

                for j in range(NUM_AGENTS):
                    if not collision_history[j]:
                        self.memory.append(np.stack(state_history[j], axis = 1), actions[j], rewards[j], terminal, next_states[j], infos[j])

                train_states, train_actions, train_next_states, train_rewards, train_terminal, train_infos = self.memory.sample_batch(self.batch_size)

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

                for k in range(NUM_AGENTS):
                    state_history[k][:-1] = state_history[k][1:]
                    state_history[k][-1] = next_states[k]

                episode_reward += np.average(rewards)

                if terminal:
                    summary = tf.Summary(value=[tf.Summary.Value(tag="Average episode reward", simple_value=episode_reward),])
                    self.ddpg.writer.add_summary(summary, i)
                    print(episode_reward / x)
                    break

            if i % 50 == 0:
                self.ddpg.saveNetwork("/media/yoshi/Seagate Expansion Drive/drlmodel/ep" + str(i))

    def burnIn(self, burn_in):
        states = self.processStates(self.env.reset())
        state_history = [[s.copy() for i in range(TRACE_LENGTH)] for s in states]

        for i in range(burn_in):
            actions = []

            for j in range(NUM_AGENTS):
                actions.append(self.ddpg.getOnlineActionProb(np.stack(state_history[j], axis=1))[0] + self.noise[j]())

            next_states, rewards, terminal, infos = self.env.step(actions)
            next_states = self.processStates(next_states)
            # plt.imshow(next_states[0][0,:,:,0], cmap='gray')
            # plt.show()

            for j in range(NUM_AGENTS):
                self.memory.append(np.stack(state_history[j], axis = 1), actions[j], rewards[j], terminal, next_states[j], infos[j])

            sleep(0.1)
            if terminal:
                states = self.processStates(self.env.reset())
                state_history = [[s.copy() for i in range(TRACE_LENGTH)] for s in states]

            else:
                for k in range(NUM_AGENTS):
                    state_history[k][:-1] = state_history[k][1:]
                    state_history[k][-1] = next_states[k]



    def processStates(self, states):
        return [self.processState(s) for s in states]


    def processState(self, state):
        img = rgb2grey(state) * 255
        return tuple(pyramid_gaussian(img, downscale=2, max_layer = 1))[1][None,:,:,None]

if __name__ == '__main__':
    runner = Runner(env = 'carsim-v0',
                 actor_layers = [1000, 1000],
                 critic_a_layers = [1000, 1000],
                 critic_s_layers = [1000, 1000],
                 lstm_layers = [1000],
                 trace_length= TRACE_LENGTH,
                 actor_lr = 1e-4,
                 critic_lr = 1e-3,
                 tau = 0.001,
                 batch_size = 64,
                 gamma = 0.99)

    runner.train(50000, BURN_IN)
