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
IMAGE_SIZE = [3]
MEMORY_SIZE = 250000
NUM_AGENTS = 4
MAX_EPISODES = 50000
TRACE_LENGTH = 1
BURN_IN = 100

class Runner():
    def __init__(self,
                 env,
                 actor_layers,
                 critic_a_layers,
                 critic_s_layers,
                 critic_layers,
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
                     critic_layers,
                     lstm_layers,
                     trace_length,
                     actor_lr,
                     critic_lr,
                     IMAGE_SIZE,
                     [1],
                     tau,
                     batch_size)

        self.ddpg.buildNetwork()

        self.memory = Memory(IMAGE_SIZE, MEMORY_SIZE, [1], TRACE_LENGTH)
        self.noise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(1), sigma=0.1)
        self.gamma = gamma
        self.batch_size = batch_size

    def train(self, num_episodes, burn_in):
        self.ddpg.initTargetNetwork()
        self.burnIn(burn_in)

        for i in range(num_episodes):
            states = self.processStates(self.env.reset())[None,:]
            state_history = [states.copy() for i in range(TRACE_LENGTH)]
            episode_reward = 0

            for x in range(MAX_EPISODES):
                actions = np.clip(self.ddpg.getOnlineActionProb(np.stack(state_history, axis = 1)) + self.noise(), [-3], [3])
                print(actions)
                next_states, rewards, terminal, infos = self.env.step(actions)
                next_states = next_states.squeeze()[None, :]
                
                self.memory.append(np.stack(state_history, axis = 1), actions, rewards, terminal, next_states, infos)

                train_states, train_actions, train_next_states, train_rewards, train_terminal, train_infos = self.memory.sample_batch(self.batch_size)
                # plt.imshow(train_states[0][0,:,:,0], cmap='gray')
                # plt.show()

                #train critic
                target_q = self.ddpg.getTargetStateValues(train_next_states, self.ddpg.getTargetActionProb(train_next_states, i), i)
                y_i = train_rewards * np.invert(train_terminal) * self.gamma * target_q
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

                state_history[:-1] = state_history[1:]
                state_history[-1] = next_states
                #print(rewards)
                self.env.render()
                episode_reward += np.average(rewards)
                if terminal:
                    summary = tf.Summary(value=[tf.Summary.Value(tag="Average episode reward", simple_value=episode_reward),])
                    self.ddpg.writer.add_summary(summary, i)
                    break


    def burnIn(self, burn_in):
        states = self.processStates(self.env.reset())[None,:]
        state_history = [states.copy() for i in range(TRACE_LENGTH)]

        for i in range(burn_in):
            actions = np.clip(self.ddpg.getOnlineActionProb(np.stack(state_history, axis=1)) + self.noise(), np.array([-3]), np.array([3]))
            next_states, rewards, terminal, infos = self.env.step(actions)
            next_states = next_states.squeeze()[None, :]
            
            self.memory.append(np.stack(state_history, axis = 1), actions, rewards, terminal, next_states, infos)

            if terminal:
                states = self.processStates(self.env.reset())
                state_history = [states.copy() for i in range(TRACE_LENGTH)]

            else:
                state_history[:-1] = state_history[1:]
                state_history[-1] = next_states



    def processStates(self, states):
        return states
        return [self.processState(s) for s in states]


    def processState(self, state):
        img = rgb2grey(state) * 255
        return tuple(pyramid_gaussian(img, downscale=2, max_layer = 1))[1][None,:,:,None]

if __name__ == '__main__':
    runner = Runner(env = 'Pendulum-v0',
                 actor_layers = [400, 300],
                 critic_a_layers = [],
                 critic_s_layers = [400],
                 critic_layers = [300],
                 lstm_layers = [10],
                 trace_length= TRACE_LENGTH,
                 actor_lr = 0.0001,
                 critic_lr = 0.001,
                 tau = 0.001,
                 batch_size = 64,
                 gamma = 0.99)

    runner.train(50000, BURN_IN)
