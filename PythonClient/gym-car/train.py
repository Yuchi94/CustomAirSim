import gym
import numpy as np
import gym_car
from network import *
from memory import *
from noise import *
from skimage.transform import pyramid_gaussian
from skimage.color import rgb2grey

IMAGE_SIZE = [72, 128, 1]
MEMORY_SIZE = 150000
NUM_AGENTS = 4
MAX_EPISODES = 1000

class Runner():
    def __init__(self,
                 env,
                 actor_layers,
                 critic_a_layers,
                 critic_s_layers,
                 critic_layers,
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
                     actor_lr,
                     critic_lr,
                     IMAGE_SIZE,
                     [3],
                     tau,
                     batch_size)

        self.ddpg.buildNetwork()

        self.memory = Memory(IMAGE_SIZE, MEMORY_SIZE, [3], data_type=np.int8)
        self.noise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(3), sigma=0.1)
        self.gamma = gamma
        self.batch_size = batch_size

    def train(self, num_episodes, burn_in):
        self.ddpg.updateTargetNetwork()
        self.burnIn(burn_in)

        for i in range(num_episodes):
            states = self.processStates(self.env.reset())
            episode_reward = 0
            for x in range(MAX_EPISODES):
                actions = []

                for j in range(NUM_AGENTS):
                    actions.append(self.ddpg.getOnlineActionProb(states[j])[0] + self.noise())

                next_states, rewards, terminal, infos = self.env.step(actions)
                next_states = self.processStates(next_states)

                for j in range(NUM_AGENTS):
                    self.memory.append(states[j], actions[j], rewards[j], terminal, next_states[j], infos[j])

                train_states, train_actions, train_next_states, train_rewards, train_terminal, train_infos = self.memory.sample_batch(self.batch_size)

                #train critic
                target_q = self.ddpg.getTargetStateValues(train_next_states, self.ddpg.getTargetActionProb(train_next_states))
                y_i = train_rewards * np.invert(train_infos) * self.gamma * target_q
                self.ddpg.trainCritic(train_states, train_actions, y_i)

                #train actor
                action_prob = self.ddpg.getOnlineActionProb(train_states)
                grads = self.ddpg.getActionGradients(train_states, action_prob)
                self.ddpg.trainActor(train_states, grads[0])

                #update target network
                self.ddpg.updateTargetNetwork()

                states = [ns.copy() for ns in next_states]
                episode_reward += np.average(rewards)

                if terminal:
                    print(episode_reward)
                    break


    def burnIn(self, burn_in):
        states = self.processStates(self.env.reset())

        for i in range(burn_in):
            actions = []

            for j in range(NUM_AGENTS):
                actions.append((self.ddpg.getOnlineActionProb(states[j])[0] + np.array(self.noise())).tolist())

            next_states, rewards, terminal, infos = self.env.step(actions)
            next_states = self.processStates(next_states)

            for j in range(NUM_AGENTS):
                self.memory.append(states[j], actions[j], rewards[j], terminal, next_states[j], infos[j])

            if terminal:
                states = self.processStates(self.env.reset())
            else:
                states = [ns.copy() for ns in next_states]



    def processStates(self, states):
        return [self.processState(s) for s in states]


    def processState(self, state):
        img = rgb2grey(state)
        return tuple(pyramid_gaussian(img, downscale=2, max_layer = 1))[1][None,:,:,None]

if __name__ == '__main__':
    runner = Runner(env = 'carsim-v0',
                 actor_layers = [100, 100],
                 critic_a_layers = [100],
                 critic_s_layers = [100],
                 critic_layers = [100],
                 actor_lr = 5e-4,
                 critic_lr = 5e-4,
                 tau = 0.1,
                 batch_size = 32,
                 gamma = 0.99)

    runner.train(50000, 1000)