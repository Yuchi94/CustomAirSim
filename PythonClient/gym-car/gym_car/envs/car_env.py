import gym
from gym import error, spaces, utils
from gym.utils import seeding
from AirSimClient import *
from matplotlib import pyplot as plt
import numpy as np
import time

class CarSimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.starting_port = 42451
        self.step_time = 0.1
        self.action_space = [3]
        self.observation_space = [144, 256, 4]
        self.step_limit = 1000
        self.step_counter = 0

        self.car_controls = CarControls()
        self.clients = []

        for i in range(4):
            client = CarClient(port = self.starting_port + i)
            client.confirmConnection()
            client.enableApiControl(True)
            self.clients.append(client)

    #define actions as a list: [(throttle1, steering1, brake1), (throttle2, steering2, brake2)]
    def step(self, actions):
        self.step_counter += 1
        #send controls
        self.setControls(actions)

        time.sleep(self.step_time)

        #retrieve state (images)
        states = self.getStates()

        #check for collision
        rewards, collisions = self.getRewards()

        #get done state
        done = self.getDone(collisions)

        #state, reward, done, info
        return states, rewards, done, collisions

    def reset(self):
        #set throttle, steering, brake to default values
        actions = [(0, 0, 0) for i in range(4)]
        self.setControls(actions)

        time.sleep(0.1)

        for i in range(4):
            self.clients[i].reset()

        self.step_counter = 0
        return self.getStates()

    def render(self, mode='human', close=False):
        pass

    def setControls(self, actions):
        for i in range(4):
            throttle = actions[i][0]
            steering = actions[i][1]
            brake = actions[i][2]
            self.car_controls.throttle = throttle
            self.car_controls.steering = steering
            self.car_controls.brake = 0 # if brake < 0.1 else 1
            self.clients[i].setCarControls(self.car_controls)

    def getStates(self):
        states = []
        for i in range(4):
            responses = self.clients[i].simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])
            response = responses[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            img = img1d.reshape(response.height, response.width, 4)

            states.append(img)

        return states

    def getDone(self, collisions):

        return True in collisions or self.step_counter > self.step_limit

    def getRewards(self):
        speed_weight = 1
        collision_penalty = 300

        rewards = [0, 0, 0, 0]
        collisions = []

        for i in range(4):
            car_state = self.clients[i].getCarState()
            collision = car_state.collision.has_collided

            rewards[i] += car_state.speed * speed_weight
            rewards[i] -= collision * collision_penalty
            collisions.append(collision)

        return rewards, collisions
