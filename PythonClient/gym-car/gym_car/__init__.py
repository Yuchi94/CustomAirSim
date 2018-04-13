from gym.envs.registration import register

register(
    id='carsim-v0',
    entry_point='gym_car.envs:CarSimEnv',
)
