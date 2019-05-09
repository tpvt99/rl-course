import tensorflow as tf
import gym
import mujoco_py
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import pickle
import numpy as np
import time

from behavior_cloning import MyModel


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_file", type = str)
    parser.add_argument("env_name", type = str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    args = parser.parse_args()

    print("Loading and building policy of " + args.env_name)
    # Load policy here
    model = MyModel()
    with open("expert_data/Humanoid-v2.pkl", "rb") as f:
        data = pickle.loads(f.read())
        observations = data['observations'].astype(np.float32)
        actions = data['actions'].astype(np.float32)
        print(actions.shape)

    obs_mean = np.mean(observations, axis=0)
    obs_meansq = np.mean(np.square(observations), axis=0)
    obs_std = np.sqrt(np.maximum(0, obs_meansq - np.square(obs_mean)))
    observations = (observations - obs_mean) / (obs_std + 1e-6)
    #print(model.variables)
    model.apply(observations[:1])

    #print(model.variables)
    model.load_weights("bc_policy/bc_model")

    print("Built and loaded")

    env = gym.make(args.env_name)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    returns = []

    for i in range(200):
        done = False
        totalr = 0
        obs = env.reset()
        #for z in range(1000):
        while not done:
            action = model(obs[None])
            print(action)
            obs, r, done, _ = env.step(action)
            if args.render:
                env.render()
            totalr += r
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__ == "__main__":
    main()