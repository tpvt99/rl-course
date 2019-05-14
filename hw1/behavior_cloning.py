import os
import tensorflow as tf
import numpy as np
import gym
import mujoco_py
import pickle
import time

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from load_policy_v2 import ExpertPolicy

NUM_ROLLOUTS = 20
ENV_NAME = "Humanoid-v2"
MAX_STEPS = 1000
BATCH_SIZE = 64

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(128, activation=tf.nn.tanh)
        self.d2 = Dense(64, activation=tf.nn.tanh)
        self.d3 = Dense(17)

    @tf.function
    def call(self, inputs, training = True):
        x = self.d1(inputs)
        x = self.d2(x)
        return self.d3(x)

class Meter():
    def __init__(self):
        self.val = 0
        self.count = 0
        self.avg = 0
        self.sum = 0

    def update(self, val, count=1):
        self.count += count
        self.val = val
        self.sum += val*count
        self.avg = self.sum / self.count

class BehaviorCloneing():
    def __init__(self, model):
        ## Constants
        self.num_rollouts = NUM_ROLLOUTS
        self.env_name = ENV_NAME
        self.max_steps = MAX_STEPS
        self.batch_size = BATCH_SIZE
        ##
        self.policy_fn = ExpertPolicy("experts/" + self.env_name + ".pkl")
        self.ds = self.load_data("expert_data/" + self.env_name + ".pkl")
        self.model = model
        self.initialization()

    def start(self):
        template = "Epoch[{0}/{1}], Loss: {2:.3f}"
        for epoch in range(200):
            for index, data in enumerate(self.ds):
                observations, actions = data
                self.training(observations, actions)
            print(template.format(epoch + 1, 200, self.meter.avg))

        print("Saving the model")
        self.model.save_weights("bc_policy/bc_model", save_format="tf")

    def keep_training(self):
        template = "Epoch[{0}/{1}], Loss: {2:.3f}"
        for obs, data in self.ds:
            self.model.apply(obs[None,0])
            break
        print("Load weight")

        self.model.load_weights("bc_policy/bc_model")

        for epoch in range(200):
            for index, data in enumerate(self.ds):
                observations, actions = data
                self.training(observations, actions)
            print(template.format(epoch + 1, 200, self.meter.avg))
        self.model.save_weights("bc_policy/bc_model", save_format="tf")

    def initialization(self):

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.meter = Meter()

    def load_data(self, filename):
        flag = True
        try:
            with open(filename, "rb") as f:
                data = pickle.loads(f.read())
                observations = data['observations'].astype(np.float32)
                actions = data['actions'].astype(np.float32)
                if observations.shape[0] != (self.num_rollouts * self.max_steps):
                    flag = False
        except FileNotFoundError:
            flag = False

        if flag == True:
            print("Expert data is generated. Done")
            observations = observations.reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
            ds = tf.data.Dataset.from_tensor_slices((observations, actions)).shuffle(observations.shape[0]).batch(self.batch_size)
            return ds
        else:
            print("Generating new expert data")
            env = gym.make(self.env_name)

            returns = []
            observations = []
            actions = []

            for i in range(self.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = self.policy_fn(obs[None, :].astype(np.float32))
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    env.render()
                    if steps % 100 == 0: print("%i/%i" % (steps, self.max_steps))
                    if steps >= self.max_steps:
                        break
                returns.append(totalr)

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))

            expert_data = {'observations': np.array(observations),
                           'actions': np.array(actions)}

            if not os.path.isdir("expert_data"):
                os.makedirs("expert_data")

            with open(os.path.join('expert_data', self.env_name + '.pkl'), 'wb') as f:
                pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

            observations = np.array(observations).astype(np.float32)
            actions = np.array(actions).astype(np.float32)

            observations = observations.reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])

            ds = tf.data.Dataset.from_tensor_slices((observations, actions)).shuffle(observations.shape[0]).batch(self.batch_size)
            return ds

        # normalizing data
        #obs_mean = np.mean(observations, axis = 0)
        #obs_meansq = np.mean(np.square(observations), axis = 0)
        #obs_std = np.sqrt(np.maximum(0, obs_meansq - np.square(obs_mean)))
        #observations = (observations - obs_mean)/ (obs_std+ 1e-6)
        #actions = actions.reshape(-1, 17)

    def training(self, observations, actions):
        with tf.GradientTape() as tape:
            predictions = self.model(observations)
            loss = tf.reduce_mean(tf.nn.l2_loss(predictions-actions))
            self.meter.update(loss.numpy())
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))



def test():
    returns = []
    env_name = ENV_NAME

    env = gym.make(env_name)
    model = MyModel()

    print("Loading and building policy of " + env_name)
    with open("expert_data/" + env_name + ".pkl", "rb") as f:
        data = pickle.loads(f.read())
        observations = data['observations'].astype(np.float32)
        actions = data['actions'].astype(np.float32)
        print(actions.shape)

    # obs_mean = np.mean(observations, axis=0)
    # obs_meansq = np.mean(np.square(observations), axis=0)
    # obs_std = np.sqrt(np.maximum(0, obs_meansq - np.square(obs_mean)))
    # observations = (observations - obs_mean) / (obs_std + 1e-6)

    model.apply(observations[:1])
    model.load_weights("bc_policy/bc_model")

    print("Built and loaded")
    max_steps = MAX_STEPS
    returns = []
    for _ in range(100):
        done = False
        totalr = 0
        obs = env.reset()
        step = 0
        # for z in range(1000):
        while not done:
            action = model(obs[None, :].astype(np.float32))
            obs, r, done, _ = env.step(action)
            env.render()
            totalr += r
            step += 1
            if step % 100 == 0: print("Iter {} / {}".format(step, max_steps))
            if step >= max_steps: break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__ == "__main__":
    model = MyModel()
    bc = BehaviorCloneing(model)
    bc.keep_training()
    test()
