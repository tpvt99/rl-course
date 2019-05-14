import gym
import tensorflow as tf
import pickle
import numpy as np
import math

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from load_policy_v2 import ExpertPolicy


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

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(64, activation='tanh')
        self.d2 = Dense(32, activation='tanh')
        self.d3 = Dense(2)

    def call(self, inputs, training = True):
        x = self.d1(inputs)
        x = self.d2(x)
        return self.d3(x)


class Dagger():
    def __init__(self, model):
        self.initialization()
        self.model = model
        self.policy_fn = ExpertPolicy("experts/" + "Reacher-v2.pkl")

    def initialization(self):
        self.optimizer = tf.keras.optimizers.Adam()
        self.batch_size = 64

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.meter = Meter()

    def start(self):
        env = gym.make("Reacher-v2")
        steps = 500
        epochs = 100
        num_rollouts = 10

        observations = []
        actions = []
        template = "Iteration {} Epoch[{}/{}], Loss: {:.3f} Accuracy: {:.4f}"

        for i in range(100):
            new_obs = []
            new_act = []
            if i == 0: # use the expert policy to generate data and train the next policy
                # first, collect data from expert policy
                for _ in range(num_rollouts):
                    obs = env.reset()
                    done = False
                    step = 0
                    while not done:
                        action = self.policy_fn(obs[None,:].astype(np.float32))
                        new_obs.append(obs)
                        new_act.append(action)
                        obs, r, done, _ = env.step(action)
                        step += 1
                        env.render()
                        if step % 100 == 0: print("Iter {} / {}".format(step, steps))
                        if step >= steps:
                            break
            else: # use the policy to generate observations, but use the expert policy to generate actions
                for _ in range(num_rollouts*2):
                    obs = env.reset()
                    done = False
                    step = 0
                    while not done:
                        action = self.model(obs[None,:])
                        obs, r, done, _ = env.step(action)
                        action = self.policy_fn(obs[None,:].astype(np.float32))
                        new_obs.append(obs)
                        new_act.append(action)
                        step+=1
                        if step % 100 == 0: print("Iter {} / {}".format(step, steps))
                        if step >= steps:
                            break
            ds, observations, actions = self.aggregate(new_obs, new_act, observations, actions)

            #second, use the data to train the policy
            for epoch in range(epochs):
                for index, data in enumerate(ds):
                    observation, action = data
                    self.train_policy(observation, action)
                print(template.format(i+1, epoch+1, epochs, self.meter.avg, self.train_accuracy.result()))

            self.model.save_weights("dagger/dagger_model", save_format="tf")
            print("ohla")
            # Testing new policy
            for i in range(10):
                done = False
                obs = env.reset()
                while not done:
                    action = self.model(obs[None,:])
                    obs, r, done, _ = env.step(action)
                    env.render()

    # def shuffle_data(self, observations, actions, seed=1):
    #     np.random.seed(seed)
    #     permutations = np.random.choice(observations.shape[0], observations.shape[0], replace = False)
    #     observations = observations[permutations,:]
    #     actions = actions[permutations,:]
    #     idx = observations.shape[0] // self.batch_size
    #
    #     for i in range(idx):
    #         input_batch = observations[i*self.batch_size : (i+1)*self.batch_size, :]
    #         output_batch = actions[i*self.batch_size : (i+1)*self.batch_size, :]
    #         yield input_batch, output_batch
    #     if idx*self.batch_size < observations.shape[0]:
    #         input_batch = observations[idx*self.batch_size:, :]
    #         output_batch = actions[idx*self.batch_size:, :]
    #         yield input_batch, output_batch

    def aggregate(self, new_obs, new_act,  observations, actions): # all are numpy type
        new_obs = np.array(new_obs).astype(np.float32)
        new_act = np.array(new_act).astype(np.float32)
        new_obs = new_obs.reshape(-1, new_obs.shape[-1])
        new_act = new_act.reshape(-1, new_act.shape[-1])
        print("SHAPE: " + str(new_act.shape))

        # obs_mean = np.mean(new_obs, axis = 0)
        # obs_meansq = np.mean(np.square(new_obs), axis = 0)
        # obs_std = np.sqrt(np.maximum(0, obs_meansq - np.square(obs_mean)))
        # new_obs = (new_obs - obs_mean)/ (obs_std+ 1e-6)
        # new_act = new_act.reshape(-1, new_act.shape[-1])

        if observations == [] and actions == []:
            observations = new_obs
            actions = new_act
        else:
            observations = np.concatenate((observations, new_obs))
            actions = np.concatenate((actions, new_act))

        ds = tf.data.Dataset.from_tensor_slices((observations, actions)).shuffle(observations.shape[0]).batch(self.batch_size)
        return ds, observations, actions


    def train_policy(self, observations, actions):
        with tf.GradientTape() as tape:
            predictions = self.model(observations)
            loss = tf.reduce_mean(tf.nn.l2_loss(predictions-actions))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.meter.update(loss)


if __name__ == "__main__":
    model = MyModel()
    dagger = Dagger(model)
    dagger.start()