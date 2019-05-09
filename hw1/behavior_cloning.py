import os
import tensorflow as tf
import numpy as np
import gym
import mujoco_py
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.python.util.tf_export import tf_export


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(64, activation='tanh')
        self.d2 = Dense(64, activation='tanh')
        self.d3 = Dense(17)

    @tf.function
    def call(self, inputs, training = True):
        x = self.d1(inputs)
        x = self.d2(x)
        return self.d3(x)

class BehaviorCloneing():
    def __init__(self, model, filename):
        self.ds = self.load_data(filename)
        self.model = model
        self.initialization()

    def start(self):
        template1 = "Epoch[{0} | {1}/{2}], Loss: {3:.3f} Accuracy: {4:.3f}"
        for epoch in range(10):
            for index, data in enumerate(self.ds):
                observations, actions = data
                self.training(observations, actions)
                print(template1.format((index+1)*10, epoch + 1, 10, self.train_loss.result(), self.train_accuracy.result()))

            print("Saving the model")
            self.model.save_weights("bc_policy/bc_model", save_format="tf")

    @staticmethod
    @tf_export("loss_hihi")
    def loss(inputs, outputs):
        return tf.reduce_mean(tf.nn.l2_loss(inputs-outputs))

    def keep_training(self):
        template1 = "Epoch[{0} | {1}/{2}], Loss: {3:.3f} Accuracy: {4:.3f}"
        for obs, data in self.ds:
            model.apply(obs[None,0])
            break
        model.load_weights("bc_policy/bc_model")

        for epoch in range(20):
            for index, data in enumerate(self.ds):
                observations, actions = data
                self.training(observations, actions)
                print(template1.format((index+1)*10, epoch + 1, 20, self.train_loss.result(), self.train_accuracy.result()))
            self.model.save_weights("bc_policy/bc_model", save_format="tf")

    def initialization(self):

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def load_data(self, filename):
        with open(filename, "rb") as f:
            data = pickle.loads(f.read())
            observations = data['observations'].astype(np.float32)
            actions = data['actions'].astype(np.float32)

        # normalizing data
        obs_mean = np.mean(observations, axis = 0)
        obs_meansq = np.mean(np.square(observations), axis = 0)
        obs_std = np.sqrt(np.maximum(0, obs_meansq - np.square(obs_mean)))
        observations = (observations - obs_mean)/ (obs_std+ 1e-6)
        actions = actions.reshape(-1, 17)
        ds = tf.data.Dataset.from_tensor_slices((observations, actions)).shuffle(20000).batch(32)
        return ds

    def training(self, observations, actions):
        @tf.function
        def train_step(inputs, outputs):
            with tf.GradientTape() as tape:
                predictions = self.model(inputs)
                loss = tf.reduce_mean(tf.nn.l2_loss(predictions-outputs))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.train_loss(loss)
            self.train_accuracy(tf.argmax(outputs, axis=1)[:,None], predictions)
        train_step(observations, actions)

    def testing(self, observations, actions):
        @tf.function
        def test_step(inputs, outputs):
            predictions = self.model(inputs)
            loss = self.loss_object(outputs, predictions)

            self.test_loss(loss)
            self.test_accuracy(outputs, predictions)

        test_step(observations, actions)

if __name__ == "__main__":
    model = MyModel()
    bc = BehaviorCloneing(model, "expert_data/Humanoid-v2.pkl")
    # switch between bc.start() or bc.keep_training() to start training or keep training
    bc.keep_training()
