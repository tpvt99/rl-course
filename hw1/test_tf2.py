# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
# from tensorflow.keras import Model
# import numpy as np
# from tensorflow.keras.layers import Layer
#
# mnist = tf.keras.datasets.fashion_mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train/255.0, x_test/255.0
#
# x_train = x_train[..., tf.newaxis]
# x_test = x_test[..., tf.newaxis]
#
# print(x_train.shape)
# train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
#
# class CustomConv2d(Layer):
#     def __init__(self):
#         super(CustomConv2d, self).__init__()
#         self.w = tf.Variable(initial_value=np.random.choice(2, 3 * 3 * 64 * 32).reshape(3, 3, 32, 64), name="conv2W",
#                               trainable=True, dtype=tf.float64)
#
#     def call(self, inputs):
#         return tf.nn.conv2d(input = inputs, filters = self.w, strides = (1,1), padding = "VALID")
#
# class MyModel(Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.conv1 = Conv2D(filters = 32, kernel_size = 3, activation = 'relu')
#         self.pool1 = MaxPool2D(pool_size = 2)
#         self.conv2 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu')
#         self.pool2 = MaxPool2D(pool_size = 2)
#         self.flatten = Flatten()
#         self.d1 = Dense(128, activation = 'relu')
#         self.d2 = Dense(10, activation = 'softmax')
#         self.conv3 = CustomConv2d()
#
#     @tf.function
#     def call(self, x):
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.conv3(x)
#         x = self.pool2(x)
#         x = self.flatten(x)
#         x = self.d1(x)
#         x = self.d2(x)
#         return x
#
# model = MyModel()
#
#
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam()
#
# train_loss = tf.keras.metrics.Mean(name = 'train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
#
# test_loss = tf.keras.metrics.Mean(name = 'test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')
#
# @tf.function
# def train_step(images, labels):
#     with tf.GradientTape() as tape:
#         predictions = model.call(images)
#         loss = loss_object(labels, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     print(model.conv3.w)
#     train_loss(loss)
#     train_accuracy(labels, predictions)
#
# def test_step(images, labels):
#     predictions = model(images)
#     t_loss = loss_object(labels, predictions)
#
#     test_loss(t_loss)
#     test_accuracy(labels, predictions)
#
# for epoch in range(10):
#     for images, labels in train_ds:
#         train_step(images, labels)
#
#     for test_images, test_labels in test_ds:
#         test_step(test_images, test_labels)
#
#     template = "Epoch[{0}/{1}], Loss: {2:.3f} Accuracy: {3:.3f}, Test loss: {4:.3f} Test Accuracy: {5:.3f}"
#     print(template.format(epoch+1, 10, train_loss.result(), train_accuracy.result(),
#                           test_loss.result(), test_accuracy.result()))

import gym
import numpy as np

env = gym.make("Humanoid-v2")
for _ in range(20):
    obs = env.reset()
    for t in range(1000):
        print('----- OBSERVATION -----')
        action = env.action_space.sample()
        print(action.shape)
        print('----- ACTION -----')
        obs, rewards, done, info = env.step(action)
        env.render()

env.close()