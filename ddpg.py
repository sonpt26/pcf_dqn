import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras.layers import Input, Dense, Concatenate, Flatten
from keras.models import Model
import tensorflow as tf
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from network import NetworkEnv

env = NetworkEnv()
state_shape = env.get_state_shape()
action_shape = env.get_action_shape()
num_states = env.observation_space.shape[0]
print("Shape of State Space ->  {}".format(state_shape))
num_actions = env.action_space.shape[0]
print("Shape of Action Space ->  {}".format(action_shape))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

"""
To implement better exploration by the Actor network, we use noisy perturbations,
specifically
an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper.
It samples noise from a correlated normal distribution.
"""


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, *state_shape))
        self.action_buffer = np.zeros((self.buffer_capacity, *action_shape))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, *state_shape))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity        
        self.state_buffer[index] = np.array(obs_tuple[0])
        self.action_buffer[index] = np.array(obs_tuple[1])
        self.reward_buffer[index] = np.array(obs_tuple[2])
        self.next_state_buffer[index] = np.array(obs_tuple[3])

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -keras.ops.mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = keras.ops.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = keras.ops.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = keras.ops.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = keras.ops.cast(reward_batch, dtype="float32")
        next_state_batch = keras.ops.convert_to_tensor(
            self.next_state_buffer[batch_indices]
        )

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(target, original, tau):
    target_weights = target.get_weights()
    original_weights = original.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

    target.set_weights(target_weights)


def get_actor():
    # # Initialize weights between -3e-3 and 3-e3
    # last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    # inputs = layers.Input(shape=state_shape)
    # out = layers.Dense(256, activation="relu")(inputs)
    # out = layers.Dense(256, activation="relu")(out)
    # outputs = layers.Dense(state_shape[0], activation="tanh", kernel_initializer=last_init)(out)

    # # Our upper bound is 2.0 for Pendulum.
    # outputs = outputs * upper_bound
    # model = keras.Model(inputs, outputs)
    # return model
    inputs = Input(shape=state_shape)
    x = Flatten()(inputs)  # Flatten the input if needed
    x = Dense(64, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(action_shape[0], activation="tanh")(x)
    model = Model(inputs, outputs)
    return model


def get_critic():
    # # State as input
    # state_input = layers.Input(shape=state_shape)
    # state_out = layers.Dense(16, activation="relu")(state_input)
    # state_out = layers.Dense(32, activation="relu")(state_out)

    # # Action as input
    # action_input = layers.Input(shape=action_shape)
    # action_out = layers.Dense(32, activation="relu")(action_input)

    # # Both are passed through seperate layer before concatenating
    # concat = layers.Concatenate()([state_out, action_out])

    # out = layers.Dense(256, activation="relu")(concat)
    # out = layers.Dense(256, activation="relu")(out)
    # outputs = layers.Dense(1)(out)

    # # Outputs single value for give state-action
    # model = keras.Model([state_input, action_input], outputs)

    # return model
    state_input = Input(shape=state_shape)
    action_input = Input(shape=action_shape)

    state_x = Flatten()(state_input)  # Flatten the state input if needed
    state_x = Dense(64, activation="relu")(state_x)
    state_x = Dense(64, activation="relu")(state_x)

    action_x = Dense(64, activation="relu")(action_input)

    concat = Concatenate()([state_x, action_x])
    x = Dense(64, activation="relu")(concat)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1, activation="linear")(x)

    model = Model([state_input, action_input], outputs)
    return model


"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""


def policy(state, noise_object):
    sampled_actions = keras.ops.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return np.squeeze(legal_action)


"""
## Training hyperparameters
"""

std_dev = 0.3
ou_noise = OUActionNoise(mean=np.zeros(action_shape), std_deviation=float(std_dev) * np.ones(action_shape))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = keras.optimizers.Adam(critic_lr)
actor_optimizer = keras.optimizers.Adam(actor_lr)

total_episodes = 50
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 4 min to train
for ep in range(total_episodes):
    prev_state, _ = env.reset()
    episodic_reward = 0
    print("==================TRAINING EPISODE", ep, "==================")
    print("==================AVG REWARD", np.mean(ep_reward_list), "==================")
    while True:
        tf_prev_state = keras.ops.expand_dims(
            keras.ops.convert_to_tensor(prev_state), 0
        )

        action = policy(tf_prev_state, ou_noise)
        print("action", action)
        # Recieve state and reward from environment.
        state, reward, done, _ = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()

        update_target(target_actor, actor_model, tau)
        update_target(target_critic, critic_model, tau)

        # End this episode when `done` or `truncated` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()


# Save the weights
actor_model.save_weights("pendulum_actor.weights.h5")
critic_model.save_weights("pendulum_critic.weights.h5")

target_actor.save_weights("pendulum_target_actor.weights.h5")
target_critic.save_weights("pendulum_target_critic.weights.h5")
