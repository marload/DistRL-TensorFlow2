import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, ReLU
from tensorflow.keras.optimizers import Adam

import gym
import argparse
import numpy as np
from collections import deque
import random
import math

tf.keras.backend.set_floatx('float64')
wandb.init(name='IQN', project="dist-rl-tf2")

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--atoms', type=int, default=8)
parser.add_argument('--quantile_dim', type=int, default=64)

args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(
            np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class Network(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.quantile_dim = args.quantile_dim
        self.atoms = args.atoms
        self.batch_size = args.batch_size

        self.feature_extraction = tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu')
        ])
        self.phi = Dense(self.quantile_dim, activation=None, use_bias=False)
        self.phi_bias = tf.cast(tf.Variable(
            tf.zeros(self.quantile_dim)), tf.float64)
        self.fc = Dense(128, activation='relu')
        self.fc_q = Dense(self.action_dim, activation=None)
        self.relu = ReLU()

    def call(self, state):
        x = self.feature_extraction(state)
        feature_dim = x.shape[1]
        tau = np.random.rand(self.atoms, 1)
        pi_mtx = tf.constant(np.expand_dims(
            np.pi * np.arange(0, self.quantile_dim), axis=0))
        cos_tau = tf.cos(tf.matmul(tau, pi_mtx))
        phi = self.relu(self.phi(cos_tau) +
                        tf.expand_dims(self.phi_bias, axis=0))
        phi = tf.expand_dims(phi, axis=0)
        x = tf.reshape(x, (-1, feature_dim))
        x = tf.expand_dims(x, 1)
        x = x * phi
        x = self.fc(x)
        x = self.fc_q(x)
        q = tf.transpose(x, [0, 2, 1])
        return q, tau


class ActionValueModel:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = args.atoms
        self.huber_loss = tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.NONE)
        self.opt = tf.keras.optimizers.Adam(args.lr)
        self.model = self.create_model()

    def create_model(self):
        net = Network(self.state_dim, self.action_dim)
        net.build((None, self.state_dim))
        return net

    def quantile_huber_loss(self, target, pred, actions, tau):
        pred = tf.reduce_sum(pred * tf.expand_dims(actions, -1), axis=1)
        pred_tile = tf.tile(tf.expand_dims(pred, axis=2), [1, 1, self.atoms])
        target_tile = tf.tile(tf.expand_dims(
            target, axis=1), [1, self.atoms, 1])
        huber_loss = self.huber_loss(target_tile, pred_tile)
        tau = tf.reshape(np.array(tau), [1, self.atoms])
        inv_tau = 1.0 - tau
        tau = tf.tile(tf.expand_dims(tau, axis=1), [1, self.atoms, 1])
        inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, self.atoms, 1])
        error_loss = tf.math.subtract(target_tile, pred_tile)
        loss = tf.where(tf.less(error_loss, 0.0), inv_tau *
                        huber_loss, tau * huber_loss)
        loss = tf.reduce_mean(tf.reduce_sum(
            tf.reduce_mean(loss, axis=2), axis=1))
        return loss

    def train(self, states, target, actions):
        with tf.GradientTape() as tape:
            theta, tau = self.model(states)
            loss = self.quantile_huber_loss(target, theta, actions, tau)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict(self, state):
        return self.model(state, training=False)

    def get_action(self, state, ep):
        state = np.reshape(state, [1, self.state_dim])
        eps = 1. / ((ep / 10) + 1)
        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)
        else:
            return self.get_optimal_action(state)

    def get_optimal_action(self, state):
        z, tau = self.model(state, training=False)
        q = np.mean(z, axis=2)[0]
        return np.argmax(q)


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.buffer = ReplayBuffer()
        self.batch_size = args.batch_size
        self.atoms = args.atoms
        self.gamma = args.gamma
        self.q = ActionValueModel(self.state_dim, self.action_dim)
        self.q_target = ActionValueModel(self.state_dim, self.action_dim)
        self.target_update()

    def target_update(self):
        weights = self.q.model.get_weights()
        self.q_target.model.set_weights(weights)

    def replay(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()
        q, tau = self.q_target.predict(next_states)  # tau
        next_actions = np.argmax(np.mean(q, axis=2), axis=1)
        theta = []
        for i in range(self.batch_size):
            if dones[i]:
                theta.append(np.ones(self.atoms) * rewards[i])
            else:
                theta.append(rewards[i] + self.gamma * q[i][next_actions[i]])
        self.q.train(states, theta, actions)

    def train(self, max_epsiodes=500):
        for ep in range(max_epsiodes):
            done, total_reward, steps = False, 0, 0
            state = self.env.reset()
            while not done:
                action = self.q.get_action(state, ep)
                next_state, reward, done, _ = self.env.step(action)
                action_ = np.zeros(self.action_dim)
                action_[action] = 1
                self.buffer.put(state, action_,
                                -1 if done else 0, next_state, done)

                if self.buffer.size() > 1000:
                    self.replay()
                if steps % 5 == 0:
                    self.target_update()

                state = next_state
                total_reward += reward
                steps += 1
            wandb.log({'reward': total_reward})
            print('EP{} reward={}'.format(ep, total_reward))


def main():
    env = gym.make('CartPole-v1')
    agent = Agent(env)
    agent.train()


if __name__ == "__main__":
    main()
