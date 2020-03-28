import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda
from tensorflow.keras.optimizers import Adam

import gym
import argparse
import numpy as np
from collections import deque
import random

tf.keras.backend.set_floatx('float64')
wandb.init(name='DQN', project="deep-rl-tf2")

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)
parser.add_argument('--atoms', type=int, default=51)
parser.add_argument('--v_min', type=float, default=-10.)
parser.add_argument('--v_max', type=float, default=10.)

args = parser.parse_args()

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done
    
    def size(self):
        return len(self.buffer)

class ActionStateModel:
    def __init__(self, state_dim, aciton_dim, z):
        self.state_dim  = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps
        self.z = z
        
        self.model = self.create_model()
    
    def create_model(self):
        input_state = Input((self.state_dim,))
        dense_1 = Dense(32, activation='relu')
        dense_2 = Dense(16, activation='relu')
        outputs = []
        for _ in range(args.actions):
            outputs.append(Dense(args.atoms))
        model = tf.keras.Model(input_state, outputs)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(args.lr))
        return model
    
    def predict(self, state):
        return self.model.predict(state)
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        return self.get_optimal_action(state)

    def get_optimal_action(self, state):
        z = self.predict(state)
        z_concat = np.vstack(z)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)
    

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.delta_z = (args.v_max - args.v_min) / float(args.atoms-1)
        self.z = [args.v_min + self.delta_z * i for i in range(args.atoms)]

        self.model = ActionStateModel(self.state_dim, self.action_dim, self.z)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim, self.z)
        self.target_update()

        self.buffer = ReplayBuffer()

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)
    
    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(args.batch_size), actions] = rewards + (1-done) * next_q_values * args.gamma
            self.model.train(states, targets)
    
    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            done, total_reward = False, 0
            state = self.env.reset()
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state, action, reward*0.01, next_state, done)
                total_reward += reward
                state = next_state
            if self.buffer.size() >= args.batch_size:
                self.replay()
            self.target_update()
            print('EP{} EpisodeReward={}'.format(ep, total_reward))
            wandb.log({'Reward': total_reward})


def main():
    env = gym.make('CartPole-v1')
    agent = Agent(env)
    agent.train(max_episodes=1000)

if __name__ == "__main__":
    main()
    