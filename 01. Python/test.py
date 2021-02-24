import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool3D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from rl.agents import CEMAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from Environment_Class import TwentyFortyEightEnvironment
import datetime


def test_env(env, episodes=1, render_every_step=True):
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = env.action_space.sample()
            if render_every_step:
                print(f'Turn: {env.turn}')
                if action == 0:
                    action_name = 'down'
                elif action == 1:
                    action_name = 'left'
                elif action == 2:
                    action_name = 'up'
                else:
                    action_name = 'right'
                print(f'Action: {action_name}')
                env.render()

            n_state, reward, done, info = env.step(action)
            score += reward

        print(f'Episode: {episode} \t Score: {score}')
        env.render()


def build_model(env):
    model = Sequential()
    model.add(Conv2D(input_shape= (1,) + env.observation_space.shape, kernel_size=(2, 2), filters=4*15, activation='tanh'))
    model.add(MaxPool3D(pool_size=(1, 3, 3)))
    # model.add(Flatten(input_dim=states))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=100000, window_length=1)
    dqn = CEMAgent(model=model, batch_size=50, memory=memory, nb_actions=actions, nb_steps_warmup=100)
    return dqn


env = TwentyFortyEightEnvironment()
# test_env(env)

log_dir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/', save_best_only=True, save_freq=30000)

states = env.observation_space.shape
actions = env.action_space.n

model = build_model(env)
model.summary()

dqn = build_agent(model, actions)
dqn.compile()

dqn.fit(env, nb_steps=5000000, visualize=False, verbose=1, callbacks=[tensorboard_callback, checkpointer])
aux = dqn.test(env, 5, visualize=True)
