import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from Environment_Class import TwentyFortyEightEnvironment
import datetime


def test_env(env, episodes = 1, render_every_step = True):
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
    # model.add(Flatten(input_dim=states))
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    # model.add(Dense(4, input_shape=(1, 4, 4),  activation='relu'))
    model.add(Dense(4, activation='relu'))

    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10,
                   target_model_update=1e-2)
    return dqn


env = TwentyFortyEightEnvironment()
# test_env(env)

states = env.observation_space.shape
actions = env.action_space.n

model = build_model(env)
model.summary()

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
dqn.fit(env, nb_steps=100000, visualize=False, verbose=1, callbacks=[tensorboard_callback])
aux = dqn.test(env, 1, visualize=True)
print(aux)