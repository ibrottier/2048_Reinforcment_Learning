import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from Environment_Class import TwentyFortyEightEnvironment


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


def build_model(states, actions):
    model = Sequential()
    # model.add(Flatten(input_dim=states))
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
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

model = build_model(states, actions)
model.summary()

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)