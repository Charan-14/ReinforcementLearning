import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset() #reseting env first

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 500
EPSILON = 0.5
START_EPSILON = 1
END_EPSILON = EPISODES//2

epsilon_decaying_value = EPSILON/(END_EPSILON-START_EPSILON)

observe_high = env.observation_space.high
observe_low = env.observation_space.low
actions = env.action_space.n

DISCRETE_OS_SIZE = [20]*len(env.observation_space.high)
discrete_stepsize = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

# print("high_value - ", observe_high)
# print("low_value - ", observe_low)
# print("No of actions - ", actions)
# print("stepsize - ", discrete_stepsize)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_stepsize
    return tuple(discrete_state.astype(np.int)) 


q_table = np.random.uniform(low=-2, high = 0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) 


for epis in range(EPISODES):
    
    if epis % SHOW_EVERY ==0:
        print(epis)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    done = False

    while not done:
        if np.random.random()>EPSILON:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, _ = env.step(action) #State - pos.vel, Reward- , done = true/false tos top loop
        new_discrete_state = get_discrete_state(new_state)
        
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state+(action, )]
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE*(reward+DISCOUNT*max_future_q)
            q_table[discrete_state+(action, )] = new_q
        elif new_state[0]>= env.goal_position:
            q_table[discrete_state+(action,)] = 0
            print(f"Made it on episode{epis}")
        discrete_state = new_discrete_state
    if END_EPSILON >= epis >= START_EPSILON:
        EPSILON-=epsilon_decaying_value

env.close()