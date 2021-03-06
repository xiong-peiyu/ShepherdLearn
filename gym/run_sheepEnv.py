"""
Dueling DQN & Natural DQN comparison
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from RL_brain1 import DQNPrioritizedReplay


env = gym.make('sheep-v0')
env = env.unwrapped
env.seed(21)
MEMORY_SIZE = 500000
ACTION_SPACE = 4
REWARD_DISTANCE = 1000
REWARD_RADIUS= 50
IDLE_DISTANCE = 300

sess = tf.Session()
with tf.variable_scope('random'):
    RL_random = DQNPrioritizedReplay(
        n_actions=ACTION_SPACE, n_features=env.FEATURE_Count, e_greedy=0.0,memory_size=MEMORY_SIZE,
        e_greedy_increment=None, sess=sess, prioritized=False,
    )
with tf.variable_scope('natural_DQN'):
        RL_natural = DQNPrioritizedReplay(
        n_actions=ACTION_SPACE, n_features=env.FEATURE_Count, e_greedy=0.70,memory_size=MEMORY_SIZE,
        e_greedy_increment=0.0000002, sess=sess, prioritized=False,
    )

with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=ACTION_SPACE, n_features=env.FEATURE_Count, e_greedy=0.70,memory_size=MEMORY_SIZE,
        e_greedy_increment=0.0000002, sess=sess, prioritized=True, output_graph=False,
    )

sess.run(tf.global_variables_initializer())


def train(RL):

    total_steps = 0
    steps = []
    episodes = []
    for i_episode in range(500):
        print('this is '+str(i_episode)+' episode')
        observation = env._reset()
        observation = np.asarray(observation)

        while True:
            #env.render()

            action = RL.choose_action(observation)

            #f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # [-2 ~ 2] float actions
            observation_, reward, done, info = env.step(action)

            DogX, DogY, SheepCOMX, SheepCOMY, distance_to_sheep_centroid, distance_to_target, ave_distance_to_centroid,env.TARGET_X, env.TARGET_Y = observation_

            # check the sheep movement in this step
            #sheepMovementX = observation_[2] - observation[2]
            #sheepMovementY = observation_[3] - observation[3]

            # check if the sheep is closer to the final destination
            #movement = observation_[5] - observation[5]
           #print('the movement of the sheep: ', movement)
            #reward = np.sign(movement-IDLE_DISTANCE)
           # print('reward from the movement of the sheep is ', reward)

            # when the com is within a radius to the final destination there is a reward to the dog
            if (distance_to_target <= REWARD_DISTANCE and ave_distance_to_centroid <= REWARD_RADIUS):

               r1 = REWARD_DISTANCE - distance_to_target
               r2 = 1 / ave_distance_to_centroid
               reward = reward + r1 + r2
            elif (distance_to_target <= REWARD_DISTANCE):
                reward = reward + 1 / 2 * 1 / (REWARD_DISTANCE - distance_to_target)

            RL.store_transition(observation, action, reward, observation_)
            #print('reward is ', reward)
            if total_steps > 5000:
                #used to be MEMORY_SIZE
                if total_steps%1000 == 0:
                    RL.learn()
            if i_episode > 0:
                if total_steps-steps[i_episode-1]>8000:
                    done = True
                    reward = reward - 20000

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = np.asarray(observation_)
            total_steps += 1

    return np.vstack((episodes, steps))
his_random = train(RL_random)
his_natural = train(RL_natural)
his_prio = train(RL_prio)


print('finished training ')
# compare based on first success
plt.plot(his_random[0, :], his_random[1, :] - his_random[1, 0], c='k', label='random Actions')
plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid()
plt.show()

print('showed the plot ')

print(his_random[1, :] - his_random[1, 0])
print(his_natural[1, :] - his_natural[1, 0])
print(his_prio[1, :] - his_prio[1, 0])