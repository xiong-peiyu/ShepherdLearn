import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
#matplotlib inline
from helper import *
import os
from random import choice
from time import sleep
from time import time
from ACNetwork import *
import gym
import matplotlib.pyplot as plt
import tensorflow as tf


max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
a_size = 4 # the dog can choose to go UP/DOWN/LEFT/RIGHT
load_model = False
model_path = './model'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network(s_size, a_size, 'global', None)  # Generate global network
    num_workers = multiprocessing.cpu_count()  # Set workers to number of available CPU threads
    workers = []
    # Create worker classes
    # num_workers
    for i in range(num_workers):
        NewEnv = gym.make('sheep-v0')
        NewEnv = NewEnv.unwrapped
        NewEnv.seed(i)
        workers.append(Worker(NewEnv, i, s_size, a_size, trainer, model_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=5)
    print('finish appending all the workers to the list')

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)