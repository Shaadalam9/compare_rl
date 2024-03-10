import numpy as np
import hyperparams as params
import ddpg_train
import time
import tensorflow as tf

max_batch_train = 10

for batch_train in range(0, max_batch_train):
    params.model_name = 'model_' + '{:03}'.format(batch_train + 1)

    if batch_train < max_batch_train:
        params.duration = 160
        params.ac_initial_learning_rate = 0.0003
        params.ac_decay_steps= 60000
        params.ac_decay_rate = 0
        params.cr_initial_learning_rate = 0.001
        params.cr_decay_steps = 60000
        params.cr_decay_rate = 0
        params.ac_layer = (256,256)
        params.cr_obs_layer = (32,32)
        params.cr_act_layer = (16,16,)
        params.cr_joint_layer = (256,256)
        params.discount_factor = 0.96
        params.std_episodes = 11000
        params.std_mul = 0.15
        params.target_update_tau = 0.01
        params.target_update_period = 1
        params.replay_buffer_max_length = 100000
        params.num_parallel_calls = 2
        params.sample_batch_size = 128
        params.num_steps = 2
        params.prefetch = 3
        params.max_episodes = 10001
        params.random_seed = np.random.randint(100000)
        params.DDPG_update_time_steps = 10
        params.DDPG_policy_store_frequency = 1000
        params.DDPG_loss_avg_interval = 100

    tf.keras.utils.set_random_seed(params.random_seed)
    tf.config.experimental.enable_op_determinism()
    ddpg_train.ddpg_train(params)

print(f'Execution Completed Successfully')

