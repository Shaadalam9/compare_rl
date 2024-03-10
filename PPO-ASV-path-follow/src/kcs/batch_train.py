import numpy as np
import hyperparams as params
import ppo_train
import time
import tensorflow as tf

max_batch_train = 1

for batch_train in range(0, max_batch_train):
    params.model_name = 'model_' + '{:03}'.format(batch_train + 1)

    if batch_train < max_batch_train:
        params.duration = 160
        params.initial_learning_rate = 0.001
        params.decay_rate = 0.5
        params.decay_steps= 3000
        params.actor_fc_layers = (128,128)
        params.value_fc_layers = (128,128)
        params.gamma = 0.96
        params.clip = 0.2
        params.lambda_val = 0.95
        params.episodes_per_iteration = 50
        params.epochs = 10
        params.replay_buffer_max_length = params.episodes_per_iteration*params.duration
        params.iterations = 100
        params.random_seed = 43738#np.random.randint(100000)
        params.PPO_policy_store_frequency = 5
        params.PPO_policy_store_min = 50
        params.PPO_loss_avg_interval = 1
        params.destination_reward = 100
        params.greedy_eval = False
        
    tf.keras.utils.set_random_seed(params.random_seed)
    tf.config.experimental.enable_op_determinism()
    ppo_train.ppo_train(params)

print(f'Execution Completed Successfully')


