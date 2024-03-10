import numpy as np
import os

# Wind Parameters
wind_flag = 0
wind_speed = 0
wind_dir = 0

# Wave Parameters
wave_flag = 0
wave_height = 0
wave_period = 0
wave_dir = 0

# TRAINING HYPERPARAMETERS

model_name = 'model_001'
duration = 160

# Learning rate parameters - Exponential decay
ac_initial_learning_rate = 0.0003
ac_decay_steps = 60000
ac_decay_rate = 0.9
cr_initial_learning_rate= 0.002
cr_decay_steps= 60000
cr_decay_rate= 0.9

ac_layer = (250,250)
cr_obs_layer = (32,32,)
cr_act_layer = (16,16,)
cr_joint_layer = (250,250)

discount_factor = 0.95
target_update_tau = 0.01
target_update_period = 1

std_mul = 0.15
std_episodes = 9000
replay_buffer_max_length = 100000
num_parallel_calls = 2
sample_batch_size = 128
num_steps = 2
prefetch = 3

max_episodes = 10001
random_seed = 12345

DDPG_update_time_steps = 20              # Updates DDPG parameters every these many time steps
DDPG_policy_store_frequency = 1000       # Stores DDPG policy every these many episodes
DDPG_loss_avg_interval = 100             # Computes DDPG loss and returns by averaging over these many episodes


def print_params(path):
    fid = open(os.path.join(path,'parameters.txt'),'w')
    fid.write(f'model_name:{model_name}\nduration:{duration}\n\n')
    fid.write(f'actor_initial_learning_rate:{ac_initial_learning_rate}\nactor_decay_steps:{ac_decay_steps}\nactor_decay_rate:{ac_decay_rate}\n\n')
    fid.write(f'critic_initial_learning_rate:{cr_initial_learning_rate}\ncritic_decay_steps:{cr_decay_steps}\ncritic_decay_rate:{cr_decay_rate}\n\n')
    fid.write(f'ac_layer:{ac_layer}\ncr_obs_layers:{cr_obs_layer}\ncr_act_layer:{cr_act_layer}\ncr_joint_layer:{cr_joint_layer}\n\n')
    fid.write(f'discount_factor:{discount_factor}\ntarget_update_tau:{target_update_tau}\ntarget_update_period:{target_update_period}\n\n')
    fid.write(f'replay_buffer_max_length:{replay_buffer_max_length}\nnum_parallel_calls:{num_parallel_calls}\nsample_batch_size:{sample_batch_size}\nnum_steps:{num_steps}\nprefetch:{prefetch}\n\n')
    fid.write(f'max_episodes:{max_episodes}\nNoise_episodes:{std_episodes}\nNoise_Multiplier:{std_mul}\nrandom_seed:{random_seed}\n\n')
    fid.write(f'DDPG_update_time_steps:{DDPG_update_time_steps}\nDDPG_policy_store_frequency:{DDPG_policy_store_frequency}\nDDPG_loss_avg_interval:{DDPG_loss_avg_interval}\n')
    fid.close()
