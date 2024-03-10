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

# Learning rate parameters - Constant
initial_learning_rate = 0.001
decay_rate = 0.8
decay_steps= 3000
destination_reward = 0

actor_fc_layers = (128,128)
value_fc_layers = (128,128)

greedy_eval = True

gamma = 0.95
clip = 0.2
lambda_val = 0.95
entropy_regularization = 0.01
epochs = 10

episodes_per_iterations = 50
iterations = 100
replay_buffer_max_length=duration*episodes_per_iterations
random_seed = 12345

PPO_policy_store_frequency = 10       # Stores PPO policy every these many episodes
PPO_loss_avg_interval = 100             # Computes PPO loss and returns by averaging over these many episodes



def print_params(path):
    fid = open(os.path.join(path,'parameters.txt'),'w')
    fid.write(f'model_name:{model_name}\nduration:{duration}\n\n')
    fid.write(f'Initial_learning_rate:{initial_learning_rate}\nDecay_steps:{decay_steps}\nDecay_rate:{decay_rate}\n\n')
    fid.write(f'Actor_fc_layer:{actor_fc_layers}\nValue_fc_layer:{value_fc_layers}\ndiscount_factor:{gamma}\nDestiantion_Reward:{destination_reward}\n')
    fid.write(f'Clip_Ratio:{clip}\nLambda_Val:{lambda_val}\nreplay_buffer_max_length:{replay_buffer_max_length}\n\n')
    fid.write(f'Entropy_Reg:{entropy_regularization}\nEpochs:{epochs}\n\n')
    fid.write(f'Greedy_Eval:{greedy_eval}\n\n')
    fid.write(f'episodes_per_iterations:{episodes_per_iterations}\nnum_iterations:{iterations}\nrandom_seed:{random_seed}\n\n')
    fid.write(f'PPO_policy_store_frequency:{PPO_policy_store_frequency}\nPPO_loss_avg_interval:{PPO_loss_avg_interval}\n')
    fid.close()

