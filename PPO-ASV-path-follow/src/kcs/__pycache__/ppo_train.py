import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_agents
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import wrappers
from tf_agents.policies import random_tf_policy
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PolynomialDecay
from environment import ship_environment
import os
import time


def collect_data(environment, policy, buffer):
    global timestep_counter
    time_step = environment.reset()
    episode_return = 0
    
    # time_step.step_type:  0-> initial step,1->intermediate, 2-> terminal step
    while not np.equal(time_step.step_type, 2):

        action_step = policy.action(time_step)
        action=action_step.action
        
        next_time_step = environment.step(action_step.action)
        episode_return += next_time_step.reward.numpy()[0]
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        buffer.add_batch(traj)
        time_step=next_time_step
        timestep_counter += 1

    return episode_return,agent


def ppo_train(params):
    mname = params.model_name
    if not os.path.exists('saved_models/' + mname):
        if not os.path.exists('saved_models'):
            os.mkdir('saved_models')
        os.mkdir('saved_models/' + mname)
        os.mkdir('saved_models/' + mname + '/plots')

    params.print_params('saved_models/' + mname)

    env = wrappers.TimeLimit(ship_environment(train_test_flag=0, wind_flag=params.wind_flag,
                                              wind_speed=params.wind_speed, wind_dir=params.wind_dir,
                                              wave_flag=params.wave_flag, wave_height=params.wave_height,
                                              wave_period=params.wave_period, wave_dir=params.wave_dir),
                             duration=params.duration)
    tf_env = tf_py_environment.TFPyEnvironment(env)


    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=params.actor_fc_layers,
        activation_fn=tf.keras.activations.tanh)

    value_net = value_network.ValueNetwork(
        tf_env.observation_spec(),
        fc_layer_params=params.value_fc_layers,
        activation_fn=tf.keras.activations.tanh)

    agent = ppo_clip_agent.PPOClipAgent(
        time_step_spec=tf_env.time_step_spec(),
        action_spec=tf_env.action_spec(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
        actor_net = actor_net,
        value_net = value_net,
        greedy_eval = True,
        importance_ratio_clipping = params.clip,
        lambda_value = params.lambda_val,
        discount_factor = params.gamma,
        entropy_regularization = 0.01,
        num_epochs = params.epochs,
        use_gae= True,
        use_td_lambda_return= True,
        normalize_rewards = True,
        normalize_observations= True,
        debug_summaries = False,
        )

    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=params.replay_buffer_max_length)

        
    timestep_counter=0
    ep_counter=0

    avg_losses, avg_returns = [], []


    for i in range(params.iterations):

        replay_buffer.clear()
        returns = []

        print('ITERATION NO: ',i+1)

        for ep in range(params.episodes_per_iteration):
            ep_return=collect_data(tf_env, agent.policy, replay_buffer, dataset, agent, ep_counter, params)
            returns.append(ep_return)
            print('Episode {}: returns={}'.format(ep+1,ep_return))

        avg_returns.append(np.mean(returns))
        print('ITERATION NO:{} END; AVG RETURNS={}'.format (i + 1,np.mean(avg_returns[-1])))

        iterator=iter(dataset)
        experience=replay_buffer.gather_all()
        agent.train(experience)
        
        
       
        if (i+1) % params.PPO_policy_store_frequency == 0 and i+1 >= 80:
            policy_dir = os.path.join('saved_models', mname, mname + '_ep_' + str(i))
            tf_policy_saver = policy_saver.PolicySaver(agent.policy)
            tf_policy_saver.save(policy_dir)

      
    train_loss = np.array(train_loss)
    interval = params.PPO_loss_avg_interval
    avg_losses, avg_returns = [], []


    for i in range(len(train_loss) - interval):
        avg_returns.append(sum(returns[i:i + interval]) / interval)
        avg_losses.append(sum(train_loss[i:i + interval]) / interval)

    mat_dict = {'returns':np.array(returns), 'loss':np.array(train_loss),
                'avg_returns':np.array(avg_returns), 'avg_loss':np.array(avg_losses)}

    savemat('saved_models/'+mname+'/'+mname+'.mat', mat_dict)

    plt.figure()
    plt.title("Returns vs. Episodes")
    plt.ylabel("Returns")
    plt.plot(avg_returns)
    plt.xlabel("Episodes")
    plt.grid()
    plt.savefig('saved_models/'+mname+'/plots/'+mname+'_return.png', dpi=600)

    plt.figure()
    plt.title("Train-loss vs. Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.plot(avg_losses)
    plt.grid()
    plt.savefig('saved_models/'+mname+'/plots/'+mname+'_loss.png', dpi=600)

if __name__ =="__main__":
    import hyperparams as params
    ppo_train(params)

