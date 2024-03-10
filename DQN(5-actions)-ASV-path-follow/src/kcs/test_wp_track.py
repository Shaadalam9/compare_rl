import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import wrappers
from scipy.integrate import solve_ivp
import os
from environment import ship_environment
import re
from scipy.io import savemat

def wp_track(model_name, wind_flag=0, wind_speed=0,wind_dir=0,
             wave_flag=0, wave_height=0, wave_period=0, wave_dir=0,
             npoints=None, x_wp=None, y_wp=None, psi0=None,traj_str='',
             xdes=None, ydes=None):

    dir_list = os.listdir('saved_models/' + model_name)
    regex = re.compile(model_name + '_ep_')
    new_dir_list = list(filter(regex.match, dir_list))

    for policy_name in new_dir_list:

        env = wrappers.TimeLimit(ship_environment(train_test_flag=1,
                                                  wind_flag=wind_flag, wind_speed=wind_speed, wind_dir=wind_dir,
                                                  wave_flag=wave_flag, wave_height=wave_height,
                                                  wave_period=wave_period, wave_dir=wave_dir,
                                                  test_x_waypoints=x_wp,
                                                  test_y_waypoints=y_wp,
                                                  nwp=npoints,
                                                  initial_obs_state=[1, 0, 0, x_wp[0], y_wp[0], psi0, 0]), duration=5000)
        
        tf_env = tf_py_environment.TFPyEnvironment(env)

        train_step_counter = tf.Variable(0)

        # Reset the train step
        episodes = 1

        saved_policy = tf.compat.v2.saved_model.load('saved_models/'+ model_name + '/' + policy_name)

        for _ in range(episodes):

            time_step = tf_env.reset()
            time1 = 0

            while not np.equal(time_step.step_type, 2):
                action_step = saved_policy.action(time_step)
                time_step = tf_env.step(action_step.action)
                time1 += 1
            print(time1)
        
        plt.figure()
        X = env.x_traj
        Y = env.y_traj
        Psi = env.psi_traj
        x_ship = np.array([-0.5, -0.5, 0.25, 0.5, 0.25, -0.5, -0.5, 0.5, 0.25, 0, 0])
        y_ship = 16.1 / 230 * np.array([-1, 1, 1, 0, -1, -1, 0, 0, 1, 1, -1])

        plt.plot(X,Y)
        plt.plot(x_wp[0], y_wp[0], 'ro')

        # Waypoints
        x_wp = env.test_x_waypoints
        y_wp = env.test_y_waypoints

        plt.plot(xdes,ydes)
        plt.scatter(x_wp,y_wp)

        # Plot ship geometry on path
        m = 9
        for i in range(1,m):
            m_indx = i * (len(X) // m)
            psi_new = Psi[m_indx]
            x_new_ship = X[m_indx] + x_ship * np.cos(psi_new) - y_ship * np.sin(psi_new)
            y_new_ship = Y[m_indx] + x_ship * np.sin(psi_new) + y_ship * np.cos(psi_new)
            plt.plot(x_new_ship, y_new_ship, 'r')

        plt.xlabel("X/L")
        plt.ylabel("Y/L")
        plt.axis('equal')
        # plt.grid(b=True,which='major',color='#666666',linestyle='-')
        # # plt.minorticks_on()
        # plt.grid(b=True,which='minor',color='#999999',linestyle='-',alpha=0.2)
        # plt.rc('xtick', labelsize=14)
        # plt.rc('ytick', labelsize=14)
        # plt.legend(["Path Followed","Start Waypoint","Predefined Path"],loc="lower left")
        plt.grid()
        
        ax = plt.gca()
        # ax.invert_yaxis()
        
        delta_dot = np.array(env.r_traj)
        delta_dot = np.sqrt((sum(np.square(delta_dot)))/len(delta_dot))

        delta_c = np.array(env.action_traj)
        delta_c = np.sqrt((sum(np.square(delta_c)))/len(delta_c))


        if wind_flag == 0 and wave_flag == 0:
            plt.title(f'Calm Water',size = 18)
            plt_str = f'saved_models/{model_name}/plots/{traj_str}_{policy_name}.png'
            plt.savefig(plt_str, dpi=600)
        elif wind_flag == 1 and wave_flag == 0:
            plt.title("$V_{w}^{'}=6$  $\beta_{w}=45\degree$",size = 18)
            plt_str = f'saved_models/{model_name}/plots/{traj_str}_{policy_name}_WS{wind_speed}_WD{wind_dir}.png'
            plt.savefig(plt_str, dpi=600)
        elif wind_flag == 0 and wave_flag == 1:
            plt.title("$W_{h}=6$ $T_{w}=2$ $Ψ_{w}=45\degree$",size = 18)
            plt_str = f'saved_models/{model_name}/plots/{traj_str}_{policy_name}_WH{wind_speed}_WD{wave_dir}.png'
            plt.savefig(plt_str, dpi=600)
        elif wind_flag == 1 and wave_flag == 1:
            plt.title("$V_{w}^{'}=6$  $Ψ_{w}=45\degree$ ; $W_{h}=6$ $T_{w}=2$ $Ψ_{w}=45\degree$",size = 18)
            plt_str = f'saved_models/{model_name}/plots/{traj_str}_{policy_name}_WH{wind_speed}_WD{wind_dir}.png'
            plt.savefig(plt_str, dpi=600)

        print("rmse:" , (np.sqrt(np.mean(np.square(env.cross_trk_err_traj)))))
        print("Actual Delta",(np.mean(np.abs(env.delta_traj))))


        # print(env.course_ang_err_traj)
        # ype = np.transpose(np.vstack(env.cross_trk_err_traj))
        # cross_ang_err = np.transpose(np.vstack(env.course_ang_err_traj))
        # comm_rudd = np.transpose(np.vstack(env.comm_rudd))
        X = np.transpose(np.vstack(X))
        Y = np.transpose(np.vstack(Y))
        # savemat('rud_ang.mat',{'rud':comm_rudd})
        # savemat('ype_RL_5.mat',{'ype_RL_5':ype})
        # savemat('cae_5.mat',{'cae_5':cross_ang_err})
        # savemat('act_rudd_6.mat',{'delta_6':env.delta_traj})

        savemat('X_RL_5_case_VI.mat',{'X_5_1':X, 'Y_5_1':Y})
        # savemat('Y_RL_3.mat',{'Y_3':Y})

