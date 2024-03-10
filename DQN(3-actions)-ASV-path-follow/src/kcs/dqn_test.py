import numpy as np
import test_wp_track as wpt
import math
model_name = 'model_012'

def cardioid(flag=0):
    npoints = 18

    if flag == 0:
        # Starboard turning cardioid
        theta = np.linspace(0, 2 * np.pi, num=npoints, endpoint=True)
        theta_des = np.linspace(0, 2 * np.pi, num=1000)
        psi0 = 0
    else:
        # Port turning cardioid
        theta = np.linspace(0, -2 * np.pi, num=npoints, endpoint=True)
        theta_des = np.linspace(0, -2 * np.pi, num=1000)
        psi0 = 0

    a_cardioid = 10
    x_wp = a_cardioid * (2 * np.cos(theta) - np.cos(2 * theta))
    y_wp = a_cardioid * (2 * np.sin(theta) - np.sin(2 * theta))
    
    x_des = a_cardioid * (2 * np.cos(theta_des) - np.cos(2 * theta_des))
    y_des = a_cardioid * (2 * np.sin(theta_des) - np.sin(2 * theta_des))
    
    return npoints, x_wp, y_wp, psi0, x_des, y_des

def astroid(flag=0):
    npoints = 18

    if flag == 0:
        # Starboard turning astroid
        theta = np.linspace(0, 2 * np.pi, num=npoints, endpoint=True)
        theta_des = np.linspace(0, 2 * np.pi, num=1000)
        psi0 = 0
    else:
        # Port turning astroid
        theta = np.linspace(0, -2 * np.pi, num=npoints, endpoint=True)
        theta_des = np.linspace(0, -2 * np.pi, num=1000)
        psi0 = 0

    a_astroid = 20
    x_wp = a_astroid * np.cos(theta) ** 3
    y_wp = a_astroid * np.sin(theta) ** 3

    x_des = a_astroid * np.cos(theta_des) ** 3
    y_des = a_astroid * np.sin(theta_des) ** 3

    return npoints, x_wp, y_wp, psi0, x_des, y_des

def spiral_trajectory(radius=10, num_turns=4):
    npoints = 40

    rad_des = np.linspace(0, radius, num=1000)
    rad = np.linspace(0, radius, num=npoints, endpoint=True)

    theta_des = np.linspace(0, 2 * np.pi * num_turns, num=1000)
    theta = np.linspace(0, 2 * np.pi * num_turns, num=npoints, endpoint=True)

    x_des = rad_des * np.cos(theta_des)
    y_des = rad_des * np.sin(theta_des)

    x_wp = rad * np.cos(theta)
    y_wp = rad * np.sin(theta)

    x_wp = x_wp[::-1]
    y_wp = y_wp[::-1]

    psi0 = -np.pi/2

    return npoints, x_wp, y_wp, psi0, x_des, y_des

def straight():
    npoints = 9
    psi0 = 0
    x_wp = np.linspace(0, 64, num=npoints, endpoint=True)
    y_wp = 0 * x_wp
    xdes = x_wp
    ydes = y_wp
    return npoints, x_wp, y_wp, psi0, xdes, ydes

def sine_wave(amplitude=20, frequency=5):
    npoints = 40
    c = 15
    theta_des = np.linspace(0, 2 * np.pi, num=1000)
    theta= np.linspace(0, 2 * np.pi, num=npoints, endpoint=True)

    x_des = c * theta_des
    y_des = amplitude * np.sin(frequency * theta_des)

    x_wp = c * theta
    y_wp = amplitude * np.sin(frequency * theta)

    psi0 = np.pi/2

    return npoints, x_wp, y_wp, psi0, x_des, y_des

def star():
    npoints = 6

    x_wp = [0,5.878,-9.511,9.511,-5.878,0]
    y_wp = [10,-8.09,3.09,3.09,-8.09,10]

    xdes = x_wp
    ydes = y_wp

    psi0 = np.arctan2(y_wp[1]-y_wp[0],x_wp[1]-x_wp[0])

    return npoints, x_wp, y_wp, psi0, xdes, ydes



def single_wp(quadrant=1, len=10):
    npoints = 2
    psi0 = 0

    if quadrant == 1:
        x_wp = np.array([0, len])
        y_wp = np.array([0, len])
    elif quadrant == 2:
        x_wp = np.array([0, -len])
        y_wp = np.array([0, len])
    elif quadrant == 3:
        x_wp = np.array([0, -len])
        y_wp = np.array([0, -len])
    elif quadrant == 4:
        x_wp = np.array([0, len])
        y_wp = np.array([0, -len])

    xdes = x_wp
    ydes = y_wp
    return npoints, x_wp, y_wp, psi0, xdes, ydes

# No wind condition

# Test Wave periods range from 5 s to 20 s
# Test Wave heights ranges from 1m to 4m

# Single waypoint tracking in four quadrants
n, xwp, ywp, psi0, xdes, ydes = single_wp(quadrant=1)
wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='quadrant_01', xdes=xdes, ydes=ydes)
n, xwp, ywp, psi0, xdes, ydes = single_wp(quadrant=2)
wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='quadrant_02', xdes=xdes, ydes=ydes)
n, xwp, ywp, psi0, xdes, ydes = single_wp(quadrant=3)
wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='quadrant_03', xdes=xdes, ydes=ydes)
n, xwp, ywp, psi0, xdes, ydes = single_wp(quadrant=4)
wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='quadrant_04', xdes=xdes, ydes=ydes)

# Cardiod starboard turn
n, xwp, ywp, psi0, xdes, ydes = cardioid(flag=0)
wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='cardiod_stbd', xdes=xdes, ydes=ydes)

# Cardiod port turn
n, xwp, ywp, psi0, xdes, ydes = cardioid(flag=1)
wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='cardiod_port', xdes=xdes, ydes=ydes)

# astroid starboard turn
n, xwp, ywp, psi0, xdes, ydes = astroid(flag=0)
wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='astroid_stbd', xdes=xdes, ydes=ydes)

# astroid port turn
n, xwp, ywp, psi0, xdes, ydes = astroid(flag=1)
wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='astroid_port', xdes=xdes, ydes=ydes)

#Spiral maneuver
n, xwp, ywp, psi0, xdes, ydes = spiral_trajectory()
wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='spiral', xdes=xdes, ydes=ydes)

# Star turn
n, xwp, ywp, psi0, xdes, ydes = star()
wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='star', xdes=xdes, ydes=ydes)

#sine wave
n, xwp, ywp, psi0, xdes, ydes = sine_wave()
wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='sine_wave', xdes=xdes, ydes=ydes)

