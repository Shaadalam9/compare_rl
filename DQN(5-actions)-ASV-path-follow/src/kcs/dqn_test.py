import numpy as np
import test_wp_track as wpt
from scipy.io import savemat

model_name = 'model_002'

def ellipse(flag=0):
    npoints = 15

    if flag == 0:
        # Starboard turning ellipse
        theta = np.linspace(0, 2 * np.pi, num=npoints, endpoint=True)
        theta_des = np.linspace(0, 2 * np.pi, num=1000)
        psi0 = np.pi / 2
    else:
        # Port turning ellipse
        theta = np.linspace(0, -2 * np.pi, num=npoints, endpoint=True)
        theta_des = np.linspace(0, -2 * np.pi, num=1000)
        psi0 = -np.pi / 2

    a_ellipse = 14
    b_ellipse = 12
    x_wp = a_ellipse * np.cos(theta)
    y_wp = b_ellipse * np.sin(theta)
    #print("waypoint\n",theta)
    xdes = a_ellipse * np.cos(theta_des)
    ydes = b_ellipse * np.sin(theta_des)
    #print("Des\n",theta_des)

    # y_wp = [-4.575, 0.0,-8.299287821523833, -8.299300598373565, -50.85898197229759, -50.85890367472103, -8.299287821523833, 0.0 ]
    # y_wp = [y / 3.05 for y in y_wp]
    # x_wp = [10.675, 0.0, -40.40444638923671, -82.96397368162344, -82.9639281437996, -40.40440085132246, -40.40444638923671, 0.0]
    # x_wp = [x / 3.05 for x in x_wp]
    # psi0 = 145*np.pi/180
    #x_wp = [x / 3.05 for x in x_wp]
    #y_wp = [10.6171,0, -40.4040, -82.9640,-82.9640,-40.4040,  -40.4040, 0]
    #y_wp = [y / 3.05 for y in y_wp]
    # xdes = x_wp
    # ydes = y_wp

    return npoints, x_wp, y_wp, psi0, xdes, ydes


def spiral_trajectory(radius=15, num_turns=4):
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

    X_1 = np.transpose(np.vstack(x_des))
    Y_1 = np.transpose(np.vstack(y_des))

    savemat('X_main.mat',{'X':X_1, 'Y':Y_1})
    
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

    X_1 = np.transpose(np.vstack(x_des))
    Y_1 = np.transpose(np.vstack(y_des))

    savemat('X_main.mat',{'X':X_1, 'Y':Y_1})

    return npoints, x_wp, y_wp, psi0, x_des, y_des


def straight():
    npoints = 9
    psi0 = 0
    x_wp = np.linspace(0, 64*2, num=npoints, endpoint=True)
    y_wp = 0 * x_wp
    xdes = x_wp
    ydes = y_wp

    X_1 = np.transpose(np.vstack(xdes))
    Y_1 = np.transpose(np.vstack(ydes))

    savemat('X_main.mat',{'X':X_1, 'Y':Y_1})

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

    X_1 = np.transpose(np.vstack(x_des))
    Y_1 = np.transpose(np.vstack(y_des))

    savemat('X_main.mat',{'X':X_1, 'Y':Y_1})

    return npoints, x_wp, y_wp, psi0, x_des, y_des


def star():
    npoints = 6

    x_wp = [0,5.878,-9.511,9.511,-5.878,0]
    y_wp = [10,-8.09,3.09,3.09,-8.09,10]

    xdes = x_wp
    ydes = y_wp

    psi0 = np.arctan2(y_wp[1]-y_wp[0],x_wp[1]-x_wp[0])

    return npoints, x_wp, y_wp, psi0, xdes, ydes


def eight(flag=0):
    npoints = 15
    psi0 = 0
    circ_dia = 9

    if flag == 0:
        theta1 = np.linspace(-0.5 * np.pi, 1.5 * np.pi, npoints - npoints // 2, endpoint=True)
        theta1_des = np.linspace(-0.5 * np.pi, 1.5 * np.pi, 1000)
        x1 = circ_dia * np.cos(theta1)
        y1 = circ_dia * np.sin(theta1) + circ_dia
        x1des = circ_dia * np.cos(theta1_des)
        y1des = circ_dia * np.sin(theta1_des) + circ_dia

        theta2 = np.linspace(0.5 * np.pi, 2.5 * np.pi, npoints // 2, endpoint=False)
        theta2_des = np.linspace(0.5 * np.pi, 2.5 * np.pi, 1000)
        x2 = circ_dia * np.cos(theta2)
        y2 = circ_dia * np.sin(theta2) - circ_dia
        x2des = circ_dia * np.cos(theta2_des)
        y2des = circ_dia * np.sin(theta2_des) - circ_dia

        x_wp = np.append(x1, x2[::-1])
        y_wp = np.append(y1, y2[::-1])

        xdes = np.append(x1des, x2des[::-1])
        ydes = np.append(y1des, y2des[::-1])

    else:
        theta1 = np.linspace(-0.5 * np.pi, 1.5 * np.pi, npoints - npoints // 2, endpoint=True)
        theta1_des = np.linspace(-0.5 * np.pi, 1.5 * np.pi, 1000)
        x1 = circ_dia * np.cos(theta1)
        y1 = circ_dia * np.sin(theta1) + circ_dia
        x1des = circ_dia * np.cos(theta1_des)
        y1des = circ_dia * np.sin(theta1_des) + circ_dia

        theta2 = np.linspace(2.5 * np.pi, 0.5 * np.pi, npoints // 2, endpoint=False)
        theta2_des = np.linspace(2.5 * np.pi, 0.5 * np.pi, 1000)
        x2 = circ_dia * np.cos(theta2)
        y2 = circ_dia * np.sin(theta2) - circ_dia
        x2des = circ_dia * np.cos(theta2_des)
        y2des = circ_dia * np.sin(theta2_des) - circ_dia

        x_wp = np.append(x2, x1)
        y_wp = np.append(y2, y1)
        xdes = np.append(x2des, x1des[::-1])
        ydes = np.append(y2des, y1des[::-1])
    

    X_1 = np.transpose(np.vstack(xdes))
    Y_1 = np.transpose(np.vstack(ydes))

    savemat('X_main.mat',{'X':X_1, 'Y':Y_1})
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

# Single waypoint tracking in four quadrants
# n, xwp, ywp, psi0, xdes, ydes = single_wp(quadrant=1)
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='quadrant_01', xdes=xdes, ydes=ydes)
# n, xwp, ywp, psi0, xdes, ydes = single_wp(quadrant=2)
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='quadrant_02', xdes=xdes, ydes=ydes)
# n, xwp, ywp, psi0, xdes, ydes = single_wp(quadrant=3)
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='quadrant_03', xdes=xdes, ydes=ydes)
# n, xwp, ywp, psi0, xdes, ydes = single_wp(quadrant=4)
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='quadrant_04', xdes=xdes, ydes=ydes)

# Ellipse starboard turn
# n, xwp, ywp, psi0, xdes, ydes = ellipse(flag=0)
# wpt.wp_track(model_name, wind_flag=1, wind_speed=12, wind_dir=90, wave_flag=0, wave_height=2, wave_period=5, wave_dir=np.pi/2, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='ellipse_stbd', xdes=xdes, ydes=ydes)

# Ellipse port turn
# n, xwp, ywp, psi0, xdes, ydes = ellipse(flag=1)
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='ellipse_port', xdes=xdes, ydes=ydes)

# Cardiod starboard turn
# n, xwp, ywp, psi0, xdes, ydes = cardioid(flag=0)
# wpt.wp_track(model_name, wind_flag=1, wind_speed=3, wind_dir=0, wave_flag=1, wave_height=2, wave_period=15, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='cardiod_stbd', xdes=xdes, ydes=ydes)

# Cardiod port turn
# n, xwp, ywp, psi0, xdes, ydes = cardioid(flag=1)
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='cardiod_port', xdes=xdes, ydes=ydes)

# astroid starboard turn
# n, xwp, ywp, psi0, xdes, ydes = astroid(flag=0)
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='astroid_stbd', xdes=xdes, ydes=ydes)

# astroid port turn
# n, xwp, ywp, psi0, xdes, ydes = astroid(flag=1)
# wpt.wp_track(model_name, wind_flag=1, wind_speed=4, wind_dir=45, wave_flag=1, wave_height=3, wave_period=20, wave_dir=315, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='astroid_port', xdes=xdes, ydes=ydes)

# sine wave
n, xwp, ywp, psi0, xdes, ydes = sine_wave()
wpt.wp_track(model_name, wind_flag=1, wind_speed=3, wind_dir=90, wave_flag=1, wave_height=3, wave_period=8, wave_dir=90, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='sine_wave', xdes=xdes, ydes=ydes)

# Star turn
# n, xwp, ywp, psi0, xdes, ydes = star()
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='star', xdes=xdes, ydes=ydes)

# Spiral maneuver
# n, xwp, ywp, psi0, xdes, ydes = spiral_trajectory()
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='spiral', xdes=xdes, ydes=ydes)

# Straight line
# n, xwp, ywp, psi0, xdes, ydes = straight()
# wpt.wp_track(model_name, wind_flag=1, wind_speed=6, wind_dir=-np.pi/2, wave_flag=1, wave_height=3, wave_period=10, wave_dir=np.pi/4, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='straight', xdes=xdes, ydes=ydes)

# Eight bottom first
# n, xwp, ywp, psi0, xdes, ydes = eight(flag=0)
# wpt.wp_track(model_name, wind_flag=1, wind_speed=6, wind_dir=90, wave_flag=1, wave_height=3, wave_period=10, wave_dir=45, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='eight_bottom', xdes=xdes, ydes=ydes)

# Eight top first
# n, xwp, ywp, psi0, xdes, ydes = eight(flag=1)
# wpt.wp_track(model_name, wind_flag=0, wind_speed=6, wind_dir=0, wave_flag=0, wave_height=0, wave_period=0, wave_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='eight_top', xdes=xdes, ydes=ydes)
