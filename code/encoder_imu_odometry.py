import numpy as np
from load_data import *
from sklearn.neighbors import NearestNeighbors
import pdb
import matplotlib.pyplot as plt

def sync_data(imu_stamps,encoder_stamps,imu_angular_velocity):
    '''
    Use encoder time stamps to pair with imu angular velocity, 
    so the final time stamps will be encoder time stamps
    '''
    neighbbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(imu_stamps.reshape(-1,1))
    distances, indices = neighbbors.kneighbors(encoder_stamps.reshape(-1,1))
    yaw_rate = imu_angular_velocity[2][indices]
    return yaw_rate

def Encoder(encoder_counts,time_intervals):
    '''
    Encoder Counts: [FR, FL, RR, RL]

    '''
    # Encoder counts Drop first tick to match time intervals
    FR = encoder_counts[0][1:]
    FL = encoder_counts[1][1:]
    RR = encoder_counts[2][1:]
    RL = encoder_counts[3][1:]
    # Distamce between wheels (mm to m)
    L = ((476.25-311.15)/2 + 311.15)/1000
    D_Left_wheel = (FL + RL)/2*0.0022
    D_Right_wheel = (FR + RR)/2*0.0022
    V_Left_wheel = D_Left_wheel/time_intervals
    V_Right_wheel = D_Right_wheel/time_intervals
    # r = L/2 * (V_Right_wheel + V_Left_wheel)/(V_Right_wheel - V_Left_wheel)
    V = (V_Right_wheel + V_Left_wheel)/2
    return V
    
def Discrete_time_differential_drive_model(V,angular_velocity,time_intervals):
    '''
    Discrete time differential-drive kinematic model 
    using Euler Discretization

    x = []
    V: Linear velocity
    theta: Angular velocity
    dt: time interval
    '''
    # Drop first omega to match time intervals
    angular_velocity = angular_velocity[1:]
    x = np.array([0,0,0])
    theta = 0
    theta_history = []
    theta_history.append(theta)
    x_history = []
    x_history.append(x[:2])
    for i in range(len(time_intervals)-1):
        x_t1 = x + np.array([V[i]*np.cos(theta), V[i]*np.sin(theta), angular_velocity[i][0]]) * time_intervals[i]
        theta = x_t1[2]
        theta_history.append(theta)
        x = x_t1
        x_history.append(x[:2])
    return np.asarray(x_history), np.asarray(theta_history)

def odometry_from_motion_model(encoder_stamps,encoder_counts,imu_stamps,imu_angular_velocity):
    ts_1_to_T = encoder_stamps[1:]
    ts_0_to_T_minus_1 = encoder_stamps[0:-1]
    time_intervals = ts_1_to_T - ts_0_to_T_minus_1
    vel = Encoder(encoder_counts,time_intervals)
    yaw_rate = sync_data(imu_stamps,encoder_stamps,imu_angular_velocity)
    X, Theta= Discrete_time_differential_drive_model(vel,yaw_rate,time_intervals)
    return X, Theta

dataset = 20
with np.load("../data/Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

with np.load("../data/Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

# print(encoder_counts.shape)
# print(encoder_stamps.shape)
# print(imu_stamps.shape)
# ts_1_to_T = encoder_stamps[1:]
# ts_0_to_T_minus_1 = encoder_stamps[0:-1]
# time_intervals = ts_1_to_T - ts_0_to_T_minus_1
# vel = Encoder(encoder_counts,time_intervals)
# yaw_rate = sync_data(imu_stamps,encoder_stamps,imu_angular_velocity)
# X = Discrete_time_differential_drive_model(vel,yaw_rate,time_intervals)
ODOMETRY, THETA= odometry_from_motion_model(encoder_stamps,encoder_counts,imu_stamps,imu_angular_velocity)
# print(ODOMETRY)
# plt.plot(ODOMETRY[:,0],ODOMETRY[:,1])
# plt.show()