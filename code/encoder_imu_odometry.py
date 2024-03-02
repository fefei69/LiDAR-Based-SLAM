import numpy as np
from load_data import *
from sklearn.neighbors import NearestNeighbors
import pdb
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def transform_pose_matrix_to_cells(pose_matrix):
    init_pose = homegenous_transformation(np.eye(3),np.zeros(3))
    X = []
    Y = []
    # pdb.set_trace()
    for i in range(len(pose_matrix)):
        position = pose_matrix[i] @ init_pose 
        x = position[:3, 3][0]
        y = position[:3, 3][1]
        sx = np.ceil((x -(-30)) / 0.05 ).astype(np.int16)-1
        sy = np.ceil((y -(-30)) / 0.05 ).astype(np.int16)-1
        X.append(sx)
        Y.append(sy)
    return X, Y

def transform_pose_matrix_to_xy(pose_matrix,need_theata=False):
    init_pose = homegenous_transformation(np.eye(3),np.zeros(3))
    X = []
    Y = []
    Theta = []
    for i in range(len(pose_matrix)):
        position = pose_matrix[i] @ init_pose 
        x = position[:3, 3][0]
        y = position[:3, 3][1]
        X.append(x)
        Y.append(y)
        r = Rotation.from_matrix(position[:3, :3])
        deg = r.as_euler('zyx', degrees=False)
        Theta.append(deg[0])
    if need_theata == True:
        return X, Y, Theta
    else:
        return X, Y

def relative_pose_from_odometry():
    # Relative pose from Odometry: x, y positions
    Odo_1_to_T = ODOMETRY[1:]
    Odo_0_to_T_minus_1 = ODOMETRY[0:-1]
    Odometry_differece = Odo_1_to_T - Odo_0_to_T_minus_1
    # Relative pose from Odometry: Theta
    THETA_1_to_T = THETA[1:]
    THETA_0_to_T_minus_1 = THETA[0:-1]
    Theta_differece = THETA_1_to_T - THETA_0_to_T_minus_1
    return Odometry_differece, Theta_differece  

def generate_relative_pose_matrix(pose):
    relative_poses = []
    # relative pose from t+1 to t since the icp inputis target to source
    for i in range(len(pose)-1):
        pose_t1_inv = np.linalg.inv(pose[i+1])
        pose_t1_to_t = pose_t1_inv @ pose[i]
        relative_poses.append(pose_t1_to_t)
    return relative_poses

def generate_relative_pose_matrix_normal_convention(pose):
    relative_poses_t_to_t1 = []
    # relative pose from t to t+1 since the icp inputis target to source
    for i in range(len(pose)-1):
        pose_t1_inv = np.linalg.inv(pose[i])
        pose_t1_to_t = pose_t1_inv @ pose[i+1]
        relative_poses_t_to_t1.append(pose_t1_to_t)
    return relative_poses_t_to_t1

def generate_pose_matrix():
    init_pose = homegenous_transformation(np.eye(3),np.zeros(3))
    X = []
    Y = []
    poses = []
    for i,theta in enumerate(THETA):
        p = np.array([ODOMETRY[:,0][i], ODOMETRY[:,1][i], 0])
        pose_matrix = homegenous_transformation(Rotation.from_euler('z',theta).as_matrix(),p)
        position = pose_matrix @ init_pose 
        # pdb.set_trace()
        # position[:3, 3]
        x = position[:3, 3][0]
        y = position[:3, 3][1]
        X.append(x)
        Y.append(y)
        poses.append(pose_matrix)
    return poses
def homegenous_transformation(R, t):
    T = np.eye(4)
    # Open3D might apply T directly (T @ pc)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
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
# print(ODOMETRY[:,0],ODOMETRY[:,1])
POSE = generate_pose_matrix()
relative_pose = generate_relative_pose_matrix(POSE)
relative_pose_t_to_t1 = generate_relative_pose_matrix_normal_convention(POSE)
# read numpy
ld_pose = np.load('results/Estimated_trajectory_dataset20_change_icp_input.npy')
# print(ld_pose.shape)
ld_x,ld_x = transform_pose_matrix_to_xy(ld_pose)

# Odometry_differece, Theta_differece = relative_pose_from_odometry()
# Test if the relative pose is correct
# print(Odometry_differece[1000])
# print(Rotation.from_euler('z',Theta_differece[1000]).as_matrix())
# h = homegenous_transformation(Rotation.from_euler('z',Theta_differece[1000]).as_matrix(),np.append(Odometry_differece[1000],0))
# print(relative_pose[1000]-h)
# Visualize odometry
# plt.plot(ODOMETRY[:,0],ODOMETRY[:,1])
# X,Y = transform_pose_matrix_to_xy(POSE)
# np.save('IMU_Od_POSE.npy',np.asarray(POSE))
# pose_store = np.load('results/Estimated_trajectory_dataset20_change_icp_input.npy')
# X,Y = transform_pose_matrix_to_xy(pose_store)
# x,y = transform_pose_matrix_to_xy(POSE)
# print(pose_store)
# print(POSE)
# plt.plot(X,Y)
# plt.plot(x,y)

# Testing map
# map = np.load("map/MAP_values.npy")
# pose_store = np.load('results/Estimated_trajectory_dataset20_change_icp_input.npy')
# sx,sy = transform_pose_matrix_to_cells(pose_store)
# pdb.set_trace()
# plt.plot(sx,sy)
# plt.imshow(map,cmap="cividis",interpolation='nearest')
# plt.colorbar()
# plt.title("Occupancy grid map")
# plt.show()
