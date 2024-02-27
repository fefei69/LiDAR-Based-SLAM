import numpy as np
from icp_warm_up.utils import read_canonical_model, load_pc, data_association, visualize_icp_result, icp, homegenous_transformation
import matplotlib.pyplot as plt
from encoder_imu_odometry import *
from pr2_utils import *
import pdb
import time
from scipy.spatial.transform import Rotation
from tqdm import tqdm
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

def sync_data(lid_stamps,encoder_stamps,lidar_ranges):
    '''
    Use encoder time stamps to pair with lidar ranges,
    so the final time stamps will be encoder time stamps
    '''
    neighbbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(lid_stamps.reshape(-1,1))
    distances, indices = neighbbors.kneighbors(encoder_stamps.reshape(-1,1))
    lidar = lidar_ranges[:,indices]
    return np.squeeze(lidar)

def transform_lidar_to_xy(lidar_ranges):
    theta = np.linspace(-135/180*np.pi,135/180*np.pi,len(lidar_ranges))
    X_lidar = []
    Y_lidar = []
    for i, r in enumerate(lidar_ranges):
        x = r * np.cos(theta[i])
        y = r * np.sin(theta[i])
        # Transform from lidar frame to robot body frame
        X_lidar.append(x-0.135)          
        Y_lidar.append(y)
    return np.array(X_lidar), np.array(Y_lidar)

with np.load("../data/Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamsp = data["time_stamps"]  # acquisition times of the lidar scans


theta = np.linspace(-135/180*np.pi,135/180*np.pi,len(lidar_ranges))
ts_1_to_T = encoder_stamps[1:]
ts_0_to_T_minus_1 = encoder_stamps[0:-1]
encoder_time_intervals = ts_1_to_T - ts_0_to_T_minus_1
synced_lidar_ranges = sync_data(lidar_stamsp,encoder_stamps,lidar_ranges)
X_lidar, Y_lidar = transform_lidar_to_xy(synced_lidar_ranges[:,1])
print(X_lidar.shape)
pos_diff, theta_diff = relative_pose_from_odometry()
Old_Tansformation = homegenous_transformation(np.eye(3),np.zeros(3))
# initail pose
X_init = np.array([0,0,0,1])
X_trajectory = [X_init]
# Drop first lidar data to match time intervals
synced_lidar_ranges = synced_lidar_ranges[:,1:]
# pdb.set_trace()
# initial pose
old_pose = POSE[0]
pose_history = [old_pose]
start_time = time.time()
rel_pose = homegenous_transformation(np.eye(3),np.zeros(3))
if __name__ == "__main__":
    for i in tqdm(range(synced_lidar_ranges.shape[1]-1)):
        # Lidar point cloud at time t and t+1
        X_lidar_t, Y_lidar_t = transform_lidar_to_xy(synced_lidar_ranges[:,i])
        X_lidar_t_1, Y_lidar_t_1 = transform_lidar_to_xy(synced_lidar_ranges[:,i+1])
        lidar_pc_t = np.stack((X_lidar_t,Y_lidar_t,np.zeros(X_lidar_t.shape))).T
        lidar_pc_t_1 = np.stack((X_lidar_t_1,Y_lidar_t_1,np.zeros(X_lidar_t_1.shape))).T
        # init_R = theta_diff[i]
        init_R = relative_pose_t_to_t1[i][:3,:3]
        # init_R = rel_pose[:3,:3]
        # print(Rotation.from_euler('z',init_R).as_matrix())
        # make initial translation 3 dimensional
        # init_T = np.append(pos_diff[i],0)
        init_T = relative_pose_t_to_t1[i][:3,3]
        # init_T = rel_pose[:3,3]
        # For testing
        # rot = Rotation.from_euler('z',1.5).as_matrix()
        # lidar_pc_source_test = lidar_pc_t
        # lidar_pc_target_test = lidar_pc_t_1 @ rot.T + np.array([1,2,0]) @ rot.T
        # print("target rotated")
        # visualize_icp_result(lidar_pc_source_test, lidar_pc_target_test, np.eye(4))
        # iCP optimal_pc is transformed lidar_pc_t_1, while T is from time t to t+1
        optimal_pc, T = icp_for_scan_matching(lidar_pc_t_1, lidar_pc_t, init_R, init_T, down_sample_rate=1,max_iterations=15,rot_mat=True)
        # For testing
        # optimal_pc, T = icp_for_scan_matching(lidar_pc_source_test, lidar_pc_target_test, rot.T, np.array([1,2,0]) @ rot, down_sample_rate=1,rot_mat=True)
        # print("after icp")
        # visualize_icp_result(lidar_pc_t, optimal_pc, np.eye(4))
        # visualize_icp_result(lidar_pc_source_test, lidar_pc_target_test, np.eye(4))
        # pdb.set_trace()
        # Optimized_pose =  old_pose @ relative_pose_t_to_t1[i]
        Optimized_pose =  old_pose @ T
        old_pose = Optimized_pose
        # rel_pose = np.linalg.inv(Optimized_pose) @ POSE[i]
        pose_history.append(Optimized_pose)
        # New_Tansformation = Old_Tansformation @ T
        # Old_Tansformation = New_Tansformation
        # New_X =  New_Tansformation @ X_init
        # X_trajectory.append(New_X)
        # print("Lidar data: ",i)
        # pdb.set_trace()
    # X_trajectory = np.asarray(X_trajectory)
    X_trajectory, Y_trajectory = transform_pose_matrix_to_xy(pose_history)
    end_time = time.time()
    print("Takes:  %s seconds " % (time.time() - start_time))
    # X_lidar_t, Y_lidar_t = transform_lidar_to_xy(synced_lidar_ranges[:,1000])
    # lidar_pc_test = np.stack((X_lidar_t,Y_lidar_t,np.zeros(X_lidar_t.shape))).T
    # visualize_icp_result(lidar_pc_test, lidar_pc_t_1, np.eye(4))
    # visualize_icp_result(lidar_pc_t, lidar_pc_t_1, New_Tansformation)
    # visualize_icp_result(lidar_pc_test, lidar_pc_t_1, New_Tansformation)
    # print("pos_diff",pos_diff[i])
    # print("theta_diff",theta_diff[i])
    # print("X_trajectory",X_trajectory)
    # pdb.set_trace()
    plt.plot(X_trajectory,Y_trajectory,"b",label="Estimated trajectory",linewidth=2.0)
    # plt.plot(X_trajectory[:,0],X_trajectory[:,1])
    plt.plot(ODOMETRY[:,0],ODOMETRY[:,1],"orange",label="Ground truth",linewidth=1.5)
    plt.legend()
    plt.savefig(f"results/Estimated_trajectory_dataset{dataset}_change_icp_input_many_icpiteration.png")
    plt.show()