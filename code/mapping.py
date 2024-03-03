# Occupancy mapping
from pr2_utils import *
from lidar_scan_matching import *

def visualize_mapping():
    # visualize mapping
    map_value = np.load("map/MAP_values_scan_matching_dataset21.npy")
    plt.plot(sx,sy,color='orange',label="Robot Trajectory")
    plt.imshow(map_value,cmap="cividis",interpolation='nearest')
    plt.colorbar()
    plt.title("Occupancy grid map of lidar scan matching dataset 21")
    plt.legend()
    plt.savefig("map/Occupancy_grid_map_scan_matching_dataset21.png")
    plt.show()

if __name__ == "__main__":
    print("Dataset:",dataset)
    # Trajectory from imu odometry
    # pose_store = np.load(f'robot_trajectory/IMU_Odometry_Pose_dataset{dataset}.npy')

    # Trajectory from scan matching
    # pose_store = np.load(f'robot_trajectory/lidar/Estimated_trajectory_dataset{dataset}_change_icp_input.npy')

    # Trajectory from gtsam
    pose_store = np.load(f'robot_trajectory/gtsam_optimized/dataset{dataset}_gtsam_icp_imu_icp.npy') 
    pose_store = state_matrix_to_pose_matrix(pose_store)
    # All starting points 
    sx,sy = transform_pose_matrix_to_cells(pose_store)
    Occupancy_Mapping(synced_lidar_ranges,pose_store,sx,sy)



