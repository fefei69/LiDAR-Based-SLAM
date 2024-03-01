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
    # trajectory from scan matching
    pose_store = np.load('results/Estimated_trajectory_dataset21_change_icp_input.npy')
    # All starting points
    sx,sy = transform_pose_matrix_to_cells(pose_store)
    Occupancy_Mapping(synced_lidar_ranges,pose_store,sx,sy)

# print(occupied)
# print(lidar_in_world_frame.shape)
# plt.plot(lidar_in_world_frame[:,0],lidar_in_world_frame[:,1],"k.")
# plt.plot(lidar_pc[:,0],lidar_pc[:,1],"r.")
# plt.show()
# test_mapCorrelation(synced_lidar_ranges[:,0])
# plt.plot(ODOMETRY[:,0],ODOMETRY[:,1],"r",label="Odometry",linewidth=2.0)
# plt.show()

