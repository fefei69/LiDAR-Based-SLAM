import matplotlib.pyplot as plt
import numpy as np
from encoder_imu_odometry import transform_pose_matrix_to_xy, transform_pose_matrix_to_cells
from pr2_utils import state_matrix_to_pose_matrix
def imu_results():
    # IMU odometry
    pose_20 = np.load("robot_trajectory/IMU_Odometry_Pose_dataset20.npy")
    pose_21 = np.load("robot_trajectory/IMU_Odometry_Pose_dataset21.npy")
    x_tr20,y_tr20 = transform_pose_matrix_to_xy(pose_20)
    x_tr21,y_tr21 = transform_pose_matrix_to_xy(pose_21)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the trajectory for dataset 20 in the first subplot
    axs[0].plot(x_tr20, y_tr20, label="IMU Odometry (Dataset 20)")
    axs[0].set_title("IMU Odometry")
    axs[0].set_xlabel("X Position (m)")
    axs[0].set_ylabel("Y Position (m)")
    axs[0].grid()
    axs[0].legend(fontsize=8)

    # Plot the trajectory for dataset 21 in the second subplot
    axs[1].plot(x_tr21, y_tr21, label="IMU Odometry (Dataset 21)")
    axs[1].set_title("IMU Odometry")
    axs[1].set_xlabel("X Position (m)")
    axs[1].set_ylabel("Y Position (m)")
    axs[1].grid()
    axs[1].legend(fontsize=8)
    # plt.savefig("images/IMU_Odometry_Trajectory.png")
    plt.show()

def scan_matching_results():  
    # Scan matching
    # dataset 20
    x_tr20_lid = np.load("robot_trajectory/lidar/Estimated_trajectory_X_trajectory_dataset20.npy")
    y_tr20_lid = np.load("robot_trajectory/lidar/Estimated_trajectory_Y_trajectory_dataset20.npy")
    pose_20 = np.load("robot_trajectory/IMU_Odometry_Pose_dataset20.npy")
    x_tr20,y_tr20 = transform_pose_matrix_to_xy(pose_20)

    # dataset 21
    x_tr21_lid = np.load("robot_trajectory/lidar/Estimated_trajectory_X_trajectory_dataset21.npy")
    y_tr21_lid = np.load("robot_trajectory/lidar/Estimated_trajectory_Y_trajectory_dataset21.npy")
    pose_21 = np.load("robot_trajectory/IMU_Odometry_Pose_dataset21.npy")
    x_tr21,y_tr21 = transform_pose_matrix_to_xy(pose_21)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the trajectory for dataset 21 lidar in the first subplot
    axs[0].plot(x_tr20_lid, y_tr20_lid, color="orange", label="Scan Matching Trajectory")
    axs[0].plot(x_tr20, y_tr20,color="blue", label="IMU Odometry")
    axs[0].set_title("Dataset 20 Lidar Scan Matching")
    axs[0].set_xlabel("X Position (m)")
    axs[0].set_ylabel("Y Position (m)")
    axs[0].grid()
    axs[0].legend(fontsize=8)

    # Plot the trajectory for dataset 21 IMU odometry in the second subplot
    axs[1].plot(x_tr21_lid, y_tr21_lid, color="orange", label="Scan Matching Trajectory")
    axs[1].plot(x_tr21, y_tr21, color="blue", label="IMU Odometry")
    axs[1].set_title("Dataset 21 Lidar Scan Matching")
    axs[1].set_xlabel("X Position (m)")
    axs[1].set_ylabel("Y Position (m)")
    axs[1].grid()
    axs[1].legend(fontsize=8)
    # plt.tight_layout()
    # plt.savefig("images/Scan_Matching_Trajectory.png")
    plt.show()
    
def occupancy_grid_map_results(imu=True, scan_matching=False, gtsam=False):
    # Occupancy grid map
    # imu odometry
    map_imu20 = np.load("map/Occupancy_mapping/imu/MAP_values_imu_dataset20.npy")
    map_imu21 = np.load("map/Occupancy_mapping/imu/MAP_values_imu_dataset21.npy")
    # IMU odometry
    pose_20 = np.load("robot_trajectory/IMU_Odometry_Pose_dataset20.npy")
    pose_21 = np.load("robot_trajectory/IMU_Odometry_Pose_dataset21.npy")
    x_tr20,y_tr20 = transform_pose_matrix_to_cells(pose_20)
    x_tr21,y_tr21 = transform_pose_matrix_to_cells(pose_21)

    # lidar scan matching
    map_icp20 = np.load("map/Occupancy_mapping/icp/MAP_values_scan_matching_dataset20.npy")
    map_icp21 = np.load("map/Occupancy_mapping/icp/MAP_values_scan_matching_dataset21.npy")
    # lidar dataset 20
    x_lid_pose20 = np.load("robot_trajectory/lidar/Estimated_trajectory_dataset20_change_icp_input.npy")
    x_tr20_lid,y_tr20_lid = transform_pose_matrix_to_cells(x_lid_pose20)
    # lidar dataset 21
    x_lid_pose21 = np.load("robot_trajectory/lidar/Estimated_trajectory_dataset21_change_icp_input.npy")
    x_tr21_lid,y_tr21_lid = transform_pose_matrix_to_cells(x_lid_pose21)

    # GTSAM
    map_gtsam20 = np.load("map/Occupancy_mapping/gtsam/MAP_values_gtsam_icp_imu_icp_dataset20.npy")
    map_gtsam21 = np.load("map/Occupancy_mapping/gtsam/MAP_values_gtsam_icp_imu_icp_dataset21.npy")
    # GTSAM optimized trajectory 20
    tr20_gtsam = np.load("robot_trajectory/gtsam_optimized/dataset20_gtsam_icp_imu_icp.npy")
    tr20_gtsam = state_matrix_to_pose_matrix(tr20_gtsam)
    x_tr20_gtsam, y_tr20_gtsam = transform_pose_matrix_to_cells(tr20_gtsam)
    # GTSAM optimized trajectory 21
    tr21_gtsam = np.load("robot_trajectory/gtsam_optimized/dataset21_gtsam_icp_imu_icp.npy")
    tr21_gtsam = state_matrix_to_pose_matrix(tr21_gtsam)
    x_tr21_gtsam, y_tr21_gtsam = transform_pose_matrix_to_cells(tr21_gtsam)

    fig, axs = plt.subplots(1, 2, figsize=(24, 12))
    if imu == True:
        # Plot IMU odometry for dataset 20
        im0 = axs[0].imshow(map_imu20, cmap="cividis", interpolation='nearest')
        axs[0].plot(x_tr20, y_tr20, color="orange", label="IMU Odometry (Dataset 20)")
        axs[0].set_title("Occupancy Map of IMU Odometry (Dataset 20)")
        axs[0].set_xlabel("X Position (m)")
        axs[0].set_ylabel("Y Position (m)")
        axs[0].legend()
        fig.colorbar(im0, ax=axs[0],shrink=0.6)

        # Plot IMU odometry for dataset 21
        im1 = axs[1].imshow(map_imu21, cmap="cividis", interpolation='nearest')
        axs[1].plot(x_tr21, y_tr21, color="orange", label="IMU Odometry (Dataset 21)")
        axs[1].set_title("Occupancy Map of IMU Odometry (Dataset 21)")
        axs[1].set_xlabel("X Position (m)")
        axs[1].set_ylabel("Y Position (m)")
        axs[1].legend()
        fig.colorbar(im1, ax=axs[1],shrink=0.6)
        plt.show()
    if scan_matching == True:
        # Plot lidar scan matching for dataset 20
        im0 = axs[0].imshow(map_icp20, cmap="cividis", interpolation='nearest')
        axs[0].plot(x_tr20_lid, y_tr20_lid, color = "orange", label="Lidar Scan Matching (Dataset 20)")
        axs[0].set_title("Occupancy Map of Lidar Scan Matching (Dataset 20)")
        axs[0].set_xlabel("X Position (m)")
        axs[0].set_ylabel("Y Position (m)")
        axs[0].legend()
        fig.colorbar(im0, ax=axs[0],shrink=0.6)
        # Plot lidar scan matching for dataset 21
        im1 = axs[1].imshow(map_icp21, cmap="cividis", interpolation='nearest')
        axs[1].plot(x_tr21_lid, y_tr21_lid, color = "orange",label="Lidar Scan Matching (Dataset 21)")
        axs[1].set_title("Occupancy Map of Lidar Scan Matching (Dataset 21)")
        axs[1].set_xlabel("X Position (m)")
        axs[1].set_ylabel("Y Position (m)")
        axs[1].legend()
        fig.colorbar(im1, ax=axs[1],shrink=0.6)
        plt.show()

        
    if gtsam == True:
        # Plot GTSAM for dataset 20
        im0 = axs[0].imshow(map_gtsam20, cmap="cividis", interpolation='nearest')
        axs[0].plot(x_tr20_gtsam, y_tr20_gtsam, color="orange", label="GTSAM Optimized Trajectory (Dataset 20)")
        axs[0].imshow(map_gtsam20, cmap="cividis", interpolation='nearest')
        axs[0].set_title("Occupancy Map of GTSAM Optimized Trajectory (Dataset 20)")
        axs[0].set_xlabel("X Position (m)")
        axs[0].set_ylabel("Y Position (m)")
        axs[0].legend()
        fig.colorbar(im0, ax=axs[0],shrink=0.6)
        # Plot GTSAM for dataset 21
        im1 = axs[1].imshow(map_gtsam21, cmap="cividis", interpolation='nearest')
        axs[1].plot(x_tr21_gtsam, y_tr21_gtsam, color="orange", label="GTSAM Optimized Trajectory (Dataset 21)")
        axs[1].imshow(map_gtsam21, cmap="cividis", interpolation='nearest')
        axs[1].set_title("Occupancy Map of GTSAM Optimized Trajectory (Dataset 21)")
        axs[1].set_xlabel("X Position (m)")
        axs[1].set_ylabel("Y Position (m)")
        axs[1].legend()
        fig.colorbar(im1, ax=axs[1],shrink=0.6)
        plt.show()
    # plt.savefig("images/Occupancy_Grid_Map_imu.png")
    # plt.savefig("images/Occupancy_Grid_Map_gtsam.png")
    plt.show()
    return 0

def texture_mapping_results(imu=True, scan_matching=False, gtsam=False):
    # Texture mapping
    # imu odometry
    map_imu20 = np.load("map/Texture_mapping/texture_mapping_dataset20_imu.npy")
    map_imu21 = np.load("map/Texture_mapping/texture_mapping_dataset21_imu.npy")
    # IMU odometry
    pose_20 = np.load("robot_trajectory/IMU_Odometry_Pose_dataset20.npy")
    pose_21 = np.load("robot_trajectory/IMU_Odometry_Pose_dataset21.npy")
    x_tr20,y_tr20 = transform_pose_matrix_to_cells(pose_20)
    x_tr21,y_tr21 = transform_pose_matrix_to_cells(pose_21)

    # lidar scan matching
    map_icp20 = np.load("map/Texture_mapping/texture_mapping_dataset20_change_kinect_to_body.npy")
    map_icp21 = np.load("map/Texture_mapping/texture_mapping_dataset21_change_kinect_to_body.npy")
    # lidar dataset 20
    x_lid_pose20 = np.load("robot_trajectory/lidar/Estimated_trajectory_dataset20_change_icp_input.npy")
    x_tr20_lid,y_tr20_lid = transform_pose_matrix_to_cells(x_lid_pose20)
    # lidar dataset 21
    x_lid_pose21 = np.load("robot_trajectory/lidar/Estimated_trajectory_dataset21_change_icp_input.npy")
    x_tr21_lid,y_tr21_lid = transform_pose_matrix_to_cells(x_lid_pose21)

    # GTSAM
    map_gtsam20 = np.load("map/Texture_mapping/texture_mapping_dataset20_gtsam_icp_imu_icp.npy")
    map_gtsam21 = np.load("map/Texture_mapping/texture_mapping_dataset21_gtsam_icp_imu_icp.npy")
    # GTSAM optimized trajectory 20
    tr20_gtsam = np.load("robot_trajectory/gtsam_optimized/dataset20_gtsam_icp_imu_icp.npy")
    tr20_gtsam = state_matrix_to_pose_matrix(tr20_gtsam)
    x_tr20_gtsam, y_tr20_gtsam = transform_pose_matrix_to_cells(tr20_gtsam)
    # GTSAM optimized trajectory 21
    tr21_gtsam = np.load("robot_trajectory/gtsam_optimized/dataset21_gtsam_icp_imu_icp.npy")
    tr21_gtsam = state_matrix_to_pose_matrix(tr21_gtsam)
    x_tr21_gtsam, y_tr21_gtsam = transform_pose_matrix_to_cells(tr21_gtsam)

    fig, axs = plt.subplots(1, 2, figsize=(24, 12))
    if imu == True:
        # Plot IMU odometry for dataset 20
        im0 = axs[0].imshow(map_imu20, cmap="cividis", interpolation='nearest')
        axs[0].plot(x_tr20, y_tr20, color = plt.cm.cividis(0.15), label="IMU Odometry (Dataset 20)",linewidth=3.0)
        axs[0].set_title("Occupancy Map of IMU Odometry (Dataset 20)")
        axs[0].set_xlabel("X Position (m)")
        axs[0].set_ylabel("Y Position (m)")
        axs[0].legend()

        # Plot IMU odometry for dataset 21
        im1 = axs[1].imshow(map_imu21, cmap="cividis", interpolation='nearest')
        axs[1].plot(x_tr21, y_tr21, color = plt.cm.cividis(0.15), label="IMU Odometry (Dataset 21)",linewidth=3.0)
        axs[1].set_title("Occupancy Map of IMU Odometry (Dataset 21)")
        axs[1].set_xlabel("X Position (m)")
        axs[1].set_ylabel("Y Position (m)")
        axs[1].legend()
        # plt.savefig("images/Texture_Map_imu.png")
        plt.show()

    if scan_matching == True:
        # Plot lidar scan matching for dataset 20
        im0 = axs[0].imshow(map_icp20, cmap="cividis", interpolation='nearest')
        axs[0].plot(x_tr20_lid, y_tr20_lid, color = plt.cm.cividis(0.15), label="Lidar Scan Matching (Dataset 20)",linewidth=3.0)
        axs[0].set_title("Occupancy Map of Lidar Scan Matching (Dataset 20)")
        axs[0].set_xlabel("X Position (m)")
        axs[0].set_ylabel("Y Position (m)")
        axs[0].legend()
        # Plot lidar scan matching for dataset 21
        im1 = axs[1].imshow(map_icp21, cmap="cividis", interpolation='nearest')
        axs[1].plot(x_tr21_lid, y_tr21_lid, color = plt.cm.cividis(0.15),label="Lidar Scan Matching (Dataset 21)",linewidth=3.0)
        axs[1].set_title("Occupancy Map of Lidar Scan Matching (Dataset 21)")
        axs[1].set_xlabel("X Position (m)")
        axs[1].set_ylabel("Y Position (m)")
        axs[1].legend()
        # plt.savefig("images/Texture_Map_icp.png")
        plt.show()
        
    if gtsam == True:
        # Plot GTSAM for dataset 20
        im0 = axs[0].imshow(map_gtsam20, cmap="cividis", interpolation='nearest')
        axs[0].plot(x_tr20_gtsam, y_tr20_gtsam, color = plt.cm.cividis(0.15), label="GTSAM Optimized Trajectory (Dataset 20)",linewidth=3.0)
        axs[0].imshow(map_gtsam20, cmap="cividis", interpolation='nearest')
        axs[0].set_title("Occupancy Map of GTSAM Optimized Trajectory (Dataset 20)")
        axs[0].set_xlabel("X Position (m)")
        axs[0].set_ylabel("Y Position (m)")
        axs[0].legend()
        # Plot GTSAM for dataset 21
        im1 = axs[1].imshow(map_gtsam21, cmap="cividis", interpolation='nearest')
        axs[1].plot(x_tr21_gtsam, y_tr21_gtsam, color = plt.cm.cividis(0.15), label="GTSAM Optimized Trajectory (Dataset 21)",linewidth=3.0)
        axs[1].imshow(map_gtsam21, cmap="cividis", interpolation='nearest')
        axs[1].set_title("Occupancy Map of GTSAM Optimized Trajectory (Dataset 21)")
        axs[1].set_xlabel("X Position (m)")
        axs[1].set_ylabel("Y Position (m)")
        axs[1].legend()
        # plt.savefig("images/Texture_Map_gtsam.png")
        plt.show()
    return 0
    
def gtsam_results(loop_closure=True):
    # IMU odometry
    pose_20 = np.load("robot_trajectory/IMU_Odometry_Pose_dataset20.npy")
    pose_21 = np.load("robot_trajectory/IMU_Odometry_Pose_dataset21.npy")
    x_tr20, y_tr20 = transform_pose_matrix_to_xy(pose_20)
    x_tr21, y_tr21 = transform_pose_matrix_to_xy(pose_21)
    # lidar dataset 20
    x_lid_pose20 = np.load("robot_trajectory/lidar/Estimated_trajectory_dataset20_change_icp_input.npy")
    x_tr20_lid, y_tr20_lid = transform_pose_matrix_to_xy(x_lid_pose20)
    # lidar dataset 21
    x_lid_pose21 = np.load("robot_trajectory/lidar/Estimated_trajectory_dataset21_change_icp_input.npy")
    x_tr21_lid, y_tr21_lid = transform_pose_matrix_to_xy(x_lid_pose21)
    # GTSAM
    # GTSAM optimized trajectory 20
    if loop_closure == True:
        # icp imu icp
        tr20_gtsam_icp = np.load("robot_trajectory/gtsam_optimized/dataset20_gtsam_icp_imu_icp.npy")
        tr20_gtsam_icp = state_matrix_to_pose_matrix(tr20_gtsam_icp)
        x_tr20_gtsam_icp, y_tr20_gtsam_icp = transform_pose_matrix_to_xy(tr20_gtsam_icp)
        # imu icp icp
        tr20_gtsam_imu = np.load("robot_trajectory/gtsam_optimized/dataset20_gtsam_imu_icp_icp.npy")
        tr20_gtsam_imu = state_matrix_to_pose_matrix(tr20_gtsam_imu)
        x_tr20_gtsam_imu, y_tr20_gtsam_imu = transform_pose_matrix_to_xy(tr20_gtsam_imu)
        # GTSAM optimized trajectory 21
        # icp imu icp
        tr21_gtsam_icp = np.load("robot_trajectory/gtsam_optimized/dataset21_gtsam_icp_imu_icp.npy")
        tr21_gtsam_icp = state_matrix_to_pose_matrix(tr21_gtsam_icp)
        x_tr21_gtsam_icp, y_tr21_gtsam_icp = transform_pose_matrix_to_xy(tr21_gtsam_icp)
        # imu icp icp
        tr21_gtsam_imu = np.load("robot_trajectory/gtsam_optimized/dataset21_gtsam_imu_icp_icp.npy")
        tr21_gtsam_imu = state_matrix_to_pose_matrix(tr21_gtsam_imu)
        x_tr21_gtsam_imu, y_tr21_gtsam_imu = transform_pose_matrix_to_xy(tr21_gtsam_imu)

        fig, axs = plt.subplots(2, 2, figsize=(36, 24))
        axs[0, 0].plot(x_tr20, y_tr20, color="blue", label="IMU Odometry (Dataset 20)")
        axs[0, 0].plot(x_tr20_lid, y_tr20_lid, color="orange", label="Lidar Scan Matching (Dataset 20)")
        axs[0, 0].plot(x_tr20_gtsam_icp, y_tr20_gtsam_icp, color="green", label="GTSAM Optimized Trajectory (Dataset 20)")
        axs[0, 0].set_title("Factor Graph Optimized Trajectory Dataset 20 (Nodes: icp, edge: imu, Loop: icp)")
        # axs[0, 0].set_xlabel("X Position (m)")
        axs[0, 0].set_ylabel("Y Position (m)")
        axs[0, 0].legend(fontsize=6) 
        axs[0, 0].grid()

        axs[0, 1].plot(x_tr21, y_tr21, color="blue", label="IMU Odometry (Dataset 21)")
        axs[0, 1].plot(x_tr21_lid, y_tr21_lid, color="orange", label="Lidar Scan Matching (Dataset 21)")
        axs[0, 1].plot(x_tr21_gtsam_icp, y_tr21_gtsam_icp, color="green", label="GTSAM Optimized Trajectory (Dataset 21)")
        axs[0, 1].set_title("Factor Graph Optimized Trajectory Dataset 21 (Nodes: icp, edge: imu, Loop: icp)")
        # axs[0, 1].set_xlabel("X Position (m)")
        axs[0, 1].set_ylabel("Y Position (m)")
        axs[0, 1].legend(fontsize=6) 
        axs[0, 1].grid()

        axs[1, 0].plot(x_tr20, y_tr20, color="blue", label="IMU Odometry (Dataset 20)")
        axs[1, 0].plot(x_tr20_lid, y_tr20_lid, color="orange", label="Lidar Scan Matching (Dataset 20)")
        axs[1, 0].plot(x_tr20_gtsam_imu, y_tr20_gtsam_imu, color="green", label="GTSAM Optimized Trajectory (Dataset 20)")
        axs[1, 0].set_title("Factor Graph Optimized Trajectory Dataset 20 (Nodes: imu, edge: icp, Loop: icp)")
        axs[1, 0].set_xlabel("X Position (m)")
        axs[1, 0].set_ylabel("Y Position (m)")
        axs[1, 0].legend(fontsize=6)
        axs[1, 0].grid()

        axs[1, 1].plot(x_tr21, y_tr21, color="blue", label="IMU Odometry (Dataset 21)")
        axs[1, 1].plot(x_tr21_lid, y_tr21_lid, color="orange", label="Lidar Scan Matching (Dataset 21)")
        axs[1, 1].plot(x_tr21_gtsam_imu, y_tr21_gtsam_imu, color="green", label="GTSAM Optimized Trajectory (Dataset 21)")
        axs[1, 1].set_title("Factor Graph Optimized Trajectory Dataset 21 (Nodes: imu, edge: icp, Loop: icp)")
        axs[1, 1].set_xlabel("X Position (m)")
        axs[1, 1].set_ylabel("Y Position (m)")
        axs[1, 1].legend(fontsize=6)
        axs[1, 1].grid()
        # plt.tight_layout()
        # plt.savefig("images/GTSAM_Optimized_Trajectory.png")
        plt.show()
    else:
        # GTSAM no loop closure
        # imu icp
        tr20_gtsam_icp_noloop = np.load("robot_trajectory/gtsam_optimized/dataset20_gtsam_no_loop.npy")
        tr20_gtsam_icp_noloop = state_matrix_to_pose_matrix(tr20_gtsam_icp_noloop)
        x_tr20_gtsam_icp, y_tr20_gtsam_icp = transform_pose_matrix_to_xy(tr20_gtsam_icp_noloop)
        tr21_gtsam_icp_noloop = np.load("robot_trajectory/gtsam_optimized/dataset21_gtsam_no_loop.npy")
        tr21_gtsam_icp_noloop = state_matrix_to_pose_matrix(tr21_gtsam_icp_noloop)
        x_tr21_gtsam_icp, y_tr21_gtsam_icp = transform_pose_matrix_to_xy(tr21_gtsam_icp_noloop)
        fig, axs = plt.subplots(1, 2, figsize=(24, 12))

        axs[0].plot(x_tr20, y_tr20, color="blue", label="IMU Odometry (Dataset 20)")
        axs[0].plot(x_tr20_lid, y_tr20_lid, color="orange", label="Lidar Scan Matching (Dataset 20)")
        axs[0].plot(x_tr20_gtsam_icp, y_tr20_gtsam_icp, color="green", label="GTSAM Optimized Trajectory (Dataset 20)")
        axs[0].set_title("Factor Graph Optimized Trajectory Dataset 20 (Nodes: imu, edge: icp, Loop: None)")
        axs[0].set_xlabel("X Position (m)")
        axs[0].set_ylabel("Y Position (m)")
        axs[0].legend(fontsize=8)
        axs[0].grid()

        axs[1].plot(x_tr21, y_tr21, color="blue", label="IMU Odometry (Dataset 21)")
        axs[1].plot(x_tr21_lid, y_tr21_lid, color="orange", label="Lidar Scan Matching (Dataset 21)")
        axs[1].plot(x_tr21_gtsam_icp, y_tr21_gtsam_icp, color="green", label="GTSAM Optimized Trajectory (Dataset 21)")
        axs[1].set_title("Factor Graph Optimized Trajectory Dataset 21 (Nodes: imu, edge: icp, Loop: None)")
        axs[1].set_xlabel("X Position (m)")
        axs[1].set_ylabel("Y Position (m)")
        axs[1].legend(fontsize=8)
        axs[1].grid()
        # plt.tight_layout()
        # plt.savefig("images/GTSAM_Optimized_Trajectory_no_loop.png")
        plt.show()
    return 0


if __name__ == "__main__":
    imu_results()
    scan_matching_results()
    occupancy_grid_map_results()
    texture_mapping_results()
    gtsam_results()