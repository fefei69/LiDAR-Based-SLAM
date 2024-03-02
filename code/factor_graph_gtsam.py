#****** Test Script to test GTSAM - python installation ******#
import gtsam
import numpy as np
import matplotlib.pyplot as plt
from encoder_imu_odometry import *

def intial_estimation(lidar_x, imu_x, imu_y, imu_theta):
    # Constructing nodes
    initial_estimate = gtsam.Values()
    # Create a 2D pose with x, y, and theta (rotation)
    for i in range(len(lidar_x)):
    #    initial_estimate.insert(i, gtsam.Pose2(lidar_x[i], lidar_y[i], lidar_theta[i]))
       initial_estimate.insert(i, gtsam.Pose2(imu_x[i], imu_y[i], imu_theta[i]))
    return initial_estimate

def create_prior():
    # Create a prior factor on a Pose2
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas([1e-6, 1e-6, 1e-4])
    # pose_key = gtsam.symbol('x', 1)
    prior_factor = gtsam.PriorFactorPose2(1, gtsam.Pose2(0, 0, 0), prior_noise)
    print("Prior factor created:", prior_factor)
    return prior_factor

if __name__ == "__main__":
    # Run basic tests
    pose_from_icp = np.load(f'results/Estimated_trajectory_dataset{dataset}_change_icp_input.npy')
    pose_from_imu = np.load('IMU_Od_POSE.npy')
    prior_noise = create_prior()
    building_nodes_w_icp = True
    building_edges_w_icp = False
    loop_closure = True
    gauss_newton_optimizer = False
    # Create a factor graph
    graph = gtsam.NonlinearFactorGraph()
    graph.add(prior_noise)
    # For simplicity, we will use the same noise model for odometry and loop closures
    noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.5])
    loop_model = gtsam.noiseModel.Diagonal.Sigmas([0.01, 0.01, 0.02])
    lidar_x, lidar_y, lidar_theta = transform_pose_matrix_to_xy(pose_from_icp,need_theata=True)
    imu_x, imu_y, imu_theta = transform_pose_matrix_to_xy(pose_from_imu,need_theata=True)
    if building_nodes_w_icp == True:
        # Create a 2D pose with x, y, and theta (rotation)
        initial_estimate = intial_estimation(lidar_x, lidar_x, lidar_y, lidar_theta)
    else:
        initial_estimate = intial_estimation(lidar_x, imu_x, imu_y, imu_theta)
    if building_edges_w_icp == True:
        # building edges with icp
        relative_pose_icp = generate_relative_pose_matrix_normal_convention(pose_from_icp)
        rel_x, rel_y, rel_theta = transform_pose_matrix_to_xy(relative_pose_icp,need_theata=True)
        # Constructing edges
        for j in range(len(relative_pose_icp)):
            graph.add(gtsam.BetweenFactorPose2(j, j+1, gtsam.Pose2(rel_x[j], rel_y[j], rel_theta[j]), noise_model))
    else:
        # building edges with imu
        relative_pose_imu = generate_relative_pose_matrix_normal_convention(pose_from_imu)
        rel_x, rel_y, rel_theta = transform_pose_matrix_to_xy(relative_pose_imu,need_theata=True)
        # Constructing edges
        for j in range(len(relative_pose_imu)):
            graph.add(gtsam.BetweenFactorPose2(j, j+1, gtsam.Pose2(rel_x[j], rel_y[j], rel_theta[j]), noise_model))
    # loop closure
    if loop_closure == True:
        loop_interval = 2
        for i in range(int((len(pose_from_icp)-len(pose_from_icp)%loop_interval)/loop_interval)):
            rel_pose = np.linalg.inv(pose_from_icp[i+loop_interval]) @ pose_from_icp[i]
            # rel_pose = np.linalg.inv(pose_from_icp[i]) @ pose_from_icp[i+loop_interval]
            init_pose = homegenous_transformation(np.eye(3),np.zeros(3))
            position = rel_pose @ init_pose 
            rel_lp_x = position[:3, 3][0]
            rel_lp_y = position[:3, 3][1]
            r = Rotation.from_matrix(position[:3, :3])
            rel_lp_theta = r.as_euler('zyx', degrees=False)
            # pdb.set_trace()
            graph.add(gtsam.BetweenFactorPose2(i+loop_interval, i, gtsam.Pose2(rel_lp_x, rel_lp_y, rel_lp_theta[0]), loop_model))
            # Forward loop closure
            # graph.add(gtsam.BetweenFactorPose2(i, i+loop_interval, gtsam.Pose2(rel_lp_x, rel_lp_y, rel_lp_theta[0]), loop_model))
    # Create the optimizer
    if gauss_newton_optimizer == True:
        param = gtsam.GaussNewtonParams()
        param.setVerbosity("Termination")
        optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate,param)
    else:
        param = gtsam.LevenbergMarquardtParams()
        param.setVerbosity("Termination")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate,param)
    result = optimizer.optimize()
    # Take the values out
    x = [result.atPose2(i).x() for i in range(result.size())]
    y = [result.atPose2(i).y() for i in range(result.size())]
    theta = [result.atPose2(i).theta() for i in range(result.size())]
    np.save('results/dataset20_gtsam_imu_icp_icp.npy',np.array([x,y,theta,np.ones(len(x))]).T)
    print("initial error: ",graph.error(initial_estimate))
    print("Final error: ",graph.error(result))
    plt.figure(figsize=(20,15))
    # plt.figure(dpi=100) 
    plt.plot(lidar_x, lidar_y,'x',label="Lidar",linewidth=.1)
    plt.plot(imu_x, imu_y, '.g',label="IMU",linewidth=0.1)
    plt.plot(x, y, '*r',label="GTSAM",linewidth=0.1)
    plt.grid()
    plt.xlabel("X Postiton",fontsize=20)
    plt.ylabel("Y Position",fontsize=20)
    plt.title("Factor Graph Optimization with GTSAM",fontsize=20)
    plt.tight_layout()
    plt.legend(fontsize=20)
    # plt.savefig('wsl_results/loop_closure_gtsam_icp_imu_icp.png')