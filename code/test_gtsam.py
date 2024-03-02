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
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.2, 0.3])
    # pose_key = gtsam.symbol('x', 1)
    prior_factor = gtsam.PriorFactorPose2(1, gtsam.Pose2(0, 0, 0), prior_noise)
    print("Prior factor created:", prior_factor)

    return prior_factor

if __name__ == "__main__":
    # Run basic tests
    pose_from_icp = np.load('results/Estimated_trajectory_dataset20_change_icp_input.npy')
    pose_from_imu = np.load('IMU_Od_POSE.npy')
    prior_noise = create_prior()
    # Create a factor graph
    graph = gtsam.NonlinearFactorGraph()
    graph.add(prior_noise)
    # For simplicity, we will use the same noise model for odometry and loop closures
    noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.1])
    loop_model = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.05])
    # Create a 2D pose with x, y, and theta (rotation)
    lidar_x, lidar_y, lidar_theta = transform_pose_matrix_to_xy(pose_from_icp,need_theata=True)
    imu_x, imu_y, imu_theta = transform_pose_matrix_to_xy(pose_from_imu,need_theata=True)
    initial_estimate = intial_estimation(lidar_x, imu_x, imu_y, imu_theta)
    # initial_estimate = intial_estimation(lidar_x, lidar_x, lidar_y, lidar_theta)

    relative_pose_icp = generate_relative_pose_matrix_normal_convention(pose_from_icp)
    rel_x, rel_y, rel_theta = transform_pose_matrix_to_xy(relative_pose_icp,need_theata=True)
    # Constructing edges
    for j in range(len(relative_pose_icp)):
        graph.add(gtsam.BetweenFactorPose2(j, j+1, gtsam.Pose2(rel_x[j], rel_y[j], rel_theta[j]), noise_model))
    # loop closure
    loop_interval = 10
    interval_posex = []
    interval_posey = []
    for i in range(int((len(pose_from_icp)-len(pose_from_icp)%loop_interval)/loop_interval)):
        rel_pose = np.linalg.inv(pose_from_icp[i+loop_interval]) @ pose_from_icp[i]
        # rel_pose = np.linalg.inv(pose_from_icp[i]) @ pose_from_icp[i+loop_interval]
        init_pose = homegenous_transformation(np.eye(3),np.zeros(3))
        position = rel_pose @ init_pose 
        rel_lp_x = position[:3, 3][0]
        rel_lp_y = position[:3, 3][1]
        interval_posex.append(rel_lp_x)
        interval_posey.append(rel_lp_y)
        r = Rotation.from_matrix(position[:3, :3])
        rel_lp_theta = r.as_euler('zyx', degrees=False)
        # pdb.set_trace()
        graph.add(gtsam.BetweenFactorPose2(i+loop_interval, i, gtsam.Pose2(rel_lp_x, rel_lp_y, rel_lp_theta[0]), loop_model))
        # graph.add(gtsam.BetweenFactorPose2(i, i+loop_interval, gtsam.Pose2(rel_lp_x, rel_lp_y, rel_lp_theta[0]), loop_model))
    # graph.add(gtsam.BetweenFactorPose2(0, len(lidar_x)-1, gtsam.Pose2(0, 0, 0), noise_model))
    param = gtsam.LevenbergMarquardtParams()
    # param = gtsam.GaussNewtonParams()
    param.setVerbosity("Termination")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate,param)
    # optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate,param)
    result = optimizer.optimize()
    # Take the values out
    x = [result.atPose2(i).x() for i in range(result.size())]
    y = [result.atPose2(i).y() for i in range(result.size())]
    pdb.set_trace()
    print("initial error: ",graph.error(initial_estimate))
    print("Final error: ",graph.error(result))
    plt.figure(figsize=(15,15))
    plt.plot(lidar_x, lidar_y,'x',label="Lidar",linewidth=1)
    plt.plot(imu_x, imu_y, '.g',label="IMU",linewidth=2.0)
    plt.plot(x, y, '*r',label="GTSAM",linewidth=0.1)
    # plt.plot(interval_posex, interval_posey, 'o',label="Loop Closure",linewidth=1.0)
    # plt.plot(rel_x, rel_y, 'x',label="ICP",linewidth=1.0)
    plt.legend()
    plt.savefig('wsl_results/loop_closure_gtsam.png')
    # result.print("Final Result:\n")