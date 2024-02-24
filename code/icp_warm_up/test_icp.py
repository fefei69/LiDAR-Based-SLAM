
import numpy as np
from utils import read_canonical_model, load_pc, data_association, visualize_icp_result, icp, homegenous_transformation
from scipy.spatial.transform import Rotation
if __name__ == "__main__":
    obj_name = 'drill' # drill or liq_container
    num_pc = 1 # number of point clouds
    down_sample_rate = 1
    source_pc = read_canonical_model(obj_name)
    # A point at origin for testing
    zero = np.zeros((1,3))
    # print(source_pc_mean)
    for i in range(num_pc):
        target_pc = load_pc(obj_name, 3)
        diff_center = np.mean(source_pc,axis=0,keepdims=True) - np.mean(target_pc,axis=0,keepdims=True)
        target = target_pc + diff_center
        optimal_pc, pose = icp(source_pc, target_pc, down_sample_rate, data_num=3)
        # Rot = Rotation.from_euler('z',-1).as_matrix()
        # R_optimal = target @ Rot.T
        # optimal_target_pc = target @ R_optimal.T
        # estimated_pose, you need to estimate the pose with ICP
        # pose = np.eye(4)
        # target = target_pc @ pose.T
        # T = homegenous_transformation(pose,np.zeros((3,)))
        # print(T)
        print(pose)
        # visualize the estimated result
        visualize_icp_result(source_pc, target, pose)

