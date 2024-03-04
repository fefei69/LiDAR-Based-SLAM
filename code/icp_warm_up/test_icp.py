
import numpy as np
from utils import read_canonical_model, load_pc, data_association, visualize_icp_result, icp, homegenous_transformation,Kabsch_Algorithm
from scipy.spatial.transform import Rotation
import open3d as o3d
import copy
if __name__ == "__main__":
    obj_name = 'drill' # drill or liq_container
    num_pc = 4 # number of point clouds
    down_sample_rate = 10
    source_pc = read_canonical_model(obj_name)
    source_pc_orig = source_pc.copy()
    zero_center_source = source_pc_orig - np.mean(source_pc_orig,axis=0,keepdims=True)
    # A point at origin for testing
    zero = np.zeros((1,3))
    # print(source_pc_mean)
    for i in range(num_pc):
        target_pc = load_pc(obj_name, i)
        target_pc_orig = target_pc.copy()
        zero_center_target = target_pc_orig - np.mean(target_pc_orig,axis=0,keepdims=True)
        diff_center = np.mean(source_pc,axis=0,keepdims=True) - np.mean(target_pc,axis=0,keepdims=True)
        target = target_pc + diff_center
        optimal_pc, T = icp(source_pc, target_pc, down_sample_rate, obj_name,data_num=i,)
        # visualize the estimated result : transform the source_pc to be aligned with the target_pc
        # when transforming source: Rotation: R.T   Translation: -T
        # when transforming target: Rotation: R Translation: +T
        visualize_icp_result(source_pc_orig, target_pc_orig,T,i)

